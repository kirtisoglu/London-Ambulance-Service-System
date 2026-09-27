"""
Build the LAS benchmark / initial state ``s_LAS`` (Implementation.md §6.1).

Pipeline:
  1. Load `data/raw/london_graph.json` and wrap as falcomchain.Graph.
  2. Set the proxy travel-time table on Assignment.travel_times.
  3. Verify Assumption 6.1 at LAS settings; repair with `fast_center`
     if it fails.
  4. Build a contiguous super_assignment from
     `data/derived/lsoa_to_sector_voronoi_contiguous.csv`.
  5. Call `Partition.from_random_assignment(super_assignment=...)` to
     produce one (P¹, P²) state with the LAS sectors fixed.
  6. Wrap as ChainState with `compute_energy +
     compute_artificial_facility_penalty` as energy_fn.
  7. Pickle the resulting ChainState to
     `data/derived/s_LAS.pkl` and print a summary.

This script materialises *one* feasible state with the LAS sector
layout. It is the seed for `chain_fixed`/`chain_variable` and the
reference comparator for ensemble percentile claims.

Run:
    python -m falcomchain_experiments.las.build_s_las
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from falcomchain.graph import Graph
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.candidates.feasibility import (
    check_facility_density,
    repair_facility_density,
)
from falcomchain.markovchain.state import ChainState
from falcomchain.markovchain.energy import (
    compute_energy,
    compute_artificial_facility_penalty,
)
from falcomchain.markovchain.facility import SuperFacilityAssignment

from .proxy_travel import ProxyTravelTimes

# --- Repository paths -----------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO / "data/raw/london_graph.json"
# v2 benchmark: LSOA → sector by NHS-ICS / LAS borough membership.
# Replaces the Voronoi approximation (see analysis/12_data_and_design_v2.md).
SECTOR_CSV = REPO / "data/derived/lsoa_to_sector_borough.csv"
# v2 L2 candidates: 21 Group HQs + 2 EOCs (from LAS-List-Of-Departments-2025
# and LAS website).  Replaces the 5 "geographically central station" proxies.
L2_FACILITIES_CSV = REPO / "data/raw/LAS_L2_facilities.csv"
OUT_GRAPH_PKL = REPO / "data/derived/s_LAS_graph.pkl"
OUT_STATE_JSON = REPO / "data/derived/s_LAS_state.json"

# --- LAS parameters --------------------------------------------------
# Capacity calibration v3 — derived from the LAS-published per-crew
# productivity target rather than from fleet-inventory arithmetic.
# LAS Annual Report and Accounts 2024/25 (p. 22, p. 45) reports that
# the Patients Per Shift quality-improvement initiative achieved a
# sustained operational target of 5.2 patients per crew per 12-hour
# shift in Q3 of 2024/25, with a Q4 trajectory of 5.4.  Annualised
# across the standard 365 single-shift-per-team rota:
#   w = 5.2 patients/shift × 365 shifts/team/year ≈ 1,898 calls/team/year
# This matches the LAS 2024/25 throughput of 1,084,922 patients seen
# by face-to-face crews (1,084,922 / 1,898 ≈ 572 team-shifts/day,
# consistent with the on-shift fleet under a two-shifts-per-vehicle
# operating pattern; the per-Group DCA inventory of 583 cross-checks
# via the 0.706 inventory-to-on-shift ratio of Coates & McCormack 2015).
# Per-Group DCA counts remain in
# ``data/derived/las_fleet_per_group.csv`` (columns n_dca, n_dca_op,
# n_fru, n_mru, n_app); they are no longer on the critical path for
# the energy since FalCom's two energies (E_minisum, E_fair) do not
# use capacity, but they support the §7.4 capacity-vs-demand-imbalance
# diagnostic reported at the super-district level.
DEMAND_TARGET = 1898      # patients per team per year (5.2/shift × 365)
EPSILON_BASE = 0.30       # ε¹  (relaxed; tighten to 0.15 once per-sector lands)
EPSILON_SUPER = 0.30      # ε²
# ====================================================================
# TODO(owner-decision): per-district capacity bounds [C_MIN, C_MAX]
# ====================================================================
# Real per-Group on-shift range is 7–30 DCAs (mean 20).  Three candidate
# bands, listed in falcom.tex Section 5A under \DESIGN{...}; final choice
# deferred to a video-call design review with the paper owner.  The
# values below are a tight default that keeps the recursive partitioner
# fast (|P¹| ∈ [19, 57] for w=2579); they should be revisited after the
# design call before committing the chain runs.
# ====================================================================
C_MIN = 15                # placeholder — see TODO above
C_MAX = 25                # placeholder — see TODO above
LAMBDA_PEN = 10_000.0     # penalty weight on artificial-facility usage
PROXY_KMH = 25.0          # assumed average urban drive speed (proxy)


def load_graph() -> Graph:
    """Load `london_graph.json` and wrap as falcomchain.Graph."""
    with open(GRAPH_PATH) as f:
        raw = json.load(f)
    g = nx.Graph()
    for n in raw["nodes"]:
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for src_idx, neighbors in enumerate(raw["adjacency"]):
        for nb in neighbors:
            g.add_edge(src_idx, nb["id"], **{k: v for k, v in nb.items() if k != "id"})
    return Graph.from_networkx(g)


def load_super_assignment(graph: Graph) -> dict[int, str]:
    """Read the borough-derived sector CSV and key it by node id.

    LSOAs marked ``Outside LAS`` (the 48 Brentwood-Essex nodes) are
    assigned to the closest London sector ``South East``, matching
    the existing demand-imputation convention (analysis/10).
    """
    code_to_id = {graph.nodes[n]["LSOA21CD"]: n for n in graph.nodes}
    df = pd.read_csv(SECTOR_CSV)
    df["sector"] = df["sector"].replace("Outside LAS", "South East")
    return {code_to_id[c]: s for c, s in zip(df["LSOA21CD"], df["sector"])}


def install_l2_candidates(graph: Graph) -> int:
    """Overwrite ``candidate_l2`` flags on the graph from the v2 L2 CSV.

    EOC rows (``facility_type == 'eoc'``) are excluded — under the
    dynamic-nested L2 design they are NOT L2 facility candidates
    (they're LAS-wide dispatch infrastructure, not Group-scoped
    coordination sites).  Only the 21 Group HQ rows are installed
    as L2 candidates.  EOCs remain in the CSV so map renderers can
    still draw them as documented infrastructure.
    Returns the number of nodes flagged as L2 candidates.
    """
    df = pd.read_csv(L2_FACILITIES_CSV)
    df = df[df["facility_type"] != "eoc"]
    code_to_id = {graph.nodes[n]["LSOA21CD"]: n for n in graph.nodes}
    target_nodes = {code_to_id[c] for c in df["LSOA21CD"] if c in code_to_id}
    # Reset and reapply.
    for n in graph.nodes:
        graph.nodes[n]["candidate_l2"] = 0
    for n in target_nodes:
        graph.nodes[n]["candidate_l2"] = 1
    return len(target_nodes)


def energy_with_artificial_penalty(state) -> float:
    return (
        compute_energy(state)
        + compute_artificial_facility_penalty(state, lambda_pen=LAMBDA_PEN)
    )


def super_facility_fn_factory(graph, nested: bool = False):
    """Return a `super_facility_fn(state)` that assigns each
    superdistrict to whichever of $F^2$ minimises the maximum travel
    time to any base node in the superdistrict.

    The ``nested`` flag selects the candidate set.  It defaults to
    ``False`` so every prior experiment that calls this factory
    without the keyword (compare, gamma_sweep, workload_sensitivity,
    pilot, …) sees no behavioural change.

    - ``nested=False`` (DEFAULT).  $F^2$ is the static set of nodes
      flagged ``candidate_l2 == 1`` on the graph, searched globally
      over all of them for every superdistrict.  Bypasses
      :meth:`SuperFacilityAssignment.from_state`'s hardcoded
      "no super-candidate inside" short-circuit, which fits dense
      $F^2$ sets but leaves sparse LAS-style layouts unassigned.

    - ``nested=True`` (LAS v3 design).  Constrain
      $F^2(s) = \\{f^1(D) : D \\in \\Pc^1\\}$ — i.e. each
      superdistrict's L2 facility must be one of the L1 facilities the
      chain has actually opened inside that superdistrict.  An LSOA
      node thus plays double duty when chosen at both levels: L1
      catchment facility and L2 group HQ for the same node.
      Operationally matches LAS: a Group HQ is always a real
      dispatching station inside the Group.  EOCs and other
      non-dispatching $F^2$ candidates are ignored under this mode.
    """
    f2 = tuple(n for n in graph.nodes if graph.nodes[n].get("candidate_l2") == 1)

    def _super_facility_fn_static(state):
        sfa = SuperFacilityAssignment()
        partition = state.partition
        travel_times = state.assignment.travel_times
        if travel_times is None:
            return sfa
        for super_id, l1_ids in partition.super_parts.items():
            base_nodes = set()
            for l1_id in l1_ids:
                if l1_id in partition.parts:
                    base_nodes |= partition.parts[l1_id]
            if not base_nodes:
                continue
            best_node = None
            best_radius = float("inf")
            for c in f2:
                try:
                    r = max(travel_times[(c, v)] for v in base_nodes)
                except KeyError:
                    continue
                if r < best_radius:
                    best_radius = r
                    best_node = c
            if best_node is not None:
                sfa._centers[super_id] = best_node
                sfa._radii[super_id] = best_radius
        return sfa

    def _super_facility_fn_open_l1(state):
        """L2 facility for each superdistrict = the open L1 facility
        inside it that minimises the maximum travel time to its base
        nodes.  Falls through gracefully (no assignment for that
        superdistrict) when the chain hasn't yet opened any L1 facility
        in it — happens only mid-bootstrap, never in a converged state.
        """
        sfa = SuperFacilityAssignment()
        partition = state.partition
        facility = state.facility
        travel_times = state.assignment.travel_times
        if travel_times is None or facility is None:
            return sfa
        l1_centers = facility.centers  # district_id -> open L1 facility node
        for super_id, l1_ids in partition.super_parts.items():
            # candidates: open L1 facilities of the L1 districts that
            # belong to this superdistrict.
            cands = [l1_centers.get(d) for d in l1_ids if l1_centers.get(d) is not None]
            if not cands:
                continue
            base_nodes = set()
            for l1_id in l1_ids:
                if l1_id in partition.parts:
                    base_nodes |= partition.parts[l1_id]
            if not base_nodes:
                continue
            best_node = None
            best_radius = float("inf")
            for c in cands:
                try:
                    r = max(travel_times[(c, v)] for v in base_nodes)
                except KeyError:
                    continue
                if r < best_radius:
                    best_radius = r
                    best_node = c
            if best_node is not None:
                sfa._centers[super_id] = best_node
                sfa._radii[super_id] = best_radius
        return sfa

    fn = _super_facility_fn_open_l1 if nested else _super_facility_fn_static
    return fn, f2


def build(
    verbose: bool = True,
    *,
    travel_times=None,
    epsilon_base: float | None = None,
    epsilon_super: float | None = None,
    run_repair: bool = True,
    c_max: int | None = None,
    fix_super_assignment: bool = False,
    graph=None,
    nested_super_facilities: bool = False,
) -> tuple[ChainState, Graph, dict[int, str], dict]:
    """Build s_LAS in one process. Returns (state, graph, voronoi_sector,
    summary). The caller may consume the state directly (e.g. as the
    initial state of a smoke chain) or persist the assignment dicts via
    `main()` for downstream analysis.

    By default ``Assignment.travel_times`` is set to a fresh
    ``ProxyTravelTimes`` and the module-level ``EPSILON_*`` constants
    are used. Pass keyword overrides to swap the travel-time table
    (e.g. ``RealTravelTimes`` over the OSMnx matrix) or to tighten /
    loosen the demand-balance tolerance band for pilot experiments.
    """
    eps_base = EPSILON_BASE if epsilon_base is None else float(epsilon_base)
    eps_super = EPSILON_SUPER if epsilon_super is None else float(epsilon_super)
    c_max_eff = C_MAX if c_max is None else int(c_max)

    log = print if verbose else (lambda *a, **k: None)
    if graph is None:
        log(f"[1/6] Load graph from {GRAPH_PATH.relative_to(REPO)}")
        t0 = time.perf_counter()
        g = load_graph()
        log(f"      graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
            f"({time.perf_counter() - t0:.2f}s)")
    else:
        g = graph
        n_real = sum(
            1 for n in g.nodes
            if g.nodes[n].get("candidate", 0) == 1
            and not g.nodes[n].get("candidate_artificial", 0)
        )
        n_art = sum(1 for n in g.nodes if g.nodes[n].get("candidate_artificial", 0))
        log(f"[1/6] Using pre-loaded graph: {g.number_of_nodes()} nodes, "
            f"{g.number_of_edges()} edges; "
            f"{n_real} real + {n_art} artificial candidates pre-installed")

    if travel_times is None:
        log(f"[2/6] Install proxy travel-time table (kmh={PROXY_KMH})")
        Assignment.travel_times = ProxyTravelTimes(g, kmh=PROXY_KMH, cache=True)
    else:
        log(f"[2/6] Install caller-supplied travel-time table "
            f"({type(travel_times).__name__})")
        Assignment.travel_times = travel_times

    log(f"[2b/6] Install v2 L2 candidate set (21 Group HQs + 2 EOCs)")
    n_l2 = install_l2_candidates(g)
    log(f"        |F²| = {n_l2} flagged on graph")

    log(f"[3/6] Borough-derived sector lookup (benchmark s_LAS)")
    voronoi_sector = load_super_assignment(g)

    if run_repair:
        log(f"[4/6] Repair Assumption 6.1 globally via fast_center "
            f"(w={DEMAND_TARGET}, ε={eps_base}, c_min={C_MIN})")
        t0 = time.perf_counter()
        rep_before = check_facility_density(
            g, demand_target=DEMAND_TARGET, epsilon=eps_base, c_min=C_MIN,
        )
        if not rep_before.passes:
            added = repair_facility_density(
                g, demand_target=DEMAND_TARGET, epsilon=eps_base,
                c_min=C_MIN, strategy="fast_center",
            )
            log(f"      added {len(added)} artificial candidates "
                f"({time.perf_counter() - t0:.1f}s)")
    else:
        log(f"[4/6] Skipping repair_facility_density "
            f"(run_repair=False; chain runs on REAL candidates only)")
        t0 = time.perf_counter()
        rep_before = check_facility_density(
            g, demand_target=DEMAND_TARGET, epsilon=eps_base, c_min=C_MIN,
        )
        log(f"      Assumption 6.1 check: "
            f"{'PASSES' if rep_before.passes else 'FAILS'} "
            f"(eps={eps_base}, c_min={C_MIN})  "
            f"({time.perf_counter() - t0:.1f}s)")

    if fix_super_assignment:
        log(f"[5/6] Build Partition with fixed Voronoi sector super-assignment "
            f"(L1 random within each of 5 sectors, c_max={c_max_eff})")
        t0 = time.perf_counter()
        partition = Partition.from_random_assignment(
            graph=g, epsilon=eps_base, demand_target=DEMAND_TARGET,
            assignment_class=Assignment, capacity_level=c_max_eff, c_min=C_MIN,
            super_assignment=voronoi_sector,
        )
    else:
        log(f"[5/6] Build Partition globally; init_super_partition derives L2 "
            f"(c_max={c_max_eff})")
        t0 = time.perf_counter()
        partition = Partition.from_random_assignment(
            graph=g, epsilon=eps_base, demand_target=DEMAND_TARGET,
            assignment_class=Assignment, capacity_level=c_max_eff, c_min=C_MIN,
            init_super_partition=True, epsilon_super=eps_super,
        )
    log(f"      |P¹|={len(partition.parts)}  "
        f"|P²|={len(set(partition.super_assignment.values()))}  "
        f"({time.perf_counter() - t0:.1f}s)")

    log(f"[6/6] Wrap as ChainState; compute E(s_LAS) and L2 facilities")
    super_facility_fn, f2_nodes = super_facility_fn_factory(
        g, nested=nested_super_facilities,
    )
    log(f"      |F²| = {len(f2_nodes)} (super-facility candidates, picked "
        f"globally by minimax travel time per superdistrict)")
    state = ChainState.initial(
        partition=partition, energy=0.0, beta=1.0,
        energy_fn=energy_with_artificial_penalty,
        super_facility_fn=super_facility_fn,
    )
    e_pure = compute_energy(state)
    e_pen = compute_artificial_facility_penalty(state, lambda_pen=LAMBDA_PEN)
    n_art_used = sum(
        1 for c in state.facility.centers.values()
        if c is not None and g.nodes[c].get("candidate_artificial", 0)
    )
    n_real_used = sum(
        1 for c in state.facility.centers.values()
        if c is not None and not g.nodes[c].get("candidate_artificial", 0)
    )
    # Tally L2 facility usage across the 7 super-candidates.
    super_centers = state.super_facility.centers if state.super_facility else {}
    f2_usage = {f: 0 for f in f2_nodes}
    for f in super_centers.values():
        if f in f2_usage:
            f2_usage[f] += 1
    summary = {
        "n_districts_l1": len(partition.parts),
        "n_super_l2": len(set(partition.super_assignment.values())),
        "teams_total": sum(partition.teams.values()),
        "energy_total": state.energy,
        "energy_pure": e_pure,
        "energy_penalty": e_pen,
        "n_real_used": n_real_used,
        "n_artificial_used": n_art_used,
        "f2_count": len(f2_nodes),
        "f2_usage_at_init": {int(k): int(v) for k, v in f2_usage.items()},
    }
    log(f"      E(s_LAS) = {state.energy:,.0f}  "
        f"(pure={e_pure:,.0f}, penalty={e_pen:,.0f})")
    log(f"      L1 centres: {n_real_used} real / {n_art_used} artificial "
        f"of {len(partition.parts)} districts")
    log(f"      L2 super-facility usage at init (out of |F²|={len(f2_nodes)}):")
    for f, count in sorted(f2_usage.items(), key=lambda kv: -kv[1]):
        log(f"        node {f:5d}  {g.nodes[f].get('LSOA21NM',''):24s}  "
            f"used by {count} superdistrict(s)")
    return state, g, voronoi_sector, summary


def main() -> int:
    state, g, voronoi_sector, summary = build(verbose=True)
    partition = state.partition

    print(f"[7/7] Persist graph + assignment dicts")
    OUT_GRAPH_PKL.parent.mkdir(parents=True, exist_ok=True)
    # Pickle the *graph* alone (networkx-only, no chain refs) ...
    with open(OUT_GRAPH_PKL, "wb") as f:
        pickle.dump(g, f)
    # ... and persist the rest as plain JSON so reload is dependency-free.
    state_blob = {
        "params": {
            "demand_target": DEMAND_TARGET,
            "epsilon_base": EPSILON_BASE,
            "epsilon_super": EPSILON_SUPER,
            "c_min": C_MIN,
            "c_max": C_MAX,
            "lambda_pen": LAMBDA_PEN,
            "proxy_kmh": PROXY_KMH,
        },
        # node id -> level-1 district id
        "l1_assignment": {
            int(node): int(part)
            for part, nodes in partition.parts.items()
            for node in nodes
        },
        # level-1 district id -> teams (capacity)
        "l1_teams": {int(k): int(v) for k, v in partition.teams.items()},
        # level-1 district id -> level-2 super-district id
        "super_assignment": {
            int(k): int(v) for k, v in partition.super_assignment.items()
        },
        # voronoi-by-station (string sector labels) — kept for the
        # downstream ensemble comparator.
        "voronoi_sector": {int(k): v for k, v in voronoi_sector.items()},
        "summary": summary,
    }
    with open(OUT_STATE_JSON, "w") as f:
        json.dump(state_blob, f)
    print(f"      wrote {OUT_GRAPH_PKL.relative_to(REPO)} "
          f"({OUT_GRAPH_PKL.stat().st_size / 1024:.0f} KB)")
    print(f"      wrote {OUT_STATE_JSON.relative_to(REPO)} "
          f"({OUT_STATE_JSON.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
