"""
LAS chain runner v3.

Loads the CDBA candidate set produced by ``run_cdba.py`` and the
candidate-LSOA travel-time matrix produced by
``compute_cdba_travel_times.py``, then runs the FalCom hierarchical
ReCom chain at the locked capacity-block calibration: the capacity
unit is a block of k = 3 ambulances, w_unit = 10,887 calls/unit/year,
eps^1 = eps^2 = 0.15, c^1 in [1,3] units, c^2 in [2,6] units,
gamma^1 = gamma^2 = 0 (cut edges selected uniformly at random).

Run:
    python -m falcomchain_experiments.las.run_chain_v3 --steps 200            # smoke
    python -m falcomchain_experiments.las.run_chain_v3                        # default 10,000
    python -m falcomchain_experiments.las.run_chain_v3 --seeds 0 1 2 3        # 4-chain ensemble
    python -m falcomchain_experiments.las.run_chain_v3 --optimizer --beta 0.05
"""

from __future__ import annotations

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import networkx as nx
import pandas as pd

import sys
sys.path.insert(0, "/Users/kirtisoglu/GitHub/FalcomChain")

from falcomchain.graph import Graph
from falcomchain.markovchain import MarkovChain
from falcomchain.markovchain.accept import always_accept
from falcomchain.markovchain.energy import (
    compute_energy,
    compute_energy_eccentricity,
    compute_plan_summary,
)
from falcomchain.markovchain.facility import SuperFacilityAssignment
from falcomchain.markovchain.proposals import hierarchical_recom
from falcomchain.markovchain.state import ChainState
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment

from .cdba_travel import CdbaTravelTimes
from .build_s_las import load_graph, install_l2_candidates

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CDBA_CSV = REPO / "data/derived/cdba_candidates_w10887_eps15_cmin1_dmax12.csv"
DEFAULT_TRAVEL = REPO / "data/derived/cdba_travel_times_w10887_eps15_cmin1_dmax12.parquet"
OUT_DIR = REPO / "data/derived/chain_v3"

# Capacity-block calibration (locked 2026-07-02): the capacity unit is a
# block of k = 3 ambulances. Derivation in Section 7.4 of the paper:
# safe capacity range wants c_max <= 3, granularity floor wants
# eps * w_unit >= 1,526 (max LSOA demand), window disjointness wants
# eps < 1/(2*c_max - 1) = 1/5; jointly feasible only for k >= 3.
# w_unit tuned so that n_units = ceil(total/w) = 97 with terminal debt 47.
W = 10887                          # calls per unit (3 ambulances) per year
EPS_BASE = 0.15
EPS_SUPER = 0.15
C_MIN_BASE = 1
C_MAX_BASE = 3
C_MIN_SUPER = 2 * C_MIN_BASE       # units: 2 * c^1_min = 2
C_MAX_SUPER = 2 * C_MAX_BASE       # units: 2 * c^1_max = 6
MIN_DISTRICTS_SUPER = 2            # kappa^2_min: L1-district count per super
GAMMA_BASE = 0.0
GAMMA_SUPER = 0.0


def install_cdba_candidates(g, cdba_csv: Path) -> tuple[int, int]:
    """Mark candidates on the graph from the CDBA CSV.
    Returns (n_real, n_artificial)."""
    df = pd.read_csv(cdba_csv)
    code_to_id = {g.nodes[n]["LSOA21CD"]: n for n in g.nodes}
    for n in g.nodes:
        g.nodes[n]["candidate"] = 0
        g.nodes[n]["candidate_artificial"] = 0
    n_real = n_art = 0
    for _, row in df.iterrows():
        nid = code_to_id.get(row["LSOA21CD"])
        if nid is None:
            continue
        g.nodes[nid]["candidate"] = 1
        if int(row["is_real"]) == 0:
            g.nodes[nid]["candidate_artificial"] = 1
            n_art += 1
        else:
            n_real += 1
    return n_real, n_art


def super_facility_open_l1(state):
    """Dynamic-nested L2: each super-district's L2 facility is the open L1
    facility inside it that minimises max travel time to its base nodes."""
    sfa = SuperFacilityAssignment()
    partition = state.partition
    facility = state.facility
    travel_times = state.assignment.travel_times
    if travel_times is None or facility is None:
        return sfa
    l1_centers = facility.centers
    for super_id, l1_ids in partition.super_parts.items():
        cands = [l1_centers.get(d) for d in l1_ids if l1_centers.get(d) is not None]
        if not cands:
            continue
        base_nodes = set()
        for l1_id in l1_ids:
            if l1_id in partition.parts:
                base_nodes |= partition.parts[l1_id]
        if not base_nodes:
            continue
        best_node, best_radius = None, float("inf")
        for c in cands:
            try:
                r = max(travel_times[(c, v)] for v in base_nodes)
            except KeyError:
                continue
            if r < best_radius:
                best_radius, best_node = r, c
        if best_node is not None:
            sfa._centers[super_id] = best_node
            sfa._radii[super_id] = best_radius
    return sfa


def build_initial_state(
    cdba_csv: Path, travel_path: Path, seed: int, *, verbose: bool = True
) -> tuple[ChainState, Graph]:
    log = print if verbose else (lambda *a, **k: None)
    random.seed(seed)

    log(f"[1/5] Load graph and restrict to LAS-served LSOAs")
    g_full = load_graph()
    las_lsoas = set(pd.read_csv(
        REPO / "data/derived/lsoa_to_group.csv"
    )["LSOA21CD"])
    keep = [n for n in g_full.nodes if g_full.nodes[n].get("LSOA21CD") in las_lsoas]
    g = Graph.from_networkx(nx.Graph(g_full.subgraph(keep)))
    log(f"      full: |V|={g_full.number_of_nodes()}, "
        f"LAS subgraph: |V|={g.number_of_nodes()}, |E|={g.number_of_edges()}")

    log(f"[2/5] Install CDBA candidates from {cdba_csv.name}")
    n_real, n_art = install_cdba_candidates(g, cdba_csv)
    log(f"      {n_real} real + {n_art} artificial = {n_real + n_art} L1 candidates")

    log(f"[3/5] Install L2 candidates (21 Group HQs, dynamic-nested)")
    n_l2_static = install_l2_candidates(g)
    log(f"      |F^2_static| = {n_l2_static} (used to seed super-partition only)")

    log(f"[4/5] Install travel times from {travel_path.name}")
    Assignment.travel_times = CdbaTravelTimes(g, matrix_path=travel_path)
    log(f"      {Assignment.travel_times.n_pairs:,} pairs loaded")

    log(f"[5/5] Build initial Partition (seed={seed})")
    t0 = time.perf_counter()
    # Identity level-2 start: the first accepted hierarchical_recom
    # proposal resamples the level-2 partition, so the hierarchy enters
    # the chain dynamics immediately (see paper Section 7.4, Protocol).
    partition = Partition.from_random_assignment(
        graph=g, epsilon=EPS_BASE, demand_target=W,
        assignment_class=Assignment, capacity_level=C_MAX_BASE,
        c_min=C_MIN_BASE,
        init_super_partition=False,
    )
    n_p1 = len(partition.parts)
    n_p2 = len(set(partition.super_assignment.values()))
    teams_total = sum(partition.teams.values())
    log(f"      |P^1|={n_p1}, |P^2|={n_p2}, total teams={teams_total} "
        f"({time.perf_counter()-t0:.1f}s)")

    state = ChainState.initial(
        partition=partition, energy=0.0, beta=0.0,
        energy_fn=compute_energy,
        super_facility_fn=super_facility_open_l1,
    )
    e0 = compute_energy(state)
    log(f"      E_0 = {e0:,.0f}")
    return state, g


def run_one_chain(
    cdba_csv: Path,
    travel_path: Path,
    seed: int,
    total_steps: int,
    *,
    optimizer: bool = False,
    beta: float = 0.05,
    max_districts: int | None = None,
    log_every: int = 100,
    init_retries: int = 30,
    verbose: bool = True,
    snapshots: bool = False,
    snapshot_path: Path | None = None,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    state, g = None, None
    n_init_attempts = 0
    for retry in range(init_retries):
        n_init_attempts += 1
        s = seed * 1000 + retry * 17 + 1
        try:
            state, g = build_initial_state(
                cdba_csv, travel_path, seed=s, verbose=verbose and retry == 0,
            )
            log(f"  initial state built on attempt {n_init_attempts} (seed={s})")
            break
        except Exception as e:
            log(f"  init attempt {retry} (seed={s}) failed: {str(e)[:90]}")
    if state is None:
        raise RuntimeError(f"All {init_retries} init attempts failed for seed={seed}")

    log(f"\n=== Chain run (T={total_steps:,}, "
        f"{'optimizer beta='+str(beta) if optimizer else 'always_accept'}) ===")
    base_proposal = partial(
        hierarchical_recom,
        epsilon_base=EPS_BASE,
        epsilon_super=EPS_SUPER,
        demand_target=W,
        c_min_base=C_MIN_BASE,
        c_min_super=C_MIN_SUPER,
        c_max_super=C_MAX_SUPER,
        min_districts_super=MIN_DISTRICTS_SUPER,
        gamma_base=GAMMA_BASE,
        gamma_super=GAMMA_SUPER,
    )

    def proposal(state):
        """MarkovChain treats RuntimeError as a clean rejection; convert
        any other proposal-internal error (e.g. the rare stranded root
        cut surfacing as IndexError) into RuntimeError so the chain
        holds its state instead of crashing. No library changes."""
        try:
            return base_proposal(state)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"proposal failed: {type(e).__name__}: {e}") from e

    if optimizer:
        from math import exp
        rng = random.Random(seed * 7919 + 13)
        warmup = max(1, total_steps // 10)
        n_calls = [0]

        def boltzmann_accept(proposed, current):
            """Search-run acceptance (implementation detail, not part of the
            paper's protocol): free acceptance during a warm-up phase so the
            chain leaves the degenerate identity-L2 start and establishes a
            real hierarchy, then alpha = exp(-beta * dE) descent on raw
            energy units. MarkovChain calls accept(proposed, current)."""
            n_calls[0] += 1
            if (max_districts is not None
                    and len(proposed.partition.parts) > max_districts):
                return False
            if n_calls[0] <= warmup:
                return True
            delta = proposed.energy - current.energy
            if delta <= 0:
                return True
            return rng.random() < exp(-beta * delta)
        accept = boltzmann_accept
    else:
        accept = always_accept

    chain = MarkovChain(
        proposal=proposal, constraints=[], accept=accept,
        initial_state=state, total_steps=total_steps,
    )
    accepts = [0]
    steps_seen = [0]

    def _acc_cb(_state, accepted):
        steps_seen[0] += 1
        if accepted:
            accepts[0] += 1

    chain.callbacks = [_acc_cb]

    def serialize_state(s):
        """Light assignment blob for snapshots / best-state tracking."""
        p = s.partition
        return {
            "l1_assignment": {int(n): int(part)
                              for part, nodes in p.parts.items()
                              for n in nodes},
            "super_assignment": {int(k): int(v)
                                 for k, v in p.super_assignment.items()},
            "teams": {int(k): int(v) for k, v in p.teams.items()},
            "centers": {int(k): (int(v) if v is not None else None)
                        for k, v in (s.facility.centers if s.facility
                                     else {}).items()},
        }

    rows = []
    snaps = []
    best_e = float("inf")
    best_blob = None
    best_step = 0
    # Matched-facility-count optimum: best valid state opening no more
    # sites than the 66 real LAS stations, for a fair gap against s_LAS.
    best66_e = float("inf")
    best66_blob = None
    best66_step = 0
    t_chain0 = time.perf_counter()
    last_print = t_chain0
    last_state = state
    from collections import Counter as _Counter
    for i, st in enumerate(chain, start=1):
        last_state = st
        if optimizer and st.energy is not None and st.energy < best_e:
            # Only valid hierarchical states are eligible as s*: every
            # super-district must contain >= MIN_DISTRICTS_SUPER districts
            # (excludes the identity-L2 bootstrap state).
            ssizes = _Counter(st.partition.super_assignment.values())
            if ssizes and min(ssizes.values()) >= MIN_DISTRICTS_SUPER:
                best_e = float(st.energy)
                best_blob = serialize_state(st)
                best_step = i
                if len(st.partition.parts) <= 66 and best_e < best66_e:
                    best66_e = best_e
                    best66_blob = best_blob
                    best66_step = i
        elif (optimizer and st.energy is not None
              and st.energy < best66_e
              and len(st.partition.parts) <= 66):
            ssizes = _Counter(st.partition.super_assignment.values())
            if ssizes and min(ssizes.values()) >= MIN_DISTRICTS_SUPER:
                best66_e = float(st.energy)
                best66_blob = serialize_state(st)
                best66_step = i
        if i % log_every == 0 or i == 1 or i == total_steps:
            now = time.perf_counter()
            rate = i / max(1e-9, now - t_chain0)
            n_p1 = len(st.partition.parts)
            n_p2 = len(set(st.partition.super_assignment.values()))
            try:
                e = float(compute_energy(st))
            except Exception:
                e = float("nan")
            n_art_used = sum(
                1 for c in st.facility.centers.values()
                if c is not None and g.nodes[c].get("candidate_artificial", 0)
            )
            rows.append({
                "step": i, "P1": n_p1, "P2": n_p2,
                "E": e, "n_artificial_used": n_art_used,
                "elapsed": now - t_chain0,
            })
            if snapshots and i % log_every == 0:
                blob = serialize_state(st)
                blob["step"] = i
                blob["E"] = e
                snaps.append(blob)
            if now - last_print >= 5 or i in (1, total_steps):
                log(f"  step {i:6,}/{total_steps:,}  "
                    f"|P^1|={n_p1:>3}  |P^2|={n_p2:>2}  "
                    f"E={e:>14,.0f}  art={n_art_used:>3}  "
                    f"({rate:.2f} steps/s)")
                last_print = now

    chain_elapsed = time.perf_counter() - t_chain0
    if snapshots and snapshot_path is not None:
        snapshot_path.write_text(json.dumps(
            {"seed": seed, "log_every": log_every, "snapshots": snaps}))
        log(f"  wrote {snapshot_path}")
    log(f"  done in {chain_elapsed:.1f}s ({total_steps/chain_elapsed:.2f} steps/s)")

    final = compute_plan_summary(
        last_state, coverage_thresholds=(8.0, 15.0),
        demand_target=W, demand_tolerance=EPS_BASE,
    )

    # Persist the final state's full assignment so maps and EMS metrics
    # can be rendered without re-running the chain.
    lp = last_state.partition
    final_state_blob = {
        "l1_assignment": {
            int(node): int(part)
            for part, nodes in lp.parts.items() for node in nodes
        },
        "super_assignment": {
            int(k): int(v) for k, v in lp.super_assignment.items()
        },
        "teams": {int(k): int(v) for k, v in lp.teams.items()},
        "l1_centers": {
            int(k): (int(v) if v is not None else None)
            for k, v in (last_state.facility.centers if last_state.facility
                         else {}).items()
        },
        "l2_centers": {
            int(k): (int(v) if v is not None else None)
            for k, v in (last_state.super_facility.centers
                         if last_state.super_facility else {}).items()
        },
    }

    return {
        "seed": seed,
        "n_init_attempts": n_init_attempts,
        "params": {
            "w": W, "eps_base": EPS_BASE, "eps_super": EPS_SUPER,
            "c_min_base": C_MIN_BASE, "c_max_base": C_MAX_BASE,
            "c_min_super": C_MIN_SUPER, "c_max_super": C_MAX_SUPER,
            "min_districts_super": MIN_DISTRICTS_SUPER,
            "gamma_base": GAMMA_BASE, "gamma_super": GAMMA_SUPER,
            "optimizer": optimizer, "beta": beta if optimizer else None,
        },
        "runtime_seconds": round(chain_elapsed, 2),
        "steps_per_second": round(total_steps / chain_elapsed, 3),
        "acceptance_rate": round(accepts[0] / max(1, steps_seen[0]), 4),
        "n_accepted": accepts[0],
        "final_summary_global": final["global"],
        "final_state": final_state_blob,
        "best_state": ({"E": best_e, "step": best_step, **best_blob}
                       if best_blob is not None else None),
        "best_state_66": ({"E": best66_e, "step": best66_step,
                           **best66_blob}
                          if best66_blob is not None else None),
        "diagnostics": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdba", default=str(DEFAULT_CDBA_CSV))
    ap.add_argument("--travel", default=str(DEFAULT_TRAVEL))
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--optimizer", action="store_true")
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max-districts", type=int, default=None,
                    help="search-run constraint: reject proposals opening "
                         "more than this many districts")
    ap.add_argument("--snapshots", action="store_true",
                    help="save thinned assignment snapshots for ensemble "
                         "post-processing (boundary frequency, stability)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    cdba_csv = Path(args.cdba)
    travel_path = Path(args.travel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        tag = f"opt_b{args.beta}" if args.optimizer else f"sample_s{seed}"
        out = out_dir / f"chain_v3_{tag}_T{args.steps}.json"
        snap_path = out_dir / f"chain_v3_snapshots_{tag}_T{args.steps}.json"
        print(f"\n{'='*72}\n=== Chain seed={seed}, T={args.steps}, "
              f"{'optimizer' if args.optimizer else 'always_accept'} ===\n{'='*72}")
        result = run_one_chain(
            cdba_csv, travel_path, seed=seed,
            total_steps=args.steps,
            optimizer=args.optimizer, beta=args.beta,
            max_districts=args.max_districts,
            snapshots=args.snapshots, snapshot_path=snap_path,
        )
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
