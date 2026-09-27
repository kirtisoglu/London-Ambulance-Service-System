"""
Run FalCom as a greedy optimizer on grid_100 / grid_400:
  - Initial partition via Partition.from_random_assignment (per_team rule).
  - 10,000-step chain with UNIFORM cut selection (psi_fn = 1, no phi or
    psi weighting) and GREEDY acceptance on R^1 + R^2 (accept iff
    new R^1 + R^2 <= current).

Outputs:
  - solution_{N}_falcom_initial.json   MILP-format JSON of the initial
  - solution_{N}_falcom_final.json     MILP-format JSON of the best-seen
  - solution_{N}_falcom_trajectory.json   per-step R^1+R^2 trace
  - solution_{N}_falcom_initial.html   plotly figure (initial)
  - solution_{N}_falcom_final.html     plotly figure (final)
  - solution_{N}_falcom_trajectory.html   plotly line plot of R^1+R^2

Run:
    python -m falcomchain_experiments.gurobi.run_falcom 100
    python -m falcomchain_experiments.gurobi.run_falcom 400
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
from pathlib import Path

# Partition objects keep a `.parent` reference that grows into a chain across
# Markov-chain steps. With 10k steps the chain depth at GC time exceeds
# Python's default recursion limit (1000) and shutdown finalisers crash with
# RecursionError. Raise to a generous ceiling that comfortably exceeds 10k.
sys.setrecursionlimit(50000)

import networkx as nx
import plotly.graph_objects as go

from falcomchain.graph import Graph
from falcomchain.markovchain import ChainState, MarkovChain
from falcomchain.markovchain.energy import compute_energy
from falcomchain.markovchain.proposals import hierarchical_recom
from falcomchain.markovchain.super_partitioners import (
    fixed_super_partition,
    resample_super_partition,
)
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment

# Local plotter (sits in the same gurobi/ folder).
from falcomchain_experiments.gurobi.plot_falcom_solution import plot_falcom_solution

tree_mod = importlib.import_module("falcomchain.tree.tree")

HERE = Path(__file__).resolve().parent


# ----- uniform cut weighting overrides ------------------------------------
def _uniform_psi(phi, gamma, r):
    """L1 ψ override: 1 if at least one candidate in the subtree, else 0.
    Cuts with positive psi are sampled uniformly (all weights = 1)."""
    return 1.0 if phi > 0 else 0.0


def _uniform_super_psi(subnodes, teams):
    """L2 ψ² override: constant 1 → uniform among admissible super-cuts."""
    return 1.0


# ----- travel times --------------------------------------------------------
def manhattan_travel_times(graph) -> dict:
    coords = {n: (int(graph.nodes[n]["C_X"]), int(graph.nodes[n]["C_Y"]))
              for n in graph.nodes}
    return {(i, j): abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            for i in coords for j in coords}


# ----- facility assignment + radii ----------------------------------------
def best_facility(part_nodes, candidate_set, travel_times):
    cands_in = [c for c in candidate_set if c in part_nodes]
    if not cands_in:
        return None, float("inf")
    best_f, best_r = None, float("inf")
    for f in cands_in:
        r = max(travel_times[(f, v)] for v in part_nodes)
        if r < best_r:
            best_f, best_r = f, r
    return best_f, best_r


def snapshot_partition(p):
    """Snapshot a live Partition into a plain dict (parts/super_assignment/teams).
    Holding a reference to the live Partition is unsafe — the chain mutates
    it later. ``copy.deepcopy`` overflows on the Partition's internal
    references. The snapshot is cheap, self-contained, and compatible with
    both ``compute_R1_R2`` and ``build_solution_dict``."""
    return {
        "parts": {int(did): set(int(v) for v in dnodes)
                  for did, dnodes in p.parts.items()},
        "super_assignment": {int(k): int(v)
                             for k, v in p.super_assignment.items()},
        "teams": {int(did): int(t) for did, t in p.teams.items()},
    }


def partition_from_snapshot(snap, graph, capacity_level):
    """Reconstruct a fresh, parent-less Partition from a snapshot dict.
    Used by short-bursts to restart a burst from the best-so-far partition
    without carrying a deep parent chain (which would overflow recursion)."""
    from falcomchain.tree.tree import Flip
    parts = snap["parts"]
    teams = snap["teams"]
    node_assignment = {v: did for did, nodes in parts.items() for v in nodes}
    p = Partition(
        capacity_level=capacity_level,
        assignment=node_assignment,
        flip=Flip(flips=node_assignment, team_flips=dict(teams),
                  new_ids=frozenset(parts.keys())),
        graph=graph,
    )
    # Restore the real L2 grouping (constructor sets identity).
    p.super_assignment = dict(snap["super_assignment"])
    return p


def joint_R1_R2(partition_or_snap, *, candidates_l1, candidates_l2, travel_times,
                return_facilities=False):
    """Match the MIP's objective: jointly choose L1 and L2 facilities
    per super-district to minimise

        R^1 + R^2  =  max_D r_D^1(f^1_D) + max_S r_S^2(f^2_S, {f^1_D : D in S}).

    This is what the MIP optimises; it CAN differ from the per-district
    minimax facility choice (compute_R1_R2 below). The L1 facility in
    district D affects both D's own radius and the radius of D's
    super-district, so a slightly worse L1 fac can yield a much better R^2.

    NOTE: this function is intended as an *objective-specific evaluator*
    only. It does NOT touch FalCom's per-district facility assignment
    (Assignment.best_facility / FacilityAssignment), which stays the
    partition-deterministic minimax used for ensemble statistics.

    :param return_facilities: when True, also return the chosen facilities
        so the serializer can record the SAME assignment that produced the
        score. Returns ``(R1, R2, l1_fac, l2_fac)`` where ``l1_fac`` maps
        district id -> chosen L1 facility node and ``l2_fac`` maps super id
        -> chosen L2 facility node.

    Returns ``(R1, R2)`` (or the 4-tuple above when return_facilities).
    """
    import itertools as _it
    if isinstance(partition_or_snap, dict):
        parts = partition_or_snap["parts"]
        super_assignment = partition_or_snap["super_assignment"]
    else:
        parts = partition_or_snap.parts
        super_assignment = partition_or_snap.super_assignment

    _fail = (float("inf"), float("inf"), {}, {}) if return_facilities \
        else (float("inf"), float("inf"))

    # Per district: feasible (L1 cand, radius) pairs.
    district_options = {}
    for did, dnodes in parts.items():
        cands_in = [c for c in candidates_l1 if c in dnodes]
        opts = []
        for c in cands_in:
            r = max(travel_times[(c, v)] for v in dnodes)
            opts.append((c, r))
        if not opts:
            return _fail
        opts.sort(key=lambda x: x[1])
        district_options[did] = opts

    # Group districts by super-district.
    supers = {}  # sid -> list of did
    super_base_nodes = {}
    for did, dnodes in parts.items():
        sid = super_assignment.get(did, did)
        supers.setdefault(sid, []).append(did)
        super_base_nodes.setdefault(sid, set()).update(dnodes)

    super_l2_options = {}
    for sid, snodes in super_base_nodes.items():
        s_cands = [c for c in candidates_l2 if c in snodes]
        if not s_cands:
            s_cands = [c for c in candidates_l1 if c in snodes]
        if not s_cands:
            return _fail
        super_l2_options[sid] = s_cands

    # Binary-search on R^1 over the distinct achievable district-radius
    # values. For each R^1 budget, each district is restricted to
    # candidates with radius <= R^1; then for each super-district the
    # min R^2_S is enumerated over (L1 fac per district in S) x (L2 fac).
    # The global R^2 is max over supers. Track min(R^1 + R^2).
    R1_min = max(opts[0][1] for opts in district_options.values())  # per-district minimax
    all_radii = sorted({r for opts in district_options.values() for _, r in opts})
    # Only R^1 values >= per-district-minimax can be feasible.
    candidate_R1_values = [r for r in all_radii if r >= R1_min]

    best_total = float("inf")
    best_R1 = float("inf")
    best_R2 = float("inf")
    best_l1_fac = {}   # did -> chosen L1 facility (at the minimising budget)
    best_l2_fac = {}   # sid -> chosen L2 facility
    for R1_budget in candidate_R1_values:
        # Per district: candidates with radius <= R1_budget.
        feas = {did: [(c, r) for c, r in opts if r <= R1_budget]
                for did, opts in district_options.items()}
        if any(not f for f in feas.values()):
            continue
        # Per super: enumerate L1 combos × L2 cands; track best R^2_S and
        # the (l1_combo, l2_cand) achieving it.
        global_R2 = 0
        budget_l1_fac = {}
        budget_l2_fac = {}
        feasible = True
        for sid, dids in supers.items():
            l1_choice_sets = [[c for c, _ in feas[did]] for did in dids]
            local_R2 = float("inf")
            local_l1 = None
            local_l2 = None
            for l1_combo in _it.product(*l1_choice_sets):
                for l2_cand in super_l2_options[sid]:
                    r2 = max(travel_times[(l2_cand, j1)] for j1 in l1_combo)
                    if r2 < local_R2:
                        local_R2 = r2
                        local_l1 = l1_combo
                        local_l2 = l2_cand
                        if local_R2 == 0:
                            break
                if local_R2 == 0:
                    break
            if local_R2 == float("inf"):
                feasible = False
                break
            # Record this super's chosen facilities.
            for did, fac in zip(dids, local_l1):
                budget_l1_fac[did] = fac
            budget_l2_fac[sid] = local_l2
            if local_R2 > global_R2:
                global_R2 = local_R2
        if not feasible:
            continue
        total = R1_budget + global_R2
        if total < best_total:
            best_total = total
            best_R1 = R1_budget
            best_R2 = global_R2
            best_l1_fac = dict(budget_l1_fac)
            best_l2_fac = dict(budget_l2_fac)

    if return_facilities:
        return best_R1, best_R2, best_l1_fac, best_l2_fac
    return best_R1, best_R2


def disaggregated_median(partition_or_snap, *, candidates_l1, candidates_l2,
                         travel_times, demands, return_facilities=False):
    """Disaggregated hierarchical median — the current MILP benchmark
    objective (experiments section). Separable across levels:

        base  = Σ_D  min_{f∈F¹∩D}   Σ_{v∈D}     d_v·dist(v,f)
        coord = Σ_S  min_{f∈F²∩S}   Σ_{v∈V¹[S]} d_v·dist(v,f)

    Each level's facility is the demand-weighted 1-median of the units it
    serves; the two terms are independent. This mirrors the FalCom library
    facility-assignment rule exactly (Section 5.4 of the paper).

    Returns ``(base, coord)`` — analogous to the script's ``(R1, R2)``
    slots, so SA minimises ``base + coord``. With ``return_facilities`` it
    also returns ``(l1_fac, l2_fac)`` dicts for consistent serialization.
    """
    if isinstance(partition_or_snap, dict):
        parts = partition_or_snap["parts"]
        super_assignment = partition_or_snap["super_assignment"]
    else:
        parts = partition_or_snap.parts
        super_assignment = partition_or_snap.super_assignment

    _fail = (float("inf"), float("inf"), {}, {}) if return_facilities \
        else (float("inf"), float("inf"))

    # ---- base-level access (per-district 1-median) ----
    base = 0.0
    l1_fac = {}
    for did, dnodes in parts.items():
        cands = [c for c in candidates_l1 if c in dnodes]
        if not cands:
            return _fail
        best_c, best_cost = None, float("inf")
        for c in cands:
            cost = sum(demands.get(v, 1.0) * travel_times[(c, v)] for v in dnodes)
            if cost < best_cost:
                best_cost, best_c = cost, c
        base += best_cost
        l1_fac[did] = best_c

    # ---- upper-level coordination (per-superdistrict 1-median over base) ----
    super_parts = {}
    for did, dnodes in parts.items():
        sid = super_assignment.get(did, did)
        super_parts.setdefault(sid, set()).update(dnodes)

    coord = 0.0
    l2_fac = {}
    for sid, snodes in super_parts.items():
        s_cands = [c for c in candidates_l2 if c in snodes]
        if not s_cands:
            s_cands = [c for c in candidates_l1 if c in snodes]
        if not s_cands:
            return _fail
        best_c, best_cost = None, float("inf")
        for c in s_cands:
            cost = sum(demands.get(v, 1.0) * travel_times[(c, v)] for v in snodes)
            if cost < best_cost:
                best_cost, best_c = cost, c
        coord += best_cost
        l2_fac[sid] = best_c

    if return_facilities:
        return base, coord, l1_fac, l2_fac
    return base, coord


def compute_R1_R2(partition_or_snap, *, candidates_l1, candidates_l2, travel_times):
    """Min-max objective at both levels (matches MIP §3 R^1 + R^2).

    R1 = max_i r_i^1   where r_i^1 = max_{v in D_i} d(f1_D_i, v)
    R2 = max_S r_S^2   where r_S^2 = max_{j1 in F^1 ∩ S} d(f1_j1, f2_S)
                       (hub-spoke distance — L2 facility to its L1 hubs)

    Returns (R1, R2). Falls back to inf if any district lacks a candidate.
    `partition_or_snap` may be a live Partition or a snapshot dict.
    """
    if isinstance(partition_or_snap, dict):
        parts = partition_or_snap["parts"]
        super_assignment = partition_or_snap["super_assignment"]
    else:
        parts = partition_or_snap.parts
        super_assignment = partition_or_snap.super_assignment

    l1_fac = {}
    R1 = 0.0
    for did, dnodes in parts.items():
        f, _ = best_facility(set(dnodes), candidates_l1, travel_times)
        if f is None:
            return float("inf"), float("inf")
        l1_fac[did] = f
        r = max(travel_times[(f, v)] for v in dnodes)
        if r > R1:
            R1 = r

    super_parts = {}
    for did, dnodes in parts.items():
        sid = super_assignment.get(did, did)
        super_parts.setdefault(sid, set()).update(dnodes)

    R2 = 0.0
    for sid, snodes in super_parts.items():
        l1_in_super = [l1_fac[did] for did, _ in parts.items()
                       if super_assignment.get(did, did) == sid]
        if not l1_in_super:
            continue
        s_cands = [c for c in candidates_l2 if c in snodes]
        if not s_cands:
            s_cands = [c for c in candidates_l1 if c in snodes]
        if not s_cands:
            return float("inf"), float("inf")
        best_r = float("inf")
        for cand in s_cands:
            r = max(travel_times[(cand, j1)] for j1 in l1_in_super)
            if r < best_r:
                best_r = r
        if best_r > R2:
            R2 = best_r
    return R1, R2


# ----- solution JSON in MILP-compatible format ----------------------------
def build_solution_dict(partition_or_snap, graph_raw, *, travel_times, c_max_l1,
                         c_max_l2, wall_time_s, meta, R1, R2, args, label,
                         l1_fac=None, l2_fac=None):
    """`partition_or_snap` may be a live Partition or a snapshot dict with
    keys 'parts', 'super_assignment', 'teams' (output format identical).

    If ``l1_fac`` / ``l2_fac`` are provided (district id -> L1 facility,
    super id -> L2 facility), the serialized assignment uses EXACTLY those
    facilities — so the written x1/y1/x2/y2 reproduce the score that chose
    them (e.g. the joint evaluator). Otherwise it falls back to per-district
    minimax (L1) and coverage (L2), which can disagree with a joint score."""
    if isinstance(partition_or_snap, dict):
        parts = partition_or_snap["parts"]
        super_assignment = partition_or_snap["super_assignment"]
        teams = partition_or_snap["teams"]
    else:
        parts = partition_or_snap.parts
        super_assignment = partition_or_snap.super_assignment
        teams = partition_or_snap.teams

    candidates_l1 = {int(n["id"]) for n in graph_raw["nodes"]
                     if n.get("candidate", 0) > 0}
    candidates_l2 = {int(n["id"]) for n in graph_raw["nodes"]
                     if n.get("super_candidate", 0) > 0}

    x1, y1, c1 = {}, {}, {}
    for did, dnodes in parts.items():
        if l1_fac is not None and did in l1_fac:
            f = l1_fac[did]
        else:
            f, _ = best_facility(set(dnodes), candidates_l1, travel_times)
        if f is None:
            raise RuntimeError(f"L1 district {did} has no candidate")
        cap = int(teams[did])
        y1[f] = 1
        c1[f] = cap
        for v in dnodes:
            x1[v] = f
    for cand in candidates_l1:
        y1.setdefault(cand, 0)
        c1.setdefault(cand, 0)

    super_parts = {}
    super_of = {}
    for did, dnodes in parts.items():
        sid = super_assignment.get(did, did)
        super_parts.setdefault(sid, set()).update(dnodes)
        super_of[did] = sid

    x2, y2, c2 = {}, {}, {}
    for sid, snodes in super_parts.items():
        s_cap = sum(int(teams[did]) for did, _ in parts.items()
                    if super_assignment.get(did, did) == sid)
        if l2_fac is not None and sid in l2_fac:
            f = l2_fac[sid]
        else:
            f, _ = best_facility(set(snodes), candidates_l2, travel_times)
            if f is None:
                f, _ = best_facility(set(snodes), candidates_l1, travel_times)
        y2[f] = 1
        c2[f] = s_cap
        for v in snodes:
            x2[v] = f
    for cand in candidates_l2:
        y2.setdefault(cand, 0)
        c2.setdefault(cand, 0)

    return {
        "instance": meta["grid_path"],
        "objective": "falcom_R1_plus_R2",
        "obj_value": float(R1 + R2),
        "R1": float(R1),
        "R2": float(R2),
        "wall_time_s": float(wall_time_s),
        "method": f"falcom_{label}",
        "params": {
            "demand_target_w": meta["demand_target_w"],
            "epsilon_l1": args.eps,
            "epsilon_l2": meta.get("epsilon_l2", args.eps),
            "c_max_l1": c_max_l1,
            "c_max_l2": c_max_l2,
            "rule": args.rule,
            "chain_steps": getattr(args, "steps",
                                   getattr(args, "burst_length", 0) *
                                   getattr(args, "num_bursts", 1)),
            "seed": args.seed,
            "cut_selection": "uniform",
            "accept_rule": getattr(args, "accept_rule", "simulated_annealing"),
        },
        "y1": {str(k): int(v) for k, v in y1.items()},
        "c1": {str(k): int(v) for k, v in c1.items()},
        "x1": {str(k): int(v) for k, v in x1.items()},
        "y2": {str(k): int(v) for k, v in y2.items()},
        "c2": {str(k): int(v) for k, v in c2.items()},
        "x2": {str(k): int(v) for k, v in x2.items()},
    }


# ----- main run ------------------------------------------------------------
def run(instance: str, args):
    inst_dir = HERE / "data"
    with open(inst_dir / f"grid_{instance}.meta.json") as f:
        meta = json.load(f)
    with open(inst_dir / meta["grid_path"]) as f:
        raw = json.load(f)
    g_nx = nx.node_link_graph(raw, edges="adjacency")
    G = Graph.from_networkx(g_nx)

    print(f"=== FalCom (SA, R^1+R^2) on grid_{instance} ===")
    print(f"  |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  "
          f"w={meta['demand_target_w']}  eps={args.eps}  "
          f"c1 in [{args.c_min_l1}, {args.c_max_l1}]  "
          f"c2 in [{args.c_min_l2}, {args.c_max_l2}]  "
          f"rule={args.rule}  steps={args.steps}")

    travel_times = manhattan_travel_times(G)
    Assignment.travel_times = travel_times

    candidates_l1 = {int(n["id"]) for n in raw["nodes"]
                     if n.get("candidate", 0) > 0}
    candidates_l2 = {int(n["id"]) for n in raw["nodes"]
                     if n.get("super_candidate", 0) > 0}

    # Seed RNGs deterministically.
    import random as _random
    _random.seed(args.seed)
    if hasattr(tree_mod, "rng"):
        try:
            tree_mod.rng.seed(args.seed)
        except Exception:
            pass

    # ------ Initial partition (retry with fresh seeds on deadlock) ------
    t0 = time.perf_counter()
    partition = None
    last_exc = None
    seed_tries = [args.seed] + [args.seed + k for k in range(1, 50) if args.seed + k != args.seed]
    for seed_try in seed_tries:
        _random.seed(seed_try)
        if hasattr(tree_mod, "rng"):
            try:
                tree_mod.rng.seed(seed_try)
            except Exception:
                pass
        try:
            partition = Partition.from_random_assignment(
                graph=G,
                epsilon=args.eps,
                demand_target=meta["demand_target_w"],
                assignment_class=Assignment,
                capacity_level=args.c_max_l1,
                c_min=args.c_min_l1,
                c_min_super=args.c_min_l2,
                c_max_super=args.c_max_l2,
                min_districts_super=args.min_districts_super,
                init_super_partition=True,
                epsilon_super=meta.get("epsilon_l2", args.eps),
                rule=args.rule,
                enforce_global_balance=False,
                psi_fn=_uniform_psi,
                super_psi_fn=_uniform_super_psi,
            )
            # Detect the silent identity-fallback emitted by
            # init_super_partition when the recursive supergraph
            # heuristic stalls (it warns and returns identity rather
            # than raising). The identity grouping violates mu^2,
            # so we replace it with a deterministic super-partition
            # built from the L1 districts -- no RNG, fully reproducible.
            sa = dict(partition.super_assignment)
            from collections import Counter
            sup_sizes = Counter(sa.values())
            if any(v < args.min_districts_super for v in sup_sizes.values()):
                from falcomchain_experiments.gurobi.deterministic_super \
                    import deterministic_super_partition
                from falcomchain.partition.partition import supergraph as _supergraph
                x1_map = dict(partition.assignment.mapping)
                c1_map = dict(partition.assignment.teams)
                new_sa = deterministic_super_partition(
                    G, x1_map, c1_map,
                    c_min_super=args.c_min_l2,
                    c_max_super=args.c_max_l2,
                    min_districts_super=args.min_districts_super,
                )
                partition.super_assignment = new_sa
                partition.supergraph = _supergraph(partition)
                print(f"  [init] super_assignment was identity "
                      f"(some super had < {args.min_districts_super} L1 "
                      f"districts); replaced with deterministic "
                      f"super-partition: {len(set(new_sa.values()))} "
                      f"super-districts.")
            if seed_try != args.seed:
                print(f"  [seed retry] succeeded at seed={seed_try}")
            break
        except RuntimeError as e:
            last_exc = e
            continue
    if partition is None:
        raise RuntimeError(f"seed build failed across {len(seed_tries)} retries; "
                           f"last error: {last_exc}")
    init_elapsed = time.perf_counter() - t0

    # Per-node demands for the disaggregated-median objective.
    demands = {int(n["id"]): n.get("demand", 0) for n in raw["nodes"]}

    _obj_desc = {
        "median": "disaggregated hierarchical median (matches MILP benchmark)",
        "joint": "joint L1+L2 facility selection (R^1+R^2 min-max)",
        "per_district": "per-district minimax then L2",
    }
    print(f"  scoring objective: {args.objective} ({_obj_desc.get(args.objective, '')})")

    def score_and_facilities(p):
        """Return (term1, term2, l1_fac, l2_fac). For 'median' the terms are
        (base access, upper coordination) and the facilities are the
        demand-weighted 1-medians; for 'joint' they are R^1, R^2 with the
        joint-chosen facilities; for 'per_district' facilities are None
        (serializer falls back)."""
        if args.objective == "median":
            return disaggregated_median(
                p, candidates_l1=candidates_l1, candidates_l2=candidates_l2,
                travel_times=travel_times, demands=demands,
                return_facilities=True,
            )
        if args.objective == "joint":
            return joint_R1_R2(
                p, candidates_l1=candidates_l1, candidates_l2=candidates_l2,
                travel_times=travel_times, return_facilities=True,
            )
        r1, r2 = compute_R1_R2(
            p, candidates_l1=candidates_l1, candidates_l2=candidates_l2,
            travel_times=travel_times,
        )
        return r1, r2, None, None

    R1_init, R2_init, l1f_init, l2f_init = score_and_facilities(partition)
    print(f"  initial: {init_elapsed:.2f}s  |P^1|={len(partition.parts)}  "
          f"caps={sorted(partition.teams.values())}  "
          f"R1={R1_init:.1f} R2={R2_init:.1f}  R1+R2={R1_init+R2_init:.1f}")

    # Save initial.
    sol_init = build_solution_dict(
        partition, raw, travel_times=travel_times,
        c_max_l1=args.c_max_l1, c_max_l2=args.c_max_l2,
        wall_time_s=init_elapsed, meta=meta, R1=R1_init, R2=R2_init,
        args=args, label="initial", l1_fac=l1f_init, l2_fac=l2f_init,
    )
    out_init_json = HERE / f"solution_{instance}_falcom_initial.json"
    with open(out_init_json, "w") as f:
        json.dump(sol_init, f, indent=2)

    # ------ Chain phase: uniform cuts + greedy R1+R2 acceptance ------
    state = ChainState.initial(
        partition=partition, energy=R1_init + R2_init, beta=1.0,
        energy_fn=compute_energy,  # required by API; not used by greedy_accept
    )
    partition.parent = None

    # Chain-level super-cut fallback: `resample_super_partition` calls
    # `capacitated_recursive_tree` on the L1-district supergraph, which only
    # considers cuts induced by random spanning trees. For tight instances
    # (e.g. grid_400's 8-supernode supergraph with team counts
    # [1,1,1,1,1,1,2,2] and c^2 in [2,5]) only ~0.12% of trees expose an
    # admissible subtree-cut, so resample raises RuntimeError on ~29% of
    # chain steps even though feasible super-partitions exist. We catch
    # that and fall through to `fixed_super_partition`, which keeps the
    # current L2 grouping and only re-cuts L1 inside one super for that
    # step. The chain step always advances with a feasible proposal.
    super_partitioner_stalls = {"n": 0}

    def _super_partitioner_with_fallback(state, **kw):
        try:
            return resample_super_partition(state, **kw)
        except RuntimeError:
            super_partitioner_stalls["n"] += 1
            # fixed_super_partition has a narrower signature (no
            # min_districts_super, since it doesn't recut at L2). Drop
            # kwargs it doesn't accept.
            fixed_kw = {k: v for k, v in kw.items()
                        if k != "min_districts_super"}
            return fixed_super_partition(state, **fixed_kw)

    proposal = lambda s: hierarchical_recom(
        s,
        epsilon_base=args.eps,
        epsilon_super=meta.get("epsilon_l2", args.eps),
        demand_target=meta["demand_target_w"],
        c_min_base=args.c_min_l1,
        c_min_super=args.c_min_l2,
        c_max_super=args.c_max_l2,
        min_districts_super=args.min_districts_super,
        rule=args.rule,
        enforce_global_balance=False,
        psi_fn=_uniform_psi,
        super_psi_fn=_uniform_super_psi,
        super_partitioner=_super_partitioner_with_fallback,
    )

    best_snap = snapshot_partition(partition)
    best_obj = R1_init + R2_init
    current_obj = R1_init + R2_init
    # Trajectory: (step, current, best). Current can rise under SA.
    traj = [(0, current_obj, best_obj)]
    step_counter = {"k": 0}
    # Objective-magnitude scale so beta_magnitude stays ~O(1) regardless of
    # whether the objective is the small R^1+R^2 (~10s) or the large
    # disaggregated median (~10^4-10^5). Worsening moves are scored on
    # delta / obj_scale. For the small objectives obj_scale≈1 (unchanged).
    n_base = max(1, len(demands))
    obj_scale = max(1.0, (R1_init + R2_init) / n_base)

    # Simulated-annealing acceptance, following the convention of
    # `SingleMetricOptimizer._simulated_annealing_acceptance_function`:
    #
    #     accept with probability  exp(-beta(t) * beta_magnitude * delta)
    #
    # where delta = score(proposed) - score(current) (minimisation, so
    # delta > 0 is a worsening move). beta(t) ∈ [0, 1] ramps from 0
    # (hot, accept everything) to 1 (cold, greedy). The optimizer module
    # only ships a jumpcycle beta_function; we use a simple linear ramp
    # beta(t) = t / num_steps to mirror the classic SA cooling profile.
    import math as _math
    import random as _rand
    total = max(1, args.steps)
    if args.cooling == "linear":
        def beta_function(t):
            # 0 → 1 monotonically over the chain
            return min(1.0, t / total)
    elif args.cooling == "cyclic":
        # Repeated sawtooth: within each cycle beta ramps 0 → 1, then
        # snaps back to 0 (reheat T). N cycles over the run.
        cycle_len = max(1, total // max(1, args.n_cycles))
        def beta_function(t):
            return (t % cycle_len) / cycle_len
    else:
        raise ValueError(f"unknown cooling schedule: {args.cooling}")

    # Per-step diagnostics: distinguishes proposal-stalls (chain caught a
    # RuntimeError from the proposal kernel, so sa_accept was never called)
    # from scoring-failures and from SA accept/reject. accept_log[i] is the
    # outcome of step i in 1..N.
    diag = {"sa_called": 0, "accepted": 0, "rejected_worsening": 0,
            "rejected_improving": 0, "score_failed": 0}
    accept_log = [None] * (args.steps + 1)  # filled inside sa_accept

    def sa_accept(proposed: ChainState, current: ChainState) -> bool:
        nonlocal current_obj, best_snap, best_obj
        diag["sa_called"] += 1
        k = step_counter["k"]
        try:
            r1, r2, _l1f, _l2f = score_and_facilities(proposed.partition)
        except Exception:
            diag["score_failed"] += 1
            accept_log[k + 1] = "score_failed"
            return False
        new_obj = r1 + r2
        delta = new_obj - current_obj
        # SA: always accept improving/equal moves; accept a worsening move
        # with prob exp(-beta * M * delta/obj_scale). Exponentiating only
        # worsening moves (delta>0 → arg≤0) avoids the OverflowError that a
        # large improving delta would cause with exp of a large positive arg.
        beta = beta_function(k)
        if delta <= 0:
            accept = True
        else:
            arg = -beta * args.beta_magnitude * (delta / obj_scale)
            accept = _rand.random() < _math.exp(arg)
        if accept:
            current_obj = new_obj
            diag["accepted"] += 1
            accept_log[k + 1] = "accepted"
            if new_obj < best_obj:
                best_obj = new_obj
                best_snap = snapshot_partition(proposed.partition)
        else:
            if delta <= 0:
                diag["rejected_improving"] += 1
                accept_log[k + 1] = "rejected_improving"
            else:
                diag["rejected_worsening"] += 1
                accept_log[k + 1] = "rejected_worsening"
        return accept

    chain = MarkovChain(
        proposal=proposal, constraints=[], accept=sa_accept,
        initial_state=state, total_steps=args.steps,
    )

    print(f"  running SA chain for {args.steps} steps "
          f"(beta(t)=t/N, beta_magnitude={args.beta_magnitude})...")
    t1 = time.perf_counter()
    log_every = max(1, args.steps // 20)
    for i, s in enumerate(chain, start=1):
        step_counter["k"] = i
        traj.append((i, current_obj, best_obj))
        if i % log_every == 0 or i == args.steps:
            elapsed = time.perf_counter() - t1
            rate = i / elapsed if elapsed > 0 else 0
            beta = beta_function(i)
            print(f"    step {i:5d}/{args.steps}  beta={beta:.3f}  "
                  f"current R1+R2={current_obj:.1f}  best={best_obj:.1f}  "
                  f"({rate:.1f} steps/s)")
    chain_elapsed = time.perf_counter() - t1
    print(f"  chain done: {chain_elapsed:.2f}s  best R1+R2 = {best_obj:.1f}")

    # ------ Diagnostics summary ------
    # Mark every step where sa_accept was not called (k+1 still None) as a
    # proposal-stall: the chain raised in the proposal kernel and caught it
    # as a rejection, leaving state unchanged.
    proposal_stalls = sum(1 for i in range(1, args.steps + 1)
                          if accept_log[i] is None)
    diag["proposal_stalls"] = proposal_stalls
    diag["super_partitioner_fallbacks"] = super_partitioner_stalls["n"]
    diag["steps_total"] = args.steps
    print(f"  diag: proposal_stalls={proposal_stalls}  "
          f"super_fallbacks={super_partitioner_stalls['n']}  "
          f"sa_called={diag['sa_called']}  "
          f"accepted={diag['accepted']}  "
          f"rej_improving={diag['rejected_improving']}  "
          f"rej_worsening={diag['rejected_worsening']}  "
          f"score_failed={diag['score_failed']}")
    # Per-thirds breakdown so we can see where the chain wedges.
    thirds = [(1, args.steps // 3 + 1),
              (args.steps // 3 + 1, 2 * args.steps // 3 + 1),
              (2 * args.steps // 3 + 1, args.steps + 1)]
    for lo, hi in thirds:
        from collections import Counter as _C
        c = _C(accept_log[i] if accept_log[i] is not None else "stall"
               for i in range(lo, hi))
        print(f"    steps [{lo:5d},{hi:5d}): {dict(c)}")

    # ------ Save final ------
    R1_final, R2_final, l1f_final, l2f_final = score_and_facilities(best_snap)
    sol_final = build_solution_dict(
        best_snap, raw, travel_times=travel_times,
        c_max_l1=args.c_max_l1, c_max_l2=args.c_max_l2,
        wall_time_s=init_elapsed + chain_elapsed, meta=meta,
        R1=R1_final, R2=R2_final, args=args, label="final",
        l1_fac=l1f_final, l2_fac=l2f_final,
    )
    out_final_json = HERE / f"solution_{instance}_falcom_final.json"
    with open(out_final_json, "w") as f:
        json.dump(sol_final, f, indent=2)

    # ------ Trajectory ------
    traj_path = HERE / f"solution_{instance}_falcom_trajectory.json"
    with open(traj_path, "w") as f:
        json.dump({"steps": [t[0] for t in traj],
                   "R1_plus_R2_current": [t[1] for t in traj],
                   "R1_plus_R2_best": [t[2] for t in traj],
                   "best": best_obj,
                   "initial": R1_init + R2_init,
                   "chain_elapsed_s": chain_elapsed,
                   "steps_total": args.steps,
                   "beta_magnitude": args.beta_magnitude,
                   "accept_rule": "simulated_annealing",
                   "diag": diag,
                   "accept_log": [accept_log[i] if accept_log[i] is not None
                                  else "stall"
                                  for i in range(1, args.steps + 1)]}, f, indent=2)

    # ------ Plotly figures ------
    fig_init = plot_falcom_solution(g_nx, sol_init)
    fig_init.update_layout(title=f"FalCom initial — grid_{instance} (R1+R2={R1_init+R2_init:.1f})")
    fig_init.write_html(HERE / f"solution_{instance}_falcom_initial.html")

    fig_final = plot_falcom_solution(g_nx, sol_final)
    fig_final.update_layout(title=f"FalCom best — grid_{instance} (R1+R2={best_obj:.1f}, {args.steps} steps)")
    fig_final.write_html(HERE / f"solution_{instance}_falcom_final.html")

    # Trajectory plot: current (noisy SA walk) + best-so-far (monotone).
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=[t[0] for t in traj],
        y=[t[1] for t in traj],
        mode="lines",
        name="R¹+R² (current)",
        line=dict(color="lightsteelblue", width=1),
        opacity=0.7,
    ))
    fig_traj.add_trace(go.Scatter(
        x=[t[0] for t in traj],
        y=[t[2] for t in traj],
        mode="lines",
        name="R¹+R² (best so far)",
        line=dict(color="firebrick", width=2),
    ))
    fig_traj.add_hline(y=best_obj, line_dash="dash", line_color="firebrick",
                       annotation_text=f"final best = {best_obj:.1f}",
                       annotation_position="bottom right")
    fig_traj.update_layout(
        title=(f"R¹ + R² trajectory — grid_{instance} "
               f"(per_team, uniform cuts, SA accept, β·M={args.beta_magnitude})"),
        xaxis_title="step",
        yaxis_title="R¹ + R²  (max-of-radii at each level)",
        template="plotly_white",
    )
    fig_traj.write_html(HERE / f"solution_{instance}_falcom_trajectory.html")

    print(f"  wrote:")
    print(f"    {out_init_json.name}")
    print(f"    {out_final_json.name}")
    print(f"    {traj_path.name}")
    print(f"    solution_{instance}_falcom_initial.html")
    print(f"    solution_{instance}_falcom_final.html")
    print(f"    solution_{instance}_falcom_trajectory.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", choices=["100", "400"], default="100", nargs="?")
    ap.add_argument("--rule", choices=["per_team", "main"], default="per_team")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c-min-l1", type=int, default=1,
                    help="Min L1 capacity per district (paper §7.5: 1).")
    ap.add_argument("--c-max-l1", type=int, default=2,
                    help="Max L1 capacity per district (paper §7.5: 2).")
    ap.add_argument("--c-min-l2", type=int, default=2,
                    help="Min L2 capacity per super-district (paper §7.5: 2).")
    ap.add_argument("--c-max-l2", type=int, default=5,
                    help="Max L2 capacity per super-district (paper §7.5: 5).")
    ap.add_argument("--min-districts-super", type=int, default=2,
                    help="Min # L1 districts per super-district (matches the "
                         "MIP's min_l1_per_l2). Default 2.")
    ap.add_argument("--beta-magnitude", type=float, default=1.0,
                    help="SA scaling: accept-prob = exp(-beta(t)·M·Δ). "
                         "Larger M makes worsening moves rarer earlier.")
    ap.add_argument("--objective", choices=["median", "per_district", "joint"],
                    default="median",
                    help="Scoring objective. 'median': disaggregated "
                         "hierarchical median (demand-weighted 1-median at "
                         "both levels) — matches the current MILP benchmark "
                         "and the FalCom library objective. 'joint': "
                         "R^1+R^2 min-max with joint L1+L2 facility selection. "
                         "'per_district': greedy minimax per district then L2.")
    ap.add_argument("--cooling", choices=["linear", "cyclic"], default="linear",
                    help="'linear': beta(t)=t/N, single ramp. 'cyclic': "
                         "sawtooth ramp 0→1 within each of --n-cycles cycles, "
                         "reheating T between cycles to escape local optima.")
    ap.add_argument("--n-cycles", type=int, default=5,
                    help="Number of reheat cycles (only used with --cooling cyclic).")
    args = ap.parse_args()
    run(args.instance, args)


if __name__ == "__main__":
    raise SystemExit(main())
