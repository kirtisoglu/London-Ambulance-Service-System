"""
Exact-enumeration validation on a grid with NON-UNIFORM demand.

The instances of ``enumeration.py`` give every unit demand 100, so every
admissible district has demand exactly ``c * d_bar``, the running debt of the
recursion is identically zero and the debt-corrected windows are never
exercised. Here the unit demands differ, so districts of one capacity have
different sizes and demands, the debt is non-zero after almost every
extraction, and the hierarchical balance rule of the paper (Proposition
"Effective balance") is the binding description of a state.

The reference set is the feasible state space of the paper, listed exactly:

- level 1: partitions into connected districts, each with a candidate and a
  capacity c in {1..c_max} whose demand lies in the compounded window
  ``(1 -+ eps2) (1 -+ eps1 (c+1)/(2c)) c d_bar``; capacities sum to ceil(d/w);
- level 2: groupings of the districts into connected superdistricts with at
  least kappa districts, capacity in [c2_min, c2_max] and demand within
  ``(1 +- eps2) c2 d_bar``.

The compounded window is an outer description (a district is cut against the
per-unit demand of the superdistrict it was re-cut in), so the chain's support
may be a proper subset of this set; the script reports which part of it each
chain visits and whether the chains visit the same part.

    PYTHONHASHSEED=0 python -m falcomchain_experiments.validation.enumeration_nonuniform \
        --instance 4x5 --steps 1000000 --out-dir data/derived/validation_nonuniform
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

from falcomchain import MarkovChain, always_accept, hierarchical_recom
from falcomchain.graph import Graph
from falcomchain.markovchain.state import ChainState
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed
from falcomchain.tree.tree import Flip

from .enumeration import normalize, spanning_tree_weight, tv

EPS = 0.15
INSTANCES = {
    # 16 units, 4 capacity units, one or two superdistricts
    "4x4": dict(rows=4, cols=4, w=400.0, units=4, c_max=2, c_super=(2, 4), kappa=2, demand_seed=7,
                candidates=[(0, 0), (0, 3), (3, 0), (3, 3), (1, 2)]),
    # 20 units, 5 capacity units, c2 in [2, 3]: two superdistricts (2 + 3 units)
    "4x5": dict(rows=4, cols=5, w=400.0, units=5, c_max=2, c_super=(2, 3), kappa=2, demand_seed=7,
                candidates=[(0, 0), (0, 4), (3, 0), (3, 4), (1, 2), (2, 1), (2, 3)]),
}
OUT_DIR = Path(__file__).resolve().parents[2] / "data/derived/validation_nonuniform"


def build_graph(spec):
    """Grid with integer demands drawn from {80..120} and adjusted to sum to units * w."""
    rows, cols = spec["rows"], spec["cols"]
    rng = np.random.default_rng(spec["demand_seed"])
    n = rows * cols
    demand = rng.integers(80, 121, size=n).astype(int)
    target = int(spec["units"] * spec["w"])
    i = 0
    while demand.sum() != target:                       # nudge units one by one, staying in 80..120
        step = 1 if demand.sum() < target else -1
        if 80 <= demand[i % n] + step <= 120:
            demand[i % n] += step
        i += 1
    g = nx.grid_2d_graph(rows, cols)
    g = nx.relabel_nodes(g, {(r, c): r * cols + c for (r, c) in list(g.nodes)})
    cand = {r * cols + c for (r, c) in spec["candidates"]}
    for v in g.nodes:
        r, c = divmod(v, cols)
        g.nodes[v].update(demand=float(demand[v]), area=1.0, C_X=float(c), C_Y=float(r),
                          candidate=1 if v in cand else 0, boundary_node=False, boundary_perim=0)
    return g, cand, demand


def windows(spec):
    """Level-1 compounded window per capacity and level-2 window per capacity (d_bar = w)."""
    d_bar = spec["w"]
    w1 = {}
    for c in range(1, spec["c_max"] + 1):
        half = EPS * (c + 1) / (2 * c)
        w1[c] = ((1 - EPS) * (1 - half) * c * d_bar, (1 + EPS) * (1 + half) * c * d_bar)
    w2 = {c: ((1 - EPS) * c * d_bar, (1 + EPS) * c * d_bar)
          for c in range(spec["c_super"][0], spec["c_super"][1] + 1)}
    return w1, w2


def connected_sets_upto(g, demand, max_demand):
    """All connected node sets of demand <= max_demand."""
    seen = set()
    frontier = [frozenset([v]) for v in g.nodes]
    seen.update(frontier)
    while frontier:
        nxt = []
        for s in frontier:
            d = sum(demand[v] for v in s)
            for v in s:
                for u in g.neighbors(v):
                    if u in s or d + demand[u] > max_demand:
                        continue
                    t = s | {u}
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
        frontier = nxt
    return seen


def enumerate_states(g, cand, demand, spec):
    """Feasible level-1 states (dict district -> capacity) and their level-2 groupings."""
    w1, w2 = windows(spec)
    tol = 1e-9
    pieces = {}
    for s in connected_sets_upto(g, demand, max(hi for _, hi in w1.values())):
        if not (s & cand):
            continue
        d = sum(demand[v] for v in s)
        for c, (lo, hi) in w1.items():
            if lo - tol <= d <= hi + tol:
                pieces[s] = c                               # the windows are disjoint at eps = 0.15
    by_node = {v: [s for s in pieces if v in s] for v in g.nodes}
    states1 = []

    def rec(uncovered, chosen, cap):
        if cap > spec["units"]:
            return
        if not uncovered:
            if cap == spec["units"]:
                states1.append(frozenset(chosen))
            return
        v = min(uncovered)
        for s in by_node[v]:
            if s <= uncovered:
                rec(uncovered - s, chosen + [s], cap + pieces[s])

    rec(frozenset(g.nodes), [], 0)

    def groupings(state):
        districts = sorted(state, key=sorted)
        adj = {d: {e for e in districts if e != d and any(g.has_edge(u, v) for u in d for v in e)}
               for d in districts}

        def ok(block):
            cap = sum(pieces[d] for d in block)
            if len(block) < spec["kappa"] or cap not in w2:
                return False
            dem = sum(demand[v] for d in block for v in d)
            if not (w2[cap][0] - tol <= dem <= w2[cap][1] + tol):
                return False
            seen, stack = {block[0]}, [block[0]]
            while stack:
                d = stack.pop()
                for e in adj[d]:
                    if e in block and e not in seen:
                        seen.add(e)
                        stack.append(e)
            return len(seen) == len(block)

        out = set()

        def rec2(remaining, blocks):
            if not remaining:
                out.add(frozenset(frozenset(b) for b in blocks))
                return
            first, rest = remaining[0], remaining[1:]
            for k in range(len(rest) + 1):
                for others in itertools.combinations(rest, k):
                    block = [first, *others]
                    if ok(block):
                        rec2([d for d in rest if d not in others], blocks + [block])

        rec2(districts, [])
        return out

    level2 = {s: groupings(s) for s in states1}
    states1 = sorted((s for s in states1 if level2[s]), key=lambda st: sorted(sorted(d) for d in st))
    return states1, level2, pieces


# --------------------------------------------------------------------------
# Chain (one process per start)
# --------------------------------------------------------------------------

def _run_chain(args):
    name, spec, start, caps, steps, seed, retry = args
    g, cand, demand = build_graph(spec)
    Assignment.travel_times = {(u, v): abs(g.nodes[u]["C_X"] - g.nodes[v]["C_X"])
                               + abs(g.nodes[u]["C_Y"] - g.nodes[v]["C_Y"]) for u in g.nodes for v in g.nodes}
    graph = Graph.from_networkx(g)
    set_seed(seed)
    flips, teams = {}, {}
    for i, d in enumerate(sorted(start, key=sorted), start=1):
        teams[i] = caps[d]
        for v in d:
            flips[v] = i
    flip = Flip(flips=flips, team_flips=teams, new_ids=frozenset(teams), merged_ids=frozenset())
    partition = Partition(capacity_level=spec["c_max"], assignment=flips, graph=graph, flip=flip)
    state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
    proposal = partial(hierarchical_recom, epsilon_base=EPS, epsilon_super=EPS, demand_target=spec["w"],
                       c_min_super=spec["c_super"][0], c_max_super=spec["c_super"][1],
                       min_districts_super=spec["kappa"],
                       max_attempts_base=retry, max_attempts_super=retry)
    chain = MarkovChain(proposal=proposal, constraints=[], accept=always_accept,
                        initial_state=state, total_steps=steps)
    c_joint, half1 = Counter(), Counter()
    cap_mismatch = 0
    debt_nonzero = 0
    t0 = time.perf_counter()
    for i, st in enumerate(chain):
        if i == 0:
            continue
        p = st.partition
        parts = p.parts
        k1 = frozenset(frozenset(v) for v in parts.values())
        k2 = frozenset(frozenset(frozenset(parts[d]) for d in ids) for ids in p.super_parts.values())
        c_joint[(k1, k2)] += 1
        if i < steps // 2:
            half1[k1] += 1
        # capacity carried by the chain against the capacity the window assigns
        for pid, members in parts.items():
            fs = frozenset(members)
            if caps.get(fs) is not None and int(p.teams[pid]) != caps[fs]:
                cap_mismatch += 1
        if any(abs(sum(demand[v] for v in m) - p.teams[pid] * spec["w"]) > 1e-9 for pid, m in parts.items()):
            debt_nonzero += 1
    rep = chain.rejection_report()
    return name, c_joint, half1, {"acceptance_rate": rep["acceptance_rate"], "causes": rep["causes"],
                                  "seconds": round(time.perf_counter() - t0, 1),
                                  "capacity_mismatch_visits": cap_mismatch,
                                  "share_of_steps_with_off_centre_district": debt_nonzero / max(1, steps - 1)}


def noise_floor(law, n, rng, reps=20):
    p = np.array(list(law.values()), dtype=float)
    p = p / p.sum()
    two = [0.5 * np.abs(rng.multinomial(n, p) / n - rng.multinomial(n, p) / n).sum() for _ in range(reps)]
    h = n // 2
    halves = [0.5 * np.abs(rng.multinomial(h, p) / h - rng.multinomial(h, p) / h).sum() for _ in range(reps)]
    return {"tv_between_two_samples": float(np.mean(two)), "tv_between_halves": float(np.mean(halves)),
            "expected_unvisited": float(((1.0 - p) ** n).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", choices=sorted(INSTANCES), default="4x5")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--retry", type=int, default=200)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    spec = INSTANCES[a.instance]
    g, cand, demand = build_graph(spec)
    t0 = time.perf_counter()
    states1, level2, caps = enumerate_states(g, cand, demand, spec)
    n1 = len(states1)
    n2 = sum(len(level2[s]) for s in states1)
    patt = Counter(tuple(sorted(caps[d] for d in s)) for s in states1)
    sizes = Counter(len(d) for s in states1 for d in s)
    print(f"{a.instance}: demands {sorted(Counter(demand.tolist()).items())[:3]}... total {demand.sum()}, "
          f"{n1} level-1 states, {n2} joint states ({time.perf_counter() - t0:.0f}s); "
          f"capacity mixes {dict(patt)}; district sizes {dict(sorted(sizes.items()))}", flush=True)
    if a.count_only:
        return 0
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all2 = {(s, s2) for s in states1 for s2 in level2[s]}
    set1 = set(states1)
    starts = {"first": states1[0], "middle": states1[n1 // 2], "last": states1[-1]}
    jobs = [(name, spec, start, caps, a.steps, 1000 + i, a.retry) for i, (name, start) in enumerate(starts.items())]
    with Pool(min(a.procs, len(jobs))) as pool:
        outs = pool.map(_run_chain, jobs)
    names = list(starts)
    joint, lvl1, halves, runs = {}, {}, {}, {}
    for name, c_joint, half1, info in outs:
        c1 = Counter()
        for (k1, _k2), v in c_joint.items():
            c1[k1] += v
        joint[name], lvl1[name] = normalize(c_joint), normalize(c1)
        second = Counter({k: c1[k] - half1.get(k, 0) for k in c1})
        info.update(
            visited_level1=len(c1), visited_joint=len(c_joint),
            unsound_level1_visits=int(sum(v for k, v in c1.items() if k not in set1)),
            unsound_joint_visits=int(sum(v for k, v in c_joint.items() if k not in all2)),
            tv_first_vs_second_half=tv(normalize(half1), normalize(second)))
        runs[name] = info
        print(name, json.dumps(info), flush=True)
    pairs = [(names[i], names[j]) for i in range(3) for j in range(i + 1, 3)]
    pooled1 = normalize(Counter({s: sum(lvl1[n].get(s, 0.0) for n in names) for s in set1}))
    pooledj = normalize(Counter({k: sum(joint[n].get(k, 0.0) for n in names) for k in all2}))
    visited_by = {n: {s for s in set1 if lvl1[n].get(s, 0.0) > 0} for n in names}
    union = set.union(*visited_by.values())
    inter = set.intersection(*visited_by.values())
    rng = np.random.default_rng(0)
    st_w = {s: spanning_tree_weight(g, s) for s in union}
    z = sum(st_w.values())
    tree_on_support = {s: w / z for s, w in st_w.items()}
    unif_on_support = {s: 1.0 / len(union) for s in union}
    pos = [v for v in pooled1.values() if v > 0]
    # which window a district of the visited states uses: demand relative to c * d_bar
    rel = [sum(demand[v] for v in d) / (caps[d] * spec["w"]) for s in union for d in s]
    rel_all = [sum(demand[v] for v in d) / (caps[d] * spec["w"]) for s in set1 for d in s]
    summary = {
        "instance": {"name": a.instance, **{k: v for k, v in spec.items()}, "eps": EPS,
                     "demand": demand.tolist(), "retry_budget": a.retry},
        "steps": a.steps,
        "feasible_level1": n1, "feasible_joint": n2,
        "capacity_mixes": {str(k): v for k, v in patt.items()},
        "district_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "runs": runs,
        "visited_level1_union": len(union), "visited_level1_by_all_chains": len(inter),
        "visited_joint_union": int(sum(1 for v in pooledj.values() if v > 0)),
        "mass_on_states_not_visited_by_all_chains": float(sum(pooled1[s] for s in union - inter)),
        "tv_between_starts_level1": {f"{x}-{y}": tv(lvl1[x], lvl1[y]) for x, y in pairs},
        "tv_between_starts_joint": {f"{x}-{y}": tv(joint[x], joint[y]) for x, y in pairs},
        "iid_noise_floor": {"level1": noise_floor({k: v for k, v in pooled1.items() if v > 0}, a.steps, rng),
                            "joint": noise_floor({k: v for k, v in pooledj.items() if v > 0}, a.steps, rng)},
        "pooled_tv_to_uniform_on_support": tv({s: pooled1[s] for s in union}, unif_on_support),
        "pooled_tv_to_spanning_tree_law_on_support": tv({s: pooled1[s] for s in union}, tree_on_support),
        "max_over_min_pooled_frequency": max(pos) / min(pos),
        "relative_load_visited": [float(min(rel)), float(max(rel))],
        "relative_load_feasible_set": [float(min(rel_all)), float(max(rel_all))],
    }
    (out_dir / f"enumeration_nonuniform_{a.instance}.json").write_text(json.dumps(summary, indent=2))
    rows = [{"capacities": sorted(caps[d] for d in s), "sizes": sorted(len(d) for d in s),
             "pooled": pooled1[s], **{f"emp_{n}": lvl1[n].get(s, 0.0) for n in names}} for s in states1]
    (out_dir / f"enumeration_nonuniform_{a.instance}_states.json").write_text(json.dumps(rows))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("instance", "runs")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
