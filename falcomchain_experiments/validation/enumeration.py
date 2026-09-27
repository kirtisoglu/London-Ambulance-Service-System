"""
Exact-enumeration validation of the FalCom sampler on a 3 x 4 grid.

Instance: 12 units of demand 100, four candidate sites, w = 300 (four units of
capacity), eps = 0.15, c^1 in {1, 2}, c^2 in [2, 4], kappa^2_min = 2. Under
these numbers a level-1 district is admissible iff it is a connected set of
exactly 3 units (one capacity unit) or exactly 6 units (two units) that
contains a candidate, so every feasible hierarchical state can be listed by
exact cover. The sampler (hierarchical proposal, counting predicate, uniform
cut selection, always-accept) is run from several very different initial
states and its empirical distribution is compared with

- the full list of feasible states (support and soundness),
- itself across starts and across halves of a run (start-independence,
  total-variation distance), and
- two reference laws: uniform over feasible level-1 states and the
  spanning-tree law (product of spanning-tree counts of the districts).

Run from the repository root::

    python -m falcomchain_experiments.validation.enumeration --steps 200000
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter
from functools import partial
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

ROWS, COLS = 3, 4
DEMAND = 100.0
W = 300.0
EPS = 0.15
C_MAX = 2
C_MIN_SUPER, C_MAX_SUPER, KAPPA = 2, 4, 2
CANDIDATES = {(0, 0), (0, 3), (2, 1), (1, 2)}
OUT_DIR = Path(__file__).resolve().parents[2] / "data/derived/validation"


def build_graph():
    g = nx.grid_2d_graph(ROWS, COLS)
    g = nx.relabel_nodes(g, {rc: r * COLS + c for (r, c) in list(g.nodes) for rc in [(r, c)]})
    cand = {r * COLS + c for (r, c) in CANDIDATES}
    for n in g.nodes:
        r, c = divmod(n, COLS)
        g.nodes[n].update(demand=DEMAND, area=1.0, C_X=float(c), C_Y=float(r),
                          candidate=1 if n in cand else 0, boundary_node=False, boundary_perim=0)
    return g, cand


# ---------------------------------------------------------------------------
# Exact enumeration
# ---------------------------------------------------------------------------

def connected_sets(g, size):
    out = set()
    for combo in itertools.combinations(sorted(g.nodes), size):
        if nx.is_connected(g.subgraph(combo)):
            out.add(frozenset(combo))
    return out


def enumerate_level1(g, cand):
    """All partitions into connected 3- and 6-sets that contain a candidate."""
    pieces = [s for size in (3, 6) for s in connected_sets(g, size) if s & cand]
    by_node = {n: [s for s in pieces if n in s] for n in g.nodes}
    states = set()

    def rec(uncovered, chosen):
        if not uncovered:
            states.add(frozenset(chosen))
            return
        n = min(uncovered)
        for s in by_node[n]:
            if s <= uncovered:
                rec(uncovered - s, chosen + [s])

    rec(frozenset(g.nodes), [])
    return sorted(states, key=lambda st: sorted(sorted(d) for d in st))


def enumerate_level2(g, state):
    """Set partitions of the districts into connected super-districts holding
    at least KAPPA districts and C_MIN_SUPER..C_MAX_SUPER units."""
    districts = sorted(state, key=sorted)
    units = {d: len(d) // 3 for d in districts}
    adj = {d: {e for e in districts if e != d and any(g.has_edge(u, v) for u in d for v in e)}
           for d in districts}

    def ok(block):
        u = sum(units[d] for d in block)
        if len(block) < KAPPA or not (C_MIN_SUPER <= u <= C_MAX_SUPER):
            return False
        seen, stack = {block[0]}, [block[0]]
        while stack:
            d = stack.pop()
            for e in adj[d]:
                if e in block and e not in seen:
                    seen.add(e); stack.append(e)
        return len(seen) == len(block)

    results = set()

    def rec(remaining, blocks):
        if not remaining:
            results.add(frozenset(frozenset(b) for b in blocks)); return
        first = remaining[0]
        rest = remaining[1:]
        for k in range(len(rest) + 1):
            for others in itertools.combinations(rest, k):
                block = [first, *others]
                if ok(block):
                    rec([d for d in rest if d not in others], blocks + [block])

    rec(districts, [])
    return results


def spanning_tree_weight(g, state):
    w = 1.0
    for d in state:
        sub = g.subgraph(d)
        L = nx.laplacian_matrix(sub).toarray().astype(float)
        w *= round(abs(np.linalg.det(L[1:, 1:])))
    return w


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def make_partition(graph, state):
    flips, teams = {}, {}
    for i, d in enumerate(sorted(state, key=sorted), start=1):
        teams[i] = len(d) // 3
        for n in d:
            flips[n] = i
    flip = Flip(flips=flips, team_flips=teams, new_ids=frozenset(teams), merged_ids=frozenset())
    return Partition(capacity_level=C_MAX, assignment=flips, graph=graph, flip=flip)


def key1(partition):
    return frozenset(frozenset(v) for v in partition.parts.values())


def key2(partition):
    parts = partition.parts
    return frozenset(frozenset(frozenset(parts[d]) for d in ids) for ids in partition.super_parts.values())


def run_chain(graph, start_state, steps, seed):
    set_seed(seed)
    partition = make_partition(graph, start_state)
    state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
    proposal = partial(hierarchical_recom, epsilon_base=EPS, epsilon_super=EPS, demand_target=W,
                       c_min_super=C_MIN_SUPER, c_max_super=C_MAX_SUPER, min_districts_super=KAPPA,
                       max_attempts_base=200, max_attempts_super=200)
    chain = MarkovChain(proposal=proposal, constraints=[], accept=always_accept,
                        initial_state=state, total_steps=steps)
    c1, c2, first_half = Counter(), Counter(), Counter()
    t0 = time.perf_counter()
    for i, st in enumerate(chain):
        if i == 0:
            continue        # the start (identity level-2 grouping) is not a sample
        k1 = key1(st.partition)
        c1[k1] += 1
        c2[key2(st.partition)] += 1
        if i < steps // 2:
            first_half[k1] += 1
    return c1, c2, first_half, chain.rejection_report(), time.perf_counter() - t0


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def normalize(counter):
    tot = sum(counter.values())
    return {k: v / tot for k, v in counter.items()}


def plot_from_json(out_dir: Path) -> None:
    """Figure for the paper: (a) per-state sampled frequency against the
    spanning-tree law on log-log axes, coloured by district-size pattern;
    (b) the mass each pattern receives under the sampled law, the uniform
    law and the spanning-tree law."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads((out_dir / "enumeration_3x4_states.json").read_text())
    summary = json.loads((out_dir / "enumeration_3x4.json").read_text())
    n1 = len(rows)
    patterns = sorted({tuple(r["sizes"]) for r in rows}, key=lambda t: (len(t), t))
    colors = {p: c for p, c in zip(patterns, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for pat in patterns:
        xs = [r["law_tree"] for r in rows if tuple(r["sizes"]) == pat]
        ys = [r["pooled_empirical"] for r in rows if tuple(r["sizes"]) == pat]
        ax.scatter(xs, ys, s=18, alpha=0.8, color=colors[pat],
                   label=f"district sizes {pat} ({len(xs)} states)")
    lo = min(min(r["law_tree"], r["pooled_empirical"]) for r in rows) * 0.7
    hi = max(max(r["law_tree"], r["pooled_empirical"]) for r in rows) * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="spanning-tree law")
    ax.axhline(1 / n1, color="gray", lw=0.8, ls=":", label="uniform law")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("spanning-tree law"); ax.set_ylabel("sampled frequency (three chains pooled)")
    ax.set_title(f"(a) all {n1} feasible level-1 states")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    pm = summary["pattern_mass"]
    keys = [str(p) for p in patterns]
    import numpy as np
    x = np.arange(len(keys)); wdt = 0.26
    ax.bar(x - wdt, [pm[k]["empirical"] for k in keys], wdt, label="sampled", color="#333333")
    ax.bar(x, [pm[k]["uniform"] for k in keys], wdt, label="uniform law", color="#bbbbbb")
    ax.bar(x + wdt, [pm[k]["spanning_tree_law"] for k in keys], wdt, label="spanning-tree law", color="#7f9fc5")
    ax.set_xticks(x); ax.set_xticklabels([f"sizes {k}" for k in keys], fontsize=8)
    ax.set_ylabel("probability mass of the pattern")
    ax.set_title("(b) mass by district-size pattern")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_enumeration_3x4.png", dpi=160)
    fig.savefig(out_dir / "fig_enumeration_3x4.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--plot-only", action="store_true", help="re-draw the figure from the saved JSON files")
    a = ap.parse_args()
    if a.plot_only:
        plot_from_json(Path(a.out_dir))
        return 0
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    g, cand = build_graph()
    Assignment.travel_times = {(u, v): abs(g.nodes[u]["C_X"] - g.nodes[v]["C_X"]) + abs(g.nodes[u]["C_Y"] - g.nodes[v]["C_Y"])
                               for u in g.nodes for v in g.nodes}
    states1 = enumerate_level1(g, cand)
    level2 = {s: enumerate_level2(g, s) for s in states1}
    states1 = [s for s in states1 if level2[s]]          # a level-1 state needs a feasible level-2
    n1 = len(states1)
    n2 = sum(len(level2[s]) for s in states1)
    sizes = Counter(tuple(sorted(len(d) for d in s)) for s in states1)
    print(f"feasible level-1 states: {n1} (by district sizes {dict(sizes)}); "
          f"feasible (level-1, level-2) states: {n2}", flush=True)
    st_weight = {s: spanning_tree_weight(g, s) for s in states1}
    Z = sum(st_weight.values())
    law_tree = {s: w / Z for s, w in st_weight.items()}
    law_uniform = {s: 1.0 / n1 for s in states1}
    all2 = {(s, s2) for s in states1 for s2 in level2[s]}

    graph = Graph.from_networkx(g)
    starts = {"first": states1[0], "last": states1[-1], "middle": states1[n1 // 2]}
    results, empirical = {}, {}
    for name, start in starts.items():
        c1, c2, half, rej, secs = run_chain(graph, start, a.steps, seed=hash(name) % 10_000)
        emp = normalize(c1)
        joint_keys = set()
        for k2 in c2:
            k1 = frozenset(d for blk in k2 for d in blk)
            joint_keys.add((k1, k2))
        results[name] = {
            "steps": a.steps, "seconds": round(secs, 1), "acceptance_rate": rej["acceptance_rate"],
            "rejection_causes": rej["causes"],
            "visited_level1": len(c1), "feasible_level1": n1,
            "unsound_level1_visits": int(sum(v for k, v in c1.items() if k not in st_weight)),
            "visited_joint": len(joint_keys), "feasible_joint": n2,
            "unsound_joint_visits": int(sum(1 for k in joint_keys if k not in all2)),
            "tv_first_vs_second_half": tv(normalize(half), normalize(Counter({k: c1[k] - half.get(k, 0) for k in c1}))),
            "tv_to_uniform": tv(emp, law_uniform), "tv_to_spanning_tree_law": tv(emp, law_tree),
        }
        empirical[name] = emp
        print(name, json.dumps(results[name]), flush=True)
    names = list(starts)
    cross = {f"{names[i]}-{names[j]}": tv(empirical[names[i]], empirical[names[j]])
             for i in range(len(names)) for j in range(i + 1, len(names))}
    pooled = normalize(Counter({s: sum(empirical[n].get(s, 0.0) for n in names) for s in states1}))
    # How the sampled law relates to the two reference laws, split by district-size pattern:
    # the mass each pattern receives, and the spanning-tree law *conditioned on the pattern*
    # (tree law within a pattern, pattern mass taken from the chain).
    pattern_of = {s: tuple(sorted(len(d) for d in s)) for s in states1}
    patterns = sorted(set(pattern_of.values()))
    pattern_mass = {}
    for pat in patterns:
        members = [s for s in states1 if pattern_of[s] == pat]
        pattern_mass[str(pat)] = {
            "states": len(members),
            "empirical": sum(pooled[s] for s in members),
            "uniform": len(members) / n1,
            "spanning_tree_law": sum(law_tree[s] for s in members),
            "empirical_max_over_min": max(pooled[s] for s in members) / min(pooled[s] for s in members),
        }
    law_tree_within = {}
    for pat in patterns:
        members = [s for s in states1 if pattern_of[s] == pat]
        z = sum(st_weight[s] for s in members)
        m = pattern_mass[str(pat)]["empirical"]
        for s in members:
            law_tree_within[s] = m * st_weight[s] / z
    summary = {"instance": {"rows": ROWS, "cols": COLS, "w": W, "eps": EPS, "c_max": C_MAX,
                            "c_super": [C_MIN_SUPER, C_MAX_SUPER], "kappa": KAPPA,
                            "candidates": sorted(cand)},
               "feasible_level1": n1, "feasible_joint": n2, "district_size_patterns": {str(k): v for k, v in sizes.items()},
               "runs": results, "tv_between_starts": cross,
               "pooled_tv_to_uniform": tv(pooled, law_uniform), "pooled_tv_to_spanning_tree_law": tv(pooled, law_tree),
               "max_over_min_pooled_frequency": max(pooled.values()) / min(pooled.values()),
               "pattern_mass": pattern_mass,
               "pooled_tv_to_spanning_tree_law_within_pattern": tv(pooled, law_tree_within)}
    (out_dir / "enumeration_3x4.json").write_text(json.dumps(summary, indent=2))
    # per-state table for a figure
    rows = []
    for s in states1:
        rows.append({"sizes": sorted(len(d) for d in s), "spanning_tree_weight": st_weight[s],
                     "law_tree": law_tree[s], "pooled_empirical": pooled[s],
                     **{f"emp_{n}": empirical[n].get(s, 0.0) for n in names}})
    (out_dir / "enumeration_3x4_states.json").write_text(json.dumps(rows))
    try:
        plot_from_json(out_dir)
    except Exception as exc:  # noqa: BLE001
        summary["figure_error"] = str(exc)
        (out_dir / "enumeration_3x4.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
