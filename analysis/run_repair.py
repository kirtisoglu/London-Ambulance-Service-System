"""
Run repair_facility_density on the LAS graph at several settings to
quantify how many artificial candidates are needed for Assumption 6.1
to hold.

The "weighted_center" strategy is O(|V|^2) per iteration and does not
finish in a reasonable time on a single 5000-node violating component.
We use "highest_demand" first (very fast: O(|V|) per iteration) to
get a count, then only run "weighted_center" at one chosen setting
if we want spatially-balanced placements for the paper.

Output feeds 02_repair.md.
"""

import copy
import json
import statistics
import sys
import time
from pathlib import Path

import networkx as nx

from falcomchain.candidates.feasibility import (
    check_facility_density,
    repair_facility_density,
)


GRAPH_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "london_graph.json"
)


def load_london_graph(path=GRAPH_PATH):
    with open(path) as f:
        raw = json.load(f)
    g = nx.Graph()
    for n in raw["nodes"]:
        g.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for src, neighbors in enumerate(raw["adjacency"]):
        for nbr in neighbors:
            g.add_edge(src, nbr["id"])
    for n in g.nodes:
        g.nodes[n]["super_candidate"] = g.nodes[n].pop("candidate_l2", 0)
    return g


def main():
    base = load_london_graph()
    n_real = sum(1 for v, d in base.nodes(data=True) if d.get("candidate"))
    print(f"Loaded LAS graph: {base.number_of_nodes()} nodes, "
          f"{n_real} real candidates")

    # Operationally calibrated settings (see analysis/06_parameters.md):
    # demand_target = total_demand / (|P¹| · c_avg) with fleet anchor
    # |P¹| · c_avg ≈ 446 (LAS 2017/18 ambulance count). Population
    # demand total = 9.17 M, so w ≈ 20,000 across capacity regimes.
    # ε = 0.15 satisfies the granularity floor w_max/(c_min·w) at all
    # listed settings.
    settings = [
        # (demand_target, c_min, epsilon)
        (10_000, 4, 0.15),  # fine: ~225 districts, 4 teams each
        (20_000, 2, 0.15),  # RECOMMENDED PRIMARY: ~225 districts, (2,4) teams
        (20_000, 3, 0.15),  # alt: ~150 districts, 3 teams each
        (20_000, 4, 0.15),  # alt: ~115 districts, 4 teams each
        (40_000, 2, 0.15),  # coarse: ~115 districts, (2,4) teams
    ]

    strategies = ["highest_demand", "fast_center", "balanced_separator"]
    try:
        import pymetis  # noqa: F401
        strategies.append("metis_separator")
    except ImportError:
        print("(pymetis not available; skipping metis_separator)")

    for strategy in strategies:
        print(f"\n=== Strategy: {strategy} ===")
        print(
            f"\n{'demand_target':>14} {'c_min':>5} {'eps':>5} "
            f"{'added':>6} {'total':>6} {'time(s)':>8} {'passes?':>8}"
        )
        print("-" * 60)
        for dt, c_min, eps in settings:
            g = copy.deepcopy(base)
            t0 = time.time()
            added = repair_facility_density(
                g,
                demand_target=dt,
                epsilon=eps,
                strategy=strategy,
                c_min=c_min,
            )
            dt_elapsed = time.time() - t0
            total = sum(
                1 for v, d in g.nodes(data=True) if d.get("candidate")
            )
            post = check_facility_density(
                g, demand_target=dt, epsilon=eps, c_min=c_min
            )
            print(
                f"{dt:>14,} {c_min:>5} {eps:>5.2f} "
                f"{len(added):>6} {total:>6} {dt_elapsed:>8.1f} "
                f"{('OK' if post.passes else 'FAILS'):>8}",
                flush=True,
            )


if __name__ == "__main__":
    main()
