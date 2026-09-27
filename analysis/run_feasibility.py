"""
Reproducible script for the analysis in 01_feasibility.md.

Run from the repo root:

    python3 analysis/run_feasibility.py

Requires FalcomChain installed (or accessible via PYTHONPATH).
"""

import json
import statistics
import sys
from pathlib import Path

import networkx as nx

from falcomchain.candidates.feasibility import check_facility_density


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
    # Rename candidate_l2 -> super_candidate (FalcomChain's expected name).
    for n in g.nodes:
        g.nodes[n]["super_candidate"] = g.nodes[n].pop("candidate_l2", 0)
    return g


def main():
    g = load_london_graph()
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"Connected: {nx.is_connected(g)}")
    print(f"Avg degree: {2 * g.number_of_edges() / g.number_of_nodes():.2f}")

    n_cand = sum(1 for v, d in g.nodes(data=True) if d.get("candidate"))
    n_super = sum(1 for v, d in g.nodes(data=True) if d.get("super_candidate"))
    print(f"\nLevel-1 candidates: {n_cand}")
    print(f"Level-2 candidates: {n_super}")

    demands = [g.nodes[v]["demand"] for v in g.nodes]
    print(f"\nDemand statistics:")
    print(f"  total:  {sum(demands):,}")
    print(f"  mean:   {statistics.mean(demands):,.0f}")
    print(f"  median: {statistics.median(demands):,.0f}")
    print(f"  stdev:  {statistics.stdev(demands):,.0f}")
    print(f"  min:    {min(demands):,}")
    print(f"  max:    {max(demands):,}  (= w_max)")
    w_max = max(demands)

    print(f"\nAssumption 6.1 at various (demand_target, c_min) settings:")
    print(
        f"{'demand_target':>14} {'c_min':>6} "
        f"{'threshold':>10} {'pass?':>6} {'worst':>11} {'eps_floor':>10}"
    )
    print("-" * 65)
    for dt in (5_000, 10_000, 25_000, 50_000, 100_000):
        for c_min in (1, 2, 3):
            rep = check_facility_density(
                g, demand_target=dt, epsilon=0.1, c_min=c_min
            )
            eps_floor = w_max / (c_min * dt)
            print(
                f"{dt:>14,} {c_min:>6} {int(rep.threshold):>10,} "
                f"{('passes' if rep.passes else 'FAILS'):>6} "
                f"{int(rep.worst_demand):>11,} {eps_floor:>10.4f}"
            )

    print(
        f"\nGranularity floor (paper Eq. 31): "
        f"epsilon >= w_max / (c_min * demand_target)"
    )


if __name__ == "__main__":
    main()
