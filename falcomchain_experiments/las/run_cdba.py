"""
Reproducible CDBA driver for the LAS case study.

Runs three-phase CDBA on the LAS-served subgraph and writes the
augmented candidate set to
``data/derived/cdba_candidates_w{W}_eps{EPS}_cmin{CMIN}_dmax{DMAX}.csv``.

Inputs (paths relative to repo root):
  data/raw/london_graph.json         5,042-node LSOA adjacency graph
  data/raw/LAS_stations.csv          66 real LAS station LSOA codes
  data/derived/lsoa_to_group.csv     4,994 LAS-served LSOAs (drops 48 Brentwood)

Parameters (edit the constants at the top, or pass --w / --eps / --cmin / --dmax):
  W      max workload per team per year   (calls/year, default 3,650)
  EPS    base-level tolerance epsilon^1   (default 0.15)
  CMIN   minimum capacity per L1 district (default 3)
  DMAX   max graph diameter per component (default 12)

The strict per-component demand cap is V_max = c_min * (1 - eps^1) * w.
After Phase 3 every connected component C of G[V \ F^1] satisfies
d(C) < V_max, which establishes Assumption 6.1 of the paper.

Run:
    python -m falcomchain_experiments.las.run_cdba
    python -m falcomchain_experiments.las.run_cdba --cmin 2
    python -m falcomchain_experiments.las.run_cdba --w 4380 --cmin 3
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.readwrite import json_graph

from .cdba_scaffold import cdba_three_phase

REPO = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO / "data/raw/london_graph.json"
STATIONS_PATH = REPO / "data/raw/LAS_stations.csv"
GROUP_PATH = REPO / "data/derived/lsoa_to_group.csv"
OUT_DIR = REPO / "data/derived"

W_DEFAULT = 3650
EPS_DEFAULT = 0.15
CMIN_DEFAULT = 3
DMAX_DEFAULT = 12


def load_las_subgraph() -> nx.Graph:
    """Load london_graph.json and restrict to the 4,994 LAS-served LSOAs."""
    with open(GRAPH_PATH) as f:
        raw = json.load(f)
    G = json_graph.adjacency_graph(raw)
    las_lsoas = set(pd.read_csv(GROUP_PATH)["LSOA21CD"])
    keep = [v for v in G.nodes if G.nodes[v].get("LSOA21CD") in las_lsoas]
    return G.subgraph(keep).copy()


def mark_real_candidates(graph: nx.Graph) -> int:
    """Mark the 66 LAS-station LSOAs as real candidates. Returns the count."""
    station_lsoas = set(pd.read_csv(STATIONS_PATH)["LSOA21CD"].astype(str))
    for v in graph.nodes:
        graph.nodes[v]["candidate"] = 0
    n = 0
    for v in graph.nodes:
        if graph.nodes[v].get("LSOA21CD") in station_lsoas:
            graph.nodes[v]["candidate"] = 1
            n += 1
    return n


def run(w: int, eps: float, c_min: int, d_max: int) -> dict:
    vol_max = c_min * (1 - eps) * w
    out_path = (
        OUT_DIR
        / f"cdba_candidates_w{w}_eps{int(round(eps*100)):d}_cmin{c_min}_dmax{d_max}.csv"
    )

    print(f"w = {w}, eps^1 = {eps}, c_min = {c_min}, d_max = {d_max}")
    print(f"Vol_max = c_min * (1 - eps^1) * w = {vol_max:.0f}")

    g = load_las_subgraph()
    total_demand = sum(g.nodes[v]["demand"] for v in g.nodes)
    print(f"LAS subgraph: |V| = {g.number_of_nodes()}, |E| = {g.number_of_edges()},"
          f" total demand = {total_demand:,}")

    n_real = mark_real_candidates(g)
    print(f"real candidates: {n_real}\n")

    t0 = time.perf_counter()
    artificials = cdba_three_phase(g, V_max=vol_max, d_max=d_max, log=True)
    elapsed = time.perf_counter() - t0

    final_F = n_real + len(artificials)

    sub = g.subgraph(v for v in g.nodes if not g.nodes[v].get("candidate"))
    comps = list(nx.connected_components(sub))
    demands = [sum(g.nodes[v]["demand"] for v in C) for C in comps]
    max_d = max(demands) if demands else 0

    print(f"\n=== CDBA summary @ w={w}, eps={eps}, c_min={c_min}, d_max={d_max} ===")
    print(f"  real candidates       : {n_real}")
    print(f"  artificial candidates : {len(artificials)}")
    print(f"  |F^1|                 : {final_F}  "
          f"({100*final_F/g.number_of_nodes():.1f}% of LSOAs)")
    print(f"  wall time             : {elapsed:.1f}s")
    print(f"  candidate-free comps  : {len(comps)}")
    print(f"  max component demand  : {max_d:,} < Vol_max {vol_max:.0f}")
    assert max_d < vol_max, "Phase 3 failed to establish Assumption 6.1"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "LSOA21CD", "LAT", "LONG", "demand", "is_real"])
        for v in g.nodes:
            if g.nodes[v].get("candidate", 0) == 1:
                n = g.nodes[v]
                writer.writerow([
                    v, n.get("LSOA21CD", ""), n.get("LAT", ""), n.get("LONG", ""),
                    n.get("demand", ""),
                    0 if n.get("candidate_artificial", 0) == 1 else 1,
                ])
    print(f"\nsaved to {out_path.relative_to(REPO)}")

    return {
        "params": {"w": w, "eps": eps, "c_min": c_min, "d_max": d_max,
                   "vol_max": vol_max},
        "n_real": n_real,
        "n_artificial": len(artificials),
        "n_total": final_F,
        "wall_seconds": elapsed,
        "max_residual_component_demand": max_d,
        "out_path": str(out_path.relative_to(REPO)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="LAS CDBA reproducible driver")
    ap.add_argument("--w", type=int, default=W_DEFAULT,
                    help="max workload per team per year (calls/yr)")
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT,
                    help="base-level tolerance epsilon^1")
    ap.add_argument("--cmin", type=int, default=CMIN_DEFAULT,
                    help="minimum capacity c_min^1 (teams per L1 district)")
    ap.add_argument("--dmax", type=int, default=DMAX_DEFAULT,
                    help="max graph diameter per candidate-free component")
    args = ap.parse_args()
    run(w=args.w, eps=args.eps, c_min=args.cmin, d_max=args.dmax)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
