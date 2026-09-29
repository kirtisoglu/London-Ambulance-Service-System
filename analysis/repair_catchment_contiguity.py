#!/usr/bin/env python3
"""Make every station catchment of the benchmark state s_LAS contiguous.

The nearest-station rule of run_build_station_catchments.py assigns LSOAs by
straight-line distance within each Group, which can leave a catchment in two
or more pieces on the rook-adjacency graph (river bends, parks, railway land).
This script moves every LSOA of a detached piece to the neighbouring station
whose catchment shares the most rook edges with the piece, preferring a
station of the same Group; when the receiving station belongs to another
Group, the LSOA's Group and sector follow it. It iterates until all station
catchments and all Groups are connected, then rewrites
data/derived/lsoa_to_station.csv and logs the moves in
data/derived/lsoa_to_station_contiguity_moves.csv.

    python analysis/repair_catchment_contiguity.py            # dry run
    python analysis/repair_catchment_contiguity.py --apply    # rewrite the CSV
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parents[1]
GRAPH = REPO / "data/raw/london_graph.json"
CSV_PATH = REPO / "data/derived/lsoa_to_station.csv"
LOG_PATH = REPO / "data/derived/lsoa_to_station_contiguity_moves.csv"


def load_graph():
    g = json.load(open(GRAPH))
    G = nx.Graph()
    for nd in g["nodes"]:
        G.add_node(nd["LSOA21CD"], demand=float(nd["demand"]), name=nd["LSOA21NM"])
    ids = [nd["LSOA21CD"] for nd in g["nodes"]]
    for i, nbrs in enumerate(g["adjacency"]):
        for nb in nbrs:
            G.add_edge(ids[i], ids[int(nb["id"])])
    return G


def pieces(G, rows, key):
    """Connected components of every class of `key`, largest first."""
    members = collections.defaultdict(list)
    for r in rows.values():
        members[r[key]].append(r["LSOA21CD"])
    out = {}
    for k, nodes in members.items():
        comps = sorted(nx.connected_components(G.subgraph(nodes)), key=len, reverse=True)
        out[k] = comps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    G = load_graph()
    rows = {r["LSOA21CD"]: dict(r) for r in csv.DictReader(open(CSV_PATH))}
    station_group = {}
    for r in rows.values():
        station_group.setdefault(r["station_name"], collections.Counter())[r["group"]] += 1
    station_group = {s: c.most_common(1)[0][0] for s, c in station_group.items()}
    station_sector = {r["station_name"]: r["sector"] for r in rows.values()}
    moves = []
    for it in range(10):
        comps = pieces(G, rows, "station_name")
        detached = [(s, c) for s, cs in comps.items() for c in cs[1:]]
        if not detached:
            break
        for s, piece in detached:
            votes = collections.Counter()
            for n in piece:
                for m in G.neighbors(n):
                    if m in rows and m not in piece:
                        votes[rows[m]["station_name"]] += 1
            same = {t: v for t, v in votes.items() if station_group.get(t) == station_group.get(s)}
            pool = same or dict(votes)
            target = max(pool, key=pool.get)
            for n in sorted(piece):
                old = rows[n]
                moves.append(dict(LSOA21CD=n, LSOA21NM=old["LSOA21NM"], demand=G.nodes[n]["demand"],
                                  from_station=s, to_station=target,
                                  from_group=old["group"], to_group=station_group[target],
                                  iteration=it + 1, same_group=int(target in same)))
                rows[n]["station_name"] = target
                rows[n]["station_code"] = next(r["station_code"] for r in rows.values() if r["station_name"] == target)
                rows[n]["group"] = station_group[target]
                rows[n]["sector"] = station_sector[target]
    comps = pieces(G, rows, "station_name")
    gcomps = pieces(G, rows, "group")
    bad_s = [s for s, cs in comps.items() if len(cs) > 1]
    bad_g = [g for g, cs in gcomps.items() if len(cs) > 1]
    print(f"moves: {len(moves)} LSOAs, demand {sum(m['demand'] for m in moves):.0f}, "
          f"cross-group {sum(1 for m in moves if not m['same_group'])}")
    for m in moves:
        print(f"  {m['LSOA21NM']}: {m['from_station']} -> {m['to_station']}"
              f"{'' if m['same_group'] else f' (Group {m['from_group']} -> {m['to_group']})'}")
    print(f"after repair: non-contiguous stations {bad_s}, non-contiguous Groups {bad_g}")
    if args.apply:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys()))
            w.writeheader()
            for r in rows.values():
                w.writerow(r)
        with open(LOG_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(moves[0].keys()) if moves else ["LSOA21CD"])
            w.writeheader()
            for m in moves:
                w.writerow(m)
        print(f"wrote {CSV_PATH.relative_to(REPO)} and {LOG_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
