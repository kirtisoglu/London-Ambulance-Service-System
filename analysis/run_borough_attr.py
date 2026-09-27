"""
Assign a numeric `borough` attribute to every node in london_graph.json.

Borough name is derived from `LSOA21NM` (the trailing " NNNX" stripped).
Borough id is the 1-based alphabetical rank across all borough names
present in the graph (33 London boroughs + 1 Essex district, Brentwood).
The mapping is also written to data/derived/borough_id.csv for
downstream tools.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO / "data/raw/london_graph.json"
LOOKUP_PATH = REPO / "data/derived/borough_id.csv"

LSOA_SUFFIX = re.compile(r" \d{3}[A-Z]$")


def borough_of(lsoa_name: str) -> str:
    return LSOA_SUFFIX.sub("", lsoa_name)


def main() -> int:
    print("loading graph...")
    with open(GRAPH_PATH) as f:
        graph = json.load(f)
    nodes = graph["nodes"]

    boroughs = sorted({borough_of(n["LSOA21NM"]) for n in nodes})
    borough_id = {b: i + 1 for i, b in enumerate(boroughs)}
    print(f"  {len(boroughs)} distinct boroughs")

    for n in nodes:
        n["borough"] = borough_id[borough_of(n["LSOA21NM"])]

    print(f"writing {GRAPH_PATH.relative_to(REPO)} ...")
    with open(GRAPH_PATH, "w") as f:
        json.dump(graph, f)

    LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {LOOKUP_PATH.relative_to(REPO)} ...")
    with open(LOOKUP_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["borough_id", "borough_name"])
        for b, i in borough_id.items():
            w.writerow([i, b])

    print("done.")
    print()
    print("id  borough")
    for b, i in borough_id.items():
        print(f"{i:>2}  {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
