"""
Export the per-LSOA demand vector to a portable CSV.

`run_calls_to_demand.py` already writes the borough-disaggregated
incident counts back into `node["demand"]` on the graph. This script
just dumps that field next to LSOA21CD and population for use by
analysis code that prefers a flat table to a graph file.

Inputs:
  data/raw/london_graph.json

Output:
  data/derived/demand_lsoa.csv
      columns: LSOA21CD, LSOA21NM, population, demand, borough_id
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO / "data/raw/london_graph.json"
OUT_CSV = REPO / "data/derived/demand_lsoa.csv"


def main() -> int:
    with open(GRAPH_PATH) as f:
        raw = json.load(f)

    rows = [
        {
            "LSOA21CD": n["LSOA21CD"],
            "LSOA21NM": n["LSOA21NM"],
            "population": n["population"],
            "demand": n["demand"],
            "borough_id": n["borough"],
        }
        for n in raw["nodes"]
    ]
    df = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"wrote {OUT_CSV.relative_to(REPO)}: {len(df)} LSOAs")
    print(
        f"demand: sum={df['demand'].sum():,.0f}, "
        f"mean={df['demand'].mean():.1f}, "
        f"max={df['demand'].max():.0f}"
    )
    print(
        f"population: sum={df['population'].sum():,}, "
        f"mean={df['population'].mean():.1f}, "
        f"max={df['population'].max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
