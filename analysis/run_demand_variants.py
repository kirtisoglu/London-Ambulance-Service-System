"""
Alternative within-borough apportionment of the borough incident totals
(sensitivity check for the London demand proxy).

The paper's demand file ``data/derived/demand_lsoa.csv`` apportions each
borough's annual incidents to its LSOAs in proportion to population. This
script writes ``data/derived/demand_lsoa_uniform.csv``, in which each
borough's total is split equally across its LSOAs (largest-remainder
rounding, so borough totals are preserved exactly). Run from the repository
root::

    python analysis/run_demand_variants.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/derived/demand_lsoa.csv"
DST = REPO / "data/derived/demand_lsoa_uniform.csv"


def largest_remainder(total: float, n: int) -> list[int]:
    base = math.floor(total / n)
    rem = int(round(total)) - base * n
    return [base + 1 if i < rem else base for i in range(n)]


def main() -> int:
    df = pd.read_csv(SRC)
    out = []
    for borough, grp in df.groupby("borough_id", sort=False):
        shares = largest_remainder(float(grp["demand"].sum()), len(grp))
        for (code, name), share in zip(zip(grp["LSOA21CD"], grp["LSOA21NM"]), shares):
            out.append((code, name, share, borough))
    res = pd.DataFrame(out, columns=["LSOA21CD", "LSOA21NM", "demand", "borough_id"])
    res.to_csv(DST, index=False)
    print(f"{DST.name}: {len(res)} LSOAs, total {res['demand'].sum():,.0f} "
          f"(population-weighted file: {df['demand'].sum():,.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
