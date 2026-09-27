"""
Convert the existing per-LSOA population proxy into call-volume demand
(Option 2 from analysis/06_parameters.md).

Steps:
1. Load `data/raw/london_graph.json`. The very first run finds
   ``demand`` = 2021 LSOA population on each node and renames it to
   ``population``. Subsequent (idempotent) runs skip that rename
   because ``population`` is already present.
2. Read borough-level monthly incident counts from
   `data/raw/ambulance-borough-monthly.xlsx`, sum the 2010-2015
   columns, annualise.
3. Round each borough's annual total to an integer, then disaggregate
   it to its LSOAs by **largest-remainder apportionment** with
   per-LSOA weights = population.  This produces an INTEGER demand
   per LSOA that (a) sums exactly to the borough's annual total and
   (b) is as close as possible to ``B_b * pop_v / pop_b``.  Largest
   remainder is the same method used for parliamentary seat
   apportionment.
   For 48 Brentwood LSOAs (Essex, outside LAS coverage but in the
   graph) impute a total at the London-wide incidents-per-capita
   rate so the graph stays connected and feasibility analysis isn't
   distorted by zero-demand pockets; then apply largest-remainder
   within Brentwood as well.
4. Write the result back into `node["demand"]` (the per-LSOA call
   count) and overwrite `data/raw/london_graph.json`.
5. Add a `demand` column next to `population` in
   `data/raw/LSOA_2021_London_population.csv`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO / "data/raw/london_graph.json"
EXCEL_PATH = REPO / "data/raw/ambulance-borough-monthly.xlsx"
POP_CSV = REPO / "data/raw/LSOA_2021_London_population.csv"

# LSOA21NM ends with " NNNX" (3 digits + letter). Strip to get borough.
LSOA_SUFFIX = re.compile(r" \d{3}[A-Z]$")


def borough_of(lsoa_name: str) -> str:
    return LSOA_SUFFIX.sub("", lsoa_name)


def load_borough_annual_2010_2015() -> dict[str, float]:
    """Annualised mean incident count per borough over 2010-2015."""
    df = pd.read_excel(EXCEL_PATH, "All Ambulance Attended")
    borough_rows = df[df["Code"].astype(str).str.startswith("E09")]

    # Pick all monthly columns whose label contains a year in 2010..2015.
    # Columns look like "Jan 2010", "Feb 2011", ... and one anomalous
    # "May2014" (per analysis/04_data_sources.md it's actually May 2016
    # mislabelled — exclude it from the 2010-2015 window).
    months_2010_2015 = []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        if c == "May2014":
            continue
        for y in range(2010, 2016):
            if str(y) in c:
                months_2010_2015.append(c)
                break

    print(f"  using {len(months_2010_2015)} monthly columns 2010-2015")
    monthly = borough_rows.set_index("Borough")[months_2010_2015].sum(axis=1)
    annual = monthly * 12 / len(months_2010_2015)
    return annual.to_dict()


def largest_remainder(weights: dict[int, float], total: int) -> dict[int, int]:
    """Largest-remainder apportionment.

    ``weights[k]`` are the relative weights for splitting ``total``
    across keys (they need not sum to ``total``).  Returns per-key
    non-negative integers summing exactly to ``total``, each within
    1 of the population-proportional float.  Ties are broken by key
    for determinism.
    """
    keys = list(weights.keys())
    w = [weights[k] for k in keys]
    s = sum(w)
    if s <= 0 or total == 0:
        return {k: 0 for k in keys}
    scaled = [wi * total / s for wi in w]
    floors = [int(x) for x in scaled]
    leftover = total - sum(floors)
    # Sort indices by (remainder desc, key asc) and bump top `leftover`.
    order = sorted(
        range(len(keys)),
        key=lambda i: (-(scaled[i] - floors[i]), keys[i]),
    )
    out = dict(zip(keys, floors))
    for i in order[:leftover]:
        out[keys[i]] += 1
    return out


def main() -> int:
    print("loading graph...")
    with open(GRAPH_PATH) as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    print(f"  {len(nodes)} nodes")

    # 1. Ensure `population` exists on every node.  First-run case:
    #    the graph carries `demand = LSOA population`; rename it.
    #    Subsequent runs: `population` is already there, skip.
    if all("population" in n for n in nodes):
        print("  `population` already present on every node — skipping rename")
    elif all("demand" in n for n in nodes):
        print("  first-run rename: `demand` (LSOA population) -> `population`")
        for n in nodes:
            n["population"] = n.pop("demand")
    else:
        sys.exit("nodes lack both `population` and `demand` — aborting")

    # 2. Borough annual incidents 2010-2015 (rounded to integer so that
    #    largest-remainder produces integer LSOA demands that sum
    #    exactly to each borough's official annual count).
    print("loading borough-monthly incidents...")
    borough_annual_raw = load_borough_annual_2010_2015()
    borough_annual = {b: int(round(v)) for b, v in borough_annual_raw.items()}
    print(f"  {len(borough_annual)} boroughs; total annual incidents = "
          f"{sum(borough_annual.values()):,}")

    # 3. Borough-level population sums from the graph
    borough_pop: dict[str, int] = defaultdict(int)
    for n in nodes:
        borough_pop[borough_of(n["LSOA21NM"])] += n["population"]

    # London-wide incidents-per-capita (used as imputation rate for
    # boroughs absent from the LAS data, e.g. Brentwood / Essex).
    london_total_incidents = sum(borough_annual.values())
    london_total_pop = sum(
        pop for b, pop in borough_pop.items() if b in borough_annual
    )
    london_rate = london_total_incidents / london_total_pop
    print(f"  London-wide rate = {london_rate:.4f} incidents per person per year")

    missing = sorted(b for b in borough_pop if b not in borough_annual)
    if missing:
        missing_lsoa_count = sum(
            1 for n in nodes if borough_of(n["LSOA21NM"]) in missing
        )
        print(f"  imputing London rate for {len(missing)} non-LAS borough(s) "
              f"({missing_lsoa_count} LSOAs): {missing}")

    # 4. Per-LSOA INTEGER demand via largest-remainder per borough.
    #    For each borough, pop-weighted apportionment of the borough's
    #    rounded annual total across its LSOAs; sums match exactly.
    #    Brentwood and any other absent boroughs get an imputed total
    #    from the London-wide rate and the same apportionment.
    new_demand: dict[int, int] = {}
    imputed_count = 0
    by_borough: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        by_borough[borough_of(n["LSOA21NM"])].append(n)
    for b, members in by_borough.items():
        if b in borough_annual:
            total = borough_annual[b]
        else:
            total = int(round(london_rate * borough_pop[b]))
            imputed_count += len(members)
        weights = {m["id"]: float(m["population"]) for m in members}
        alloc = largest_remainder(weights, total)
        for m in members:
            m["demand"] = int(alloc[m["id"]])
            new_demand[m["id"]] = m["demand"]

    total_alloc = sum(new_demand.values())
    print(f"  new graph demand total = {total_alloc:,} (integer; mean per LSOA "
          f"= {total_alloc/len(nodes):.1f}; imputed = {imputed_count})")
    # Verify per-borough sums match the rounded borough totals exactly.
    by_b_sums: dict[str, int] = defaultdict(int)
    for n in nodes:
        by_b_sums[borough_of(n["LSOA21NM"])] += n["demand"]
    bad = [b for b in borough_annual if by_b_sums[b] != borough_annual[b]]
    if bad:
        sys.exit(f"BUG: largest-remainder failed to match {len(bad)} borough "
                 f"totals exactly: {bad[:5]}")
    print(f"  per-borough sums match official totals exactly for all "
          f"{len(borough_annual)} LAS boroughs.")

    # 5. Save graph (overwrite)
    print(f"writing {GRAPH_PATH.relative_to(REPO)} ...")
    with open(GRAPH_PATH, "w") as f:
        json.dump(graph, f)

    # 6. Update LSOA_2021_London_population.csv
    print(f"updating {POP_CSV.relative_to(REPO)} ...")
    pop_df = pd.read_csv(POP_CSV)
    lsoa_to_demand = {
        n["LSOA21CD"]: n["demand"] for n in nodes
    }
    pop_df["demand"] = pop_df["LSOA21CD"].map(lsoa_to_demand)
    n_missing = pop_df["demand"].isna().sum()
    if n_missing:
        print(f"  warning: {n_missing} LSOAs in CSV had no graph match "
              f"(left as NaN)")
    pop_df.to_csv(POP_CSV, index=False)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
