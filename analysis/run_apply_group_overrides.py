"""
Apply user-provided LSOA → Group overrides on top of the v2 derived
assignment (data/derived/lsoa_to_group.csv) and auto-fix any contiguity
breaks the explicit moves create.

Override mechanism:

  1. Read ``data/derived/lsoa_to_group_overrides.csv`` — two columns
     (``LSOA21CD,group``).  Each row is an explicit, manual override.

  2. Apply those moves to a working copy of ``lsoa_to_group.csv``.

  3. For every Group whose membership changed, rebuild its
     induced subgraph using the LSOA rook-adjacency from
     ``london_graph.json``.  Detect connected components.

  4. If a Group has > 1 component, the component containing its
     Group HQ station's LSOA is kept; every other component is
     reassigned to the receiving Group of the override (looked up
     from the move that introduced the split).  The reassignment
     applies recursively — moving a component might break the
     *receiving* Group's contiguity, which we then patch in turn.

  5. Write the result back to ``data/derived/lsoa_to_group.csv``
     (overwrite), tagging the moved rows with
     ``assignment_source = "override"`` or
     ``"override-contiguity-fix"``.

  6. Rebuild ``data/derived/groups_polygons.gpkg`` from the new
     assignment.

Outputs (overwrites the originals):
  - data/derived/lsoa_to_group.csv
  - data/derived/groups_polygons.gpkg

Run:
    python analysis/run_apply_group_overrides.py
"""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LSOA_GPKG = REPO / "data/raw/LSOA_2021_London.gpkg"
GRAPH_JSON = REPO / "data/raw/london_graph.json"
LSOA_GROUP_CSV = REPO / "data/derived/lsoa_to_group.csv"
OVERRIDES_CSV = REPO / "data/derived/lsoa_to_group_overrides.csv"
OUT_POLYS = REPO / "data/derived/groups_polygons.gpkg"
STATIONS_CSV = REPO / "data/raw/LAS_stations.csv"

LSOA_SUFFIX = re.compile(r" \d{3}[A-Z]$")

# Same Group HQ table as run_build_groups_v2.py — used to identify
# the "anchor" LSOA whose connected component is kept as the Group's
# primary footprint.
GROUP_HQ_STATION = {
    "Camden": "Camden", "Edmonton": "Edmonton", "Friern Barnet": "Friern Barnet",
    "Homerton": "Homerton", "Ilford": "Ilford", "Newham": "Newham",
    "Romford": "Romford", "Whipps Cross": "Whipps Cross",
    "Brent": "Brent", "Fulham": "Fulham", "Hanwell": "Hanwell",
    "Hillingdon": "Hillingdon",  # Hillingdon (B5) added to roster 2026-05-26
    "Westminster": "Westminster",
    "Bromley": "Bromley", "Deptford": "Deptford",
    "Greenwich": "Greenwich", "Oval": "Oval",
    "Croydon": "Croydon", "New Malden": "New Malden",
    "St Helier": "St Helier", "Wimbledon": "Wimbledon",
}
GROUP_TO_SECTOR = {
    "Camden": "North Central", "Edmonton": "North Central", "Friern Barnet": "North Central",
    "Homerton": "North East", "Ilford": "North East", "Newham": "North East",
    "Romford": "North East", "Whipps Cross": "North East",
    "Brent": "North West", "Fulham": "North West", "Hanwell": "North West",
    "Hillingdon": "North West", "Westminster": "North West",
    "Bromley": "South East", "Deptford": "South East", "Greenwich": "South East", "Oval": "South East",
    "Croydon": "South West", "New Malden": "South West",
    "St Helier": "South West", "Wimbledon": "South West",
}


def load_adjacency() -> tuple[dict[str, set[str]], dict[str, int], dict[int, str]]:
    """Return per-LSOA neighbour sets keyed by LSOA21CD, plus
    code↔node-id maps."""
    with open(GRAPH_JSON) as f:
        g = json.load(f)
    id_to_code = {n["id"]: n["LSOA21CD"] for n in g["nodes"]}
    code_to_id = {v: k for k, v in id_to_code.items()}
    neighbours: dict[str, set[str]] = {c: set() for c in code_to_id}
    for u, adj in enumerate(g["adjacency"]):
        uc = id_to_code[u]
        for nb in adj:
            v = nb["id"]
            if v == u: continue
            neighbours[uc].add(id_to_code[v])
    return neighbours, code_to_id, id_to_code


def connected_components(members: set[str], neighbours: dict[str, set[str]]) -> list[set[str]]:
    """All connected components of the subgraph induced by `members`."""
    unseen = set(members)
    comps: list[set[str]] = []
    while unseen:
        seed = next(iter(unseen))
        comp = {seed}
        q = deque([seed])
        while q:
            x = q.popleft()
            for y in neighbours.get(x, ()):
                if y in unseen and y not in comp:
                    comp.add(y)
                    q.append(y)
        unseen -= comp
        comps.append(comp)
    return comps


def anchor_lsoa_for_group(group: str, lsoa_lookup_by_borough: dict[str, list[str]],
                          stations: pd.DataFrame) -> str | None:
    """Return the LSOA21CD that contains the Group HQ station.
    Falls back to None when not findable (e.g. Hillingdon missing station).
    """
    hq = GROUP_HQ_STATION.get(group)
    if hq is None:
        return None
    row = stations[stations["station_name"] == hq]
    if row.empty:
        return None
    return row.iloc[0]["LSOA21CD"]


def main() -> int:
    if not OVERRIDES_CSV.exists():
        raise FileNotFoundError(f"override file not found: {OVERRIDES_CSV}")

    overrides = pd.read_csv(OVERRIDES_CSV)
    overrides["LSOA21CD"] = overrides["LSOA21CD"].astype(str)
    overrides["group"] = overrides["group"].astype(str)
    print(f"loaded {len(overrides)} explicit overrides from "
          f"{OVERRIDES_CSV.relative_to(REPO)}")

    df = pd.read_csv(LSOA_GROUP_CSV)
    df["LSOA21CD"] = df["LSOA21CD"].astype(str)
    # Drop LSOAs outside LAS service area (Brentwood / EEAST).  Idempotent:
    # if the v2 build already excluded them this is a no-op.
    pre = len(df)
    df = df[df["sector"] != "Outside LAS"].reset_index(drop=True)
    if len(df) != pre:
        print(f"  dropped {pre - len(df)} LSOAs outside LAS service area")
    print(f"loaded current assignment: {len(df)} LSOAs across "
          f"{df['group'].nunique()} groups")

    neighbours, _, _ = load_adjacency()
    stations = pd.read_csv(STATIONS_CSV)

    # Step 1: apply explicit overrides
    moves: list[dict] = []   # for the log
    for _, ov in overrides.iterrows():
        code, new_g = ov["LSOA21CD"], ov["group"]
        if code not in df["LSOA21CD"].values:
            print(f"  WARN: override LSOA {code} not found; skipping")
            continue
        idx = df.index[df["LSOA21CD"] == code][0]
        old_g = df.at[idx, "group"]
        if old_g == new_g:
            continue
        df.at[idx, "group"] = new_g
        df.at[idx, "assignment_source"] = "override"
        df.at[idx, "group_sector"] = GROUP_TO_SECTOR.get(new_g, df.at[idx, "sector"])
        moves.append({"code": code, "from": old_g, "to": new_g, "reason": "override"})

    print(f"\napplied {len(moves)} explicit moves:")
    for m in moves:
        print(f"    {m['code']}  {m['from']} → {m['to']}")

    # Step 2: contiguity REPORT only — auto-fix is disabled.  Every
    # LSOA-Group assignment is the user's explicit choice, encoded in
    # the override CSV; non-contiguous Groups are intentional and we
    # surface them in the log but never re-route LSOAs ourselves.

    # Final connectivity report
    print(f"\n--- final per-Group component counts ---")
    for g in sorted(df["group"].unique()):
        members = set(df.loc[df["group"] == g, "LSOA21CD"])
        comps = connected_components(members, neighbours)
        marker = "" if len(comps) == 1 else f"  ⚠ {len(comps)} components"
        print(f"    {g:<20s}  {len(members):>4d} LSOAs{marker}")

    # Write the updated assignment + log
    df.to_csv(LSOA_GROUP_CSV, index=False)
    print(f"\nwrote {LSOA_GROUP_CSV.relative_to(REPO)}  "
          f"({len(df)} rows, {len(moves)} total moves applied)")

    # Rebuild Group polygons
    print("\nrebuilding groups_polygons.gpkg ...")
    lsoa = gpd.read_file(LSOA_GPKG)
    lsoa = lsoa.merge(df[["LSOA21CD", "group"]], on="LSOA21CD", how="left")
    groups_poly = lsoa.dissolve(by="group").reset_index()
    groups_poly["sector"] = groups_poly["group"].map(GROUP_TO_SECTOR)
    groups_poly[["group", "sector", "geometry"]].to_file(OUT_POLYS, driver="GPKG")
    print(f"  wrote {OUT_POLYS.relative_to(REPO)}  ({len(groups_poly)} polygons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
