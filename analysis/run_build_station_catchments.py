"""
Derive the L1 station service-area partition for ``s_LAS``: each LSOA
goes to the nearest LAS station IN ITS GROUP (BNG straight-line).

Hierarchy invariant: every LSOA is inside one station catchment, which
is inside one Group, which is inside one Sector — by construction.

Outputs:
  - data/derived/lsoa_to_station.csv  — per-LSOA station + group + sector
  - data/derived/stations_polygons.gpkg — 66 dissolved catchment polygons

Run:
    python analysis/run_build_station_catchments.py
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree

# Single source of truth for station → Group is lsoa_to_group.csv:
# each station's Group is derived from its own LSOA21CD (the LSOA it
# physically sits inside).  Earlier, this script imported a hand-written
# STATION_TO_GROUP dict from run_build_groups_v2 — but that dict drifts
# out of sync with lsoa_to_group_overrides.csv when a station's LSOA
# gets reassigned to a different Group (e.g. B7 Ruislip moved from Brent
# to Hillingdon when Hillingdon Group was added; N3 Waterloo moved from
# Deptford to Oval when Lambeth-area LSOAs were re-routed).  Deriving
# from the LSOA truth eliminates that drift surface.

REPO = Path(__file__).resolve().parents[1]
LSOA_GPKG     = REPO / "data/raw/LSOA_2021_London.gpkg"
STATIONS_CSV  = REPO / "data/raw/LAS_stations.csv"
LSOA_GROUP    = REPO / "data/derived/lsoa_to_group.csv"
OUT_STN_CSV   = REPO / "data/derived/lsoa_to_station.csv"
OUT_STN_POLYS = REPO / "data/derived/stations_polygons.gpkg"

LSOA_SUFFIX = re.compile(r" \d{3}[A-Z]$")


def main() -> int:
    lsoa = gpd.read_file(LSOA_GPKG).to_crs(27700)   # BNG metres
    lsoa["x"] = lsoa.geometry.centroid.x
    lsoa["y"] = lsoa.geometry.centroid.y
    lsoa["borough"] = lsoa["LSOA21NM"].str.replace(LSOA_SUFFIX, "", regex=True)

    grp = pd.read_csv(LSOA_GROUP)
    grp = grp[["LSOA21CD", "group", "group_sector"]]
    lsoa = lsoa.merge(grp, on="LSOA21CD", how="inner")
    print(f"  LSOAs (LAS catchment): {len(lsoa)}")

    stations = pd.read_csv(STATIONS_CSV)
    # Derive station → Group from each station's LSOA21CD via
    # lsoa_to_group.csv (the post-override single source of truth).
    truth_group = pd.read_csv(LSOA_GROUP).set_index("LSOA21CD")["group"].to_dict()
    stations_g = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs=4326,
    ).to_crs(27700)
    stations_g["x"] = stations_g.geometry.x
    stations_g["y"] = stations_g.geometry.y
    stations_g["group"] = stations_g["LSOA21CD"].map(truth_group)
    miss = stations_g[stations_g["group"].isna()]
    if not miss.empty:
        raise RuntimeError(f"stations whose LSOA21CD is not in "
                           f"lsoa_to_group.csv: "
                           f"{sorted(miss['station_name'].tolist())}")
    print(f"  stations: {len(stations_g)} across {stations_g['group'].nunique()} groups")

    # Per-Group nearest-station assignment.
    out_rows = []
    for g_name, sub_l in lsoa.groupby("group"):
        sub_s = stations_g[stations_g["group"] == g_name].reset_index(drop=True)
        if sub_s.empty:
            print(f"  WARN: Group {g_name!r} has no stations — LSOAs left "
                  f"unassigned: {len(sub_l)}")
            continue
        tree = cKDTree(sub_s[["x", "y"]].values)
        _, idx = tree.query(sub_l[["x", "y"]].values, k=1)
        for (_, lr), si in zip(sub_l.iterrows(), idx):
            sr = sub_s.iloc[int(si)]
            out_rows.append({
                "LSOA21CD":     lr["LSOA21CD"],
                "LSOA21NM":     lr["LSOA21NM"],
                "borough":      lr["borough"],
                "sector":       lr["group_sector"],
                "group":        g_name,
                "station_name": sr["station_name"],
                "station_code": sr["station_code"],
            })
    out = pd.DataFrame(out_rows).sort_values("LSOA21CD").reset_index(drop=True)
    out.to_csv(OUT_STN_CSV, index=False)
    print(f"\n  wrote {OUT_STN_CSV.relative_to(REPO)}  ({len(out)} rows)")

    print(f"\n  LSOAs per station (descending; showing top 10 + bottom 10):")
    counts = out["station_code"].value_counts()
    names = stations_g.set_index("station_code")["station_name"]
    for code in list(counts.index[:10]) + ["…"] + list(counts.index[-10:]):
        if code == "…":
            print("       …")
        else:
            print(f"       {code:<4s} {names[code]:<22s}  {counts[code]:>4d} LSOAs")

    # Dissolve into station catchment polygons.
    poly = lsoa[["LSOA21CD", "geometry"]].merge(
        out[["LSOA21CD", "station_code", "station_name", "group", "sector"]],
        on="LSOA21CD", how="inner",
    )
    catchments = poly.dissolve(by="station_code").reset_index()
    catchments = catchments[["station_code", "station_name", "group", "sector", "geometry"]]
    catchments.to_file(OUT_STN_POLYS, driver="GPKG")
    print(f"\n  wrote {OUT_STN_POLYS.relative_to(REPO)}  "
          f"({len(catchments)} station catchment polygons)")

    # Contiguity report — within-Group Voronoi can in principle produce
    # disconnected pockets if the borough Voronoi (one tier up) has
    # carved a region awkwardly.  Flag any here.
    import json
    from collections import deque
    with open(REPO / "data/raw/london_graph.json") as f:
        gj = json.load(f)
    id_to_code = {n["id"]: n["LSOA21CD"] for n in gj["nodes"]}
    nb: dict[str, set[str]] = {c: set() for c in id_to_code.values()}
    for u, adj in enumerate(gj["adjacency"]):
        for a in adj:
            v = a["id"]
            if v != u:
                nb[id_to_code[u]].add(id_to_code[v])
    n_split = 0
    for code, members in out.groupby("station_code")["LSOA21CD"]:
        members = set(members)
        unseen = set(members); comps = []
        while unseen:
            s = next(iter(unseen)); c = {s}; q = deque([s])
            while q:
                x = q.popleft()
                for y in nb.get(x, ()):
                    if y in unseen and y not in c:
                        c.add(y); q.append(y)
            unseen -= c; comps.append(c)
        if len(comps) > 1:
            print(f"  ⚠ {code} ({names.get(code, '?')}): {len(comps)} components, "
                  f"sizes={sorted(len(x) for x in comps)}")
            n_split += 1
    if n_split == 0:
        print(f"\n  All 66 station catchments are contiguous.")
    else:
        print(f"\n  {n_split} catchments have multiple components (Voronoi-within-Group "
              f"limitation; not auto-fixed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
