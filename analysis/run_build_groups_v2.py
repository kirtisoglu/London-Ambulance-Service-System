"""
Build LSOA → Group at LSOA-exact precision by combining two truth
sources, in priority order:

  Tier 1 (exact).  When a London borough lies *entirely* inside a
    single LAS Group (per the 2018 FiveSectorMap + 2025 Departments
    doc + 2023 Operational Estate raster), every LSOA in that borough
    inherits that Group.  No approximation: borough boundaries are
    contiguous on the LSOA queen graph by construction.

  Tier 2 (Voronoi within split borough).  For boroughs that are
    *split* across two or more Groups, fall back to station-Voronoi
    within that borough: each LSOA gets the Group of its nearest LAS
    station (BNG straight-line) drawn from the stations *of either
    Group that share this borough*.  This is the residual
    approximation; we tabulate exactly how many LSOAs are affected.

Split-borough catalogue (from cross-referencing the 2023 map and
the per-Group station lists in LAS-List-Of-Departments-2025.pdf):

  Hounslow         (NW)  split between Hanwell (Feltham, Isleworth, …)
                         and Fulham (Chiswick)
  Lewisham         (SE)  split between Deptford (Deptford station)
                         and Bromley (Forest Hill, Lee)
  Lambeth          (SE)  split between Deptford (Waterloo) and Oval
                         (Brixton, Oval, Streatham)
  Tower Hamlets    (NE)  Poplar → Homerton; rest of TH stays Homerton
                         (single-Group borough modulo small overlap)
  Bromley          (SE)  Mottingham is geographically in Bromley but
                         the 2023 map places it in Greenwich Group
                         (kept as Greenwich for exactness)

Outputs (overwrite previous v1 from run_build_groups.py):
  data/derived/lsoa_to_group.csv           — per-LSOA Group label, with
                                             `assignment_source ∈
                                             {borough, voronoi-within-borough}`
  data/derived/groups_polygons.gpkg        — 21 dissolved Group polygons
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
LSOA_GPKG = REPO / "data/raw/LSOA_2021_London.gpkg"
STATIONS_CSV = REPO / "data/raw/LAS_stations.csv"
SECTOR_CSV = REPO / "data/derived/lsoa_to_sector_borough.csv"
OUT_GROUPS = REPO / "data/derived/lsoa_to_group.csv"
OUT_POLYS = REPO / "data/derived/groups_polygons.gpkg"

LSOA_SUFFIX = re.compile(r" \d{3}[A-Z]$")

# ----------------------------------------------------------------------
# Authoritative borough → Group mapping (Tier 1).  Each entry lists the
# borough name as it appears in the LSOA21NM prefix and the Group it
# belongs to.  Boroughs NOT in this dict are "split" and resolved by
# Tier-2 Voronoi inside the borough.
# ----------------------------------------------------------------------
BOROUGH_TO_GROUP: dict[str, str] = {
    # North Central
    "Camden":              "Camden",
    "Islington":           "Camden",
    "Enfield":             "Edmonton",
    "Haringey":            "Edmonton",
    "Barnet":              "Friern Barnet",
    # North East
    "Hackney":             "Homerton",
    "City of London":      "Homerton",
    "Tower Hamlets":       "Homerton",   # Poplar, Smithfield, Shoreditch all → Homerton Group
    "Newham":              "Newham",
    "Havering":            "Romford",
    "Barking and Dagenham": "Romford",
    "Redbridge":           "Ilford",
    "Waltham Forest":      "Whipps Cross",
    # North West
    "Brent":               "Brent",
    "Harrow":              "Brent",
    "Hammersmith and Fulham": "Fulham",
    "Kensington and Chelsea": "Fulham",
    "Ealing":              "Hanwell",
    # "Hounslow":          (split — Tier 2: Hanwell vs Fulham)
    "Hillingdon":          "Hillingdon",
    "Westminster":         "Westminster",
    # South East
    "Bromley":             "Bromley",
    # "Lewisham":          (split — Tier 2: Bromley vs Deptford vs Greenwich)
    "Greenwich":           "Greenwich",
    "Bexley":              "Greenwich",
    # "Lambeth":           (split — Tier 2: Deptford vs Oval)
    "Southwark":           "Deptford",
    # South West
    "Croydon":             "Croydon",
    "Kingston upon Thames": "New Malden",
    "Richmond upon Thames": "New Malden",
    "Sutton":              "St Helier",
    "Wandsworth":          "Wimbledon",
    "Merton":              "Wimbledon",
    # Brentwood (Essex) is served by East of England Ambulance Service (EEAST),
    # NOT LAS — left out of the Group lookup so its LSOAs are excluded from
    # every Group polygon and downstream analysis.
}

SPLIT_BOROUGHS: dict[str, list[str]] = {
    # borough -> list of Groups that can claim LSOAs in it
    "Hounslow": ["Hanwell", "Fulham"],
    "Lewisham": ["Bromley", "Deptford"],   # Greenwich Group covers Mottingham (in Bromley borough)
    "Lambeth":  ["Deptford", "Oval"],
}

GROUP_TO_SECTOR: dict[str, str] = {
    "Camden": "North Central", "Edmonton": "North Central", "Friern Barnet": "North Central",
    "Homerton": "North East", "Ilford": "North East", "Newham": "North East",
    "Romford": "North East", "Whipps Cross": "North East",
    "Brent": "North West", "Fulham": "North West", "Hanwell": "North West",
    "Hillingdon": "North West", "Westminster": "North West",
    "Bromley": "South East", "Deptford": "South East", "Greenwich": "South East", "Oval": "South East",
    "Croydon": "South West", "New Malden": "South West",
    "St Helier": "South West", "Wimbledon": "South West",
}

STATION_TO_GROUP: dict[str, str] = {
    "Bloomsbury": "Camden", "Camden": "Camden", "Islington": "Camden",
    "Bounds Green": "Edmonton", "Chase Farm": "Edmonton", "Edmonton": "Edmonton",
    "Ponders End": "Edmonton", "Tottenham": "Edmonton",
    "Barnet": "Friern Barnet", "Friern Barnet": "Friern Barnet", "Mill Hill": "Friern Barnet",
    "Homerton": "Homerton", "Poplar": "Homerton", "Shoreditch": "Homerton", "Smithfield": "Homerton",
    "Ilford": "Ilford",
    "Newham": "Newham", "Silvertown": "Newham", "West Ham": "Newham",
    "Becontree": "Romford", "Hornchurch": "Romford", "Romford": "Romford",
    "Walthamstow": "Whipps Cross", "Whipps Cross": "Whipps Cross",
    "Brent": "Brent", "Kenton": "Brent", "Pinner": "Brent", "Wembley": "Brent", "Ruislip": "Brent",
    "Chiswick": "Fulham", "Fulham": "Fulham", "North Kensington": "Fulham",
    "Feltham": "Hanwell", "Greenford": "Hanwell", "Hanwell": "Hanwell", "Isleworth": "Hanwell",
    "Hayes": "Hillingdon", "Heathrow Airport": "Hillingdon", "Hillingdon": "Hillingdon",
    # NB Hillingdon (B5) added 2026-05-26; was previously absent from the
    # roster and proxied by Hayes.
    "St John's Wood": "Westminster", "St Johns Wood": "Westminster", "Westminster": "Westminster",
    "Beckenham": "Bromley", "Bromley": "Bromley", "Forest Hill": "Bromley", "Lee": "Bromley",
    "St Paul's Cray": "Bromley", "St Pauls Cray": "Bromley",
    "Deptford": "Deptford", "Waterloo": "Deptford",
    "Barnehurst": "Greenwich", "Greenwich": "Greenwich",
    "Mottingham": "Greenwich", "Woolwich": "Greenwich",
    "Brixton": "Oval", "Oval": "Oval", "Streatham": "Oval",
    "Coulsdon": "Croydon", "Croydon": "Croydon",
    "New Addington": "Croydon", "South Croydon": "Croydon",
    "New Malden": "New Malden", "Richmond": "New Malden",
    "Tolworth": "New Malden", "Twickenham": "New Malden",
    "St Helier": "St Helier",
    # Sutton was previously mapped to St Helier but R2 Sutton is a Make
    # Ready Hub, not an ambulance station — removed 2026-05-26.
    "Battersea": "Wimbledon", "Putney": "Wimbledon", "Wimbledon": "Wimbledon",
}


def main() -> int:
    lsoa = gpd.read_file(LSOA_GPKG)  # EPSG:27700
    lsoa["borough"] = lsoa["LSOA21NM"].str.replace(LSOA_SUFFIX, "", regex=True)
    sec_df = pd.read_csv(SECTOR_CSV)
    # Keep "Outside LAS" as-is; Brentwood LSOAs will drop out of the Group
    # lookup below since their borough is intentionally absent from
    # BOROUGH_TO_GROUP and SPLIT_BOROUGHS.
    sector_by_lsoa = dict(zip(sec_df["LSOA21CD"], sec_df["sector"]))
    lsoa["sector"] = lsoa["LSOA21CD"].map(sector_by_lsoa)

    # Station table with sector override (csv field is unreliable)
    stations = pd.read_csv(STATIONS_CSV)
    stations = stations.merge(
        sec_df[["LSOA21CD", "sector"]].rename(columns={"sector": "sector_v2"}),
        on="LSOA21CD", how="left",
    )
    stations["sector"] = stations["sector_v2"].fillna(stations["sector"])
    stations["group"] = stations["station_name"].map(STATION_TO_GROUP)
    stations_g = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs=4326,
    ).to_crs(27700)
    stations_g["x"] = stations_g.geometry.x
    stations_g["y"] = stations_g.geometry.y

    # Assign each LSOA
    lsoa["x"] = lsoa.geometry.centroid.x
    lsoa["y"] = lsoa.geometry.centroid.y

    out_rows = []
    n_tier1 = n_tier2 = n_dropped = 0
    for _, r in lsoa.iterrows():
        borough = r["borough"]
        if borough in BOROUGH_TO_GROUP:
            g = BOROUGH_TO_GROUP[borough]
            src = "borough"
            n_tier1 += 1
        elif borough in SPLIT_BOROUGHS:
            candidate_groups = SPLIT_BOROUGHS[borough]
            cand = stations_g[stations_g["group"].isin(candidate_groups)]
            if len(cand) == 0:
                # shouldn't happen
                g, src = candidate_groups[0], "fallback"
            else:
                tree = cKDTree(cand[["x", "y"]].values)
                _, idx = tree.query([[r["x"], r["y"]]], k=1)
                g = cand.iloc[int(idx[0])]["group"]
                src = "voronoi-within-borough"
            n_tier2 += 1
        else:
            # Boroughs outside LAS service area (Brentwood, etc.) are dropped
            # from the Group assignment entirely.
            n_dropped += 1
            continue
        out_rows.append({
            "LSOA21CD": r["LSOA21CD"],
            "LSOA21NM": r["LSOA21NM"],
            "borough":  borough,
            "sector":   r["sector"],
            "group":    g,
            "group_sector": GROUP_TO_SECTOR.get(g, r["sector"]),
            "assignment_source": src,
        })
    if n_dropped:
        print(f"      dropped {n_dropped} LSOAs (outside LAS service area)")
    out = pd.DataFrame(out_rows).sort_values("LSOA21CD").reset_index(drop=True)
    out.to_csv(OUT_GROUPS, index=False)
    print(f"  wrote {OUT_GROUPS.relative_to(REPO)}  ({len(out)} rows)")
    print(f"      Tier 1 (borough-exact):       {n_tier1} LSOAs "
          f"({100*n_tier1/len(out):.1f}%)")
    print(f"      Tier 2 (Voronoi-in-borough):  {n_tier2} LSOAs "
          f"({100*n_tier2/len(out):.1f}%) — across "
          f"{out.loc[out.assignment_source=='voronoi-within-borough', 'borough'].nunique()} "
          f"split boroughs")

    # Per-Group LSOA counts
    print("\n  LSOAs per Group:")
    g_lsoa = out["group"].value_counts().sort_values(ascending=False)
    for g, n in g_lsoa.items():
        print(f"      {g:<20s}  {n:>3d} LSOAs  ({GROUP_TO_SECTOR.get(g, '?')})")

    # Per-split-borough breakdown
    split_df = out[out["assignment_source"] == "voronoi-within-borough"]
    if not split_df.empty:
        print("\n  Tier-2 LSOAs by split-borough and resolved Group:")
        for b, sub in split_df.groupby("borough"):
            counts = sub["group"].value_counts()
            print(f"      {b}: {dict(counts)}")

    # Dissolve to Group polygons (drop LSOAs without a Group — Brentwood etc.)
    lsoa_with_group = lsoa.merge(out[["LSOA21CD", "group"]], on="LSOA21CD", how="inner")
    groups_poly = lsoa_with_group.dissolve(by="group").reset_index()
    groups_poly["sector"] = groups_poly["group"].map(GROUP_TO_SECTOR)
    groups_poly[["group", "sector", "geometry"]].to_file(OUT_POLYS, driver="GPKG")
    print(f"\n  wrote {OUT_POLYS.relative_to(REPO)}  ({len(groups_poly)} polygons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
