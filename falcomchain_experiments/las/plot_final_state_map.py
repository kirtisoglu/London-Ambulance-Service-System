"""
Render a chain final state as an interactive Folium map.

Reads the final_state blob saved by run_chain_v3.py, joins it to the LSOA
polygons, and draws: LSOA fills colored by super-district, thin district
boundaries, thick super-district boundaries, and markers for the open L1
facilities (stations) and L2 facilities (Group-HQ role).

Run:
    python -m falcomchain_experiments.las.plot_final_state_map --seed 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "data/derived/chain_v3"
GPKG = REPO / "data/raw/LSOA_2021_London.gpkg"
GRAPH = REPO / "data/raw/london_graph.json"

# Qualitative cycle for super-districts (cartographic adjacency use; the
# hierarchy is disambiguated by the thick super boundaries, not hue alone).
QUAL = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
    "#e87ba4", "#eb6834", "#67a9cf", "#7fbc41", "#c98500", "#8073ac",
    "#d6604d", "#35978f", "#bf812d", "#de77ae", "#4393c3", "#5aae61",
    "#9970ab", "#f46d43", "#01665e", "#8c510a", "#762a83", "#b2182b",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10_000)
    args = ap.parse_args()

    run = json.loads(
        (IN_DIR / f"chain_v3_sample_s{args.seed}_T{args.steps}.json").read_text())
    fs = run["final_state"]
    l1 = {int(k): int(v) for k, v in fs["l1_assignment"].items()}
    sup = {int(k): int(v) for k, v in fs["super_assignment"].items()}
    teams = {int(k): int(v) for k, v in fs["teams"].items()}
    l1_centers = {int(k): v for k, v in fs["l1_centers"].items() if v is not None}
    l2_centers = {int(k): v for k, v in fs["l2_centers"].items() if v is not None}

    graph = json.loads(GRAPH.read_text())
    node_info = {n["id"]: n for n in graph["nodes"]}

    rows = []
    for node, dist in l1.items():
        info = node_info[node]
        rows.append({
            "LSOA21CD": info["LSOA21CD"],
            "district": dist,
            "super": sup.get(dist, -1),
            "units": teams.get(dist, 0),
        })
    df = pd.DataFrame(rows)

    gdf = gpd.read_file(GPKG)[["LSOA21CD", "geometry"]].merge(df, on="LSOA21CD")
    gdf = gdf.to_crs(epsg=4326)

    supers = sorted(gdf["super"].unique())
    color_of = {s: QUAL[i % len(QUAL)] for i, s in enumerate(supers)}

    m = folium.Map(location=[51.49, -0.11], zoom_start=10, tiles="cartodbpositron")

    # district polygons (dissolved) with super color fill + thin border
    dist_gdf = gdf.dissolve(by="district", as_index=False, aggfunc="first")
    folium.GeoJson(
        dist_gdf.to_json(),
        style_function=lambda f: {
            "fillColor": color_of[f["properties"]["super"]],
            "fillOpacity": 0.55,
            "color": "#ffffff",
            "weight": 1.2,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["district", "super", "units"],
            aliases=["district", "super-district", "capacity (units of 3 amb.)"]),
    ).add_to(m)

    # super-district outlines, thick
    super_gdf = gdf.dissolve(by="super", as_index=False)
    folium.GeoJson(
        super_gdf.to_json(),
        style_function=lambda f: {
            "fill": False, "color": "#333333", "weight": 2.6},
    ).add_to(m)

    # facility markers
    for dist, node in l1_centers.items():
        info = node_info[node]
        is_l2 = node in set(l2_centers.values())
        folium.CircleMarker(
            location=[info["LAT"], info["LONG"]],
            radius=7 if is_l2 else 4,
            color="#0b0b0b", weight=2,
            fill=True, fill_color="#ffffff" if not is_l2 else "#0b0b0b",
            fill_opacity=1.0,
            tooltip=(f"{'L2 (Group HQ role) + ' if is_l2 else ''}L1 facility, "
                     f"district {dist}: {info['LSOA21NM']}"),
        ).add_to(m)

    g = run["final_summary_global"]
    title = (f"<div style='position:fixed;top:12px;left:60px;z-index:1000;"
             f"background:#fcfcfb;border:1px solid rgba(11,11,11,.15);"
             f"border-radius:8px;padding:10px 14px;"
             f"font:13px system-ui'><b>FalCom LAS final state, chain seed "
             f"{args.seed}</b> (T = {args.steps:,})<br>"
             f"|P&sup1;| = {g.get('n_districts')} districts, "
             f"|P&sup2;| = {g.get('n_super')} super-districts "
             f"(real LAS: 66 stations / 21 Groups)<br>"
             f"fill = super-district; white lines = districts; "
             f"dark lines = super-districts;<br>"
             f"dots = open L1 facilities (filled = also L2 facility)</div>")
    m.get_root().html.add_child(folium.Element(title))

    out = IN_DIR / f"las_final_state_s{args.seed}.html"
    m.save(str(out))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
