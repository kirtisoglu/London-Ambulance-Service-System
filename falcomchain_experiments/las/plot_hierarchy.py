"""
Plot the LAS three-level operational hierarchy locally in the browser
via falcomplot.

Renders four layers on a Leaflet basemap of Greater London:
  1. LSOA polygons (5,042 base units of G¹) -- light grey fill.
  2. Group boundaries (19 group polygons) -- thin dark-green solid.
     Each group is the (sector, station-code-letter) pair, e.g. NW-A.
  3. Sector boundaries (5 super-polygons) -- bold dashed blue.
  4. Facility markers:
       red  -- 63 LAS ambulance stations (level-1 candidates F¹).
       blue -- 7 super-facility candidates (5 sector HQs + 2 EOCs).

Both group and sector polygons are derived by Voronoi-by-station
(closest LAS station in BNG metres) and repaired to contiguity by
greedy majority-neighbour reassignment.

Output: `data/derived/las_hierarchy_map.html`. The script opens it in
the default browser; pass `--no-open` to skip.

Run:
    python -m falcomchain_experiments.las.plot_hierarchy
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import json

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from falcomplot import mapping

REPO = Path(__file__).resolve().parents[2]
LSOA_GPKG = REPO / "data/raw/LSOA_2021_London.gpkg"
# v2 inputs — borough-derived sectors + Departments-doc Group catchments.
SECTOR_CSV = REPO / "data/derived/lsoa_to_sector_borough.csv"
GROUP_CSV = REPO / "data/derived/lsoa_to_group.csv"
STATIONS_GPKG = REPO / "data/derived/stations_polygons.gpkg"
FLEET_PER_GROUP_CSV = REPO / "data/derived/las_fleet_per_group.csv"    # 21 rows, raw Group totals
# Note: LAS publishes vehicles at the Group level only, and FalCom's
# energy functions do not use capacity, so we keep fleet at Group
# granularity and do not display per-L1-station vehicle counts.
STATIONS_CSV = REPO / "data/raw/LAS_stations.csv"
L2_CSV = REPO / "data/raw/LAS_L2_facilities.csv"
GRAPH_JSON = REPO / "data/raw/london_graph.json"
METRICS_JSON = REPO / "data/derived/voronoi_metrics.json"  # optional (legacy)
OUT_HTML = REPO / "data/derived/las_hierarchy_map.html"

# Centred on the City of London.
LONDON_CENTER = (51.5074, -0.1278)
LONDON_ZOOM = 10
# Half-step per +/- click so zoom feels less abrupt; requires
# zoomSnap == zoomDelta in Leaflet 1.x.
ZOOM_DELTA = 0.5

LAS_CATEGORIES = {
    "L1 Station": {"color": "#111111", "radius": 5, "order": 0},
    # L2 markers re-coloured to hot pink so they don't blend with the
    # new NC yellow palette.
    "L2 HQ / EOC": {"color": "#ec4899", "radius": 9, "order": 1},
}

# ---------------------------------------------------------------------------
# Station-icon helpers
# ---------------------------------------------------------------------------
# Tiny self-contained SVG of an ambulance silhouette (no external assets).
# Encoded once and reused for every L1 / L2 marker.  The marker colour and
# size come from arguments at render time so we can tint by balance status
# or facility type without juggling separate icon files.
import base64

_AMBULANCE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="{fill}"
  stroke="#111" stroke-width="0.6">
  <rect x="2" y="9" width="14" height="9" rx="1.2"/>
  <path d="M16 11h3l2 3v4h-5z"/>
  <circle cx="6" cy="20" r="1.7" fill="#111"/>
  <circle cx="6" cy="20" r="0.8" fill="#fff"/>
  <circle cx="18" cy="20" r="1.7" fill="#111"/>
  <circle cx="18" cy="20" r="0.8" fill="#fff"/>
  <path d="M9 11h2v2h2v2h-2v2H9v-2H7v-2h2z" fill="#fff" stroke="none"/>
</svg>"""


def _ambulance_data_url(fill: str) -> str:
    """Return a `data:image/svg+xml;base64,...` URL for the ambulance SVG
    tinted with the given fill colour."""
    svg = _AMBULANCE_SVG.format(fill=fill).encode("utf-8")
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")


def _make_icon(fill: str, size: int = 22) -> folium.CustomIcon:
    """Folium CustomIcon wrapping the tinted ambulance SVG."""
    return folium.CustomIcon(
        icon_image=_ambulance_data_url(fill),
        icon_size=(size, size),
        icon_anchor=(size // 2, size // 2),
    )

# Tooltip routing: every Leaflet tooltip on the map is suppressed and
# its HTML is instead rendered in a single fixed-position info-panel
# anchored to the viewport (top-left).  Because the panel is
# `position: fixed`, it does NOT move when the basemap pans/zooms —
# the previous per-feature tooltips were anchored to feature lat/lon
# and slid off-screen when the user dragged the map.
#
# Wiring: after Leaflet loads, we walk every layer on the Folium map,
# read each layer's bound tooltip content, and replace cursor-follow
# behaviour with `mouseover → show(html)` / `mouseout → hide()` calls
# targeting the panel.  The panel listens at the document level so it
# updates regardless of which FeatureGroup / GeoJson / Marker is hovered.
TOOLTIP_GLOBAL_CSS = """
<style>
  /* Hide all native Leaflet tooltips — we route their content to a
     viewport-fixed info-panel instead. */
  .leaflet-tooltip { display: none !important; }

  /* Fixed-position info-panel: anchored to the viewport, NOT the map
     pane, so it never moves when the basemap pans/zooms. */
  #falcom-info-panel {
    position: fixed;
    top: 70px;
    left: 20px;
    z-index: 1100;
    background: rgba(255, 255, 255, 0.92);
    -webkit-backdrop-filter: blur(3px);
    backdrop-filter: blur(3px);
    border: 1px solid rgba(17, 24, 39, 0.85);
    border-radius: 4px;
    padding: 8px 10px;
    color: #111827;
    font-family: system-ui, sans-serif;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.18);
    max-width: 360px;
    pointer-events: none;
    display: none;
  }
  #falcom-info-panel.is-visible { display: block; }
</style>
<div id="falcom-info-panel"><div id="falcom-info-content"></div></div>
<script>
  (function () {
    function init() {
      if (typeof L === "undefined" || !L.Map) { setTimeout(init, 80); return; }
      // Folium injects its map as window.map_<hash>.  Find it.
      var mapInst = null;
      for (var k in window) {
        try {
          if (k.indexOf("map_") === 0 && window[k] instanceof L.Map) {
            mapInst = window[k]; break;
          }
        } catch (e) {}
      }
      if (!mapInst) { setTimeout(init, 80); return; }

      var panel   = document.getElementById("falcom-info-panel");
      var content = document.getElementById("falcom-info-content");
      if (!panel || !content) return;

      function show(html) {
        content.innerHTML = html;
        panel.classList.add("is-visible");
      }
      function hide() {
        panel.classList.remove("is-visible");
      }

      function resolveAtHover(tooltipSource, hoveredLayer) {
        // ._content can be: (1) a static HTML string (folium.Tooltip on a Marker)
        //                   (2) a function returning a DOM Node (folium.GeoJsonTooltip)
        //                   (3) a function returning an HTML string
        // For (2)/(3), the function expects the HOVERED layer so it can
        // read `layer.feature.properties`. Resolve at hover time so we
        // pass the per-feature sub-layer reference, not the parent group.
        var raw = tooltipSource._content;
        var resolved;
        if (typeof raw === "function") {
          try { resolved = raw(hoveredLayer); } catch (e) { resolved = ""; }
        } else {
          resolved = raw;
        }
        if (resolved && typeof resolved === "object" && resolved.nodeType) {
          return resolved.innerHTML || resolved.outerHTML || "";
        }
        return String(resolved || "");
      }

      function bindToPanel(targetLayer, tooltipSource) {
        targetLayer.on("mouseover", function () {
          show(resolveAtHover(tooltipSource, targetLayer));
        });
        targetLayer.on("mouseout", hide);
      }

      function hookLayer(layer) {
        if (!layer) return;
        var tt = layer.getTooltip && layer.getTooltip();
        if (tt) {
          if (layer.eachLayer) {
            // GeoJSON / FeatureGroup carrying a tooltip — each per-feature
            // sub-layer needs to evaluate the tooltip's content function
            // with ITS OWN layer reference (so .feature.properties resolves).
            layer.eachLayer(function (sub) { bindToPanel(sub, tt); });
          } else {
            // Leaf layer (e.g., Marker) with its own static-string tooltip.
            bindToPanel(layer, tt);
          }
        } else if (layer.eachLayer) {
          // Plain container without a tooltip (e.g., the L1/L2 marker
          // FeatureGroups) — recurse to reach the markers inside.
          try { layer.eachLayer(hookLayer); } catch (e) {}
        }
      }

      mapInst.eachLayer(hookLayer);
      // Catch layers added later (LayerControl toggles, etc.).
      mapInst.on("layeradd", function (e) { hookLayer(e.layer); });
    }
    document.addEventListener("DOMContentLoaded", init);
  })();
</script>
"""

# One base hue per sector; within a sector, group fills run from light
# to dark in the order the groups sort alphabetically. Each list has
# enough tones to cover that sector's groups (NW/SE: 3; NC/NE: 4; SW: 5).
SECTOR_PALETTES = {
    # v2 group counts (Departments-doc derived):
    # NC: 3 (Camden, Edmonton, Friern Barnet)
    # NE: 5 (Homerton, Ilford, Newham, Romford, Whipps Cross)
    # NW: 5 (Brent, Fulham, Hanwell, Hillingdon, Westminster)
    # SE: 4 (Bromley, Deptford, Greenwich, Oval)
    # SW: 4 (Croydon, New Malden, St Helier, Wimbledon)
    "North West":    ["#dbeafe", "#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"],
    "North Central": ["#fde047", "#facc15", "#ca8a04"],
    "North East":    ["#dcfce7", "#bbf7d0", "#86efac", "#22c55e", "#15803d"],
    "South East":    ["#cffafe", "#67e8f9", "#22d3ee", "#0e7490"],
    "South West":    ["#fecaca", "#fca5a5", "#ef4444", "#b91c1c"],
}
# Sector-boundary stroke: a bold solid colour matching each sector palette.
SECTOR_STROKE = {
    "North West":    "#1e3a8a",
    "North Central": "#713f12",
    "North East":    "#14532d",
    "South East":    "#164e63",
    "South West":    "#7f1d1d",
}


def assign_group_colors(groups: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a `fill_color` column to `groups` from SECTOR_PALETTES.

    Within each sector, groups are sorted by `group_id` and assigned the
    palette entries in order (light → dark). Sectors with fewer groups
    use a prefix of their palette; sectors with more would raise.
    """
    out = groups.copy()
    out["fill_color"] = ""
    for sec, sub in out.groupby("sector"):
        palette = SECTOR_PALETTES.get(sec)
        if palette is None:
            raise KeyError(f"no palette for sector {sec!r}")
        ids = sorted(sub["group_id"].tolist())
        if len(ids) > len(palette):
            raise ValueError(
                f"sector {sec} has {len(ids)} groups but palette has "
                f"{len(palette)} tones"
            )
        for i, gid in enumerate(ids):
            out.loc[out["group_id"] == gid, "fill_color"] = palette[i]
    return out


def build_facilities_gdf() -> gpd.GeoDataFrame:
    """Combine L1 stations and L2 facilities into one GeoDataFrame
    with the schema falcomplot.add_markers expects."""
    stations = pd.read_csv(STATIONS_CSV)
    l2 = pd.read_csv(L2_CSV)

    rows = []
    for _, r in stations.iterrows():
        rows.append({
            "name": r["station_name"],
            "station_code": r["station_code"],
            "sector": r["sector"],
            "category": "L1 Station",
            "source": "LAS",
            "geometry": Point(r["longitude"], r["latitude"]),
        })
    for _, r in l2.iterrows():
        rows.append({
            "name": r["facility_name"],
            "facility_id": r["facility_id"],
            "facility_type": r["facility_type"],
            "sector": r.get("sector", ""),
            "category": "L2 HQ / EOC",
            "source": "LAS",
            "geometry": Point(r["longitude"], r["latitude"]),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def build_sector_hierarchy(lsoas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve LSOAs by the v2 borough-derived sector into 5 super-polygons.
    Brentwood is filtered out (served by EEAST, not LAS)."""
    sec = pd.read_csv(SECTOR_CSV)
    sec = sec[sec["sector"] != "Outside LAS"]
    merged = lsoas.merge(sec[["LSOA21CD", "sector"]], on="LSOA21CD", how="inner")
    if merged.crs is None or merged.crs.to_epsg() != 4326:
        merged = merged.to_crs("EPSG:4326")
    return merged.dissolve(by="sector").reset_index()


def build_outer_boundary(lsoas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve every LAS-served LSOA into a single polygon — the outer
    boundary of the LAS catchment.  Brentwood is excluded (EEAST)."""
    sec = pd.read_csv(SECTOR_CSV)
    keep_codes = set(sec.loc[sec["sector"] != "Outside LAS", "LSOA21CD"])
    sub = lsoas[lsoas["LSOA21CD"].isin(keep_codes)]
    if sub.crs is None or sub.crs.to_epsg() != 4326:
        sub = sub.to_crs("EPSG:4326")
    return gpd.GeoDataFrame(geometry=[sub.geometry.union_all()], crs="EPSG:4326")


def build_graph_layers(
    lsoas: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build (edges, nodes) GeoDataFrames for the dual-graph overlay.

    `nodes` carries one Point per LSOA centroid (5,042 points).
    `edges` carries one LineString per rook-adjacency pair (14,687
    edges, deduplicated). Both are EPSG:4326 so they overlay the
    Leaflet basemap directly.
    """
    if lsoas.crs is None or lsoas.crs.to_epsg() != 4326:
        lsoas = lsoas.to_crs("EPSG:4326")
    centroid_by_code: dict[str, Point] = {}
    for _, row in lsoas.iterrows():
        centroid_by_code[row["LSOA21CD"]] = row.geometry.centroid

    with open(GRAPH_JSON) as f:
        raw = json.load(f)
    id_to_code = {n["id"]: n["LSOA21CD"] for n in raw["nodes"]}

    # Skip nodes / edges that touch LSOAs outside the LAS catchment
    # (e.g. Brentwood — present in london_graph.json but filtered out
    # of `lsoas` upstream so its centroid is absent).
    edge_rows = []
    for src_idx, neighbors in enumerate(raw["adjacency"]):
        src_code = id_to_code[src_idx]
        c0 = centroid_by_code.get(src_code)
        if c0 is None:
            continue
        for nb in neighbors:
            tgt_idx = nb["id"]
            if tgt_idx <= src_idx:
                continue  # deduplicate undirected edges
            c1 = centroid_by_code.get(id_to_code[tgt_idx])
            if c1 is None:
                continue
            edge_rows.append(LineString([c0, c1]))
    edges = gpd.GeoDataFrame(geometry=edge_rows, crs="EPSG:4326")

    node_rows = []
    node_codes = []
    for i in sorted(id_to_code):
        c = centroid_by_code.get(id_to_code[i])
        if c is None:
            continue
        node_rows.append(c)
        node_codes.append(id_to_code[i])
    nodes = gpd.GeoDataFrame(
        {"LSOA21CD": node_codes}, geometry=node_rows, crs="EPSG:4326",
    )
    return edges, nodes


def build_group_hierarchy(lsoas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve LSOAs by Group (v2 borough-aligned + Voronoi-in-split-borough)
    into 21 polygons.  The new ``lsoa_to_group.csv`` schema uses the
    Group HQ station name as the Group label (no `letter` code)."""
    grp = pd.read_csv(GROUP_CSV)
    cols = ["LSOA21CD", "group", "group_sector"]
    merged = lsoas.merge(grp[cols], on="LSOA21CD", how="left")
    if merged["group"].isna().any():
        n_missing = int(merged["group"].isna().sum())
        print(f"warn: {n_missing} LSOAs have no group assignment")
    if merged.crs is None or merged.crs.to_epsg() != 4326:
        merged = merged.to_crs("EPSG:4326")
    dissolved = merged.dissolve(by="group").reset_index()
    # Keep the existing column name `group_id` for downstream compatibility
    # with the colour palette logic; set `letter` to empty (legacy).
    dissolved = dissolved.rename(columns={"group": "group_id",
                                          "group_sector": "sector"})
    dissolved["letter"] = ""
    return dissolved[["group_id", "sector", "letter", "geometry"]]


def load_metrics() -> tuple[dict, dict, dict, dict]:
    """Load voronoi_metrics.json and return:

      station_code -> per-district metric dict (keyed by L1 station code)
      sector       -> per-super metric dict
      sector       -> max L1 radius across that sector's districts
      global_dict  -> top-level summary

    Returns empty dicts (and an empty global) if the metrics file
    hasn't been produced yet -- the map still renders, only the
    metric tooltips are skipped.
    """
    if not METRICS_JSON.exists():
        print(f"warn: {METRICS_JSON.relative_to(REPO)} not found; "
              f"metric tooltips will be omitted. Run "
              f"`python -m falcomchain_experiments.las.compute_voronoi_metrics` "
              f"first.")
        return {}, {}, {}, {}
    with open(METRICS_JSON) as f:
        m = json.load(f)
    by_station = {}
    by_sector_max_l1 = {}
    for did_str, info in m["per_district"].items():
        sc = info.get("station_code", "")
        if sc:
            by_station[sc] = info
        sec = info.get("sector", "")
        r1 = info.get("l1_radius")
        if sec and r1 is not None:
            cur = by_sector_max_l1.get(sec, 0.0)
            if r1 > cur:
                by_sector_max_l1[sec] = r1
    by_super = dict(m["per_super"])
    return by_station, by_super, by_sector_max_l1, m.get("global", {})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-open", action="store_true",
                   help="Write the HTML but don't open the browser.")
    args = p.parse_args()

    by_station, by_super, by_sector_max_l1, global_metrics = load_metrics()

    print(f"[1/4] Load LSOA polygons from {LSOA_GPKG.relative_to(REPO)}")
    lsoas_all = gpd.read_file(LSOA_GPKG)
    if lsoas_all.crs is None or lsoas_all.crs.to_epsg() != 4326:
        lsoas_all = lsoas_all.to_crs("EPSG:4326")
    # Restrict every downstream layer to the LAS catchment — Brentwood
    # (Essex, served by EEAST) is dropped.
    sec_full = pd.read_csv(SECTOR_CSV)
    keep_codes = set(sec_full.loc[sec_full["sector"] != "Outside LAS", "LSOA21CD"])
    n_dropped = int((~lsoas_all["LSOA21CD"].isin(keep_codes)).sum())
    lsoas = lsoas_all[lsoas_all["LSOA21CD"].isin(keep_codes)].reset_index(drop=True)
    print(f"      {len(lsoas_all)} total polygons, {len(lsoas)} in LAS catchment "
          f"({n_dropped} Brentwood/EEAST excluded)")

    print(f"[2/4] Build bare basemap (centre={LONDON_CENTER}, zoom={LONDON_ZOOM},"
          f" half-step zoom)")
    m = folium.Map(
        location=list(LONDON_CENTER),
        zoom_start=LONDON_ZOOM,
        tiles="CartoDB Positron",
        zoom_snap=ZOOM_DELTA,
        zoom_delta=ZOOM_DELTA,
    )
    # Outer boundary of Greater London — dissolved union of every LSOA.
    # Drawn thin and always on, so the city silhouette is visible
    # regardless of which inner layers are toggled.
    outer = build_outer_boundary(lsoas)
    outer_layer = folium.FeatureGroup(name="London outline", show=False)
    folium.GeoJson(
        outer,
        style_function=lambda _: {
            "fillColor": "#000000", "fillOpacity": 0.0,
            "color": "#111827", "weight": 1.5, "dashArray": "",
        },
    ).add_to(outer_layer)
    outer_layer.add_to(m)

    # LSOAs as a toggleable, faint layer with per-LSOA hover tooltip.
    # Turn the layer on from the LayerControl to see the 5,042-cell base
    # graph; hovering an LSOA highlights it and shows LSOA21CD/LSOA21NM
    # plus the Group / Sector it belongs to (from the v2 lookup).
    grp_lookup = pd.read_csv(GROUP_CSV)[
        ["LSOA21CD", "group", "group_sector", "assignment_source"]
    ].rename(columns={"group_sector": "sector",
                      "assignment_source": "source"})
    lsoas_for_tip = lsoas[["LSOA21CD", "LSOA21NM", "geometry"]].merge(
        grp_lookup, on="LSOA21CD", how="left",
    )
    lsoas_for_tip["group"] = lsoas_for_tip["group"].fillna("—")
    lsoas_for_tip["sector"] = lsoas_for_tip["sector"].fillna("—")
    lsoas_for_tip["source"] = lsoas_for_tip["source"].fillna("—")
    # Per-LSOA population + integer demand from london_graph.json — pre-format
    # with thousands separator so the tooltip table renders as e.g. "2,514".
    with open(GRAPH_JSON) as _gf:
        _raw_graph = json.load(_gf)
    pop_demand = pd.DataFrame([
        {"LSOA21CD": n["LSOA21CD"],
         "population": f"{int(n['population']):,}",
         "demand":     f"{int(n['demand']):,}"}
        for n in _raw_graph["nodes"]
    ])
    lsoas_for_tip = lsoas_for_tip.merge(pop_demand, on="LSOA21CD", how="left")
    lsoas_for_tip["population"] = lsoas_for_tip["population"].fillna("—")
    lsoas_for_tip["demand"] = lsoas_for_tip["demand"].fillna("—")
    lsoa_layer = folium.FeatureGroup(name="LSOAs (5,042)", show=False)
    folium.GeoJson(
        lsoas_for_tip,
        style_function=lambda _: {
            "fillColor": "#dde4ec", "fillOpacity": 0.05,
            "color": "#9aa6b2", "weight": 0.4,
        },
        highlight_function=lambda _: {
            "fillColor": "#fde047", "fillOpacity": 0.45,
            "color": "#ca8a04", "weight": 2.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["LSOA21NM", "LSOA21CD", "group", "sector",
                    "population", "demand", "source"],
            aliases=["LSOA name", "LSOA code", "Group", "Sector",
                     "Population", "Incidents/yr (est.)",
                     "assignment source"],
            sticky=True,
        ),
    ).add_to(lsoa_layer)
    lsoa_layer.add_to(m)

    # Dual graph (rook adjacency on LSOA centroids).
    print(f"      build dual-graph layer (5,042 nodes, 14,687 edges)")
    edges_gdf, nodes_gdf = build_graph_layers(lsoas)
    graph_layer = folium.FeatureGroup(
        name=f"Graph ({len(nodes_gdf)} nodes / {len(edges_gdf)} edges)",
        show=True,
    )
    folium.GeoJson(
        edges_gdf,
        style_function=lambda _: {
            "color": "#374151", "weight": 0.4, "opacity": 0.6,
        },
    ).add_to(graph_layer)
    folium.GeoJson(
        nodes_gdf,
        marker=folium.CircleMarker(
            radius=1.4, color="#111827", weight=0.6,
            fill=True, fill_color="#1f2937", fill_opacity=0.9,
        ),
    ).add_to(graph_layer)
    graph_layer.add_to(m)

    print(f"[3/5] Overlay 19 Voronoi group polygons "
          f"(per-sector palette, tones light → dark within sector)")
    groups = build_group_hierarchy(lsoas)
    groups = assign_group_colors(groups)
    print(f"      {len(groups)} groups: {sorted(groups['group_id'].tolist())}")
    for sec in sorted(SECTOR_PALETTES):
        sub = groups[groups["sector"] == sec].sort_values("group_id")
        print(f"        {sec:14s}  "
              + "  ".join(f"{r['group_id']}:{r['fill_color']}"
                           for _, r in sub.iterrows()))
    group_layer = folium.FeatureGroup(name="Groups (19)", show=False)
    folium.GeoJson(
        groups,
        style_function=lambda feat: {
            "fillColor": feat["properties"]["fill_color"],
            "fillOpacity": 0.65,
            "color": "#1a1a1a",
            "weight": 0.8,
        },
        highlight_function=lambda feat: {
            "fillColor": feat["properties"]["fill_color"],
            "fillOpacity": 0.85,
            "color": "#000000",
            "weight": 2.5,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["group_id", "sector"],
            aliases=["Group", "Sector"],
        ),
    ).add_to(group_layer)
    group_layer.add_to(m)

    # Station service areas (per-station Voronoi-within-Group catchments).
    # These are derived approximations of the current operational layout's
    # L1 districting — they don't appear on any official LAS map (LAS
    # dispatches the nearest available unit rather than committing to
    # fixed catchments) — but they're what FalCom's `s_LAS` benchmark
    # uses as the L1 partition for comparison.
    if STATIONS_GPKG.exists():
        print(f"[3b/5] Overlay {66} Voronoi station catchments (current-plan L1 service areas)")
        catchments = gpd.read_file(STATIONS_GPKG).to_crs(4326)
        catchments_layer = folium.FeatureGroup(
            name=f"Station service areas — Voronoi ({len(catchments)})",
            show=False,
        )
        folium.GeoJson(
            catchments,
            style_function=lambda feat: {
                "fillColor": "#000000",
                "fillOpacity": 0.0,
                "color": "#7f1d1d",
                "weight": 1.4,
                "dashArray": "6,3",
            },
            highlight_function=lambda feat: {
                "fillColor": "#fee2e2",
                "fillOpacity": 0.35,
                "color": "#7f1d1d",
                "weight": 2.6,
                "dashArray": "6,3",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["station_code", "station_name", "group", "sector"],
                aliases=["Code", "Station", "Group", "Sector"],
            ),
        ).add_to(catchments_layer)
        catchments_layer.add_to(m)
    else:
        print(f"      (skipping station catchments: {STATIONS_GPKG.name} not built; "
              f"run analysis/run_build_station_catchments.py)")

    print(f"[4/5] Overlay 5 Voronoi sector boundaries (toggleable, above groups)")
    sectors = build_sector_hierarchy(lsoas)
    sectors["stroke_color"] = sectors["sector"].map(SECTOR_STROKE)
    print(f"      {len(sectors)} sectors: {sorted(sectors['sector'].tolist())}")
    # Decorate each sector polygon with the FalCom metrics, so users can
    # inspect Voronoi-benchmark radii / demand / HQ directly on the map.
    def _sec_metric(sec, key, fmt="{:.2f}"):
        info = by_super.get(sec, {})
        v = info.get(key)
        if v is None:
            return "—"
        if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer()):
            return f"{int(v):,}"
        return fmt.format(v)
    sectors["n_districts"] = sectors["sector"].map(
        lambda s: by_super.get(s, {}).get("n_districts", "—"))
    sectors["demand"] = sectors["sector"].map(
        lambda s: _sec_metric(s, "demand"))
    sectors["max_l1_radius_min"] = sectors["sector"].map(
        lambda s: (f"{by_sector_max_l1[s]:.2f}"
                   if s in by_sector_max_l1 else "—"))
    sectors["mean_l1_radius_min"] = sectors["sector"].map(
        lambda s: _sec_metric(s, "mean_l1_radius"))
    sectors["l2_radius_min"] = sectors["sector"].map(
        lambda s: _sec_metric(s, "l2_radius"))
    sectors["hq_name"] = sectors["sector"].map(
        lambda s: by_super.get(s, {}).get("hq_name", "—"))
    sector_layer = folium.FeatureGroup(name="Sectors (5)", show=False)
    folium.GeoJson(
        sectors,
        style_function=lambda feat: {
            "fillColor": feat["properties"]["stroke_color"],
            "fillOpacity": 0.0,
            "color": feat["properties"]["stroke_color"],
            "weight": 4.0,
            "dashArray": "8,5",
        },
        highlight_function=lambda feat: {
            "fillColor": feat["properties"]["stroke_color"],
            "fillOpacity": 0.10,
            "color": feat["properties"]["stroke_color"],
            "weight": 6.0,
            "dashArray": "8,5",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "sector", "n_districts", "demand",
                "max_l1_radius_min", "mean_l1_radius_min",
                "l2_radius_min", "hq_name",
            ],
            aliases=[
                "Sector", "#L1 districts", "Demand (calls/yr)",
                "max L1 radius (min)", "mean L1 radius (min)",
                "L2 radius (min)", "HQ",
            ],
        ),
    ).add_to(sector_layer)
    sector_layer.add_to(m)

    print(f"[5/5] Add facility markers as toggleable FeatureGroups")
    facilities = build_facilities_gdf()
    n_l1 = int((facilities["category"] == "L1 Station").sum())
    n_l2 = int((facilities["category"] == "L2 HQ / EOC").sum())
    print(f"      {n_l1} L1 stations  /  {n_l2} L2 HQs/EOCs")

    # Vehicle counts from the April-2025 LAS fleet inventory.  LAS
    # reports at Group level only (every vehicle row is tagged with
    # the Group HQ station name), so we render Group totals on the
    # L2 Group HQ markers and DO NOT advertise per-station vehicle
    # counts on L1 markers (those would be synthetic).
    fleet_by_group: dict[str, dict] = {}
    try:
        g_df = pd.read_csv(FLEET_PER_GROUP_CSV)
        for _, r in g_df.iterrows():
            fleet_by_group[r["station_name"]] = {
                "n_dca":    int(r["n_dca"]),
                "n_dca_op": int(r["n_dca_op"]),
                "n_fru":    int(r["n_fru"]),
                "n_mru":    int(r["n_mru"]),
                "n_app":    int(r["n_app"]),
            }
    except Exception as e:
        print(f"      warn: could not load {FLEET_PER_GROUP_CSV.name}: {e}")

    def fleet_block(_station_name: str) -> str:
        """L1 markers do not show vehicle counts — LAS publishes them
        at Group level only.  Kept as a no-op so the rest of the
        tooltip pipeline doesn't have to special-case L1."""
        return ""

    def fleet_block_group(group_hq_name: str) -> str:
        """Group-total fleet block (L2 Group HQ markers).  Raw counts."""
        f = fleet_by_group.get(group_hq_name)
        if not f:
            return ""
        cells = []
        if f["n_dca"]:
            cells.append(f"DCA: <b>{f['n_dca']}</b> "
                         f"<span style='opacity:0.7'>(on-shift {f['n_dca_op']})</span>")
        if f["n_fru"]:
            cells.append(f"FRU: {f['n_fru']}")
        if f["n_mru"]:
            cells.append(f"MRU: {f['n_mru']}")
        if f["n_app"]:
            cells.append(f"APP: {f['n_app']}")
        if not cells:
            return ""
        return "<br><b>vehicles in Group (April 2025):</b><br>" + "<br>".join(cells)

    l1_cfg = LAS_CATEGORIES["L1 Station"]
    l1_group = folium.FeatureGroup(name=f"L1 Station ({n_l1})", show=False)
    # color L1 markers by balance status when balance data is present
    BAL_FILL = {
        "ok":     "#16a34a",  # green — within +/- eps
        "above":  "#dc2626",  # red — demand > (1+eps) * teams * w
        "below":  "#f59e0b",  # amber — demand < (1-eps) * teams * w
        "no_teams": "#6b7280",
    }
    for _, row in facilities[facilities["category"] == "L1 Station"].iterrows():
        code = row.get("station_code", "")
        info = by_station.get(code, {})
        if info:
            bal_status = info.get("balance_status")
            bal_line = ""
            if bal_status:
                ratio = info.get("balance_ratio")
                t_i = info.get("teams")
                target = info.get("target_demand")
                lo = info.get("lower_band")
                hi = info.get("upper_band")
                status_emoji = {"ok": "✓", "above": "⚠ over",
                                 "below": "⚠ under", "no_teams": "—"}.get(bal_status, "?")
                bal_line = (
                    f"<br><b>capacity / demand</b>: {status_emoji}<br>"
                    f"teams (model-implied): {t_i}<br>"
                    f"target demand: {target:,.0f} "
                    f"&nbsp; band: [{lo:,.0f}, {hi:,.0f}]<br>"
                    f"balance ratio: {ratio:.2f}"
                    if ratio is not None else ""
                )
            tip = (
                f"<b>{code} — {row.get('name', '')}</b><br>"
                f"sector: {row.get('sector', '')}<br>"
                f"#LSOAs: {info.get('n_nodes', '—')}<br>"
                f"demand: {info.get('demand', '—'):,}<br>"
                f"L1 radius (max t to LSOA): {info.get('l1_radius', 0):.2f} min<br>"
                f"mean t: {info.get('mean_t', 0):.2f} min<br>"
                f"t_p90: {info.get('t_p90', 0):.2f} min<br>"
                f"coverage ≤8 min: {info.get('coverage_8', 0)*100:.1f}%<br>"
                f"coverage ≤15 min: {info.get('coverage_15', 0)*100:.1f}%"
                + bal_line
                + fleet_block(row.get("name", ""))
            )
        else:
            tip = (
                f"<b>{code} — {row.get('name', '')}</b><br>"
                f"sector: {row.get('sector', '')}"
                + fleet_block(row.get("name", ""))
            )
        bal_status = info.get("balance_status") if info else None
        fill = BAL_FILL.get(bal_status, "#dc2626")  # red ambulance default
        size = 22 + (4 if bal_status in ("above", "below") else 0)
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            icon=_make_icon(fill, size=size),
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(l1_group)
    l1_group.add_to(m)

    # Map L2 facility name back to the sector it serves (if any).
    name_to_sector = {}
    for sec, info in by_super.items():
        nm = info.get("hq_name")
        if nm:
            name_to_sector[nm] = sec

    # Split L2 markers across two toggleable layers so EOC and Group HQ
    # markers that sit at the same LSOA centroid (e.g. Newham — where
    # GH-Newham and EOC-Newham share the K1 Newham building) don't
    # stack on top of each other in a single FeatureGroup.  This also
    # mirrors the model design: under dynamic-nested L2, EOCs are NOT
    # in F² — they're documented infrastructure shown separately.
    l2_facilities = facilities[facilities["category"] == "L2 HQ / EOC"]
    n_hq  = int((l2_facilities["facility_type"] == "group_hq").sum())
    n_eoc = int((l2_facilities["facility_type"] == "eoc").sum())
    l2_hq_group  = folium.FeatureGroup(name=f"L2 Group HQ ({n_hq})", show=False)
    l2_eoc_group = folium.FeatureGroup(name=f"EOC ({n_eoc})", show=False)
    for _, row in l2_facilities.iterrows():
        nm = row.get("name", "")
        sec = name_to_sector.get(nm)
        if sec is None:
            sec = row.get("sector", "")
        info = by_super.get(sec, {}) if sec else {}
        ftype = row.get("facility_type", "")
        # For Group HQs (group_hq), the L2 name carries a parenthesised
        # host-station, e.g. "Camden Group HQ (Camden)" — pull that out
        # to look up the fleet counts.
        host_station = ""
        if ftype == "group_hq" and "(" in nm and nm.endswith(")"):
            host_station = nm.rsplit("(", 1)[-1].rstrip(")")
        if info:
            l2r = info.get("l2_radius")
            tip = (
                f"<b>{row.get('facility_id', '')} — {nm}</b><br>"
                f"type: {ftype}<br>"
                f"serves sector: {sec}<br>"
                f"#L1 districts: {info.get('n_districts', '—')}<br>"
                f"demand: {info.get('demand', '—'):,}<br>"
                f"L2 radius (max t to L1 facility): "
                f"{l2r:.2f} min" if l2r is not None else ""
            )
        else:
            tip = (
                f"<b>{row.get('facility_id', '')} — {nm}</b><br>"
                f"type: {ftype}"
            )
        # L2 Group HQ markers show the Group total (not the equal-split estimate).
        tip += fleet_block_group(host_station) if host_station else ""
        # EOCs → blue; Group HQs → pink ambulance.
        if ftype == "eoc":
            target_group = l2_eoc_group
            fill = "#1d4ed8"
        else:
            target_group = l2_hq_group
            fill = "#ec4899"
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            icon=_make_icon(fill, size=30),
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(target_group)
    l2_hq_group.add_to(m)
    l2_eoc_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Inject the global tooltip CSS once, into the map root, so every
    # Leaflet tooltip (LSOA/Group/Sector layers + station/EOC markers)
    # shares the nearly-transparent, offset-from-cursor styling.
    m.get_root().html.add_child(folium.Element(TOOLTIP_GLOBAL_CSS))

    # Floating summary panel with the global Voronoi-benchmark metrics.
    # Anchored top-right, semi-transparent so it doesn't block the map.
    if global_metrics:
        g_ = global_metrics
        def _fmt(v, fmt="{:.2f}"):
            if v is None:
                return "—"
            if isinstance(v, int):
                return f"{v:,}"
            return fmt.format(v)
        # Capacity / demand balance block (only present if the metrics
        # JSON includes it — added when w and eps were passed in).
        bal = g_.get("balance") or {}
        bal_block = ""
        if bal:
            total_d = g_.get("total_demand") or 1
            pct_in  = 100.0 * bal.get("demand_balanced", 0) / total_d
            pct_out = 100.0 * bal.get("demand_imbalanced", 0) / total_d
            bal_block = (
                f'<tr><td colspan="2" style="padding-top:6px;'
                f'border-top:1px dashed #9ca3af;'
                f'font-weight:600">capacity / demand balance</td></tr>'
                f'<tr><td>w (calls/team/year)</td>'
                f'<td style="text-align:right">{_fmt(bal["demand_target"], "{:.0f}")}</td></tr>'
                f'<tr><td>ε (tolerance)</td>'
                f'<td style="text-align:right">{_fmt(bal["demand_tolerance"])}</td></tr>'
                f'<tr><td>districts in band</td>'
                f'<td style="text-align:right">{bal["n_balanced"]} / {g_["n_districts"]}</td></tr>'
                f'<tr><td>districts above band</td>'
                f'<td style="text-align:right; color:#dc2626">{bal["n_above"]}</td></tr>'
                f'<tr><td>districts below band</td>'
                f'<td style="text-align:right; color:#f59e0b">{bal["n_below"]}</td></tr>'
                f'<tr><td>demand inside band</td>'
                f'<td style="text-align:right">{pct_in:.1f}%</td></tr>'
                f'<tr><td>demand outside band</td>'
                f'<td style="text-align:right; color:#dc2626">{pct_out:.1f}%</td></tr>'
                f'<tr><td>max overload ratio</td>'
                f'<td style="text-align:right">{_fmt(bal.get("max_overload_ratio"))}</td></tr>'
                f'<tr><td>min underload ratio</td>'
                f'<td style="text-align:right">{_fmt(bal.get("min_underload_ratio"))}</td></tr>'
            )
        panel_html = f"""
        <div id="falcom-voronoi-panel" style="
            position: fixed; bottom: 10px; right: 10px; z-index: 1000;
            background: rgba(255,255,255,0.92);
            border: 1px solid #1f2937; border-radius: 6px;
            padding: 10px 12px; font-family: system-ui, sans-serif;
            font-size: 12px; line-height: 1.4; max-width: 320px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        ">
          <div style="font-weight:600; font-size:13px; margin-bottom:4px;">
            Voronoi benchmark (operational LAS imitation)
          </div>
          <div style="color:#4b5563; margin-bottom:6px; font-size:11px;">
            Metrics computed via falcomchain.compute_plan_summary.
            FalCom chain ensembles will be compared to these numbers.
          </div>
          <table style="border-collapse:collapse; width:100%;">
            <tr><td>#L1 districts</td><td style="text-align:right">{_fmt(g_.get('n_districts'))}</td></tr>
            <tr><td>#sectors (L2)</td><td style="text-align:right">{_fmt(g_.get('n_super'))}</td></tr>
            <tr><td>total demand (calls/yr)</td><td style="text-align:right">{_fmt(g_.get('total_demand'))}</td></tr>
            <tr><td><b>max L1 radius</b></td><td style="text-align:right"><b>{_fmt(g_.get('max_l1_radius'))} min</b></td></tr>
            <tr><td>mean L1 radius</td><td style="text-align:right">{_fmt(g_.get('mean_l1_radius'))} min</td></tr>
            <tr><td><b>max L2 radius</b></td><td style="text-align:right"><b>{_fmt(g_.get('max_l2_radius'))} min</b></td></tr>
            <tr><td>T_mean (demand-weighted)</td><td style="text-align:right">{_fmt(g_.get('T_mean'))} min</td></tr>
            <tr><td>T_p90</td><td style="text-align:right">{_fmt(g_.get('T_p90'))} min</td></tr>
            <tr><td>coverage ≤8 min</td><td style="text-align:right">{_fmt((g_.get('coverage_8') or 0)*100)}%</td></tr>
            <tr><td>coverage ≤15 min</td><td style="text-align:right">{_fmt((g_.get('coverage_15') or 0)*100)}%</td></tr>
            <tr><td>E_minisum (Σ d·t)</td><td style="text-align:right">{_fmt(g_.get('E_minisum'))}</td></tr>
            <tr><td>E_eccentricity (Σ r²)</td><td style="text-align:right">{_fmt(g_.get('E_eccentricity'))}</td></tr>
            {bal_block}
          </table>
        </div>
        """
        m.get_root().html.add_child(folium.Element(panel_html))

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"\nwrote {OUT_HTML.relative_to(REPO)} "
          f"({OUT_HTML.stat().st_size / 1024:.0f} KB)")

    if not args.no_open:
        url = OUT_HTML.resolve().as_uri()
        print(f"opening {url}")
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
