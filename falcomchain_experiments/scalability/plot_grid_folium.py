"""
Interactive folium map of a scalability grid via :mod:`falcomplot`.

Renders every base node as a unit square coloured by demand
(choropleth), with hover tooltips exposing the per-node attributes.
Level-1 facility candidates and level-2 super-candidates are
overlaid as circle/star markers with their own tooltips.

Usage::

    python3 plot_grid_folium.py            # default: grid_10000
    python3 plot_grid_folium.py 20000      # alternative size

Output: ``figures/grid_{N}_interactive.html``
"""

import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
from shapely.geometry import Polygon, Point

from falcomplot.mapping import build_basemap, add_choropleth


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_grid_nodes(path: Path):
    """Return (rows, cols, list_of_node_dicts) from a generate_grids.py json."""
    with open(path) as f:
        d = json.load(f)
    return d["graph"]["rows"], d["graph"]["cols"], d["nodes"]


def build_node_gdf(nodes):
    """One unit-square polygon per node, with all paper-relevant attributes."""
    rows = []
    for n in nodes:
        x, y = float(n["C_X"]), float(n["C_Y"])
        # Unit square centred at (x, y); we treat (C_X, C_Y) as (lon, lat).
        poly = Polygon([
            (x - 0.5, y - 0.5),
            (x + 0.5, y - 0.5),
            (x + 0.5, y + 0.5),
            (x - 0.5, y + 0.5),
            (x - 0.5, y - 0.5),
        ])
        is_l1 = bool(n.get("candidate", 0))
        is_l2 = bool(n.get("super_candidate", 0))
        facility_role = (
            "L2 super-candidate" if is_l2
            else "L1 candidate" if is_l1
            else "none"
        )
        rows.append({
            "node_id": n["id"],
            "C_X": int(x),
            "C_Y": int(y),
            "demand": int(n["demand"]),
            "area": int(n.get("area", 1)),
            "facility_role": facility_role,
            "candidate": int(is_l1),
            "super_candidate": int(is_l2),
            "geometry": poly,
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    gdf = gdf.set_crs("EPSG:4326")  # synthetic — declare, don't reproject
    return gdf


def add_facility_markers(m, node_gdf, *, level: int):
    """Overlay circles for L1 and stars for L2, both with tooltips."""
    if level == 1:
        sub = node_gdf[node_gdf["candidate"] == 1]
        color = "#2563eb"        # blue
        radius = 3
        fill_opacity = 0.6
        layer_name = "L1 candidates (facility candidates)"
    elif level == 2:
        sub = node_gdf[node_gdf["super_candidate"] == 1]
        color = "#dc2626"        # red
        radius = 6
        fill_opacity = 0.95
        layer_name = "L2 super-candidates (HQ / EOC equivalents)"
    else:
        raise ValueError(f"level must be 1 or 2, got {level}")

    group = folium.FeatureGroup(name=layer_name, show=True)
    for _, row in sub.iterrows():
        cx, cy = row["C_X"], row["C_Y"]
        tooltip_html = (
            f"<b>Facility</b>: {row['facility_role']}<br>"
            f"<b>Node ID</b>: {row['node_id']}<br>"
            f"<b>Position</b>: ({cx}, {cy})<br>"
            f"<b>Demand at this node</b>: {row['demand']}"
        )
        folium.CircleMarker(
            location=(cy, cx),  # folium expects (lat, lon)
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            weight=1.0,
            tooltip=folium.Tooltip(tooltip_html, sticky=False),
        ).add_to(group)
    group.add_to(m)


def render_grid(n_nodes: int):
    grid_path = DATA_DIR / f"grid_{n_nodes}.json"
    if not grid_path.exists():
        raise SystemExit(f"Grid file not found: {grid_path}")

    rows, cols, nodes = load_grid_nodes(grid_path)
    n_l1 = sum(1 for x in nodes if x.get("candidate"))
    n_l2 = sum(1 for x in nodes if x.get("super_candidate"))
    print(
        f"Loaded grid_{n_nodes}.json — {rows}×{cols} = {len(nodes):,} nodes; "
        f"L1 candidates: {n_l1:,}; L2 super-candidates: {n_l2:,}"
    )

    node_gdf = build_node_gdf(nodes)

    # Centre folium on the geometric centre of the grid.
    cx, cy = cols / 2.0, rows / 2.0
    # Approximate zoom: smaller grids need more zoom-in.
    zoom = max(3, int(8 - (max(rows, cols) / 50)))

    m = build_basemap(
        boundary=None,
        center=(cy, cx),
        zoom=zoom,
        tiles="CartoDB Positron",
    )

    # Per-node choropleth on demand with rich hover tooltip.
    add_choropleth(
        m,
        node_gdf,
        column="demand",
        label="Demand $w_v$",
        palette="YlGnBu",
        n_colors=8,
        tooltip_fields=[
            "node_id", "C_X", "C_Y", "demand", "facility_role",
        ],
    )

    add_facility_markers(m, node_gdf, level=1)
    add_facility_markers(m, node_gdf, level=2)

    # Layer control so the user can toggle L1 / L2 visibility.
    folium.LayerControl(collapsed=False).add_to(m)

    # Title overlay.
    title_html = (
        f'<div style="position:fixed;top:10px;left:60px;z-index:10000;'
        f'background:white;padding:8px 14px;border:1px solid #aaa;'
        f'border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.2);'
        f'font:14px sans-serif;">'
        f'<b>FalCom scalability grid</b> — '
        f'|V|={n_nodes:,} ({rows}×{cols}); '
        f'L1 candidates: {n_l1:,}; L2 super-candidates: {n_l2:,}.<br>'
        f'<span style="color:#555;font-size:12px;">'
        f'Hover any cell for per-node attributes. Toggle facility '
        f'layers from the top-right control.</span></div>'
    )
    m.get_root().html.add_child(folium.Element(title_html))

    out = FIG_DIR / f"grid_{n_nodes}_interactive.html"
    m.save(str(out))
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    render_grid(n)
