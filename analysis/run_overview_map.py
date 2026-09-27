"""
Build a visual overview of the LAS dataset for the docs.

Produces two artefacts in ``data/derived/``:

* ``london_overview.html`` — interactive Leaflet map with three layers:
  LSOA outlines, dissolved borough boundaries, and the LAS facility
  markers (64 stations + 7 HQ/EOC sites). Built with FalcomPlot's
  :func:`falcomplot.mapping.plott.build_basemap`,
  :func:`add_hierarchy`, and :func:`add_markers`.
* ``london_overview.png`` — static figure of the same three layers via
  geopandas/matplotlib, for embedding in the FalcomChain
  ``working_with_real_data`` doc.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

from falcomplot.mapping import plott

REPO = Path(__file__).resolve().parents[1]
LSOA_GPKG = REPO / "data/raw/LSOA_2021_London.gpkg"
STATIONS_CSV = REPO / "data/raw/LAS_stations.csv"
L2_CSV = REPO / "data/raw/LAS_L2_facilities.csv"
OUT_DIR = REPO / "data/derived"

# Strip the trailing " NNNX" from LSOA21NM to recover the borough name.
LSOA_SUFFIX = re.compile(r" \d{3}[A-Z]$")


def load_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (lsoa_4326, stations, l2_facilities) — all in EPSG:4326."""
    lsoa = gpd.read_file(LSOA_GPKG)
    lsoa["borough"] = lsoa["LSOA21NM"].str.replace(LSOA_SUFFIX, "", regex=True)
    lsoa_4326 = lsoa.to_crs(4326)

    stations = pd.read_csv(STATIONS_CSV)
    stations_gdf = gpd.GeoDataFrame(
        stations.assign(
            category="LAS station",
            source="LAS",
            name=stations["station_name"],
        ),
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs=4326,
    )

    l2 = pd.read_csv(L2_CSV)
    # Map facility_type → display category. Two distinct markers.
    l2["category"] = l2["facility_type"].map({
        "EOC": "Emergency Operations Centre",
        "Sector HQ": "Sector HQ (proxy)",
    }).fillna("Sector HQ (proxy)")
    l2["source"] = "LAS"
    l2["name"] = l2["facility_name"] if "facility_name" in l2.columns else l2["facility_id"]
    l2_gdf = gpd.GeoDataFrame(
        l2,
        geometry=gpd.points_from_xy(l2["longitude"], l2["latitude"]),
        crs=4326,
    )
    return lsoa_4326, stations_gdf, l2_gdf


def build_interactive(lsoa, stations, l2, out_path: Path) -> None:
    """Folium / Leaflet interactive map via FalcomPlot."""
    centroid = lsoa.geometry.union_all().centroid
    m = plott.build_basemap(
        boundary=lsoa,
        center=(float(centroid.y), float(centroid.x)),
        zoom=10,
    )
    plott.add_hierarchy(
        m,
        hierarchy="borough",
        basemap_gdf=lsoa,
        color="#222222",
        weight=1.6,
        fill_color="#fed8b1",
        fill_opacity=0.18,
        tooltip_fields=["borough"],
    )
    categories = {
        "LAS station": {"color": "#e63946", "radius": 5, "order": 0},
        "Emergency Operations Centre": {"color": "#1d3557", "radius": 8, "order": 1},
        "Sector HQ (proxy)": {"color": "#457b9d", "radius": 7, "order": 2},
    }
    plott.add_markers(m, stations, categories=categories)
    plott.add_markers(m, l2, categories=categories)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))


def build_static(lsoa, stations, l2, out_path: Path) -> None:
    """Static PNG with the same three layers, ready to embed in docs."""
    boroughs = lsoa.dissolve(by="borough").reset_index()

    fig, ax = plt.subplots(figsize=(9, 9))
    lsoa.plot(
        ax=ax,
        facecolor="#f3f4f6",
        edgecolor="#cbd5e1",
        linewidth=0.15,
    )
    boroughs.boundary.plot(
        ax=ax,
        edgecolor="#1f2937",
        linewidth=1.2,
    )
    stations.plot(
        ax=ax,
        marker="o",
        color="#e63946",
        markersize=22,
        edgecolor="white",
        linewidth=0.5,
        label=f"LAS station (n={len(stations)})",
        zorder=3,
    )
    l2_eoc = l2[l2["category"] == "Emergency Operations Centre"]
    l2_hq = l2[l2["category"] == "Sector HQ (proxy)"]
    if len(l2_eoc):
        l2_eoc.plot(
            ax=ax,
            marker="s",
            color="#1d3557",
            markersize=80,
            edgecolor="white",
            linewidth=0.8,
            label=f"EOC (n={len(l2_eoc)})",
            zorder=4,
        )
    if len(l2_hq):
        l2_hq.plot(
            ax=ax,
            marker="D",
            color="#457b9d",
            markersize=70,
            edgecolor="white",
            linewidth=0.8,
            label=f"Sector HQ proxy (n={len(l2_hq)})",
            zorder=4,
        )

    handles = [
        mpatches.Patch(facecolor="#f3f4f6", edgecolor="#cbd5e1",
                       label=f"LSOA (n={len(lsoa)})"),
        mpatches.Patch(facecolor="none", edgecolor="#1f2937", linewidth=1.2,
                       label=f"Borough boundary (n={lsoa['borough'].nunique()})"),
    ]
    handles += ax.get_legend_handles_labels()[0]
    ax.legend(
        handles=handles, loc="lower left", framealpha=0.92, fontsize=9,
    )
    ax.set_title("Greater London — LSOAs, boroughs, and LAS facilities",
                 fontsize=12, pad=10)
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("loading layers...")
    lsoa, stations, l2 = load_layers()
    print(f"  LSOAs: {len(lsoa):,} | boroughs: {lsoa['borough'].nunique()}")
    print(f"  L1 stations: {len(stations)} | L2 facilities: {len(l2)}")

    out_html = OUT_DIR / "london_overview.html"
    print(f"writing {out_html.relative_to(REPO)} ...")
    build_interactive(lsoa, stations, l2, out_html)

    out_png = OUT_DIR / "london_overview.png"
    print(f"writing {out_png.relative_to(REPO)} ...")
    build_static(lsoa, stations, l2, out_png)

    size_html = out_html.stat().st_size / 1024
    size_png = out_png.stat().st_size / 1024
    print(f"done. html={size_html:.0f} KB, png={size_png:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
