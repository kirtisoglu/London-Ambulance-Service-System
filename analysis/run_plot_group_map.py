"""
Render the v2 LAS group map.  Two outputs:

  * ``data/derived/figures/groups_borough_map.png`` — 21 group polygons
    coloured by sector, with stations and Group HQs overlaid (with
    on-shift DCA labels).

  * ``data/derived/figures/groups_vs_official.png`` — side-by-side
    comparison of the derived group polygons (left) and the official
    LAS 2023 Operational Estate raster (right) so the reader can
    eyeball where the station-Voronoi-within-sector approximation
    differs from the dashed-line group boundary.

Run:
    python analysis/run_plot_group_map.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GROUPS_GPKG = REPO / "data/derived/groups_polygons.gpkg"
SECTORS_GPKG = REPO / "data/derived/sectors_polygons.gpkg"
STATIONS_CSV = REPO / "data/raw/LAS_stations.csv"
L2_CSV = REPO / "data/raw/LAS_L2_facilities.csv"
FLEET_CSV = REPO / "data/derived/las_fleet_per_group.csv"
OFFICIAL_RASTER = REPO / "data/raw/las_sector_map_aug2023.jpg"

OUT_GROUPS_PNG = REPO / "data/derived/figures/groups_borough_map.png"
OUT_COMPARE_PNG = REPO / "data/derived/figures/groups_vs_official.png"

SECTOR_COLOURS = {
    "North Central": "#cdb4db",
    "North East":    "#a8dadc",
    "North West":    "#ffc8a2",
    "South East":    "#b5e2b6",
    "South West":    "#f7d488",
}


def _draw_groups(ax, groups, sectors, stations, l2, fleet):
    """Draw the derived 21-group map on a matplotlib axis."""
    # Fill groups with the sector colour; dashed boundary between groups
    # within the same sector; solid boundary on the sector edge.
    for _, g in groups.iterrows():
        gpd.GeoSeries([g.geometry]).plot(
            ax=ax, color=SECTOR_COLOURS.get(g["sector"], "#dddddd"),
            edgecolor="#444", linewidth=0.7, linestyle=(0, (4, 2)),
            alpha=0.9,
        )
    sectors.boundary.plot(ax=ax, edgecolor="#222", linewidth=1.6, zorder=4)

    # Layer: all stations as small dots
    stations.plot(ax=ax, marker="o", color="#222", markersize=10,
                  alpha=0.55, edgecolor="white", linewidth=0.25, zorder=5)

    # Layer: Group HQ markers sized by DCA count
    fleet_by = {r["station_name"]: r for _, r in fleet.iterrows()}
    group_hqs = l2[l2["facility_type"] == "group_hq"].copy()
    group_hqs["station_name"] = group_hqs["facility_name"].str.extract(r"\(([^)]+)\)$")
    group_hqs["n_dca_op"] = group_hqs["station_name"].map(
        {r["station_name"]: int(r["n_dca_op"]) for _, r in fleet.iterrows()}
    ).fillna(0).astype(int)
    group_hqs["marker_size"] = group_hqs["n_dca_op"] ** 1.4 * 6
    group_hqs.plot(
        ax=ax, marker="D", color="#1d3557",
        markersize=group_hqs["marker_size"],
        edgecolor="white", linewidth=0.9, zorder=6,
    )
    for _, r in group_hqs.iterrows():
        ax.annotate(
            f"{r['station_name']}\n{r['n_dca_op']} DCA",
            xy=(r.geometry.x, r.geometry.y),
            xytext=(5, 5), textcoords="offset points",
            fontsize=6.5, color="#0c1c34", fontweight="600",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", fc="white",
                      ec="#1d3557", lw=0.4, alpha=0.9),
            zorder=8,
        )

    # Layer: EOCs (red squares)
    eocs = l2[l2["facility_type"] == "eoc"]
    eocs.plot(
        ax=ax, marker="s", color="#e63946", markersize=140,
        edgecolor="white", linewidth=1.0, zorder=7,
    )
    for _, r in eocs.iterrows():
        ax.annotate(
            r["facility_name"],
            xy=(r.geometry.x, r.geometry.y),
            xytext=(5, -7), textcoords="offset points",
            fontsize=6.5, color="#a02430", fontweight="700",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="#e63946", lw=0.5, alpha=0.92),
            zorder=8,
        )
    ax.set_axis_off()


def main() -> int:
    groups = gpd.read_file(GROUPS_GPKG).to_crs(4326)
    sectors = gpd.read_file(SECTORS_GPKG).to_crs(4326)
    stations = pd.read_csv(STATIONS_CSV)
    l2 = pd.read_csv(L2_CSV)
    fleet = pd.read_csv(FLEET_CSV)
    stations_g = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs=4326,
    )
    l2_g = gpd.GeoDataFrame(
        l2,
        geometry=gpd.points_from_xy(l2["longitude"], l2["latitude"]),
        crs=4326,
    )

    # ----- (A) stand-alone group map -----------------------------------
    fig, ax = plt.subplots(figsize=(13, 11))
    _draw_groups(ax, groups, sectors, stations_g, l2_g, fleet)
    # sector centroid labels
    for _, s in sectors.iterrows():
        c = s.geometry.representative_point()
        ax.text(c.x, c.y, s["sector"], fontsize=15, fontweight="800",
                color="#222", ha="center", va="center", alpha=0.32, zorder=3)
    handles = [mpatches.Patch(facecolor=c, edgecolor="#444",
                              label=s) for s, c in SECTOR_COLOURS.items()]
    handles += [
        plt.Line2D([0], [0], color="#222", lw=1.5, linestyle="-",
                   label="Sector boundary (solid)"),
        plt.Line2D([0], [0], color="#444", lw=1, linestyle=(0, (4, 2)),
                   label="Group boundary (dashed; derived)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#222",
                   markersize=6, markeredgecolor="white", label="LAS station"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1d3557",
                   markersize=10, markeredgecolor="white",
                   label="Group HQ (size ∝ on-shift DCAs)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e63946",
                   markersize=10, markeredgecolor="white", label="EOC"),
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.92, fontsize=9)
    total_dca = int(fleet["n_dca_op"].sum())
    ax.set_title(
        f"LAS v2 — 21 Group catchments derived from station-Voronoi-within-sector\n"
        f"5 sectors (NHS-ICS borough membership) · {total_dca} on-shift DCAs",
        fontsize=12, pad=10,
    )
    fig.tight_layout()
    OUT_GROUPS_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GROUPS_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_GROUPS_PNG.relative_to(REPO)}  "
          f"({OUT_GROUPS_PNG.stat().st_size/1024:.0f} KB)")

    # ----- (B) side-by-side vs official raster -------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    _draw_groups(ax1, groups, sectors, stations_g, l2_g, fleet)
    ax1.set_title("Derived 21-group catchments\n(station-Voronoi within "
                  "borough-derived sector)", fontsize=11)

    img = mpimg.imread(OFFICIAL_RASTER)
    ax2.imshow(img)
    ax2.set_axis_off()
    ax2.set_title("Official LAS Operational Estate, August 2023\n"
                  "(dashed = Group area, solid = Sector)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_COMPARE_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_COMPARE_PNG.relative_to(REPO)}  "
          f"({OUT_COMPARE_PNG.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
