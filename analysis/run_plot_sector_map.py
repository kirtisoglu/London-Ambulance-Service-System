"""
Render the v2 LAS sector map: 5 sector polygons (dissolved from borough
membership) with the 63 LAS stations and 21 Group HQ markers + 2 EOCs
overlaid.  Output:

    data/derived/figures/sectors_borough_map.png

Run:
    python analysis/run_plot_sector_map.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

REPO = Path(__file__).resolve().parents[1]
SECTOR_GPKG = REPO / "data/derived/sectors_polygons.gpkg"
STATIONS_CSV = REPO / "data/raw/LAS_stations.csv"
L2_CSV = REPO / "data/raw/LAS_L2_facilities.csv"
FLEET_CSV = REPO / "data/derived/las_fleet_per_station.csv"
OUT = REPO / "data/derived/figures/sectors_borough_map.png"

SECTOR_COLOURS = {
    "North Central": "#cdb4db",
    "North East":    "#a8dadc",
    "North West":    "#ffc8a2",
    "South East":    "#b5e2b6",
    "South West":    "#f7d488",
}


def main() -> int:
    sectors = gpd.read_file(SECTOR_GPKG).to_crs(4326)
    stations = pd.read_csv(STATIONS_CSV)
    l2 = pd.read_csv(L2_CSV)
    fleet = pd.read_csv(FLEET_CSV)
    fleet_by_station = {r["station_name"]: r for _, r in fleet.iterrows()}

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

    fig, ax = plt.subplots(figsize=(11, 10.5))

    # Layer 1: filled sector polygons with light colours
    for _, row in sectors.iterrows():
        c = SECTOR_COLOURS.get(row["sector"], "#dddddd")
        gpd.GeoSeries([row.geometry]).plot(
            ax=ax, color=c, edgecolor="#333", linewidth=1.4, alpha=0.85,
        )

    # Layer 2: thin LSOA-borough boundaries are skipped (kept clean for this map)

    # Layer 3: every L1 station as a small grey dot
    stations_g.plot(
        ax=ax, marker="o", color="#444", markersize=14, alpha=0.55,
        edgecolor="white", linewidth=0.3, zorder=3,
    )

    # Layer 4: Group HQs (blue diamonds) sized by on-shift DCA count
    group_hqs = l2_g[l2_g["facility_type"] == "group_hq"].copy()
    group_hqs["station_name"] = group_hqs["facility_name"].str.extract(
        r"\(([^)]+)\)$"
    )
    group_hqs["n_dca_op"] = group_hqs["station_name"].map(
        {r["station_name"]: int(r["n_dca_op"]) for _, r in fleet.iterrows()}
    ).fillna(0).astype(int)
    group_hqs["marker_size"] = group_hqs["n_dca_op"] ** 1.4 * 7  # visual scale
    group_hqs.plot(
        ax=ax, marker="D", color="#1d3557", alpha=0.92,
        markersize=group_hqs["marker_size"],
        edgecolor="white", linewidth=1.0, zorder=5,
    )
    # Group-HQ labels with DCA count
    for _, r in group_hqs.iterrows():
        ax.annotate(
            f"{r['station_name']}\n{r['n_dca_op']} DCA",
            xy=(r.geometry.x, r.geometry.y),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, color="#0c1c34", fontweight="600",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#1d3557", lw=0.5, alpha=0.9),
            zorder=6,
        )

    # Layer 5: EOCs (red squares)
    eocs = l2_g[l2_g["facility_type"] == "eoc"]
    eocs.plot(
        ax=ax, marker="s", color="#e63946", alpha=0.95, markersize=180,
        edgecolor="white", linewidth=1.2, zorder=6,
    )
    for _, r in eocs.iterrows():
        label = r["facility_name"].split(" EOC")[0]
        ax.annotate(
            f"{label} EOC",
            xy=(r.geometry.x, r.geometry.y),
            xytext=(7, -7), textcoords="offset points",
            fontsize=8, color="#a02430", fontweight="700",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="#e63946", lw=0.6, alpha=0.92),
            zorder=7,
        )

    # Sector centroid labels
    for _, row in sectors.iterrows():
        c = row.geometry.representative_point()
        ax.text(c.x, c.y, row["sector"], fontsize=14, fontweight="800",
                color="#222", ha="center", va="center", alpha=0.55,
                zorder=2)

    # Legend
    handles = [
        mpatches.Patch(facecolor=c, edgecolor="#333", label=s)
        for s, c in SECTOR_COLOURS.items()
    ]
    handles += [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#444",
                   markersize=6, markeredgecolor="white", label="LAS station"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1d3557",
                   markersize=10, markeredgecolor="white",
                   label="Group HQ (size ∝ on-shift DCAs)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e63946",
                   markersize=10, markeredgecolor="white",
                   label="Emergency Operations Centre"),
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.92, fontsize=9)

    total_dca = int(fleet["n_dca_op"].sum())
    ax.set_title(
        f"LAS v2 — 5 sectors (NHS-ICS borough membership), "
        f"21 Group HQs, 2 EOCs\n"
        f"{total_dca} on-shift A&E ambulances across the network",
        fontsize=12, pad=10,
    )
    ax.set_axis_off()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
