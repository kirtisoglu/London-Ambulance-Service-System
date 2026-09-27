"""
Plot London LSOA population choropleth using FalcomPlot.
"""

import sys
import os
import geopandas as gpd
import pandas as pd

sys.path.insert(0, "/Users/kirtisoglu/GitHub/FalcomPlot/src")
from shapely.geometry import Point
from falcomplot.mapping import build_basemap, add_choropleth, add_hierarchy, add_markers

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# Load LSOA boundaries and population
gdf = gpd.read_file(os.path.join(DATA_DIR, "LSOA_2021_London.gpkg"))
pop = pd.read_csv(os.path.join(DATA_DIR, "LSOA_2021_London_population.csv"))

# Load facility candidates
l1 = pd.read_csv(os.path.join(DATA_DIR, "LAS_stations.csv"))
l2 = pd.read_csv(os.path.join(DATA_DIR, "LAS_L2_facilities.csv"))

# Merge population into the geodataframe
gdf = gdf.merge(pop[["LSOA21CD", "population"]], on="LSOA21CD", how="left")

# Extract borough name from LSOA name (e.g. "Barking and Dagenham 016A" -> "Barking and Dagenham")
gdf["borough"] = gdf["LSOA21NM"].str.replace(r"\s+\d{3}[A-Z]$", "", regex=True)

# Build basemap centred on London
m = build_basemap(boundary=gdf, center=(51.50, -0.12), zoom=10)

# Add population choropleth
add_choropleth(
    m,
    gdf,
    column="population",
    label="Population",
    palette="YlOrRd",
    n_colors=6,
    tooltip_fields=["LSOA21CD", "LSOA21NM"],
)

# Add borough hierarchy (dissolve LSOAs into borough boundaries)
add_hierarchy(
    m,
    hierarchy="borough",
    basemap_gdf=gdf,
    color="#ffffff",
    weight=2.5,
    fill_color="#4a90d9",
    fill_opacity=0.08,
)

# Add facility candidate markers (L1 stations + L2 HQs/EOCs)
facilities = pd.concat([
    pd.DataFrame({
        "name": l1["station_name"],
        "category": "Ambulance Station (L1)",
        "source": "LAS",
        "latitude": l1["latitude"],
        "longitude": l1["longitude"],
    }),
    pd.DataFrame({
        "name": l2["facility_name"],
        "category": l2["facility_type"].map({"sector_hq": "Sector HQ (L2)", "eoc": "EOC (L2)"}),
        "source": "LAS",
        "latitude": l2["latitude"],
        "longitude": l2["longitude"],
    }),
], ignore_index=True)
facilities = gpd.GeoDataFrame(
    facilities, geometry=[Point(xy) for xy in zip(facilities.longitude, facilities.latitude)], crs="EPSG:4326"
)

LAS_CATEGORIES = {
    "Ambulance Station (L1)": {"color": "#2196F3", "radius": 5, "order": 0},
    "Sector HQ (L2)":         {"color": "#E63946", "radius": 8, "order": 1},
    "EOC (L2)":               {"color": "#6A0572", "radius": 8, "order": 2},
}

add_markers(m, facilities, categories=LAS_CATEGORIES, default_source="LAS")

out_path = os.path.join(os.path.dirname(__file__), "london_population.html")
m.save(out_path)
print(f"Map saved to {out_path}")
