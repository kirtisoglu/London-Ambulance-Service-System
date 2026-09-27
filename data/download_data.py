"""
Download and prepare base datasets for LAS case study.

Step 1: LSOA 2021 boundaries (Generalised Clipped) from ONS Open Geography Portal
Step 2: LSOA mid-year population estimates from ONS
"""

import os
import sys
import json
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

LONDON_BOROUGHS = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "City of London", "Croydon", "Ealing", "Enfield",
    "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey",
    "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
    "Kensington and Chelsea", "Kingston upon Thames", "Lambeth",
    "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames",
    "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest",
    "Wandsworth", "Westminster",
]

# ArcGIS Feature Server for LSOA 2021 BGC V5
FEATURE_SERVER_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/"
    "FeatureServer/0/query"
)


def download_file(url, dest, description=""):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {description or url} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded // (1 << 20)} MB)", end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# Step 1: LSOA 2021 Boundaries via ArcGIS Feature Server query
# ---------------------------------------------------------------------------

def _query_lsoas_for_borough(borough, return_geometry=True):
    """Query ArcGIS Feature Server for LSOAs matching a borough name prefix."""
    features = []
    offset = 0
    while True:
        params = {
            "where": f"LSOA21NM LIKE '{borough}%'",
            "outFields": "LSOA21CD,LSOA21NM,BNG_E,BNG_N,LAT,LONG",
            "returnGeometry": str(return_geometry).lower(),
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": 2000,
            "resultOffset": offset,
        }
        r = requests.get(FEATURE_SERVER_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        batch = data.get("features", [])
        if not batch:
            break
        features.extend(batch)
        if len(batch) < 2000:
            break
        offset += len(batch)
    return features


def step1_lsoa_boundaries():
    """Download London LSOA 2021 boundaries from ArcGIS Feature Server."""
    print("\n=== Step 1: LSOA 2021 Boundaries ===")

    london_gpkg = os.path.join(RAW_DIR, "LSOA_2021_London.gpkg")
    if os.path.exists(london_gpkg):
        print(f"  Already exists: {london_gpkg}")
        gdf = gpd.read_file(london_gpkg)
        print(f"  {len(gdf)} LSOAs loaded")
        return gdf

    all_features = []
    for i, borough in enumerate(LONDON_BOROUGHS, 1):
        feats = _query_lsoas_for_borough(borough)
        print(f"  [{i:2d}/33] {borough}: {len(feats)} LSOAs")
        all_features.extend(feats)

    geojson = {"type": "FeatureCollection", "features": all_features}
    gdf = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326")
    print(f"\n  Total London LSOAs: {len(gdf)}")

    # Reproject to British National Grid for distance calculations
    gdf = gdf.to_crs("EPSG:27700")

    gdf.to_file(london_gpkg, driver="GPKG")
    print(f"  Saved: {london_gpkg}")
    print(f"  CRS: {gdf.crs}")

    return gdf


# ---------------------------------------------------------------------------
# Step 2: LSOA Population Estimates from ONS SAPE
# ---------------------------------------------------------------------------

POP_URL = (
    "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
    "populationandmigration/populationestimates/datasets/"
    "lowersuperoutputareamidyearpopulationestimates/"
    "mid2022revisednov2025tomid2024/sapelsoasyoa20222024.xlsx"
)
POP_XLSX = os.path.join(RAW_DIR, "sape_lsoa_population.xlsx")


def step2_population(london_lsoa_codes=None):
    """Download ONS SAPE LSOA population estimates and extract London data."""
    print("\n=== Step 2: LSOA Population Estimates ===")

    pop_csv = os.path.join(RAW_DIR, "LSOA_2021_London_population.csv")
    if os.path.exists(pop_csv):
        print(f"  Already exists: {pop_csv}")
        df = pd.read_csv(pop_csv)
        print(f"  {len(df)} rows loaded")
        return df

    download_file(POP_URL, POP_XLSX, "ONS SAPE LSOA population (83 MB)")

    print("  Reading Excel file (this may take a moment)...")
    xl = pd.ExcelFile(POP_XLSX)
    print(f"  Sheet names: {xl.sheet_names}")

    # Find the most recent mid-year total population sheet
    target_sheets = [s for s in xl.sheet_names if "Mid-20" in s and "Person" in s]
    if not target_sheets:
        target_sheets = [s for s in xl.sheet_names if "Mid-20" in s]
    if not target_sheets:
        print(f"  WARNING: Could not find population sheet.")
        print(f"  Available sheets: {xl.sheet_names}")
        # Try all sheets to find one with LSOA data
        for s in xl.sheet_names:
            target_sheets = [s]
            break

    sheet_name = target_sheets[-1]  # most recent
    print(f"  Reading sheet: '{sheet_name}'")

    # SAPE sheets have metadata rows at top; try different skip values
    df = None
    for skiprows in range(3, 9):
        candidate = pd.read_excel(POP_XLSX, sheet_name=sheet_name, skiprows=skiprows)
        cols_str = " ".join(str(c) for c in candidate.columns)
        if "LSOA" in cols_str or any(
            str(candidate.iloc[0:3][c].dropna().values[0]).startswith("E01")
            if len(candidate[c].dropna()) > 0 else False
            for c in candidate.columns[:3]
        ):
            df = candidate
            print(f"  Found header at skiprows={skiprows}")
            break

    if df is None:
        # Last resort: read with no skip and find the header row
        raw = pd.read_excel(POP_XLSX, sheet_name=sheet_name, header=None, nrows=15)
        print(f"  First 10 rows of sheet:\n{raw.to_string()}")
        print("  ERROR: Could not auto-detect header row.")
        return None

    print(f"  Columns (first 10): {list(df.columns)[:10]}")
    print(f"  Shape: {df.shape}")

    # Find LSOA code column
    lsoa_col = None
    for c in df.columns:
        if "LSOA" in str(c) and ("Code" in str(c) or "CD" in str(c)):
            lsoa_col = c
            break
    if not lsoa_col:
        for c in df.columns:
            vals = df[c].dropna().astype(str).head(10)
            if vals.str.match(r"^E01\d{6}$").any():
                lsoa_col = c
                break

    if not lsoa_col:
        print("  ERROR: Cannot find LSOA code column")
        print(f"  All columns: {list(df.columns)}")
        return None

    print(f"  LSOA code column: '{lsoa_col}'")

    # Find total population column
    total_col = None
    for c in df.columns:
        cs = str(c).strip()
        if cs == "Total" or ("All" in cs and "Age" in cs):
            total_col = c
            break
    if not total_col:
        # Sum single-year-of-age columns (0, 1, 2, ... 90+)
        age_cols = [c for c in df.columns if str(c).isdigit() or str(c) == "90+"]
        if age_cols:
            print(f"  Summing {len(age_cols)} age columns for total population")
            df["population"] = pd.to_numeric(
                df[age_cols].stack(), errors="coerce"
            ).unstack().sum(axis=1)
            total_col = "population"
        else:
            print("  ERROR: Cannot find population columns")
            return None

    print(f"  Population column: '{total_col}'")

    # Find LSOA name column
    name_col = None
    for c in df.columns:
        if "LSOA" in str(c) and "Name" in str(c):
            name_col = c
            break

    # Filter to London
    if london_lsoa_codes is not None and len(london_lsoa_codes) > 0:
        df_london = df[df[lsoa_col].isin(london_lsoa_codes)].copy()
    else:
        if name_col:
            mask = df[name_col].apply(
                lambda n: any(str(n).startswith(b) for b in LONDON_BOROUGHS)
            )
            df_london = df[mask].copy()
        else:
            print("  WARNING: No London filter available, saving all E01 codes")
            df_london = df[df[lsoa_col].astype(str).str.startswith("E01")].copy()

    # Build output dataframe
    cols_map = {lsoa_col: "LSOA21CD", total_col: "population"}
    if name_col:
        cols_map[name_col] = "LSOA21NM"

    df_out = df_london[list(cols_map.keys())].rename(columns=cols_map).copy()
    df_out["population"] = pd.to_numeric(df_out["population"], errors="coerce")
    df_out = df_out.dropna(subset=["LSOA21CD", "population"])

    print(f"  London LSOAs with population: {len(df_out)}")
    print(f"  Total London population: {df_out['population'].sum():,.0f}")
    print(f"  Mean LSOA population: {df_out['population'].mean():,.0f}")

    df_out.to_csv(pop_csv, index=False)
    print(f"  Saved: {pop_csv}")

    return df_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("London Ambulance Service — Data Download Script")
    print("=" * 50)

    gdf = step1_lsoa_boundaries()

    london_codes = gdf["LSOA21CD"].values if gdf is not None else None
    pop = step2_population(london_lsoa_codes=london_codes)

    print("\n" + "=" * 50)
    print("Summary:")
    if gdf is not None:
        print(f"  LSOA boundaries: {len(gdf)} London LSOAs")
        print(f"  CRS: {gdf.crs}")
    if pop is not None:
        print(f"  Population data: {len(pop)} rows")
    print("\nDone.")
