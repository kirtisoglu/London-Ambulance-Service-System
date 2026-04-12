# London-Ambulance-Service-System

## Data

### LSOA 2021 Boundaries

- **Source:** [ONS Open Geography Portal](https://geoportal.statistics.gov.uk) — Lower layer Super Output Areas (December 2021) Boundaries EW BGC V5
- **File:** `data/raw/LSOA_2021_London.gpkg`
- **Format:** GeoPackage (EPSG:27700, British National Grid)
- **Coverage:** 5,042 LSOAs across all 33 London boroughs
- **License:** Open Government Licence

### LSOA Population Estimates (Mid-2024)

- **Source:** [ONS Small Area Population Estimates (SAPE)](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates) — Mid-2024 edition
- **File:** `data/raw/LSOA_2021_London_population.csv`
- **Format:** CSV (columns: `LSOA21CD`, `total_population`, `LSOA21NM`)
- **Coverage:** 5,042 London LSOAs, total population 9,169,062 (mean 1,819 per LSOA)
- **License:** Open Government Licence

Both datasets can be downloaded by running:

```
python data/download_data.py
```