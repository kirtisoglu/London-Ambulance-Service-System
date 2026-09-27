# London-Ambulance-Service-System

Experiment code and data for the FalCom paper: hierarchical capacitated
districting of the London Ambulance Service (LAS) network at LSOA
resolution (4,994-node dual graph, 66 stations, 5 sectors).

Requires the local `falcomchain` package (FalcomChain repo) on the path;
travel-time computation additionally uses `falcomtravel` and OSMnx.

## Repository layout

- `data/` — raw inputs (`raw/`) and derived instance files (`derived/`),
  including the final ensemble results and figures in `derived/chain_v3/`.
- `analysis/` — instance-construction scripts (sectors, Groups, station
  catchments, demand attachment).
- `falcomchain_experiments/las/` — the LAS case-study pipeline (see below).
- `falcomchain_experiments/experiment1/` — detailed-balance validation on
  synthetic grids.
- `falcomchain_experiments/gurobi/` — MILP (Gurobi) vs FalCom comparison
  on synthetic grids, with numeric results (`solution_*.json`).
- `falcomchain_experiments/scalability/` — runtime and convergence
  scaling experiments up to 50,000-node grids.
- `plot_london.py` — London population/facilities overview map via
  FalcomPlot.

## LAS pipeline (reproduction order)

Instance construction (only needed to rebuild derived data from raw):

1. `analysis/run_build_v2_data.py` — parse FOI PDFs into sector/borough
   tables.
2. `analysis/run_build_groups_v2.py` — build the 21-Group layer
   (`lsoa_to_group.csv`).
3. `analysis/run_apply_group_overrides.py` — apply hand-verified
   boundary overrides.
4. `analysis/run_build_station_catchments.py` — station catchments
   (`lsoa_to_station.csv`).
5. `analysis/run_calls_to_demand.py`, `analysis/run_borough_attr.py` —
   attach demand and borough attributes to the graph.

Experiment (calibration locked 2026-07-02: capacity unit = 3-ambulance
block, w_unit = 10,887, eps = 0.15, c1 in [1,3], c2 in [2,6] units,
uniform cut selection):

1. `falcomchain_experiments/las/run_cdba.py --w 10887 --cmin 1` —
   CDBA candidate set.
2. `falcomchain_experiments/las/compute_cdba_travel_times.py` —
   OSM road-network travel-time matrix (needs network access).
3. `falcomchain_experiments/las/run_chain_v3.py --seeds 1 2 3 4 --snapshots`
   — 4-chain ensemble; `--optimizer --beta 0.05` for optimizer runs.
4. `falcomchain_experiments/las/postprocess_ensemble.py` — ensemble
   metrics, convergence and boundary-frequency figures, review page.
5. `falcomchain_experiments/las/plot_ensemble_diagnostics.py`,
   `plot_final_state_map.py`, `plot_hierarchy.py` — diagnostics and maps.

`diag_debt_mode.py` is a standalone A/B diagnostic for the debt-correction
rule (defaults reference the earlier w=3,642 calibration; its candidate
CSV is kept in `data/derived/` for that reason).

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
- **Format:** CSV (columns: `LSOA21CD`, `population`, `LSOA21NM`)
- **Coverage:** 5,042 London LSOAs, total population 9,169,062 (mean 1,819 per LSOA)
- **License:** Open Government Licence

### LAS Ambulance Stations (L1 Facility Candidates)

- **Source:** Station names, codes, and addresses from [LAS stations reference](https://lessavine.co.uk/london-ambulance-service-ambulance-stations/), cross-referenced with [LAS official website](https://www.londonambulance.nhs.uk/talking-with-us/freedom-of-information/classes-of-information/who-we-are-and-what-we-do/) and [OpenStreetMap](https://www.openstreetmap.org/) (`emergency=ambulance_station` tag). Coordinates geocoded via [postcodes.io](https://postcodes.io) and refined with OSM building-level positions where available.
- **File:** `data/raw/LAS_stations.csv`
- **Format:** CSV (columns: `station_code`, `station_name`, `latitude`, `longitude`, `sector`, `postcode`)
- **Coverage:** 66 ambulance stations across 5 LAS operational sectors (North West: 16, North Central: 14, North East: 12, South East: 9, South West: 15)
- **Notes:** ~70 stations historically documented; some consolidated during COVID. The 66 stations form the primary L1 candidate set F¹. Sector assignments inferred from station letter-code groupings, verified by geographic consistency.

### LAS Sector HQs and EOCs (L2 Facility Candidates)

- **Source:** EOC locations from [LAS website](https://www.londonambulance.nhs.uk). Sector HQ locations are not publicly documented; the most geographically central station in each sector is used as a proxy.
- **File:** `data/raw/LAS_L2_facilities.csv`
- **Format:** CSV (columns: `facility_id`, `facility_name`, `facility_type`, `latitude`, `longitude`, `sector`, `notes`)
- **Coverage:** 7 L2 candidate facilities — 5 sector HQs + 2 Emergency Operations Centres (Waterloo and Newham)
- **Notes:** Sector HQ proxies: Camden (North Central), Whipps Cross (North East), Hanwell (North West), Mottingham (South East), Battersea (South West). Exact HQ locations are not material since all facilities are snapped to LSOA centroids — the proxy and the true HQ would typically fall within the same or an adjacent LSOA.

### Data Download

LSOA boundaries and population data can be downloaded by running:

```
python data/download_data.py
```

Station and L2 facility locations are committed directly as CSV files.