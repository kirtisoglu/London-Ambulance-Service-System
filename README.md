# London-Ambulance-Service-System

Experiment code and data for the FalCom paper: hierarchical capacitated
districting of the London Ambulance Service (LAS) network at LSOA
resolution (4,994-node dual graph, 66 stations, 5 sectors).

Requires the local `falcomchain` package (FalcomChain repo) on the path;
travel-time computation additionally uses `falcomtravel` and OSMnx.

## Repository layout

- `data/` — raw inputs (`raw/`) and derived files (`derived/`): the
  real-station ensemble in `derived/real_stations/`, the enumeration
  validation in `derived/validation/`, and the earlier augmented-candidate
  ensemble in `derived/chain_v3/`.
- `analysis/` — instance-construction scripts (sectors, Groups, station
  catchments, demand attachment).
- `falcomchain_experiments/las/` — the LAS case-study pipeline (see below).
- `falcomchain_experiments/validation/` — exact-enumeration validation of
  the sampler on a 3x4 grid.
- `falcomchain_experiments/experiment1/` — earlier detailed-balance
  diagnostics on synthetic grids.
- `falcomchain_experiments/gurobi/` — MILP (Gurobi) vs FalCom comparison
  on synthetic grids, with numeric results (`solution_*.json`).
- `falcomchain_experiments/scalability/` — runtime and convergence
  scaling experiments up to 50,000-node grids.
- `plot_london.py` — London population/facilities overview map via
  FalcomPlot.

## Reproducing the paper's experiments

All experiment scripts run from the repository root with the `falcomchain`
package installed (`pip install -e ../FalcomChain`), Python 3.12, and
`PYTHONHASHSEED=0` for deterministic set iteration. Outputs land in
`data/derived/<experiment>/`, which is committed for every run reported in
the paper.

### London Ambulance Service on the 66 real stations (paper Section 7.4)

Calibration (locked; `falcomchain_experiments/las/calibration.py`): capacity
unit = block of 3 ambulances, `w_unit = 10,887` calls/year, `eps = 0.15`,
`c1 in [1, 3]`, `c2 in [2, 6]` units, `kappa = 2`, uniform cut selection.
The candidate set is the 66 real stations; no artificial candidates are
added (the counting predicate of the tree cut keeps the recursion from
stranding a candidate-free residual).

1. `python -m falcomchain_experiments.las.run_real_stations --seed S --steps 40000 --snap-every 40 --tag real`
   for `S = 1 2 3 4` (about one hour per chain on one core; the four
   chains can run in parallel). Initial plans are built sector by sector
   (`--init sectors`, the default). Each run writes
   `summary_real_sS_T40000.json`, a per-step `trace_*.csv` and a compact
   `snap_*.npz` of every 40th plan to `data/derived/real_stations/`.
2. `python -m falcomchain_experiments.las.postprocess_real_stations --tag real --steps 40000`
   computes the diagnostics reported in the paper (cross-chain KS distances,
   split-R-hat and ESS, forgetting curves, boundary frequencies, contested
   LSOAs, station opening frequencies and capacity mixes, Group
   co-membership against the 21 real Groups, and the operational layout
   `s_LAS` against the ensemble) into `ensemble_real_stations.json` and the
   `fig_*.png` figures.

`falcomchain_experiments/las/build_s_las.py` reconstructs the operational
layout `s_LAS` (Appendix B of the paper); `run_cdba.py`,
`compute_cdba_travel_times.py`, `run_chain_v3.py` and `postprocess_ensemble.py`
are the earlier pipeline on an augmented candidate set and are kept for
reference only.

### Exact-enumeration validation (paper Section 7.2.1)

`python -m falcomchain_experiments.validation.enumeration --steps 200000`
lists every feasible hierarchical state of a 3x4 grid (93 level-1 and 119
joint states), runs three chains from very different starts and compares
their empirical laws with each other and with the uniform and
spanning-tree laws. Outputs: `data/derived/validation/enumeration_3x4.json`,
`enumeration_3x4_states.json` and `fig_enumeration_3x4.{png,pdf}`;
`--plot-only` redraws the figure from the saved files.

### Multi-start agreement and scalability on synthetic grids (paper Sections 7.2.2-7.2.3)

The grids and their per-grid parameters live in
`falcomchain_experiments/gurobi/data/grid_{N}.json` and `.meta.json`
(built by `gurobi/build_instances.py`). From `falcomchain_experiments/scalability/`:

- timing sweep: `PYTHONHASHSEED=0 python run_scalability.py --sizes 100 400 1000 10000 50000 --steps 10000 --seed 42`
  then `python analyze_scalability.py` (table `results/summary.csv`, figures
  `figures/scal_t_vs_*.png`);
- multi-start diagnostics: `PYTHONHASHSEED=0 python run_scalability.py --sizes 10000 --steps 50000 --seed S --track-structural --snap-every 100 --out-dir results/multistart`
  for four seeds, then `python analyze_multistart.py --nodes 10000 --seeds 42 43 44 45`
  (`results/multistart/multistart_10000.json`, `figures/fig_multistart_10000.png`);
  the 50,000-node grid uses `--steps 20000 --snap-every 50`.

### MILP comparison (paper Section 7.3)

`falcomchain_experiments/gurobi/solve_milp.py` solves the exact model with
Gurobi on `grid_100`, `grid_400` and `grid_400_dense`;
`gurobi/run_falcom.py {100,400}` runs FalCom as an optimizer on the same
instances. Numeric results are the `gurobi/solution_*.json` files.

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