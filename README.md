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
  `analysis/repair_catchment_contiguity.py` makes every station catchment of
  the benchmark state contiguous on the rook graph (17 LSOAs moved; log in
  `data/derived/lsoa_to_station_contiguity_moves.csv`).
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
stranding a candidate-free residual). Every sampled state carries exactly
`ceil(d / w) = 97` capacity units: the sector-wise start apportions the 97
units to the five sectors by largest remainder (`--zone-units exact`, the
default), and the level-2 windows are centred on `d / 97` (`--level2-target
dbar`, the default). All commands below assume `PYTHONHASHSEED=0`.

1. Sampling, four chains:
   `python -m falcomchain_experiments.las.run_real_stations --seed S --steps 80000 --snap-every 40 --tag real --out-dir data/derived/real_stations_u97`
   for `S = 1 2 3 4` (about one hour per chain on one core). Each run
   writes `summary_real_sS_T80000.json`, a per-step `trace_*.csv` (energy,
   counts, acceptance and rejection cause at every step) and a compact
   `snap_*.npz` of every 40th plan.
2. Diagnostics and readouts:
   `python -m falcomchain_experiments.las.postprocess_real_stations --in-dir data/derived/real_stations_u97 --tag real --steps 80000`
   (cross-chain and within-chain-halves KS distances on every 20th step,
   split-R-hat and ESS, forgetting curves, boundary frequencies, contested
   LSOAs, station opening frequencies and capacity mixes, Group
   co-membership and the adjusted Rand index against the 21 real Groups with
   its null reference, and the operational layout `s_LAS` against the
   ensemble) into `ensemble_real_stations.json` and the `fig_*.png` figures;
   `python -m falcomchain_experiments.las.london_extras --in-dir data/derived/real_stations_u97 --steps 80000`
   (realized per-unit loads, shares outside the nominal windows, jump-chain
   reweighting, effective number of plans, Monte Carlo errors) into
   `london_extras.json`;
   `python -m falcomchain_experiments.las.best_plans --in-dir data/derived/real_stations_u97 --steps 80000`
   (best sampled plans and sample extremes of the station-leg metrics,
   Group-count distribution, top co-membership pairs within and across the
   real Groups, the LSOAs the road matrix cannot reach, per-chain standard
   errors) into `best_plans.json`;
   `python -m falcomchain_experiments.las.london_digest --in-dir data/derived/real_stations_u97 --steps 80000`
   prints every number quoted in the section.
3. Sensitivity to the tolerance: the same three steps with
   `--eps 0.25 --steps 40000 --out-dir data/derived/real_stations_u97_eps25`.
4. Sensitivity to the demand proxy: `python analysis/run_demand_variants.py`
   writes `data/derived/demand_lsoa_uniform.csv` (borough totals spread
   uniformly over the LSOAs of each borough); then the same steps with
   `--demand-csv data/derived/demand_lsoa_uniform.csv --out-dir data/derived/real_stations_u97_uniform`
   (the post-processing scripts take the same `--demand-csv`).
5. A structurally different start: `--init groups --seed 5 --out-dir data/derived/real_stations_u97_groupstart`
   builds the initial plan inside the 21 real Groups (the three
   single-station Groups merged into their neighbours, 18 zones) instead of
   the five sectors. To compare it with the four sector-start chains, link
   the five chains' `summary_*`, `trace_*` and `snap_*` files into one
   directory (`data/derived/real_stations_u97_5chains`, whose links are not
   committed) and run the post-processing and `best_plans` there; the
   pairwise KS distances that involve seed 5 and the per-chain readouts of
   `best_plans.json` give the comparison.
6. FalCom as an optimizer on the same instance (the annealing rule of
   paper Section 7.3, `--optimizer`), eight seeds:
   `python -m falcomchain_experiments.las.run_real_stations --seed S --steps 50000 --snap-every 5000 --optimizer --tag opt --out-dir data/derived/real_stations_u97_opt`
   for `S = 11 ... 18`, then
   `python -m falcomchain_experiments.las.summarize_optimizer_runs --in-dir data/derived/real_stations_u97_opt --steps 50000 --ensemble data/derived/real_stations_u97/ensemble_real_stations.json`
   (best plan of each seed with its station-leg metrics, `optimizer_summary_T50000.json`).

`falcomchain_experiments/las/build_s_las.py` reconstructs the operational
layout `s_LAS` (the appendix on the LAS data assembly in the paper);
`run_cdba.py`, `compute_cdba_travel_times.py`, `run_chain_v3.py` and
`postprocess_ensemble.py` are the earlier pipeline on an augmented candidate
set and are kept for reference only. The ensembles of the first submission,
whose sector-wise start carried 99 units, are kept under
`data/derived/real_stations` and `real_stations_eps25`.

### Exact-enumeration validation (paper Section 7.2.1)

`python -m falcomchain_experiments.validation.enumeration --instance 3x4 --steps 200000 --out-dir data/derived/validation_v2`
(and `--instance 4x4`; `--instance 4x6 --steps 3000000` for the 24-unit
instance with two or three superdistricts) lists every feasible hierarchical
state of the grid (3x4: 93 level-1 and 119 joint states; 4x4: 435 and 595;
4x6: 22,549 and 43,689), runs three chains from very different starts and
compares their empirical laws with each other and with the uniform and
spanning-tree laws, at level 1 and for the joint (level-1, level-2) law,
including the law of level 2 given level 1 and an iid noise floor (the
total-variation distance expected between two independent samples of the
run length). Outputs: `enumeration_<instance>.json`,
`enumeration_<instance>_states.json` and `fig_enumeration_<instance>.{png,pdf}`;
`--plot-only` redraws the figure from the saved files.

### Multi-start agreement and scalability on synthetic grids (paper Sections 7.2.2-7.2.3)

The grids and their per-grid parameters live in
`falcomchain_experiments/gurobi/data/grid_{N}.json` and `.meta.json`
(built by `PYTHONHASHSEED=0 python gurobi/build_instances.py`, sparse regime:
1.5 candidate sites per team, `F2 = F1`, no augmentation; the augmented
instances of the first draft are archived under `gurobi/data/augmented/`,
their results under `scalability/results/augmented/` and
`scalability/results/multistart_augmented/`). Zone-wise starts apportion
exactly `ceil(d / w)` units to the zones. Every step record carries the
wall-clock split between the supergraph re-partition (`t_super`) and the
base re-cut (`t_base`). From `falcomchain_experiments/scalability/`:

- timing sweep: `PYTHONHASHSEED=0 python run_scalability.py --sizes 100 400 1000 10000 50000 --steps 10000 --seed 42`
  then `python analyze_scalability.py` (table `results/summary.csv` with the
  median, mean and 95th percentile of the step time and the share of a step
  spent in the supergraph recursion, figures `figures/scal_t_vs_*.png`);
- multi-start diagnostics: `PYTHONHASHSEED=0 python run_scalability.py --sizes 10000 --steps 30000 --seed S --track-structural --snap-every 60 --out-dir results/multistart`
  for the seeds S = 42, 43, 44, 45, then `python analyze_multistart.py --nodes 10000 --seeds 42 43 44 45`
  (`results/multistart/multistart_10000.json`, `figures/fig_multistart_10000.png`;
  KS on every twentieth post-burn-in step, within-chain halves, split-R-hat
  and ESS per observable); the 50,000-node grid uses the same command with
  `--sizes 50000` (about 2.2 h per chain on the cloud container) and
  `--nodes 50000` in the analysis;
- fixed graph size, growing team count (Section 7.2.3): the variants
  `grid_10000_w10000`, `_w5000` and `_w2500` (101, 201 and 401 teams, 1.5
  candidates per team) are built by
  `PYTHONHASHSEED=0 python gurobi/build_instances.py 10000 --w W --suffix _wW`
  and run with `python run_scalability.py --sizes 10000 --variant _wW --steps 3000 --seed S --out-dir results/fixed_size`.

### MILP comparison (paper Section 7.3)

`gurobi/RUNBOOK.md` is the step-by-step procedure (Gurobi license needed) for
the sparse instances; `gurobi/summarize_milp.py` collects its outputs.

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

### LAS Group headquarters and EOCs (`LAS_L2_facilities.csv`)

- **Source:** the 21 Group headquarters from the 2025 LAS list of departments (every headquarters is an active station); the two Emergency Operations Centres (Waterloo and Newham) from the LAS website.
- **File:** `data/raw/LAS_L2_facilities.csv`
- **Format:** CSV (columns: `facility_id`, `facility_name`, `facility_type`, `LSOA21CD`, `LSOA21NM`, `latitude`, `longitude`, `sector`, `notes`)
- **Coverage:** 23 rows: 21 Group headquarters (`facility_type = group_hq`) and 2 EOCs (`eoc`).
- **Notes:** the sampler does not use this file. Level-2 facilities in the chain are the open stations of each superdistrict (demand-weighted 1-median); the Group headquarters enter only the reconstruction of the operational layout `s_LAS`, and the EOCs are listed for reference. The `sector` column of `LAS_stations.csv` follows the station letter codes; the sectors used by the paper and by `sector_zones()` are the Integrated Care System sectors of `lsoa_to_sector_borough.csv`.

### Data Download

LSOA boundaries and population data can be downloaded by running:

```
python data/download_data.py
```

Station and L2 facility locations are committed directly as CSV files.