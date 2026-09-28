# Runbook: MILP comparison on the sparse grids (paper Section 7.3)

This runbook is for a machine with a Gurobi license. It re-solves the exact
hierarchical model on the two smallest synthetic grids after the candidate
sets were rebuilt in the sparse regime (1.5 candidate sites per team, no
augmentation, `F2 = F1`), and produces the FalCom optimizer-mode runs and
figures that Section 7.3 of the paper compares against. Nothing here needs
the 10,000- and 50,000-node grids.

Everything below runs from `falcomchain_experiments/gurobi/` of the
London-Ambulance-Service-System repository. Budget: about one to two hours,
most of it Gurobi time on grid_400.

## 0. Environment

```bash
git clone https://github.com/kirtisoglu/FalcomChain.git
git clone https://github.com/kirtisoglu/London-Ambulance-Service-System.git
python3 -m venv venv && source venv/bin/activate
pip install -e FalcomChain
pip install -r London-Ambulance-Service-System/requirements.txt
pip install gurobipy            # the license must be visible to gurobipy
pip install kaleido             # optional: PNG export of the plotly figures
python -c "import gurobipy; m = gurobipy.Model(); print('gurobi ok', gurobipy.gurobi.version())"
cd London-Ambulance-Service-System/falcomchain_experiments/gurobi
export PYTHONHASHSEED=0         # required by every script below
```

## 1. Check the instances (do not rebuild them)

The sparse instances are committed. Confirm you have them and that they are
the ones the paper uses:

```bash
python - <<'PY'
import json
for name in ("100", "400"):
    m = json.load(open(f"data/grid_{name}.meta.json"))
    print(name, m["regime"], "k =", m["k_teams"], "sites =", m["n_l1_candidates"], "sha =", m["grid_sha256"][:12])
PY
```

Expected: `100 sparse k = 5 sites = 8 sha = 24fb4059e56a` and
`400 sparse k = 10 sites = 15 sha = 2589fc95a629`. If the hashes differ, stop
and report; do not rebuild.

## 2. Build the capacity variant of grid_400

```bash
python build_capacity_instance.py      # writes data/grid_400_dense.json and .meta.json
```

It takes the grid_400 demands, raises two interior units to 1.4 w so that no
single-team district can hold them, and draws the sparse candidate set with
the two peaks forced in. Record the printed `L1 = L2 = ...` count.

## 3. MILP solves

The paper's constraints are `c1 in [1,2]`, `c2 in [2,5]` (from the meta files)
and at least two open level-1 districts per open superdistrict, which the
script does not read from the meta file, so pass `--c-min-l2 2
--min-l1-per-l2 2` every time. Rename each output before the next run,
because the script always writes `solution_{instance}_median.json`.

```bash
# grid_100, exact, SHIR contiguity (seconds to a minute)
python solve_milp.py 100 --obj median --contiguity shir --c-min-l2 2 --min-l1-per-l2 2 --time-limit 3600 --threads 4 | tee log_100_shir.txt
mv solution_100_median.json solution_100_median_shir.json

# grid_400, SHIR contiguity, one hour cap
python solve_milp.py 400 --obj median --contiguity shir --c-min-l2 2 --min-l1-per-l2 2 --time-limit 3600 --threads 4 | tee log_400_shir.txt
mv solution_400_median.json solution_400_median_shir.json

# grid_400 without contiguity constraints (the relaxation the paper contrasts)
python solve_milp.py 400 --obj median --contiguity none --c-min-l2 2 --min-l1-per-l2 2 --time-limit 3600 --threads 4 | tee log_400_noshir.txt
mv solution_400_median.json solution_400_median_noshir.json

# capacity variant, c1_max = 2 forced by the peaks
python solve_milp.py 400_dense --obj median --contiguity shir --c-min-l2 2 --min-l1-per-l2 2 --time-limit 3600 --threads 4 | tee log_400_dense.txt
mv solution_400_dense_median.json solution_400_dense_median_shir.json
```

For each run note from the log: objective, MIP gap at termination, wall time,
number of open level-1 and level-2 facilities. A gap above 0 on grid_400 is
acceptable; report it as it is.

## 4. FalCom in optimizer mode on the same instances

```bash
python run_falcom.py 100 --steps 10000 --seed 42 | tee log_falcom_100.txt
python run_falcom.py 400 --steps 10000 --seed 42 | tee log_falcom_400.txt
python run_falcom_multistart.py 100 --steps 10000 --seeds 100 200 300 400 500 600 700 800 | tee log_falcom_ms_100.txt
python run_falcom_multistart.py 400 --steps 10000 --seeds 100 200 300 400 500 600 700 800 | tee log_falcom_ms_400.txt
```

Outputs land next to the scripts as `solution_{N}_falcom_initial.json`,
`solution_{N}_falcom_final.json`, `solution_{N}_falcom_trajectory.json`,
`solution_{N}_falcom_multistart_summary.json` and the matching `.html`
figures. If a run fails at initialization, retry with `--seed 43` and say so.

## 5. Figures

```bash
python plot_falcom_solution.py 100 median_shir     # figures/solution_100_median_shir.html
python plot_falcom_solution.py 400 median_shir
python plot_falcom_solution.py 400 median_noshir
python plot_falcom_solution.py 400_dense median_shir
```

If kaleido is installed, export PNGs at 300 dpi next to each HTML file:

```bash
python - <<'PY'
import plotly.io as pio, glob
for f in glob.glob("figures/*.html") + glob.glob("solution_*_falcom_*.html"):
    fig = pio.read_json(f) if f.endswith(".json") else None
PY
```

(plotly cannot re-read an HTML file; if PNGs are wanted, re-run the plot
scripts with `fig.write_image(..., scale=3)` added after `write_html`, or
leave the HTML files and they will be rendered on the paper side.)

## 6. Summarize and hand back

```bash
python summarize_milp.py            # table on stdout, results_sparse/milp_summary.json
git checkout -b milp-sparse-results
git add data/grid_400_dense.json data/grid_400_dense.meta.json solution_*.json figures results_sparse log_*.txt
git commit -m "MILP comparison on the sparse grids (Section 7.3)"
git push -u origin milp-sparse-results
```

Then report: the branch name, the table printed by `summarize_milp.py`, and
anything that deviated from this runbook (seeds changed, time limits hit,
errors). The paper's Section 7.3 is rewritten from `milp_summary.json` and
the figures.

## Sanity checks on the numbers

- grid_100 has 5 teams and 8 sites: the MILP opens 3 to 5 level-1 facilities
  and 1 or 2 level-2 facilities, all districts contiguous under SHIR.
- FalCom's best objective should be within a few percent of the SHIR optimum
  on grid_100 and within the MILP gap on grid_400. A FalCom value below a
  proven MILP optimum means the two do not solve the same model; report it.
- The no-contiguity relaxation on grid_400 is at most the SHIR value; some of
  its districts may be disconnected, which is the point of that run.

## Agent prompt (paste to the local agent)

> In the London-Ambulance-Service-System repository, follow
> `falcomchain_experiments/gurobi/RUNBOOK.md` end to end with the Gurobi
> license on this machine. Do not rebuild grid_100 or grid_400. Rename the
> MILP outputs as the runbook says before each subsequent solve. When done,
> push the branch `milp-sparse-results` and paste the table printed by
> `summarize_milp.py` together with any deviations.
