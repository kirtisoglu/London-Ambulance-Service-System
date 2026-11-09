# Paper package: "FalCom: An MCMC Sampling Framework for Facility Location and Districting Problems"

Everything behind the paper, in one place. The library is
[FalcomChain](https://github.com/kirtisoglu/FalcomChain) at tag `paper-2026-10`;
this repository holds the experiment code, the data assembly, the stored outputs
of every chain reported in the paper, and the scripts that put the numbers into
the text. All commands assume `PYTHONHASHSEED=0`, Python 3.12+ and the library
installed in editable mode (`pip install -e ../FalcomChain`), plus
`requirements.txt` of this repository.

## Layout of this folder

- `sources/`: the manuscript (`main_arxiv.tex`, article class), the EJOR build
  (`main_ejor.tex`, elsarticle), `highlights.tex`, `ref_final.bib`, `figures/`,
  `tex/table2.tex`, and the compiled article `falcom_article.pdf`.
- `pipeline/`: the shell scripts that ran the compute of the final revision on a
  four-core container, in order: `phase1_london.sh` (London ensembles at 0.15 and
  0.25), `phase2_optimizer.sh` (eight annealer seeds), `phase345.sh` (uniform-demand
  chains, the 4x6 enumeration, the grid multi-start reruns, the sweep, the
  fixed-size grids, the dense-grid annealer attempt), `phase5_fixedsize.sh`.
- `sources/tex/`: the restructured experiments. `sec_experiments.tex` (Section 7) and
  `sec_london.tex` (Section 8, the London case study) are `\input` by `main_arxiv.tex`;
  `app_london.tex` is the appendix with the detailed London diagnostics; the `tab_*.tex`
  files are the tables, generated from the stored outputs, and the `*_numbers.json`
  files hold every generated number that the text quotes. Figures of the two sections
  are in `sources/figures/sec7/`.
- `text_generators/`: scripts that read the stored outputs and write the
  corresponding paragraphs and tables of the manuscript (they expect the sources
  under `/home/user/paper_revision/src`; adjust `P` at the top of each script):
  `apply_sec721.py` (Section 7.2.1), `apply_phase4.py` (7.2.2, 7.2.3, Tables 4 and
  5, contribution (iii)), `apply_dense.py` (Table 6 dense row), `apply_sec74.py`
  (Section 7.4), `apply_phase2.py` (annealer paragraph), `apply_phase3.py`
  (demand-proxy paragraph), `apply_g7.py` (fifth chain), `apply_v4_fixes.py` and
  `trim_pass.py` (the last editorial passes), `make_ejor.py` (EJOR build and
  highlights), `make_bundle.sh` (PDFs and source zip).

## Restructured experiments (Sections 7 and 8)

The experiments are organized around five questions (correctness, stability, component
ablations, scalability, the ensemble relative to an exact optimum) followed by the London
case study as its own section. Three scripts rebuild them, in this order, from the
repository root:

1. `python paper/text_generators/make_sec7_grids.py` writes the tables and figures of
   Section 7 (`tab_enum`, `tab_stability`, `tab_rejections`, `tab_scal`, `tab_fixed`,
   `tab_milp`, `tab_ablation`, `tab_budget`; `fig_forgetting`, `fig_placement`).
2. `python paper/text_generators/make_sec8_london.py` writes those of Section 8
   (`tab_london_layout`, `tab_london_sens`; `fig_plan_vs_layout`, `fig_station_loads`,
   `fig_layout_vs_ensemble`).
3. `python paper/text_generators/apply_sec7_restructure.py` rebuilt `main_arxiv.tex`
   from the earlier manuscript (applied once; it now refuses to run without `--force`,
   because the manuscript has been edited by hand since): it replaces the old Section 7 by the two `\input`
   files, moves the detailed London diagnostics verbatim into `tex/app_london.tex`, and
   updates the abstract and the roadmap. The prose of `sec_experiments.tex` and
   `sec_london.tex` is edited by hand; the numbers it quotes are in the tables and the
   `*_numbers.json` files.

Page layout: the manuscript has no forced float barriers or page breaks, and the algorithms
float. `python paper/text_generators/check_page_gaps.py <pdf> <tmpdir> 0.05 0.07` lists the
pages with blank areas; only the last page should appear.

New runs behind the restructured sections (all with `PYTHONHASHSEED=0`):

| Paper | Script(s) | Stored output |
|---|---|---|
| 7.2 enumeration with non-uniform demand (4x4 grid, demands 80 to 120) | `python -m falcomchain_experiments.validation.enumeration_nonuniform --instance 4x4 --steps 1000000` | `data/derived/validation_nonuniform/` |
| 7.2 reachable set of the proposal on that instance, computed without the sampler | `python -m falcomchain_experiments.validation.enumeration_closure` (after the run above) | `data/derived/validation_nonuniform/enumeration_closure_4x4.json` |
| 7.4 component ablations and retry budget (grid_1000, grid_10000, London) | `falcomchain_experiments/ablation/run_job.sh` on each line of `ablation/logs/jobs.txt` (`run_grid.py`, `run_london.py`, variants in `variants.py`; `ABLATION_BUDGET_S=900` caps the chain of a variant at 15 minutes) | `falcomchain_experiments/ablation/results/<variant>/`, `data/derived/ablation_london/<variant>/` |
| 7.6 the ensemble against the exact optimum (grid_100, grid_400) | `python -m falcomchain_experiments.gurobi.sample_energy --sizes 100 400 --steps 20000 --seeds 1 2 3 4` | `falcomchain_experiments/gurobi/results_sparse/sample_energy_{100,400}.json` |

The section numbers in the table below are those of the earlier layout
(7.2.1 is now 7.2, 7.2.2 is 7.3, 7.2.3 is 7.5, 7.3 is 7.6 and 7.4 is Section 8).
The appendix on the optional candidate-aware cut selection has been removed, so the
candidate-density appendix is now Appendix C and the LAS data assembly Appendix D.

## Where each part of the paper comes from

| Paper | Script(s) | Stored output |
|---|---|---|
| 7.2.1 exact enumeration (3x4, 4x4, 4x6; joint law; iid noise floor) | `falcomchain_experiments/validation/enumeration.py --instance {3x4,4x4,4x6}` (200,000 steps; 3,000,000 for 4x6) | `data/derived/validation_v2/enumeration_*.json`, `*_states.json`, `fig_enumeration_*.pdf` |
| 7.2.2 multi-start, Table 4 | `falcomchain_experiments/scalability/run_scalability.py --sizes {10000,50000} --steps 30000 --seed {42..45} --track-structural --snap-every 60 --out-dir results/multistart`, then `analyze_multistart.py --nodes N --seeds 42 43 44 45` | `falcomchain_experiments/scalability/results/multistart/` |
| 7.2.3 sweep, Table 5, step-time split | `run_scalability.py --sizes 100 400 1000 10000 50000 --steps 10000 --seed 42`, then `analyze_scalability.py` | `scalability/results/grid_*_seed42.json`, `summary.csv`, `analysis.json` |
| 7.2.3 fixed graph size, growing team count | `gurobi/build_instances.py 10000 --w {10000,5000,2500} --suffix _wW`; `run_scalability.py --sizes 10000 --variant _wW --steps 3000 --seed {42,43} --out-dir results/fixed_size` | `scalability/results/fixed_size/` (the 401-team start did not close; see `logs` note in the paper) |
| 7.3 MILP comparison, Table 6 | `gurobi/RUNBOOK.md`, `gurobi/run_falcom_multistart.py`, `gurobi/summarize_milp.py`; dense attempt `run_falcom_multistart.py 400_dense --steps 10000` | `gurobi/results_sparse/`, `gurobi/solution_400_dense_falcom_multistart_summary.json` (all eight seeds failed to build an initial plan) |
| 7.4 London ensembles (0.15; 0.25; uniform demand; Group-based start; annealer) | `falcomchain_experiments/las/run_real_stations.py` with the options documented in the top-level README (sections 1 to 6 of "London Ambulance Service") | `data/derived/real_stations_u97`, `real_stations_u97_eps25`, `real_stations_u97_uniform`, `real_stations_u97_groupstart`, `real_stations_u97_opt` |
| 7.4 diagnostics and readouts | `las/postprocess_real_stations.py`, `las/london_extras.py`, `las/best_plans.py`, `las/summarize_optimizer_runs.py`, `las/london_digest.py` (prints every number of the section) | `ensemble_real_stations.json`, `london_extras.json`, `best_plans.json`, `optimizer_summary_T50000.json`, `fig_*.png` in each directory above |
| 7.4 benchmark state and Appendix E | `las/build_s_las.py`, `analysis/run_build_groups_v2.py`, `analysis/run_calls_to_demand.py`, `analysis/run_demand_variants.py` | `data/derived/lsoa_to_*.csv`, `demand_lsoa*.csv`, `las_fleet_per_group.csv`, `data/raw/` |
| Appendix D candidate density, Table 7 | `run_scalability.py` on `grid_{1000,10000}_{d10,d07}` | `scalability/results/density/` |

The runs of the first submission (sector-wise starts with 99 units on London,
52 and 117 on the grids) are kept under `data/derived/real_stations`,
`real_stations_eps25` and the dated entries of `scalability/results` for
comparison only; the paper reports the re-runs listed above.
