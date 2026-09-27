# Scalability Experiment — Plan

**Paper:** §6.6 (Experiment 2: Scalability) of `FalCom (38).pdf`.
**Goal:** Empirically validate Contribution (iv): FalCom handles graphs
with up to 50,000 basic units, and per-step cost is governed by the
**merged subgraph size `|H|`** rather than the **total graph size `|V|`**.
**Headline target:** at `|V| = 50,000`, FalCom produces 1,000 samples in
under 1 hour on a single workstation.

**Scope (2026-05-05 update):** focused on the three "scalability stress"
sizes — `|V| ∈ {10000, 20000, 50000}`. The smaller sizes from the
original plan (100…5000) don't probe the locality claim because their
total size is already comparable to a typical merged subgraph; we drop
them.

---

## 1. Why this experiment matters

ReCom-family chains were originally evaluated at political-redistricting
scale (a few hundred to a few thousand precincts). FalCom claims to push
the same paradigm to 50,000-node graphs without sacrificing per-step
locality. Two falsifiable claims:

1. **Sublinear in `|V|`.** The slope of `log(time/step) vs log(|V|)` is
   strictly less than 1.
2. **Linear in `|H|`.** The slope of `log(time/step) vs log(|H|)` is
   approximately 1.

Together these demonstrate that the work per step is local to the
re-partitioned superdistrict and does not scale with the full graph.

---

## 2. Instances

Three square synthetic grids per paper §6.4:

| `|V|`  | rows × cols | Notes |
|-------:|-------------|-------|
| 10,000 | 100 × 100   | small (locality should already dominate) |
| 20,000 | 100 × 200   | medium |
| 50,000 | 200 × 250   | **headline size** |

Saved as JSON to `data/grid_{N}.json` (NetworkX `node_link_graph`
format). Generation is reproducible: each grid uses seed `42 + i` so
the three instances are independent but pinned.

(Already implemented in `falcomchain_experiments/grids.py::make_grid`.)

**Per-instance attributes (paper §6.4):**
- `demand` ~ Uniform(80, 120) i.i.d.
- Candidate density `ρ = 0.05` (one per 20 nodes on average)
- Edge attribute `shared_perim = 1`

---

## 3. Hyperparameters (paper §6.4)

| Parameter | Value | Note |
|-----------|-------|------|
| `|L|` | 2 | hierarchy levels |
| `ε¹` | 0.10 | level-1 demand-balance tolerance |
| `ε²` | 0.15 | level-2 |
| `c¹_max` | 4 | max teams per level-1 district |
| `c²_max` | 3 | max level-1 districts per superdistrict |
| `γ` | 1 | candidate-aware bias (paper default for case studies) |
| `accept` | `always_accept` | sampling mode |
| `T` | **100 steps** | per the paper §6.6 protocol |

**Demand target.** Paper specifies `c¹_max = 4` but doesn't give an
explicit `w`. To pick `w` consistently across grid sizes:

- mean demand `≈ 100`; `w_max = 120`
- granularity floor: `ε¹ ≥ w_max / (c_min · w)` ⇒ at `c_min = 1`,
  `w ≥ 120 / 0.10 = 1,200`.
- choose **`w = 2,000`** (~20 base units per unit-capacity district),
  comfortably above the floor at `c_min = 1`. With `c_max = 4`, target
  district sizes range 20–80 nodes, scaling cleanly across all grid sizes.
- `c_min = 1` (matches paper's default; relaxed `c_min` not relevant
  for scalability claims).

**Repair.** Run `repair_facility_density` once per instance with
`strategy = "fast_center"` (no external dependency). Record the number
of artificial candidates added — should remain a small fraction of
`|V|` thanks to ρ = 0.05 spread.

---

## 4. Protocol

For each grid size `|V|`:

1. **Build instance** via `make_grid(|V|, seed=42 + i)` from
   `falcomchain_experiments/grids.py`.
2. **Repair** Assumption 6.1 with `strategy="fast_center"`,
   `c_min=1`, `demand_target=2000`, `epsilon=0.10`. Record
   `n_artificial`.
3. **Build initial partition** via
   `Partition.from_random_assignment(epsilon=0.10, demand_target=2000,
   c_min=1, capacity_level=4)`.
4. **Wrap in `ChainState`** with `beta=1.0`, `energy_fn=None`
   (sampling mode, paper default).
5. **Run chain** for `T = 100` steps with:
   - `proposal = partial(hierarchical_recom, epsilon_base=0.10,
     epsilon_super=0.15, demand_target=2000, gamma=1.0)`
   - `accept = always_accept`
   - per-step instrumentation (see §5).
6. **Record** to `results/{|V|}.json`.

Three independent seeds per grid size → median over 300 step-times
per `|V|`. (Single seed × 100 steps would be noisy at small grids.)

---

## 5. Per-step instrumentation

The chain doesn't currently expose `|H|` per step, so we add a
**callback** that wraps step timing AND derives `|H|` from
`state.partition.flow.part_flows`.

```python
import time
from typing import List, Dict

class ScalabilityProbe:
    def __init__(self):
        self.records: List[Dict] = []
        self._t_last = time.perf_counter()

    def __call__(self, state, accepted):
        # Called after each chain step — accepted state is `state`.
        t_now = time.perf_counter()
        dt = t_now - self._t_last

        # |H| = nodes in the superdistricts touched this step
        flow = state.partition.flow
        touched_super_ids = (
            flow.part_flows.get("in", set())
            | flow.part_flows.get("out", set())
        )
        # node count of those touched superdistricts in the current state
        H_nodes = 0
        for super_id in touched_super_ids:
            for level1_id in state.partition.super_parts.get(super_id, ()):
                H_nodes += len(state.partition.assignment.parts.get(level1_id, ()))

        self.records.append(dict(
            step=len(self.records),
            t_step=dt,
            H_size=H_nodes,
            n_districts=len(state.partition.parts),
            accepted=bool(accepted),
        ))
        self._t_last = t_now
```

**Note on `|H|`:** Strictly the paper defines `|H| = |D²ⱼ|` for the
re-partitioned superdistrict at step `j`. The implementation above
sums over all *touched* superdistricts (in/out flow); for
`hierarchical_recom` exactly one superdistrict is re-partitioned per
step under `resample_super_partition`, so the two definitions match.
Verify the assumption with a sanity assert in the smoke test.

---

## 6. Outputs

### 6.1 Raw timings — `results/{|V|}.json`

```json
{
  "V": 5000,
  "rows": 50, "cols": 100,
  "n_artificial": 47,
  "demand_target": 2000,
  "epsilon_l1": 0.10,
  "epsilon_l2": 0.15,
  "c_max_l1": 4, "c_max_l2": 3,
  "gamma": 1.0,
  "T": 100,
  "seeds": [42, 43, 44],
  "steps": [
    {"step": 0, "t_step": 0.024, "H_size": 187, "n_districts": 32, "accepted": true},
    ...
  ]
}
```

### 6.2 Aggregated CSV — `results/summary.csv`

| `|V|` | k | mean(\|H\|) | median(t_step) | total_for_1000_samples_h |
|------:|--:|------:|-----:|-----:|
| 100   | 5  | 22  | 0.005 | 0.0014 |
| 500   | 22 | 25  | 0.008 | 0.0022 |
| ...   |    |     |       |       |
| 50,000 | 2,200 | 28 | ~3.0 | ~0.83 |

(Numbers above are illustrative — actual values produced by the run.)

### 6.3 Figures — `figures/`

1. **`fig_time_vs_V.png`** — log-log plot, median t/step on y-axis,
   `|V|` on x-axis. Fitted slope shown in legend. **Target slope < 0.7**.

2. **`fig_time_vs_H.png`** — log-log plot, median t/step on y-axis,
   mean `|H|` on x-axis. Fitted slope. **Target slope ≈ 1.0 ± 0.2**.

3. **`fig_table.tex`** — LaTeX table for the paper, covering
   `|V|, k, mean |H|, median t_step, projected hours for 1,000 samples`.

---

## 7. File layout

```
falcomchain_experiments/scalability/
├── PLAN.md                  ← this file
├── run_scalability.py       ← driver (one entry point)
├── analyze_scalability.py   ← reads results/, produces figures/
├── results/                 ← per-grid JSON + summary.csv (gitignored)
│   └── .gitkeep
└── figures/                 ← published figures (committed)
    └── .gitkeep
```

---

## 8. Acceptance criteria

The experiment is a **success** if all four hold:

1. At `|V| = 50,000`, projected wall-clock for 1,000 samples is
   **under 1 hour** on the test workstation.
2. Slope of `log(t_step) vs log(|V|)` is **< 0.7**.
3. Slope of `log(t_step) vs log(mean |H|)` is in `[0.8, 1.2]`.
4. Median `|H| / |V|` is **< 0.05** at `|V| ≥ 5,000` (locality
   confirmed: re-partition touches a small fraction of the graph).

If any criterion fails, do **not** silently lower the threshold —
investigate and fix the root cause (likely a non-local operation
introduced unintentionally).

---

## 9. Time budget

| Grid size | Per-seed (100 steps) | × 3 seeds | Cumulative |
|-----------|---------------------|-----------|-----------|
| 100       | < 1 s               | < 3 s     | 3 s |
| 500       | ~ 1 s               | ~ 3 s     | 6 s |
| 1,000     | ~ 2 s               | ~ 6 s     | 12 s |
| 2,000     | ~ 5 s               | ~ 15 s    | 27 s |
| 5,000     | ~ 15 s              | ~ 45 s    | 72 s |
| 10,000    | ~ 1 min             | ~ 3 min   | ~5 min |
| 20,000    | ~ 2.5 min           | ~ 7.5 min | ~12 min |
| 50,000    | ~ 6 min             | ~ 18 min  | ~30 min |
| **Total** |                     |           | **~50 min** |

Numbers are rough projections from the paper's headline ("1,000 samples in
< 1 h at 50K"). 100-step runs at 50K should take ~6 minutes.
The full sweep should fit into a single workstation hour.

---

## 10. What to write in the paper after the run

Required for §6.6 of the paper:

- One paragraph reporting the four headline numbers (50K time,
  two slopes, |H|/|V| ratio).
- The two log-log plots.
- The summary table.
- A one-line caveat: "Single-workstation, single-thread Python; further
  speedups available via parallel chains (see §7.4 limitations)."

The paper will use these to support Contribution (iv).

---

## 11. Sequence of next concrete actions

```
[ ] Implement run_scalability.py (driver)
[ ] Implement analyze_scalability.py (figures + table)
[ ] Smoke-test on |V|=100, |V|=500 — verify ScalabilityProbe records valid |H|
[ ] Verify the |H| sanity assert ("touched superdistricts" matches the paper's |D²ⱼ|)
[ ] Run full sweep, three seeds per size
[ ] Generate figures and summary table
[ ] Update results/summary.csv
[ ] Commit figures/ to repo (results/ stays gitignored)
[ ] Add a paragraph to paper §6.6 with the headline numbers
```

---

## 12. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| `repair_facility_density` blows up at 50K | Use `fast_center` (linear per iter) — sanity check on the 5K instance first. With ρ=0.05 most components should be small to start. Worst case: run repair once and serialise the repaired graph to disk, skip from subsequent runs. |
| Memory pressure at 50K (full all-pairs travel times) | **Use `Assignment.travel_times = None` (graph-distance fallback) for the scalability experiment** — energy isn't computed in sampling mode anyway, and travel times only feed `compute_energy`. Confirm no path goes through travel_times during chain step. |
| `bipartition_tree` resample loop hits its retry limit on a hard instance | The `M` retry budget should be dimensioned by `|H|` not `|V|`. Already passes from `CutParams`; verify before running. |
| Slope of t vs \|V\| comes out > 0.7 | Indicates a non-local operation in the chain. Profile with `cProfile` to identify the culprit; common suspects: full-graph BFS in spanning-tree construction (should be subgraph-local), full-graph hash ops in flow construction. |
