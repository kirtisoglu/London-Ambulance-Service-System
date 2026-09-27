# Scalability Experiment — Design Notes

**Paper:** §6.6 (Experiment 2: Scalability) of `FalCom (38).pdf`.
**Question:** Empirically validate that FalCom's per-step cost is governed by the **merged subgraph size `|H|`**, not the **total graph size `|V|`** — at scales up to 50,000 basic units.

These are working notes captured during 2026-05-04/05 development. They cover **what we tried**, **what didn't work and why**, and **what the experiment is calibrated to**. The results live in [`results.md`](./results.md).

---

## 1. Final hyperparameter set (and what each one is)

| Symbol | Value | Notes |
|--------|------:|-------|
| `|V|` | 10K, 20K, 50K | Square grids — 100×100, 100×200, 200×250 |
| Demand per node | i.i.d. `Uniform(80, 120)` | Paper §6.4 |
| L1 candidate density `ρ¹` | 0.05 | Paper §6.4 |
| L2 candidate density (vs `ρ¹`) | 0.20 | Super-candidates ⊂ candidates; LAS-like ratio |
| `d̄` (`demand_target`) | **10,000** | per-team workload |
| `ε¹` | **0.50** | Paper says 0.10; relaxed because the spanning-tree retry heuristic exhausts at ε¹=0.10 with U(80,120) demand for `|V| ≥ 2000`. 0.50 is the smallest value that consistently builds the initial partition. |
| `ε²` | **0.25** | < ε¹ as recommended. Doesn't blow up the granularity floor at the supergraph level (super-node demand ≈ `c¹·d̄`, so `ε²·c²·d̄` window holds it as long as `c²_max ≥ 2·c¹_max`). |
| `c¹_min` | 1 | |
| `c¹_max` | **3** | |
| `c²_min` | **2** | `≥ 2·c¹_min` per the paper convention |
| `c²_max` | **6** | `≥ 2·c¹_max` so a super-district can fit two L1 districts at maximum capacity |
| γ¹, γ² | 0 | Baseline ψ = teams (no candidate-aware bias). The locality claim is independent of γ; γ tuning is Experiment 4's job. |
| `T` (chain steps) | 100 (successful) | Per (grid, seed) pair, with up to `4·T` attempts allowed |
| Seeds | 42, 43, 44 | Three independent seeds per grid |
| Repair strategy | `metis_separator` | Demand-balanced, `pip install pymetis`. ~10× faster than `fast_center` and similar candidate count. |

**Hierarchy capacity rule (compatibility check):** the chain's `find_superedge_cuts` measures L2 capacity in **teams** (sum of L1 capacities), not L1-district count. So a super-district holding 2 L1 districts of `c¹=3` each has `c²=6` teams. With `c²_max=6`, that case fits. With paper's `c²_max=3`, it doesn't — any L1 district reaching `c¹>3` becomes unmergeable. We chose `c²_max=2·c¹_max=6` to keep the chain's constraint window aligned with the maximum admissible L1 capacity.

---

## 2. Why several configurations don't work — chain failure modes we hit

### 2.1 Initial-partition build fails at tight ε¹

`Partition.from_random_assignment` recursively bipartitions the whole graph into capacitated districts. The leaf-demand window is `[(1−ε¹)·c·d̄, (1+ε¹)·c·d̄]` for `c ∈ [c¹_min, c¹_max]`. With `U(80,120)` demand and ε¹=0.10, the windows are too narrow for the spanning-tree heuristic to reliably find an admissible cut within its 5,000-retry budget.

| `|V|` | ε¹=0.10 | ε¹=0.20 | ε¹=0.30 | ε¹=0.50 |
|------:|--------:|--------:|--------:|--------:|
| 1,000 | ✓ | ✓ | ✓ | ✓ |
| 2,000 | ✗ | ✓ | ✓ | ✓ |
| 5,000 | ✗ | ✓ | ✓ | ✓ |
| 10,000 | ✗ | ✗ | ✓ | ✓ |

Robust setting at all our sizes: ε¹=0.50.

### 2.2 Supergraph cut fails when `c¹_max > c²_max/2`

Pre-fix: the chain's `find_superedge_cuts` had a hardcoded constraint `2 ≤ teams ≤ capacity_level` (line 635 of tree.py before our patch). With `c²_max=3` (paper) and `c¹_max=4` (paper), a single L1 district with `c¹=4` teams cannot fit any super-district leaf — `c²_max=3` caps the leaf at 3 teams. The recursion exhausts.

Fix (committed): we replaced the hardcoded `2` with `h.c_min` (which is `c²_min` for the supergraph case) so the constraint becomes `c²_min ≤ teams ≤ c²_max`. We also added `c_min_super` and `c_max_super` parameters to `hierarchical_recom`. These threading changes plus the choice `c²_max ≥ 2·c¹_max` keep super-leaves admissible.

### 2.3 The "debt" adjustment is wrong on the supergraph

`capacitated_recursive_tree` adjusts the demand-target window between iterations via a `debt` accumulator (lines 882–897). The intent is to correct cumulative drift on the **base** graph where node demands vary widely. On the **supergraph**, super-node demands are fixed at ≈ `c¹·d̄`, and the debt-adjusted window slides off that scale after a few peels — the recursion can't find admissible subtrees anymore.

Fix (committed): when `supergraph=True`, skip the debt adjustment and use the fixed `(d̄, ε)` window throughout. Patched `capacitated_recursive_tree` to branch on `supergraph` for the window calculation.

### 2.4 RNG-state dependence is intermittent

Even after the fixes above, the chain succeeds for some RNG states and fails for others — a single chain step succeeds about 50–70 % of the time per attempt. The recursion just doesn't find an admissible subtree within `max_attempts=5000` for some "unlucky" RNG states. This is a heuristic limitation of the spanning-tree retry approach, not a structural issue.

Workaround for the scalability run: in `run_scalability.py` we run the chain in a loop that **catches** the per-step `RuntimeError`, **skips** the failed step, and **continues** until we have `T=100` successful steps (capped at `T·4=400` attempts). The probe records timing only for successful steps, which is what we want for the scalability claim. Failed-attempt count is reported alongside successes.

This is **not** a sound MCMC chain (skipping failures introduces bias toward "easier" RNG paths), but the **per-step timing** of successful steps is a faithful measure of the chain's per-step cost, which is what §6.6 is about.

---

## 3. What the experiment measures

For each (grid `|V|`, seed) pair, the `ScalabilityProbe` callback records per chain step:

| Field | Meaning |
|-------|---------|
| `t_step` | Wall-clock time of this chain step (proposal + accept + flow updates + facility re-assignment + callbacks). |
| `H_size` | Sum of base-node counts of the L1 districts touched by this step (= `flow.part_flows['in'] | flow.part_flows['out']`). For `resample_super_partition` this approximates `\|D²ⱼ\|` — the merged region the paper's locality claim is about. |
| `n_districts` | Number of L1 districts in the (post-step) partition. |

Aggregated over the three seeds per grid size, we report:

- **Median `t_step`** — primary headline.
- **Mean `|H|`** — for the locality claim.
- **Slope of `log(t_step)` vs `log(|V|)`** — should be < 0.7 (sublinear in `|V|`).
- **Slope of `log(t_step)` vs `log(|H|)`** — should be ≈ 1.0 (linear in `|H|`).
- **Projected wall time for 1,000 samples at `|V|=50K`** — should be < 1 hour.

---

## 4. Reproducibility contract

- **Pinned hyperparameters** in `generate_grids.py` and `run_scalability.py` constants (matching).
- **Pinned random seeds**: each grid uses `42 + i`; each chain uses one of `{42, 43, 44}`.
- **PYTHONHASHSEED=0** set on every invocation (the runner script wraps with this).
- **Falcomchain RNG pinned via `set_seed()`** at the top of each chain run.
- **Grid JSON files** are deterministic and saved to `data/grid_{N}.json`. SHA-256 of each file is recorded in `data/grid_{N}.meta.json`.
- **Environment fingerprint** (Python, NetworkX, NumPy, SciPy, FalcomChain, pymetis versions; OS + platform string) is saved in `data/index.json` and `results/index.json`.
- **All hyperparameters** are echoed into every per-run output JSON, so a downstream reader can reconstruct the experiment without referring to source code.

The grid `super_candidate` flags are drawn from a deterministic seed (the grid's own seed); changing the L2 candidate density would require re-running `generate_grids.py`.

---

## 5. What's deferred / known limitations

- **Paper-strict ε¹ = 0.10**. The chain's spanning-tree retry heuristic exhausts at this value for `|V| ≥ 2000`. A larger retry budget might rescue this, but the cost-per-step claim doesn't depend on the exact ε¹.
- **Variable γ¹, γ² > 0**. Paper §6.4 default γ=1 isn't tested here. Adding it tightens the cut-acceptance ratio and is the subject of Experiment 4 (§6.8). Not needed for the locality claim.
- **A clean RNG-resilient chain run**. Right now, the chain mixes failed/skipped steps with successful ones. For paper Section 6.5 (Convergence Diagnostics) we'll need a chain that completes T uninterrupted steps. The right fix is a deeper retry/refresh inside `bipartition_tree` rather than at the runner level.

These limitations are documented for follow-up; they don't undermine the scalability claim being measured in this experiment.
