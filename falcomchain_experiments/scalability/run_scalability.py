"""
FalCom scalability experiment for paper §7 (scalability).

For each reproducible benchmark grid in ``gurobi/data/grid_{N}.json`` run
``--steps`` chain steps with ``always_accept`` and the *per-grid*
hyperparameters recorded in its ``.meta.json`` (the single source of
truth produced by ``gurobi/build_instances.py``). Record per-step wall
time and the merged-subgraph size |H|. Raw step records are written to
``results/grid_{N}_seed{S}.json``.

Why per-step |H| matters: ``hierarchical_recom`` re-partitions exactly
one superdistrict per step, so the work touches only the merged region
H (a handful of L1 districts), never all of V. The scalability claim is
that per-step cost tracks |H|, not |V|. This runner measures both.

Reproducibility:
- Run with PYTHONHASHSEED=0 (deterministic set iteration in CDBA/cuts).
- Each seed pins the falcomchain RNG via ``set_seed(seed)``.
- Per-grid parameters come from the meta file, not hard-coded globals.
- Travel times are computed lazily as Manhattan distance on grid
  coordinates — avoids the |V|² memory blow-up at 50K nodes.

Usage::

    # smoke test: one small grid, few steps
    PYTHONHASHSEED=0 python3 run_scalability.py --sizes 100 --steps 200

    # full §7 sweep
    PYTHONHASHSEED=0 python3 run_scalability.py \
        --sizes 100 400 1000 10000 50000 --steps 50000
"""

import argparse
import datetime as _dt
import importlib.metadata as md
import json
import math
import os
import platform
import sys
import time
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np

# Make the experiment package importable
_EXP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_ROOT))

from falcomchain import (
    MarkovChain,
    Partition,
    SuperFacilityAssignment,
    always_accept,
    hierarchical_recom,
    set_seed,
)
from falcomchain.markovchain.state import ChainState
from falcomchain.partition.assignment import Assignment


# ---------------------------------------------------------------------------
# Defaults (override on the CLI).  Grids + per-grid params live in
# gurobi/data; we never duplicate the hyperparameters here.
# ---------------------------------------------------------------------------
GUROBI_DATA = _EXP_ROOT / "gurobi" / "data"
ALL_SIZES = [100, 400, 1000, 10000, 50000]
N_INIT_RETRIES = 12   # reseed safety net for the seed build (rarely needed)
GAMMA = 0.0           # baseline ψ² = teams (no hub-coherence penalty)


# ---------------------------------------------------------------------------
# Lazy travel-time provider — Manhattan distance on grid coordinates.
# Avoids storing an |V|² dict (would be ~2.5 GB at |V|=50K).
# ---------------------------------------------------------------------------
class GridManhattanDist:
    """Dict-like ``__getitem__((u, v)) -> distance`` returning the
    Manhattan distance between grid coordinates of u and v."""

    __slots__ = ("_nodes",)

    def __init__(self, graph):
        self._nodes = {
            n: (data["C_X"], data["C_Y"]) for n, data in graph.nodes(data=True)
        }

    def __getitem__(self, key):
        u, v = key
        ux, uy = self._nodes[u]
        vx, vy = self._nodes[v]
        return abs(ux - vx) + abs(uy - vy)


# ---------------------------------------------------------------------------
# Per-step instrumentation
# ---------------------------------------------------------------------------
class ScalabilityProbe:
    """
    Callback for ``MarkovChain(callbacks=[...])``.

    Records per-step wall time and the merged-subgraph size |H| (sum of
    base-node counts of the level-1 districts touched by the step, i.e.
    those appearing in ``flow.part_flows['in']`` or ``['out']``). For
    ``hierarchical_recom`` exactly one superdistrict is re-partitioned
    per step, so the touched districts equal the merged region D²ⱼ.
    """

    __slots__ = ("records", "_t_last", "_demands")

    def __init__(self, demands=None):
        self.records = []
        self._t_last = time.perf_counter()
        # When set (node -> demand), the probe also records the structural
        # convergence statistics R1 (level-1 access cost) and the per-team
        # demand spread each step. None => timing-only (scalability) mode.
        self._demands = demands

    def __call__(self, state, accepted):
        t_now = time.perf_counter()
        dt = t_now - self._t_last
        self._t_last = t_now

        # True merged-subgraph size H: the base nodes of the superdistrict
        # that was merged and re-partitioned this step. After the flip the
        # merged region is covered exactly by the freshly-numbered districts
        # in ``flip.new_ids`` (base nodes are conserved across a step), so
        # |H| = Σ_{i in new_ids} |parts[i]|. We must NOT use ``part_flows``
        # here: it stores set-differences (new_ids \ merged_ids), so a re-cut
        # that reuses district IDs nets to an undercount (often 0).
        part = state.partition
        parts = part.assignment.parts
        new_ids = getattr(part.flip, "new_ids", None) or ()
        H_nodes = sum(len(parts.get(i, ())) for i in new_ids)

        # Number of L1 districts in the merged region (the superdistrict).
        merged_ids = getattr(getattr(part, "superflip", None), "merged_ids", None)
        n_merged = len(merged_ids) if merged_ids else 0

        rec = {
            "step": len(self.records),
            "t_step": dt,
            "H_size": H_nodes,
            "n_districts": len(part.parts),
            "n_merged": n_merged,
            "n_new": len(new_ids),
            "accepted": bool(accepted),
        }

        # Structural convergence observables (Section 4 state coordinates):
        #   R1   = level-1 demand-weighted access cost = sum of L1 district
        #          1-median radii (the base term of the median objective).
        #          O(k); the L1 facility is maintained incrementally.
        #   dspread = max-min per-team demand d(D)/c(D) across districts.
        if self._demands is not None:
            fac = getattr(state, "facility", None)
            if fac is not None:
                rec["R1"] = sum(r for r in fac.radii.values()
                                if r != float("inf"))
                teams = part.assignment.teams
                d = self._demands
                ptd = []
                for pid, nodes in parts.items():
                    c = teams.get(pid, 1) or 1
                    ptd.append(sum(d.get(v, 0) for v in nodes) / c)
                rec["dspread"] = (max(ptd) - min(ptd)) if ptd else 0.0

        self.records.append(rec)


# ---------------------------------------------------------------------------
# Environment + reproducibility helpers
# ---------------------------------------------------------------------------

def _ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _env_fingerprint() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "packages": {
            "networkx": _ver("networkx"),
            "numpy": _ver("numpy"),
            "scipy": _ver("scipy"),
            "matplotlib": _ver("matplotlib"),
            "falcomchain": _ver("falcomchain"),
            "pymetis": _ver("pymetis"),
        },
    }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def load_grid(path: Path) -> nx.Graph:
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="adjacency")


def run_one(n: int, seed: int, steps: int, results_dir: Path,
            with_l2_facility: bool = False,
            track_structural: bool = False,
            snap_every: int = 0, variant: str = "") -> dict:
    print(f"\n=== chain run |V|={n:,} seed={seed} steps={steps:,} ===")

    grid_path = GUROBI_DATA / f"grid_{n}{variant}.json"
    meta_path = GUROBI_DATA / f"grid_{n}{variant}.meta.json"
    meta = json.load(open(meta_path))

    # Per-grid parameters — the single source of truth.
    demand_target = meta["demand_target_w"]
    eps_l1 = meta["epsilon_l1"]
    eps_l2 = meta["epsilon_l2"]
    c_min_l1 = meta["c_min_l1"]
    c_max_l1 = meta["c_max_l1"]
    c_min_l2 = meta["c_min_l2"]
    c_max_l2 = meta["c_max_l2"]
    kappa_min = int(meta.get("min_l1_per_l2", 1))

    g = load_grid(grid_path)
    print(f"  Loaded {grid_path.name}: {g.number_of_nodes():,} nodes, "
          f"{g.number_of_edges():,} edges")
    n_l1 = sum(1 for _, d in g.nodes(data=True) if d.get("candidate"))
    n_l2 = sum(1 for _, d in g.nodes(data=True) if d.get("super_candidate"))
    print(f"  Candidates: L1={n_l1}, L2={n_l2}")
    print(f"  Params (from meta): w={demand_target}, eps1={eps_l1}, eps2={eps_l2}, "
          f"c1=[{c_min_l1},{c_max_l1}], c2=[{c_min_l2},{c_max_l2}]")

    set_seed(seed)
    Assignment.travel_times = GridManhattanDist(g)

    # Initial partition, built at the full eps_l1 = 0.15 (no relaxation).
    #
    # Subtlety: the global recursive bipartition centers each district's
    # demand window on `demand_target`. The coverage rule needs
    # k = ceil(total/w) teams, so the k districts must AVERAGE total/k,
    # which is strictly below the nominal w. If we seed with
    # demand_target = w, the greedy peeling takes districts near w and
    # starves the last one below (1-eps)*w -> infeasible, and the heuristic
    # exhausts its retry budget (empirically 0/8 at |V| in {1000, 10000}).
    # Centering on the exact per-team workload total/ceil(total/w) recenters
    # the window on the achievable average and the seed succeeds on the
    # first draw at eps=0.15 (8/8). This mirrors the chain proposal, which
    # already recenters locally via new_demand_target = super_demand /
    # super_teams (markovchain/proposals.py).
    total_demand = sum(d["demand"] for _, d in g.nodes(data=True))
    k_teams = max(1, math.ceil(total_demand / demand_target))
    seed_demand_target = total_demand / k_teams
    t0 = time.perf_counter()
    partition = None
    n_init_retries = 0
    for retry in range(N_INIT_RETRIES):
        try:
            partition = Partition.from_random_assignment(
                graph=g,
                epsilon=eps_l1,
                demand_target=seed_demand_target,
                assignment_class=None,
                capacity_level=c_max_l1,
                c_min=c_min_l1,
            )
            break
        except Exception:
            n_init_retries += 1
            set_seed(seed * 1000 + n_init_retries)
    if partition is None:
        raise RuntimeError(
            f"Could not build initial partition for |V|={n}, seed={seed} "
            f"at eps={eps_l1} ({n_init_retries} reseeds)."
        )
    t_init_partition = time.perf_counter() - t0
    print(f"  Initial partition: {len(partition.parts)} districts "
          f"({t_init_partition:.2f}s, {n_init_retries} reseeds, "
          f"seed_demand_target={seed_demand_target:.0f} (=total/{k_teams}), "
          f"eps={eps_l1})")

    # Level-2 facilities are a DETERMINISTIC readout of the partition (the
    # demand-weighted 1-median hub of each superdistrict), not a sampled
    # variable. At gamma=0 (our regime — cut selection is NOT biased by
    # candidate-aware / hub-coherence scores) the L2 hub never feeds back
    # into the proposal or acceptance, so it is not part of the sampling
    # kernel. We therefore do NOT recompute it each step (its naive full
    # recompute is O(|V|) and would mask the kernel's O(|H|) locality); the
    # median objective is evaluated post-hoc on saved samples instead. Pass
    # --with-l2-facility to time the readout-in-the-loop variant.
    super_facility_fn = (
        SuperFacilityAssignment.from_state if with_l2_facility else None
    )
    t0 = time.perf_counter()
    state = ChainState.initial(
        partition=partition,
        energy=0.0, beta=1.0,
        super_facility_fn=super_facility_fn,
    )
    t_init_state = time.perf_counter() - t0
    print(f"  ChainState init: {t_init_state:.2f}s "
          f"(L2 facility per-step: {with_l2_facility})")

    demands = ({v: g.nodes[v]["demand"] for v in g.nodes}
               if track_structural else None)
    probe = ScalabilityProbe(demands=demands)
    proposal = partial(
        hierarchical_recom,
        epsilon_base=eps_l1,
        epsilon_super=eps_l2,
        demand_target=demand_target,
        gamma_super=GAMMA,
        c_min_base=c_min_l1,
        c_min_super=c_min_l2,
        c_max_super=c_max_l2,
        min_districts_super=kappa_min,
    )
    chain = MarkovChain(
        proposal=proposal,
        constraints=lambda p: True,
        accept=always_accept,
        initial_state=state,
        total_steps=steps,
        callbacks=[probe],
    )

    # The chain swallows per-step RuntimeError internally (treats it as a
    # rejection: state unchanged, callback fires with accepted=False). But
    # the recursion can also raise IndexError / PopulationBalanceError on
    # the "empty complement" edge case; those propagate out of __next__
    # WITHOUT advancing the chain counter or firing the callback, so we
    # catch them here, count them as failed attempts, and retry (the global
    # RNG has advanced, so the next draw differs). Cap total attempts.
    print(f"  Running {steps:,} steps (max {steps * 4:,} attempts)…")
    # Compact snapshots of the level-1/level-2 plan for the multi-start
    # diagnostics (forgetting curves, boundary agreement across chains).
    node_list = sorted(g.nodes)
    node_pos = {v: i for i, v in enumerate(node_list)}
    edges = np.array([(node_pos[u], node_pos[v]) for u, v in g.edges()], dtype=np.int32)
    snaps = {"step": [], "district": [], "super": [], "capacity": []}

    def take_snapshot(step_no):
        part = chain.state.partition
        assignment = part.assignment
        sup = part.super_assignment
        teams = part.assignment.teams
        dist = np.empty(len(node_list), dtype=np.int32)
        supr = np.empty(len(node_list), dtype=np.int32)
        cap = np.empty(len(node_list), dtype=np.int8)
        for v in node_list:
            d = assignment[v]
            i = node_pos[v]
            dist[i] = int(d)
            supr[i] = int(sup.get(d, d))
            cap[i] = int(teams.get(d, 1))
        snaps["step"].append(step_no)
        snaps["district"].append(dist)
        snaps["super"].append(supr)
        snaps["capacity"].append(cap)

    if snap_every:
        take_snapshot(0)
    chain.total_steps = steps * 4
    chain_iter = iter(chain)
    probe._t_last = time.perf_counter()
    t_chain_start = time.perf_counter()
    n_step = 0
    n_failed = 0
    err_types = {}
    next_report = max(1, steps // 20)
    while n_step < steps and n_step + n_failed < steps * 4:
        try:
            next(chain_iter)
            n_step += 1
            if snap_every and n_step % snap_every == 0:
                take_snapshot(n_step)
            if n_step % next_report == 0:
                el = time.perf_counter() - t_chain_start
                print(f"    {n_step:,}/{steps:,} steps  "
                      f"({el:.1f}s, {el / n_step * 1000:.1f}ms/step, "
                      f"{n_failed} hard-fail)")
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001 - heuristic edge cases
            n_failed += 1
            t = type(exc).__name__
            err_types[t] = err_types.get(t, 0) + 1
            probe._t_last = time.perf_counter()  # don't bill retry to next step
    t_chain = time.perf_counter() - t_chain_start
    n_rejected = sum(1 for r in probe.records if not r["accepted"])
    print(f"  Chain: {n_step:,} steps ({len(probe.records)} recorded, "
          f"{n_rejected} rejected), {n_failed:,} hard-fail {err_types} in "
          f"{t_chain:.1f}s ({t_chain / max(n_step, 1) * 1000:.2f}ms/step)")

    rejection_report = chain.rejection_report()
    print(f"  Rejections by cause: {rejection_report.get('causes')}")
    if snap_every:
        snap_path = results_dir / f"snap_grid_{n}{variant}_seed{seed}.npz"
        np.savez_compressed(
            snap_path,
            step=np.array(snaps["step"], dtype=np.int32),
            district=np.stack(snaps["district"]),
            super=np.stack(snaps["super"]),
            capacity=np.stack(snaps["capacity"]),
            node_ids=np.array(node_list, dtype=np.int32),
            edges=edges,
        )
        print(f"  Wrote {snap_path} ({len(snaps['step'])} snapshots)")

    out = {
        "n_nodes": n,
        "n_edges": g.number_of_edges(),
        "seed": seed,
        "T_steps": steps,
        "demand_target": demand_target,
        "epsilon_l1": eps_l1,
        "epsilon_l2": eps_l2,
        "seed_demand_target": seed_demand_target,
        "k_teams_coverage": k_teams,
        "c_min_l1": c_min_l1,
        "c_max_l1": c_max_l1,
        "c_min_l2": c_min_l2,
        "c_max_l2": c_max_l2,
        "gamma": GAMMA,
        "l2_facility_per_step": with_l2_facility,
        "track_structural": track_structural,
        "n_l1_candidates": n_l1,
        "n_l2_candidates": n_l2,
        "grid_sha256": meta.get("grid_sha256"),
        "init_partition_time_s": t_init_partition,
        "init_state_time_s": t_init_state,
        "chain_total_time_s": t_chain,
        "n_init_partition_retries": n_init_retries,
        "n_chain_steps_succeeded": n_step,
        "n_chain_steps_recorded": len(probe.records),
        "n_chain_steps_rejected": n_rejected,
        "n_chain_steps_hard_failed": n_failed,
        "hard_fail_error_types": err_types,
        "n_districts_initial": len(partition.parts),
        "kappa_min": kappa_min,
        "snap_every": snap_every,
        "variant": variant,
        "rejection_report": rejection_report,
        "steps": probe.records,
    }
    out_path = results_dir / f"grid_{n}{variant}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"  Wrote {out_path}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=ALL_SIZES)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-l2-facility", action="store_true",
                    help="recompute the L2 super-facility readout every step "
                         "(O(|V|)); off by default since at gamma=0 it is not "
                         "part of the sampling kernel.")
    ap.add_argument("--track-structural", action="store_true",
                    help="record convergence observables R1 (level-1 access "
                         "cost) and per-team demand spread each step.")
    ap.add_argument("--out-dir", default="results",
                    help="results subdirectory (e.g. results/convergence).")
    ap.add_argument("--variant", default="",
                    help="instance-name suffix, e.g. _w2000 loads grid_{n}_w2000.json "
                         "and names the outputs accordingly.")
    ap.add_argument("--snap-every", type=int, default=0,
                    help="store a compact plan snapshot every N steps "
                         "(0 = off) for the multi-start diagnostics.")
    args = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        print(
            "WARNING: PYTHONHASHSEED is not 0. CDBA/cut set-iteration may "
            "differ across invocations. Re-run as:\n"
            "    PYTHONHASHSEED=0 python3 run_scalability.py …",
            file=sys.stderr,
        )

    base = Path(__file__).resolve().parent
    results_dir = base / args.out_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    env = _env_fingerprint()
    print(f"Environment: Python {env['python']} on {env['platform']}")
    print(f"  PYTHONHASHSEED={env['pythonhashseed']}")
    print(f"  sizes={args.sizes}  steps={args.steps:,}  seed={args.seed}")

    summary = []
    overall_t0 = time.perf_counter()
    for n in args.sizes:
        try:
            summary.append(run_one(n, args.seed, args.steps, results_dir,
                                   with_l2_facility=args.with_l2_facility,
                                   track_structural=args.track_structural,
                                   snap_every=args.snap_every,
                                   variant=args.variant))
        except RuntimeError as exc:
            print(f"  ! |V|={n} failed: {exc}")
            continue

    total = time.perf_counter() - overall_t0
    print(f"\nTotal experiment wall time: {total:.1f}s")

    idx = {
        "experiment": "scalability",
        "paper_section": "7",
        "ran_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "T_steps": args.steps,
        "sizes": args.sizes,
        "seed": args.seed,
        "environment": env,
        "total_wall_time_s": total,
        "runs": [
            {
                "n_nodes": s["n_nodes"],
                "seed": s["seed"],
                "n_chain_steps_succeeded": s["n_chain_steps_succeeded"],
                "chain_total_time_s": s["chain_total_time_s"],
                "n_districts_initial": s["n_districts_initial"],
                "n_l1_candidates": s["n_l1_candidates"],
                "n_l2_candidates": s["n_l2_candidates"],
            }
            for s in summary
        ],
    }
    with open(results_dir / "index.json", "w") as f:
        json.dump(idx, f, indent=2)
    print("Wrote results/index.json")


if __name__ == "__main__":
    main()
