"""
Short-bursts variant of FalCom on grid_100 / grid_400.

Algorithm (Cannon, Goldbloom-Helzner, Gupta, Matthews, Suwal 2023):
  - Build an initial partition.
  - For each of `num_bursts` bursts:
      * Run `burst_length` chain steps using `always_accept` (uniform
        sampling over feasible moves, no SA, no greedy).
      * Track the partition with the lowest R^1+R^2 seen so far.
  - At the end of each burst, snap the chain's current state back to
    that best-so-far partition before starting the next burst.

Outputs match the SA runner (`solution_{N}_falcom_{initial,final,trajectory}.{json,html}`).

Run:
    python -m falcomchain_experiments.gurobi.run_falcom_short_bursts 100 \\
        --burst-length 500 --num-bursts 20
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go

# Partition objects keep a `.parent` chain; raise recursion limit so a
# long chain doesn't blow the stack at GC time.
sys.setrecursionlimit(50000)

from falcomchain.graph import Graph
from falcomchain.markovchain import ChainState, MarkovChain
from falcomchain.markovchain.accept import always_accept
from falcomchain.markovchain.energy import compute_energy
from falcomchain.markovchain.proposals import hierarchical_recom
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment

from falcomchain_experiments.gurobi.plot_falcom_solution import plot_falcom_solution
from falcomchain_experiments.gurobi.run_falcom import (
    _uniform_psi, _uniform_super_psi, manhattan_travel_times,
    best_facility, compute_R1_R2, joint_R1_R2, build_solution_dict,
    snapshot_partition, partition_from_snapshot,
)

tree_mod = importlib.import_module("falcomchain.tree.tree")
HERE = Path(__file__).resolve().parent


def run(instance: str, args):
    inst_dir = HERE / "data"
    with open(inst_dir / f"grid_{instance}.meta.json") as f:
        meta = json.load(f)
    with open(inst_dir / meta["grid_path"]) as f:
        raw = json.load(f)
    g_nx = nx.node_link_graph(raw, edges="adjacency")
    G = Graph.from_networkx(g_nx)
    total_steps = args.burst_length * args.num_bursts

    print(f"=== FalCom short-bursts on grid_{instance} ===")
    print(f"  |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  "
          f"w={meta['demand_target_w']}  eps={args.eps}  "
          f"c1 in [{args.c_min_l1}, {args.c_max_l1}]  "
          f"c2 in [{args.c_min_l2}, {args.c_max_l2}]  "
          f"rule={args.rule}  bursts={args.num_bursts}×{args.burst_length} "
          f"(total {total_steps})")

    travel_times = manhattan_travel_times(G)
    Assignment.travel_times = travel_times

    candidates_l1 = {int(n["id"]) for n in raw["nodes"]
                     if n.get("candidate", 0) > 0}
    candidates_l2 = {int(n["id"]) for n in raw["nodes"]
                     if n.get("super_candidate", 0) > 0}

    score_fn = joint_R1_R2 if args.objective == "joint" else compute_R1_R2
    print(f"  scoring objective: {args.objective}")

    # ------ Seed build (with seed retry) ------
    import random as _random
    t0 = time.perf_counter()
    partition = None
    last_exc = None
    seed_tries = [args.seed] + [args.seed + k for k in range(1, 50)
                                if args.seed + k != args.seed]
    for seed_try in seed_tries:
        _random.seed(seed_try)
        if hasattr(tree_mod, "rng"):
            try: tree_mod.rng.seed(seed_try)
            except Exception: pass
        try:
            partition = Partition.from_random_assignment(
                graph=G, epsilon=args.eps,
                demand_target=meta["demand_target_w"],
                assignment_class=Assignment,
                capacity_level=args.c_max_l1, c_min=args.c_min_l1,
                c_min_super=args.c_min_l2, c_max_super=args.c_max_l2,
                min_districts_super=args.min_districts_super,
                init_super_partition=True,
                epsilon_super=meta.get("epsilon_l2", args.eps),
                rule=args.rule, enforce_global_balance=False,
                psi_fn=_uniform_psi, super_psi_fn=_uniform_super_psi,
            )
            if seed_try != args.seed:
                print(f"  [seed retry] succeeded at seed={seed_try}")
            break
        except RuntimeError as e:
            last_exc = e
            continue
    if partition is None:
        raise RuntimeError(f"seed failed across {len(seed_tries)} retries; "
                           f"last: {last_exc}")
    init_elapsed = time.perf_counter() - t0
    partition.parent = None

    def score_and_facilities(p):
        if args.objective == "joint":
            return joint_R1_R2(
                p, candidates_l1=candidates_l1, candidates_l2=candidates_l2,
                travel_times=travel_times, return_facilities=True,
            )
        r1, r2 = compute_R1_R2(
            p, candidates_l1=candidates_l1, candidates_l2=candidates_l2,
            travel_times=travel_times,
        )
        return r1, r2, None, None

    R1_init, R2_init, l1f_init, l2f_init = score_and_facilities(partition)
    print(f"  initial: {init_elapsed:.2f}s  |P^1|={len(partition.parts)}  "
          f"caps={sorted(partition.teams.values())}  "
          f"R1={R1_init:.1f} R2={R2_init:.1f}  R1+R2={R1_init+R2_init:.1f}")

    sol_init = build_solution_dict(
        partition, raw, travel_times=travel_times,
        c_max_l1=args.c_max_l1, c_max_l2=args.c_max_l2,
        wall_time_s=init_elapsed, meta=meta, R1=R1_init, R2=R2_init,
        args=args, label="initial", l1_fac=l1f_init, l2_fac=l2f_init,
    )
    out_init_json = HERE / f"solution_{instance}_falcom_initial.json"
    with open(out_init_json, "w") as f: json.dump(sol_init, f, indent=2)

    # ------ Short-bursts loop ------
    best_snap = snapshot_partition(partition)
    best_obj = R1_init + R2_init
    current_partition = partition
    traj = [(0, best_obj, best_obj)]
    step = 0

    proposal = lambda s: hierarchical_recom(
        s, epsilon_base=args.eps,
        epsilon_super=meta.get("epsilon_l2", args.eps),
        demand_target=meta["demand_target_w"],
        c_min_base=args.c_min_l1, c_min_super=args.c_min_l2,
        c_max_super=args.c_max_l2,
        min_districts_super=args.min_districts_super, rule=args.rule,
        enforce_global_balance=False,
        psi_fn=_uniform_psi, super_psi_fn=_uniform_super_psi,
    )

    print(f"  running {args.num_bursts} bursts × {args.burst_length} steps "
          f"(always_accept within bursts, snap to best between)...")
    t1 = time.perf_counter()
    for b in range(args.num_bursts):
        current_partition.parent = None
        state = ChainState.initial(
            partition=current_partition, energy=best_obj, beta=1.0,
            energy_fn=compute_energy,
        )
        chain = MarkovChain(
            proposal=proposal, constraints=[], accept=always_accept,
            initial_state=state, total_steps=args.burst_length,
        )
        burst_best_obj = best_obj
        burst_best_snap = best_snap
        for s in chain:
            step += 1
            try:
                r1, r2 = score_fn(
                    s.partition, candidates_l1=candidates_l1,
                    candidates_l2=candidates_l2, travel_times=travel_times,
                )
                obj = r1 + r2
            except Exception:
                obj = float("inf")
            traj.append((step, obj, min(obj, burst_best_obj, best_obj)))
            if obj < burst_best_obj:
                burst_best_obj = obj
                burst_best_snap = snapshot_partition(s.partition)
        # Snap back to best-so-far across all bursts (true short-bursts).
        if burst_best_obj < best_obj:
            best_obj = burst_best_obj
            best_snap = burst_best_snap
        # Reconstruct a fresh parent-less Partition from the global-best
        # snapshot so the NEXT burst starts from the best plan seen so far.
        current_partition = partition_from_snapshot(
            best_snap, G, args.c_max_l1,
        )
        elapsed = time.perf_counter() - t1
        print(f"  burst {b+1:3d}/{args.num_bursts}  "
              f"burst_best={burst_best_obj:.1f}  "
              f"global_best={best_obj:.1f}  ({elapsed:.1f}s elapsed)")

    chain_elapsed = time.perf_counter() - t1
    print(f"  short-bursts done: {chain_elapsed:.2f}s  best R^1+R^2 = {best_obj}")

    # ------ Save final ------
    R1_final, R2_final, l1f_final, l2f_final = score_and_facilities(best_snap)
    sol_final = build_solution_dict(
        best_snap, raw, travel_times=travel_times,
        c_max_l1=args.c_max_l1, c_max_l2=args.c_max_l2,
        wall_time_s=init_elapsed + chain_elapsed, meta=meta,
        R1=R1_final, R2=R2_final, args=args, label="final",
        l1_fac=l1f_final, l2_fac=l2f_final,
    )
    out_final_json = HERE / f"solution_{instance}_falcom_final.json"
    with open(out_final_json, "w") as f: json.dump(sol_final, f, indent=2)

    traj_path = HERE / f"solution_{instance}_falcom_trajectory.json"
    with open(traj_path, "w") as f:
        json.dump({"steps": [t[0] for t in traj],
                   "R1_plus_R2_current": [t[1] for t in traj],
                   "R1_plus_R2_best": [t[2] for t in traj],
                   "best": best_obj, "initial": R1_init + R2_init,
                   "chain_elapsed_s": chain_elapsed,
                   "burst_length": args.burst_length,
                   "num_bursts": args.num_bursts,
                   "accept_rule": "short_bursts_always_accept"}, f, indent=2)

    fig_init = plot_falcom_solution(g_nx, sol_init)
    fig_init.update_layout(title=f"FalCom initial — grid_{instance} (R¹+R²={R1_init+R2_init:.1f})")
    fig_init.write_html(HERE / f"solution_{instance}_falcom_initial.html")
    fig_final = plot_falcom_solution(g_nx, sol_final)
    fig_final.update_layout(title=f"FalCom short-bursts best — grid_{instance} (R¹+R²={best_obj:.1f}, "
                                  f"{args.num_bursts}×{args.burst_length} steps)")
    fig_final.write_html(HERE / f"solution_{instance}_falcom_final.html")

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=[t[0] for t in traj], y=[t[1] for t in traj],
        mode="lines", name="R¹+R² (current)",
        line=dict(color="lightsteelblue", width=1), opacity=0.7,
    ))
    fig_traj.add_trace(go.Scatter(
        x=[t[0] for t in traj], y=[t[2] for t in traj],
        mode="lines", name="R¹+R² (best so far)",
        line=dict(color="firebrick", width=2),
    ))
    fig_traj.add_hline(y=best_obj, line_dash="dash", line_color="firebrick",
                       annotation_text=f"final best = {best_obj:.1f}",
                       annotation_position="bottom right")
    # Burst-boundary markers.
    for b in range(1, args.num_bursts):
        fig_traj.add_vline(x=b * args.burst_length, line_dash="dot",
                           line_color="lightgray", line_width=1)
    fig_traj.update_layout(
        title=(f"R¹ + R² trajectory — grid_{instance} (short bursts: "
               f"{args.num_bursts}×{args.burst_length}, always_accept)"),
        xaxis_title="step", yaxis_title="R¹ + R²",
        template="plotly_white",
    )
    fig_traj.write_html(HERE / f"solution_{instance}_falcom_trajectory.html")
    print(f"  wrote initial/final/trajectory .json and .html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", choices=["100", "400"], default="100", nargs="?")
    ap.add_argument("--rule", choices=["per_team", "main"], default="per_team")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--burst-length", type=int, default=500)
    ap.add_argument("--num-bursts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c-min-l1", type=int, default=1)
    ap.add_argument("--c-max-l1", type=int, default=2)
    ap.add_argument("--c-min-l2", type=int, default=2)
    ap.add_argument("--c-max-l2", type=int, default=5)
    ap.add_argument("--objective", choices=["per_district", "joint"], default="joint")
    ap.add_argument("--min-districts-super", type=int, default=2)
    args = ap.parse_args()
    run(args.instance, args)


if __name__ == "__main__":
    raise SystemExit(main())
