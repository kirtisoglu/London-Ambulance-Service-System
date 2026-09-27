"""
Multi-start wrapper: run `run_falcom` 8 times with different RNG seeds
on grid_100, keep the best R^1+R^2 across all chains, and write the
winning seed's solution JSON / plots as
`solution_{N}_falcom_multistart_best.{json,html}`.

Run:
    python -m falcomchain_experiments.gurobi.run_falcom_multistart 100
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Reuse the runner's run() and arg-parsing pieces.
from falcomchain_experiments.gurobi import run_falcom as rf

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", choices=["100", "400"], default="100", nargs="?")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--beta-magnitude", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--rule", choices=["per_team", "main"], default="per_team")
    ap.add_argument("--c-min-l1", type=int, default=1)
    ap.add_argument("--c-max-l1", type=int, default=2)
    ap.add_argument("--c-min-l2", type=int, default=2)
    ap.add_argument("--c-max-l2", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[100, 200, 300, 400, 500, 600, 700, 800])
    ap.add_argument("--objective", choices=["median", "per_district", "joint"],
                    default="median")
    ap.add_argument("--cooling", choices=["linear", "cyclic"], default="linear")
    ap.add_argument("--n-cycles", type=int, default=5)
    ap.add_argument("--min-districts-super", type=int, default=2)
    args = ap.parse_args()

    results = []
    t_total0 = time.perf_counter()
    for k, seed in enumerate(args.seeds, start=1):
        print(f"\n========== run {k}/{len(args.seeds)}  seed={seed} ==========")
        # Build a sub-args namespace matching what run_falcom.run() expects.
        sub_args = argparse.Namespace(
            instance=args.instance,
            rule=args.rule,
            eps=args.eps,
            steps=args.steps,
            seed=seed,
            c_min_l1=args.c_min_l1,
            c_max_l1=args.c_max_l1,
            c_min_l2=args.c_min_l2,
            c_max_l2=args.c_max_l2,
            beta_magnitude=args.beta_magnitude,
            objective=args.objective,
            cooling=args.cooling,
            n_cycles=args.n_cycles,
            min_districts_super=args.min_districts_super,
        )
        t0 = time.perf_counter()
        try:
            rf.run(args.instance, sub_args)
            ok = True
            err = None
        except Exception as e:
            ok = False
            err = repr(e)
            print(f"  FAILED: {err}")
        elapsed = time.perf_counter() - t0

        if not ok:
            results.append({"seed": seed, "ok": False, "error": err,
                            "elapsed_s": elapsed})
            continue

        # Read the just-produced solution_{N}_falcom_final.json.
        final_path = HERE / f"solution_{args.instance}_falcom_final.json"
        with open(final_path) as f:
            sol = json.load(f)
        traj_path = HERE / f"solution_{args.instance}_falcom_trajectory.json"
        with open(traj_path) as f:
            traj = json.load(f)
        results.append({
            "seed": seed,
            "ok": True,
            "elapsed_s": elapsed,
            "obj_value": sol["obj_value"],
            "R1": sol["R1"],
            "R2": sol["R2"],
            "initial_R1_plus_R2": traj.get("initial"),
        })
        # Stash this run's outputs by seed.
        for suffix in ("initial.json", "final.json", "trajectory.json",
                       "initial.html", "final.html", "trajectory.html"):
            src = HERE / f"solution_{args.instance}_falcom_{suffix}"
            dst = HERE / f"solution_{args.instance}_falcom_seed{seed}_{suffix}"
            if src.exists():
                shutil.copyfile(src, dst)
    t_total = time.perf_counter() - t_total0

    # ----- Aggregate -----
    print(f"\n========== MULTI-START SUMMARY ==========")
    print(f"  total wall time: {t_total:.1f}s for {len(args.seeds)} seeds")
    print(f"  {'seed':>5s}  {'initial':>8s}  {'final':>8s}  {'R1':>5s}  {'R2':>5s}  {'elapsed':>8s}")
    for r in results:
        if r["ok"]:
            print(f"  {r['seed']:>5d}  {r['initial_R1_plus_R2']:>8.1f}  "
                  f"{r['obj_value']:>8.1f}  {r['R1']:>5.1f}  {r['R2']:>5.1f}  "
                  f"{r['elapsed_s']:>7.1f}s")
        else:
            print(f"  {r['seed']:>5d}  FAILED: {r['error']}")

    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        best = min(ok_results, key=lambda r: r["obj_value"])
        print(f"\n  BEST: seed={best['seed']}  R1+R2={best['obj_value']}  "
              f"(R1={best['R1']} R2={best['R2']})")
        # Promote the best run's outputs to *_multistart_best.*
        for suffix in ("initial.json", "final.json", "trajectory.json",
                       "initial.html", "final.html", "trajectory.html"):
            src = HERE / f"solution_{args.instance}_falcom_seed{best['seed']}_{suffix}"
            dst = HERE / f"solution_{args.instance}_falcom_multistart_best_{suffix}"
            if src.exists():
                shutil.copyfile(src, dst)
        print(f"  wrote solution_{args.instance}_falcom_multistart_best_*.{{json,html}}")

    # Write a multistart summary JSON.
    out = HERE / f"solution_{args.instance}_falcom_multistart_summary.json"
    with open(out, "w") as f:
        json.dump({"instance": args.instance, "n_seeds": len(args.seeds),
                   "total_wall_s": t_total, "results": results,
                   "params": {"steps": args.steps,
                              "beta_magnitude": args.beta_magnitude,
                              "eps": args.eps, "rule": args.rule,
                              "c_l1": [args.c_min_l1, args.c_max_l1],
                              "c_l2": [args.c_min_l2, args.c_max_l2]}}, f, indent=2)
    print(f"  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
