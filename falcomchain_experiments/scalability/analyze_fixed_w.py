#!/usr/bin/env python3
"""Compare the fixed-workload run (grid_10000_w2000, w = 2,000) with the standard
grid_10000 sweep run (w = 20,000): per-step time, merged-subgraph size, district
count, acceptance and rejection causes. Paper Section 7.2.3 (level-2 cost check).

Usage: python analyze_fixed_w.py [results/fixed_w/grid_10000_w2000_seed42.json] [results/grid_10000_seed42.json]
"""
import json
import sys
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent


def summarize(path):
    d = json.load(open(path))
    steps = d["steps"]
    acc = [s for s in steps if s.get("accepted", True)]
    t = [s["t_step"] * 1000 for s in acc]
    rep = d.get("rejection_report", {})
    return {
        "file": str(path), "variant": d.get("variant", ""), "w": d.get("demand_target"),
        "k_teams": d.get("k_teams_coverage"), "n_l1_candidates": d.get("n_l1_candidates"),
        "steps": d.get("n_chain_steps_succeeded"), "init_partition_time_s": d.get("init_partition_time_s"),
        "n_districts_initial": d.get("n_districts_initial"),
        "n_districts_mean": mean(s["n_districts"] for s in steps),
        "H_mean": mean(s["H_size"] for s in steps), "H_median": median(s["H_size"] for s in steps),
        "t_step_median_ms": median(t), "t_step_mean_ms": mean(t),
        "acceptance": rep.get("acceptance_rate"), "rejection_causes": rep.get("causes"),
        "chain_total_time_s": d.get("chain_total_time_s"),
    }


def main():
    fixed = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/fixed_w/grid_10000_w2000_seed42.json"
    std = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results/grid_10000_seed42.json"
    out = {"fixed_w": summarize(fixed), "standard": summarize(std) if std.exists() else None}
    json.dump(out, open(HERE / "results/fixed_w/fixed_w_summary.json", "w"), indent=2)
    for k, v in out.items():
        if v:
            print(f"[{k}] w={v['w']} k={v['k_teams']} sites={v['n_l1_candidates']} districts~{v['n_districts_mean']:.0f} "
                  f"|H| mean {v['H_mean']:.0f} step median {v['t_step_median_ms']:.1f} ms "
                  f"acceptance {v['acceptance']:.3f} init {v['init_partition_time_s']:.0f}s causes {v['rejection_causes']}")


if __name__ == "__main__":
    main()
