"""
Analysis driver for the scalability experiment.

Reads `results/grid_{N}_seed{S}.json` files produced by run_scalability.py,
aggregates them into a per-grid summary, and produces the two log-log
figures plus a CSV table for the paper.

Usage::

    python3 analyze_scalability.py
"""

import csv
import datetime as _dt
import json
import math
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _slope(xs, ys):
    """Least-squares slope of log-log data."""
    if len(xs) < 2:
        return float("nan")
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = mean(lx), mean(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    if den == 0:
        return float("nan")
    return num / den


def _load_runs():
    """Load every results/grid_*_seed*.json + group by V."""
    by_size = {}
    for path in sorted(RESULTS_DIR.glob("grid_*_seed*.json")):
        with open(path) as f:
            d = json.load(f)
        by_size.setdefault(d["n_nodes"], []).append(d)
    return by_size


def _per_size_aggregates(runs_for_size):
    """Combine all seeds for one grid size.

    Per-step cost is measured on *accepted* steps only: a rejected step is
    a proposal whose spanning-tree cut exhausted its retry budget, leaving
    the partition unchanged; its recorded |H| is stale (the previous
    accepted flip) and its time is not representative O(|H|) work. The
    accepted steps carry the true merged-region size and the genuine
    per-step cost the locality claim is about.
    """
    all_steps = []
    for r in runs_for_size:
        all_steps.extend(s for s in r["steps"] if s.get("accepted", True))
    if not all_steps:
        return None
    t_steps = [s["t_step"] for s in all_steps]
    H_sizes = [s["H_size"] for s in all_steps]
    n_districts = [s["n_districts"] for s in all_steps]
    n_merged = [s.get("n_merged", 0) for s in all_steps]
    n = runs_for_size[0]["n_nodes"]
    accept_rates = [r["n_chain_steps_succeeded"] and
                    (sum(1 for s in r["steps"] if s.get("accepted")) /
                     max(len(r["steps"]), 1))
                    for r in runs_for_size]
    return {
        "n_nodes": n,
        "n_seeds": len(runs_for_size),
        "n_steps_accepted": len(all_steps),
        "k_districts_median": median(n_districts),
        "k_districts_mean": mean(n_districts),
        "superdistrict_l1_mean": mean(n_merged),
        "accept_rate_mean": mean(accept_rates),
        "t_step_median_ms": 1000 * median(t_steps),
        "t_step_mean_ms": 1000 * mean(t_steps),
        "t_step_min_ms": 1000 * min(t_steps),
        "t_step_max_ms": 1000 * max(t_steps),
        "H_mean": mean(H_sizes),
        "H_median": median(H_sizes),
        "H_max": max(H_sizes),
        "H_over_V_mean": mean(H_sizes) / n,
        "projected_1000_samples_h": 1000 * median(t_steps) / 3600,
    }


def main():
    by_size = _load_runs()
    if not by_size:
        print(f"No results in {RESULTS_DIR}. Run run_scalability.py first.")
        return

    rows = []
    for n in sorted(by_size):
        agg = _per_size_aggregates(by_size[n])
        if agg is not None:
            rows.append(agg)

    # --- Print summary ---
    print(f"=== Scalability summary ({len(rows)} grid sizes) ===")
    hdr = (f"{'V':>8} {'k':>5} {'mean|H|':>8} {'|H|/|V|':>8} "
           f"{'median(t/step)':>15} {'1000-smpl h':>12} {'#acc':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['n_nodes']:>8,} {r['k_districts_median']:>5.0f} "
            f"{r['H_mean']:>8.0f} {r['H_over_V_mean']:>8.3f} "
            f"{r['t_step_median_ms']:>13.2f}ms "
            f"{r['projected_1000_samples_h']:>11.3f}h "
            f"{r['n_steps_accepted']:>7}"
        )

    # --- CSV table ---
    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # --- Slopes (regression) ---
    Vs = [r["n_nodes"] for r in rows]
    ts = [r["t_step_median_ms"] for r in rows]
    Hs = [r["H_mean"] for r in rows]

    slope_t_vs_V = _slope(Vs, ts)
    slope_t_vs_H = _slope(Hs, ts) if len(set(Hs)) > 1 else float("nan")
    print(f"\nLog-log slopes (median t/step):")
    print(f"  vs |V|:  {slope_t_vs_V:.3f}  (target < 0.7 for sublinearity)")
    print(f"  vs |H|:  {slope_t_vs_H:.3f}  (target ≈ 1 for locality)")

    # --- Figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for x, xlabel, ys, ylabel, title, slope, fname in (
            (Vs, "$|V|$ (graph size)", ts, "median time / step (ms)",
             "Per-step time vs total graph size",
             slope_t_vs_V, "fig_time_vs_V.png"),
            (Hs, r"mean $|H|$ (merged subgraph size)", ts, "median time / step (ms)",
             "Per-step time vs merged subgraph size",
             slope_t_vs_H, "fig_time_vs_H.png"),
        ):
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.loglog(x, ys, "o-", color="#316cd6", linewidth=1.6, markersize=8)
            for xi, yi, V in zip(x, ys, Vs):
                ax.annotate(f"{V:,}", (xi, yi),
                            textcoords="offset points", xytext=(7, 5),
                            fontsize=8, color="#666")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(which="both", alpha=0.3)
            ax.text(0.04, 0.92, f"fitted slope = {slope:.2f}",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", edgecolor="lightgray"))
            fig.tight_layout()
            fig.savefig(FIG_DIR / fname, dpi=300)
            plt.close(fig)
            print(f"Wrote {FIG_DIR / fname}")
    except ImportError:
        print("matplotlib not available; skipping figures")

    # --- Index of analysis ---
    idx = {
        "experiment": "scalability",
        "analysed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "rows": rows,
        "slopes": {
            "log_t_vs_log_V": slope_t_vs_V,
            "log_t_vs_log_H": slope_t_vs_H,
        },
    }
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(idx, f, indent=2)
    print(f"Wrote {RESULTS_DIR / 'analysis.json'}")


if __name__ == "__main__":
    main()
