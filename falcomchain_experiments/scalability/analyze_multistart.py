"""
Multi-start diagnostics for the synthetic-grid chains (paper Section 7.2).

Reads, for one grid size, the per-chain records written by::

    PYTHONHASHSEED=0 python3 run_scalability.py --sizes N --steps T --seed S \
        --track-structural --snap-every K --out-dir results/multistart

for several seeds S and computes, on the level-1 access cost R1 and the
per-team demand spread:

* the Kolmogorov--Smirnov distance between every pair of chains
  (post burn-in), the headline agreement statistic;
* split-R-hat and effective sample size (secondary numbers);
* forgetting curves: the share of the initial plan's level-1 boundary
  edges that are still boundary edges after t steps, against the share
  two independent post-burn-in plans have in common (the baseline the
  curve must reach if the chain forgets where it started);
* acceptance rates and rejection causes.

Usage::

    python3 analyze_multistart.py --nodes 10000 --seeds 42 43 44 45
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_convergence import ess, split_rhat

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def ks_statistic(a, b):
    a = np.sort(np.asarray(a, dtype=float)); b = np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def cut_edges(district, edges):
    """Boolean mask over `edges` of the level-1 boundary edges of one snapshot."""
    return district[edges[:, 0]] != district[edges[:, 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--in-dir", default="results/multistart")
    ap.add_argument("--burnin", type=float, default=0.2)
    args = ap.parse_args()
    in_dir = HERE / args.in_dir
    FIG_DIR.mkdir(exist_ok=True)

    runs, snaps = {}, {}
    for s in args.seeds:
        runs[s] = json.load(open(in_dir / f"grid_{args.nodes}_seed{s}.json"))
        p = in_dir / f"snap_grid_{args.nodes}_seed{s}.npz"
        if p.exists():
            snaps[s] = np.load(p)

    out = {"nodes": args.nodes, "seeds": args.seeds, "burnin_fraction": args.burnin,
           "steps": {s: r["n_chain_steps_succeeded"] for s, r in runs.items()},
           "acceptance": {s: 1 - r["n_chain_steps_rejected"] / max(r["n_chain_steps_recorded"], 1)
                          for s, r in runs.items()},
           "rejection_causes": {s: r.get("rejection_report", {}).get("causes") for s, r in runs.items()},
           "median_step_ms": {s: 1000 * float(np.median([x["t_step"] for x in r["steps"] if x["accepted"]]))
                              for s, r in runs.items()}}

    # ---- observables: pairwise KS, split-R-hat, ESS -------------------------
    for key in ("R1", "dspread"):
        seqs = {s: np.array([x[key] for x in r["steps"] if key in x], dtype=float) for s, r in runs.items()}
        L = min(len(v) for v in seqs.values()); b = int(args.burnin * L)
        post = {s: v[:L][b:] for s, v in seqs.items()}
        arr = np.stack([post[s] for s in args.seeds])
        ks = {f"{a}-{c}": ks_statistic(post[a], post[c]) for a, c in itertools.combinations(args.seeds, 2)}
        out[key] = {"means": {s: float(v.mean()) for s, v in post.items()},
                    "pairwise_ks": ks, "max_pairwise_ks": max(ks.values()),
                    "split_rhat": split_rhat(arr), "ess": ess(arr),
                    "post_burnin_length": int(L - b)}

    # ---- forgetting curves ---------------------------------------------------
    forgetting, baseline_pairs = {}, []
    if snaps:
        edges = next(iter(snaps.values()))["edges"]
        masks = {s: [cut_edges(d, edges) for d in S["district"]] for s, S in snaps.items()}
        steps = {s: [int(x) for x in S["step"]] for s, S in snaps.items()}
        for s in snaps:
            m0 = masks[s][0]
            forgetting[s] = [float((m0 & m).sum() / max(m0.sum(), 1)) for m in masks[s]]
        # baseline: share of boundary edges two independent post-burn-in plans share
        T = max(max(v) for v in steps.values()); cut = args.burnin * T
        post_masks = {s: [m for st, m in zip(steps[s], masks[s]) if st >= cut] for s in snaps}
        rng = np.random.default_rng(0)
        for a, c in itertools.combinations(list(snaps), 2):
            for _ in range(200):
                ma = post_masks[a][rng.integers(len(post_masks[a]))]
                mc = post_masks[c][rng.integers(len(post_masks[c]))]
                baseline_pairs.append((ma & mc).sum() / max(ma.sum(), 1))
        base = float(np.mean(baseline_pairs)) if baseline_pairs else float("nan")
        out["forgetting"] = {"steps": steps[args.seeds[0]], "share": forgetting,
                             "independent_baseline": base,
                             "boundary_density": float(np.mean([m.mean() for ms in post_masks.values() for m in ms]))}
        # steps needed to come within 10% (relative) of the baseline
        t_forget = {}
        for s in snaps:
            hit = [st for st, sh in zip(steps[s], forgetting[s]) if sh <= 1.1 * base]
            t_forget[s] = int(hit[0]) if hit else None
        out["forgetting"]["steps_to_baseline"] = t_forget

    # ---- figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3 if snaps else 2, figsize=(13 if snaps else 9, 3.4))
        for s, r in runs.items():
            y = np.array([x["R1"] for x in r["steps"] if "R1" in x]) / 1e6
            axes[0].plot(np.arange(len(y)), y, lw=0.5, label=f"seed {s}")
        axes[0].set_xlabel("step"); axes[0].set_ylabel(r"$R^1$ ($\times 10^6$)"); axes[0].set_title("level-1 access cost")
        axes[0].legend(fontsize=7)
        for s, r in runs.items():
            y = np.array([x["R1"] for x in r["steps"] if "R1" in x]); b = int(args.burnin * len(y))
            axes[1].hist(y[b:] / 1e6, bins=40, histtype="step", density=True, label=f"seed {s}")
        axes[1].set_xlabel(r"$R^1$ ($\times 10^6$), post burn-in"); axes[1].set_title(f"max pairwise KS = {out['R1']['max_pairwise_ks']:.3f}")
        if snaps:
            for s in snaps:
                axes[2].plot(steps[s], forgetting[s], lw=1, label=f"seed {s}")
            axes[2].axhline(out["forgetting"]["independent_baseline"], color="k", ls="--", lw=0.8, label="independent plans")
            axes[2].set_xlabel("step"); axes[2].set_ylabel("share of initial boundary kept"); axes[2].set_title("forgetting the start")
            axes[2].legend(fontsize=7)
        fig.suptitle(f"grid with {args.nodes:,} nodes, {len(runs)} independently started chains")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"fig_multistart_{args.nodes}.png", dpi=300)
    except Exception as exc:  # noqa: BLE001
        out["figure_error"] = str(exc)

    (in_dir / f"multistart_{args.nodes}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "forgetting"}, indent=1, default=str))
    if "forgetting" in out:
        print("forgetting baseline", out["forgetting"]["independent_baseline"],
              "steps_to_baseline", out["forgetting"]["steps_to_baseline"])


if __name__ == "__main__":
    main()
