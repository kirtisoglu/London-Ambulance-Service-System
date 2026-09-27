"""
Convergence diagnostics for the multi-chain FalCom run (paper Experiment 1).

Loads the per-chain records produced by

    run_scalability.py --sizes 10000 --steps 50000 --seed S \
        --track-structural --out-dir results/convergence

for several seeds S, and computes, for the structural observables native to
the Section-4 state space:

  * R1      -- level-1 demand-weighted access cost (base term of the median
               objective), the slow-mode observable;
  * dspread -- per-team demand spread max_D d(D)/c(D) - min_D ... .

Diagnostics (no external deps beyond numpy):

  * split-R-hat (Gelman et al.; split each chain in half) -- target < 1.01;
  * effective sample size (Stan/Vehtari combined-autocorrelation method with
    Geyer's initial-positive-sequence truncation).

The full chain trajectory is used (rejected steps are stay-put repeats and
are part of the lazy chain, so they are kept -- they correctly inflate
autocorrelation). A burn-in fraction is discarded first.

Usage::

    python3 analyze_convergence.py                 # grid_10000, seeds 42-45
    python3 analyze_convergence.py --selftest       # validate the estimators
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONV_DIR = HERE / "results" / "convergence"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat (Gelman et al. 2013). chains: shape (m, n)."""
    m, n = chains.shape
    if n % 2:
        chains = chains[:, :-1]
        n -= 1
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    M, N = split.shape
    means = split.mean(axis=1)
    grand = means.mean()
    B = N / (M - 1) * np.sum((means - grand) ** 2)
    W = np.mean(split.var(axis=1, ddof=1))
    if W <= 0:
        return float("nan")
    var_plus = (N - 1) / N * W + B / N
    return float(np.sqrt(var_plus / W))


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance (divide by n) at all lags via FFT. x: 1D."""
    n = len(x)
    x = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(f * np.conj(f), n=size)[:n].real / n
    return acov


def ess(chains: np.ndarray) -> float:
    """Effective sample size (Stan/Vehtari combined autocorrelation)."""
    m, n = chains.shape
    if n < 4:
        return float("nan")
    means = chains.mean(axis=1)
    W = np.mean(chains.var(axis=1, ddof=1))
    B = n / (m - 1) * np.sum((means - means.mean()) ** 2) if m > 1 else 0.0
    var_plus = (n - 1) / n * W + B / n
    if var_plus <= 0:
        return float(m * n)
    # mean over chains of the (biased) autocovariance at each lag
    acov = np.mean([_autocov(c) for c in chains], axis=0)
    rho = 1.0 - (W - acov) / var_plus           # combined autocorrelation
    rho[0] = 1.0
    # Geyer initial positive sequence: sum paired autocorrelations until <=0
    tau = 1.0
    t = 1
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p <= 0:
            break
        tau += 2.0 * p
        t += 2
    return float(m * n / tau)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    # i.i.d.: R-hat ~ 1, ESS ~ n
    x = rng.standard_normal((4, 20000))
    print(f"iid     : R-hat={split_rhat(x):.4f}  ESS={ess(x):.0f}  (n*m=80000)")
    # AR(1) phi=0.9: ESS ~ n*(1-phi)/(1+phi) = n/19
    phi = 0.9
    a = np.zeros((4, 20000))
    a[:, 0] = rng.standard_normal(4)
    for t in range(1, a.shape[1]):
        a[:, t] = phi * a[:, t - 1] + rng.standard_normal(4)
    print(f"AR(1).9 : R-hat={split_rhat(a):.4f}  ESS={ess(a):.0f}  "
          f"(expect ~{80000*(1-phi)/(1+phi):.0f})")
    # non-converged: 4 chains with different means
    b = rng.standard_normal((4, 20000)) + np.array([[0], [3], [6], [9]])
    print(f"offset  : R-hat={split_rhat(b):.3f}  (expect >> 1)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_chains(n_nodes, seeds, key, burnin):
    seqs = []
    for s in seeds:
        p = CONV_DIR / f"grid_{n_nodes}_seed{s}.json"
        d = json.load(open(p))
        vals = [r[key] for r in d["steps"] if key in r]
        seqs.append(np.asarray(vals, dtype=float))
    L = min(len(s) for s in seqs)
    arr = np.stack([s[:L] for s in seqs])         # (m, L)
    b = int(burnin * L)
    return arr[:, b:], L, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    ap.add_argument("--burnin", type=float, default=0.2)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    print(f"=== Convergence: grid_{args.nodes}, {len(args.seeds)} chains, "
          f"burn-in {args.burnin:.0%} ===")
    results = {}
    for key, label in (("R1", "R1 (level-1 access cost)"),
                       ("dspread", "per-team demand spread")):
        arr, L, b = load_chains(args.nodes, args.seeds, key, args.burnin)
        m, n = arr.shape
        rh = split_rhat(arr)
        e = ess(arr)
        per_chain_mean = arr.mean(axis=1)
        results[key] = {
            "n_per_chain_total": L, "burnin": b, "n_used": n,
            "rhat": rh, "ess": e, "ess_per_chain": e / m,
            "grand_mean": float(arr.mean()),
            "per_chain_means": per_chain_mean.tolist(),
        }
        print(f"\n{label}:")
        print(f"  draws/chain used = {n}  (of {L}, burn-in {b})")
        print(f"  split R-hat = {rh:.4f}   (target < 1.01)")
        print(f"  ESS = {e:.0f}  ({e/m:.0f} per chain)")
        print(f"  per-chain means = "
              + ", ".join(f"{x:,.0f}" for x in per_chain_mean))

    json.dump(results, open(CONV_DIR / "convergence.json", "w"), indent=2)
    print(f"\nWrote {CONV_DIR/'convergence.json'}")

    # --- figures: separate trace and running-mean panels for R1 ---
    # (separate files so the paper can lay grids out as subfigures)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, L, b = load_chains(args.nodes, args.seeds, "R1", args.burnin)
        full = np.stack([np.asarray(
            [r["R1"] for r in json.load(open(CONV_DIR / f"grid_{args.nodes}_seed{s}.json"))["steps"]
             if "R1" in r], dtype=float)[:L] for s in args.seeds])
        N = args.nodes

        fig, ax = plt.subplots(figsize=(5.6, 3.7))
        for j, s in enumerate(args.seeds):
            ax.plot(full[j], lw=0.4, alpha=0.7)
        ax.axvline(b, color="k", ls=":", lw=0.9)
        ax.set_title(f"grid\\_{N}: $R^1$ trace (4 chains)")
        ax.set_xlabel("step"); ax.set_ylabel("$R^1$ access cost")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"conv_{N}_trace.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.6, 3.7))
        for j, s in enumerate(args.seeds):
            rm = np.cumsum(full[j]) / np.arange(1, L + 1)
            ax.plot(rm, lw=1.2, label=f"seed {s}")
        ax.set_title(f"grid\\_{N}: $R^1$ running mean")
        ax.set_xlabel("step"); ax.set_ylabel("running mean $R^1$")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"conv_{N}_runmean.png", dpi=150); plt.close(fig)
        print(f"Wrote conv_{N}_trace.png and conv_{N}_runmean.png")
    except ImportError:
        print("matplotlib unavailable; skipping figure")


if __name__ == "__main__":
    main()
