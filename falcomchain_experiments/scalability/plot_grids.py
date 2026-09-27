"""
Render the synthetic scalability grids as paper figures.

For each grid in `data/grid_{N}.json`, produces:
- `fig_grid_{N}_demand.png` — demand heatmap with candidate markers
- `fig_grid_{N}_panel.png` — compact panel for the paper

Usage::

    python3 plot_grids.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_grid(path: Path):
    with open(path) as f:
        d = json.load(f)
    rows = d["graph"]["rows"]
    cols = d["graph"]["cols"]
    demand = np.zeros((rows, cols), dtype=np.int32)
    cand_x, cand_y = [], []
    super_x, super_y = [], []
    for n in d["nodes"]:
        x, y = n["C_X"], n["C_Y"]
        demand[y, x] = n["demand"]
        if n.get("super_candidate"):
            super_x.append(x); super_y.append(y)
        elif n.get("candidate"):
            cand_x.append(x); cand_y.append(y)
    return rows, cols, demand, (cand_x, cand_y), (super_x, super_y)


def plot_one(n: int):
    path = DATA_DIR / f"grid_{n}.json"
    rows, cols, demand, cand, supercand = load_grid(path)
    aspect = cols / rows
    fig_w = max(5.0, 5.0 * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))

    im = ax.imshow(demand, origin="lower", cmap="viridis",
                   vmin=80, vmax=120, interpolation="nearest")
    # Marker sizes scaled so the 50k panel doesn't drown in dots.
    base_ms = max(2.0, 60.0 / np.sqrt(n))
    ax.plot(cand[0], cand[1], "o", ms=base_ms, mfc="white", mec="white",
            alpha=0.55, lw=0)
    ax.plot(supercand[0], supercand[1], "s", ms=base_ms * 2.4,
            mfc="red", mec="black", alpha=0.95, lw=0.3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        rf"$|V| = {n:,}$ — ${rows} \times {cols}$ grid, "
        rf"${len(cand[0]) + len(supercand[0]):,}$ candidates"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("demand $w_v$")
    fig.tight_layout()
    out = FIG_DIR / f"fig_grid_{n}_demand.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return rows, cols, len(cand[0]) + len(supercand[0])


def plot_panel(sizes):
    """Side-by-side panel of all three grids (paper-ready)."""
    fig, axes = plt.subplots(1, len(sizes),
                             figsize=(5.0 * len(sizes), 5.0),
                             gridspec_kw=dict(wspace=0.1))
    if len(sizes) == 1:
        axes = [axes]
    for ax, n in zip(axes, sizes):
        rows, cols, demand, cand, supercand = load_grid(
            DATA_DIR / f"grid_{n}.json"
        )
        im = ax.imshow(demand, origin="lower", cmap="viridis",
                       vmin=80, vmax=120, interpolation="nearest",
                       aspect="equal")
        base_ms = max(1.5, 50.0 / np.sqrt(n))
        ax.plot(cand[0], cand[1], "o", ms=base_ms, mfc="white",
                mec="white", alpha=0.5, lw=0)
        ax.plot(supercand[0], supercand[1], "s", ms=base_ms * 2.2,
                mfc="red", mec="black", alpha=0.95, lw=0.3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            rf"$|V| = {n:,}$" + "\n" + rf"${rows} \times {cols}$"
        )
    fig.suptitle(
        "Synthetic scalability instances: demand heatmap with candidates "
        "(white) and super-candidates (red)",
        y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_grid_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    sizes = [10_000, 20_000, 50_000]
    for n in sizes:
        plot_one(n)
    plot_panel(sizes)


if __name__ == "__main__":
    main()
