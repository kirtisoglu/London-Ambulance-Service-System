"""
Figures for the real-station pilot.

  python -m falcomchain_experiments.las.plot_pilot_real_stations traces
  python -m falcomchain_experiments.las.plot_pilot_real_stations facility

``traces``   : per-step traces of the three 1,000-step pilot configurations.
``facility`` : opening frequency of the 66 real stations across the pooled
               post-burn-in states of the recording chains, drawn on the map.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "data/derived/pilot_real_stations"
FIG = REPO / "figures/pilot_real_stations"
FIG.mkdir(parents=True, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # validated categorical slots 1-3
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.titlecolor": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def _load(tag):
    return json.loads((PILOT / f"pilot_{tag}.json").read_text())


def traces():
    runs = [
        ("cnt1_M200_s0_T1000", "counting on, M=200 (CDBA start, switched)"),
        ("cnt0_M200_s0_T1000", "counting off, M=200 (CDBA start, switched)"),
        ("cnt1_M1000_s1_T1000", "counting on, M=1000 (real-station start)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (ax_p1, ax_acc), (ax_bad, ax_p2) = axes
    for (tag, label), col in zip(runs, SERIES):
        r = _load(tag)
        rows = r["rows"]
        t = np.array([x["step"] for x in rows])
        p1 = np.array([x["P1"] for x in rows])
        p2 = np.array([x["P2"] for x in rows])
        bad = np.array([x["no_station"] for x in rows])
        acc = np.cumsum([x["accepted"] for x in rows]) / t
        ax_p1.plot(t, p1, color=col, lw=2, label=label)
        ax_p2.plot(t, p2, color=col, lw=2)
        ax_bad.plot(t, bad, color=col, lw=2)
        ax_acc.plot(t, acc, color=col, lw=2)
        ax_acc.annotate(f"{acc[-1]:.2f}", (t[-1], acc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", color=TEXT2, fontsize=9)
    ax_p1.set_title("Number of districts  |P¹|"); ax_p1.set_ylabel("districts")
    ax_p1.axhline(66, color=GRID, lw=1, ls="--")
    ax_p1.text(1000, 66.5, "66 stations", ha="right", va="bottom", color=TEXT2, fontsize=8)
    ax_p2.set_title("Number of superdistricts  |P²|"); ax_p2.set_ylabel("superdistricts")
    ax_p2.set_ylim(0, 60)
    ax_bad.set_title("Districts without a real station"); ax_bad.set_ylabel("districts")
    ax_acc.set_title("Running acceptance rate"); ax_acc.set_ylim(0, 0.6)
    for ax in axes.ravel():
        ax.set_xlabel("step"); ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xlim(0, 1000)
    ax_p1.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle("FalCom on the 66 real LAS stations, rejection mode: 1,000-step pilot runs",
                 color=TEXT, fontsize=12)
    fig.tight_layout()
    out = FIG / "pilot_traces.png"
    fig.savefig(out, dpi=160)
    print(out)


def facility(burn_in=1000):
    import geopandas as gpd
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap

    files = sorted(PILOT.glob("pilot_cnt1_M200_s1*_T5000.json"))
    if not files:
        sys.exit("no recording runs found")
    counts, n_states, chains = None, 0, []
    per_chain = []
    for f in files:
        r = json.loads(f.read_text())
        stations = r["station_nodes"]; sset = set(stations)
        post = [x for x in r["rows"] if x["step"] > burn_in]
        # A state counts only when every district center is a real station.
        # Districts inherited from the CDBA start keep their old candidate
        # list until re-cut, so their 1-median can still be an artificial node.
        rows = [x for x in post if set(x["open"]) <= sset]
        last_bad = max([x["step"] for x in post if not set(x["open"]) <= sset], default=None)
        print(f"seed {r['seed']}: {len(rows)}/{len(post)} post-burn-in states with all centers "
              f"at real stations (last state with an artificial center: step {last_bad})")
        c = pd.Series(0.0, index=stations)
        for x in rows:
            c[x["open"]] += 1
        per_chain.append(c / max(1, len(rows)))
        counts = c if counts is None else counts + c
        n_states += len(rows); chains.append(r["seed"])
        codes = r["station_codes"]
    freq = counts / n_states
    spread = pd.concat(per_chain, axis=1).max(axis=1) - pd.concat(per_chain, axis=1).min(axis=1)

    lsoas = gpd.read_file(REPO / "data/raw/LSOA_2021_London.gpkg")[["LSOA21CD", "geometry"]]
    las = set(pd.read_csv(REPO / "data/derived/lsoa_to_group.csv")["LSOA21CD"])
    lsoas = lsoas[lsoas["LSOA21CD"].isin(las)].to_crs(27700)
    pts = lsoas.set_index("LSOA21CD").geometry.centroid
    df = pd.DataFrame({"code": [codes[str(n)] for n in freq.index],
                       "freq": freq.values, "spread": spread.values})
    df["x"] = [pts[c].x for c in df["code"]]; df["y"] = [pts[c].y for c in df["code"]]

    ramp = LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"])
    fig, ax = plt.subplots(figsize=(11, 8.5))
    lsoas.plot(ax=ax, color="#f1f0ec", edgecolor="#ffffff", linewidth=0.15)
    order = df.sort_values("freq")
    sc = ax.scatter(order["x"], order["y"], c=order["freq"], cmap=ramp, vmin=0, vmax=1,
                    s=140, edgecolor="#ffffff", linewidth=1.2, zorder=3)
    for _, r_ in order.iterrows():
        ax.text(r_["x"], r_["y"], f"{r_['freq']:.2f}", ha="center", va="center",
                fontsize=5.5, color=TEXT if r_["freq"] < 0.6 else "#ffffff", zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("opening frequency across ensemble states", color=TEXT2)
    cb.outline.set_visible(False)
    ax.set_axis_off()
    ax.set_title(
        f"Opening frequency of the 66 real LAS stations\n"
        f"{len(files)} chains × 5,000 steps, burn-in {burn_in}, {n_states:,} pooled states, "
        f"max across-chain spread {spread.max():.2f}",
        color=TEXT, fontsize=12, loc="left")
    fig.tight_layout()
    out = FIG / "pilot_facility_frequency.png"
    fig.savefig(out, dpi=170)
    print(out)
    df.sort_values("freq", ascending=False).to_csv(FIG / "pilot_facility_frequency.csv", index=False)
    print(df.sort_values("freq", ascending=False).to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    {"traces": traces, "facility": facility}[sys.argv[1]]()
