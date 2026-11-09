"""
Tables and figures of the London case-study section, from stored outputs.

    fig_plan_vs_layout.pdf   the operational layout next to one sampled plan
    fig_station_loads.pdf    station loads of the operational layout against the windows
    fig_layout_vs_ensemble.pdf  station-leg metrics: ensemble, operational layout, best annealed plan
    tab_london_layout.tex    operational layout / ensemble / best sampled / best annealed
    tab_london_sens.tex      baseline, uniform demand proxy, tolerance 0.25
    london_numbers.json      every number the two tables and the text use

    python paper/text_generators/make_sec8_london.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from falcomchain_experiments.las import postprocess_real_stations as pp  # noqa: E402
from falcomchain_experiments.las.calibration import W  # noqa: E402

DER = REPO / "data/derived"
TEX = REPO / "paper/sources/tex"
FIG = REPO / "paper/sources/figures/sec7"
TEX.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})
C = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#4a5568"]


def J(p):
    return json.loads(Path(p).read_text())


def num(x, d=0):
    return f"{x:,.{d}f}".replace(",", "{,}")


def windows(eps):
    """Widest per-capacity load windows a sampled station can have (effective balance)."""
    return {c: ((1 - eps) * (1 - eps * (c + 1) / (2 * c)) * c, (1 + eps) * (1 + eps * (c + 1) / (2 * c)) * c)
            for c in (1, 2, 3)}


class Instance:
    def __init__(self, demand_csv=None):
        S = np.load(DER / "real_stations_u97/snap_real_s1_T80000.npz")
        self.node_ids = [int(x) for x in S["node_ids"]]
        self.node_lsoa = [str(x) for x in S["node_lsoa"]]
        self.station_nodes = [int(x) for x in S["station_node_ids"]]
        self.edges = S["edges"]
        inst = pp.load_instance(self.node_ids, demand_csv)
        self.inst = inst
        self.demand = inst["demand"]
        self.d_bar = self.demand.sum() / np.ceil(self.demand.sum() / W)
        minutes = inst["minutes"]
        # T[s, i]: drive minutes from station s to LSOA i
        self.T = np.array([[minutes(s, n) for n in self.node_ids] for s in self.station_nodes])
        code_of_node = {v: k for k, v in inst["station_node"].items()}
        self.station_code = [code_of_node[s] for s in self.station_nodes]
        self.station_name = [inst["station_name"][c] for c in self.station_code]
        idx_of_code = {c: i for i, c in enumerate(self.station_code)}
        self.slas_station = np.array([idx_of_code[inst["lsoa_station"][l]] for l in self.node_lsoa])
        self.slas_group = np.array([str(inst["lsoa_group"][l]).strip() for l in self.node_lsoa])

    def metrics(self, station_vec):
        t = self.T[station_vec, np.arange(len(self.node_ids))]
        m = pp.ems_metrics(t, self.demand)
        m["E1"] = float((t * self.demand).sum())
        return m

    def loads(self, station_vec):
        return np.bincount(station_vec, weights=self.demand, minlength=len(self.station_nodes)) / self.d_bar


def ensemble_metrics(I, in_dir, steps, burn):
    rows = []
    for f in sorted(Path(in_dir).glob(f"snap_real_s*_T{steps}.npz")):
        S = np.load(f)
        station, step = S["station"], S["step"]          # read once: an npz member is decompressed on every access
        for i in np.where(step >= burn)[0]:
            m = I.metrics(station[i])
            m["n_open"] = int(len(np.unique(station[i])))
            rows.append(m)
    return pd.DataFrame(rows)


def outside(loads, eps):
    w = windows(eps)
    eff = sum(not any(lo <= u <= hi for lo, hi in w.values()) for u in loads)
    nominal = sum(not any(abs(u - c) <= eps * c for c in (1, 2, 3)) for u in loads)
    return int(eff), int(nominal)


# ---------------------------------------------------------------------------

def fig_maps(I):
    import geopandas as gpd
    g = gpd.read_file(REPO / "data/raw/LSOA_2021_London.gpkg").set_index("LSOA21CD").loc[I.node_lsoa]
    S = np.load(DER / "real_stations_u97/snap_real_s1_T80000.npz")
    plan_station, plan_super = S["station"][-1], S["super"][-1]
    xy = np.array([I.inst["xy"][n] for n in I.station_nodes])

    def colouring(label):
        """Greedy colouring of the regions so that neighbours differ."""
        adj = {}
        for u, v in I.edges:
            a, b = label[u], label[v]
            if a != b:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        col = {}
        for r in sorted(set(label.tolist()), key=lambda r: -len(adj.get(r, ()))):
            used = {col[n] for n in adj.get(r, ()) if n in col}
            col[r] = next(c for c in range(12) if c not in used)
        return np.array([col[r] for r in label])

    pal = plt.get_cmap("Set3")(np.linspace(0, 1, 12))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, st, grp, title, open_idx in [
            (axes[0], I.slas_station, I.slas_group, "(a) operational layout", np.arange(66)),
            (axes[1], plan_station, plan_super, "(b) one sampled plan", np.unique(plan_station))]:
        g.assign(c=colouring(st)).plot(ax=ax, color=pal[colouring(st)], linewidth=0, edgecolor="none")
        gg = g.assign(grp=grp).dissolve("grp")
        gg.boundary.plot(ax=ax, color="k", linewidth=0.7)
        closed = np.setdiff1d(np.arange(66), open_idx)
        ax.scatter(xy[open_idx, 0], xy[open_idx, 1], s=7, c="k", zorder=3, linewidths=0)
        if len(closed):
            ax.scatter(xy[closed, 0], xy[closed, 1], s=9, facecolors="white", edgecolors="k", linewidths=0.5, zorder=3)
        ax.set_title(f"{title}: {len(open_idx)} stations, {len(set(grp.tolist()))} Groups", fontsize=8.5)
        ax.set_axis_off()
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color="k", ms=3, label="open station"),
                            Line2D([], [], marker="o", ls="", mfc="white", mec="k", mew=0.5, ms=3.5, label="closed station"),
                            Line2D([], [], color="k", lw=0.8, label="Group boundary")],
                   fontsize=6.5, loc="lower right", bbox_to_anchor=(1.02, -0.06), handletextpad=0.3, labelspacing=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_plan_vs_layout.pdf", dpi=300)
    plt.close(fig)
    return {"plan_open": int(len(np.unique(plan_station))), "plan_groups": int(len(np.unique(plan_super))),
            "plan_step": int(S["step"][-1])}


def fig_loads(I, eps=0.15):
    loads = I.loads(I.slas_station)
    order = np.argsort(loads)
    w = windows(eps)
    inside = np.array([any(lo <= u <= hi for lo, hi in w.values()) for u in loads])
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for c, (lo, hi) in w.items():
        ax.axhspan(lo, hi, color=C[2], alpha=0.18, lw=0)
        ax.text(66.3, (lo + hi) / 2, f"{c} unit" + ("s" if c > 1 else ""), va="center", fontsize=7, color=C[2])
    x = np.arange(66)
    ax.bar(x, loads[order], 0.75, color=[C[0] if inside[i] else C[1] for i in order])
    ax.set_xticks(x)
    ax.set_xticklabels([I.station_name[i] for i in order], rotation=90, fontsize=4.6)
    ax.set_xlim(-0.8, 66)
    ax.set_ylabel("load [capacity units]")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C[0], label="inside the window of some capacity"),
                       Patch(color=C[1], label="outside every window"),
                       Patch(color=C[2], alpha=0.18, label="widest window of a sampled station")],
              fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig_station_loads.pdf")
    plt.close(fig)
    eff, nominal = outside(loads, eps)
    below = int((loads < w[1][0]).sum())
    above = int((loads > w[3][1]).sum())
    return {"range": [float(loads.min()), float(loads.max())], "outside_effective": eff, "outside_nominal": nominal,
            "below_one_unit_window": below, "above_three_unit_window": above, "between_windows": eff - below - above,
            "smallest": [(I.station_name[i], round(float(loads[i]), 2)) for i in order[:4]],
            "largest": [(I.station_name[i], round(float(loads[i]), 2)) for i in order[-2:]]}


def fig_layout(I, ens, slas, ann):
    panels = [("E1", 1e-6, "station-leg energy [million min]"), ("T_mean", 1, "mean drive time [min]"),
              ("T90", 1, "90th-percentile drive time [min]"), ("C8", 100, "demand within 8 min [%]")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=True)
    for ax, (k, sc, lab) in zip(axes, panels):
        ax.hist(ens[k] * sc, bins=40, color=C[0], alpha=0.7)
        ax.axvline(slas[k] * sc, color=C[1], lw=1.6, label="operational layout")
        ax.axvline(ann[k] * sc, color=C[2], lw=1.3, ls="--", label="best annealed plan")
        ax.set_xlabel(lab, fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("sampled plans", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(2.3, 1.22), ncol=2)
    fig.savefig(FIG / "fig_layout_vs_ensemble.pdf")
    plt.close(fig)


def variant(I, name, in_dir, steps, burn, eps):
    e = J(Path(in_dir) / "ensemble_real_stations.json")
    b = J(Path(in_dir) / "best_plans.json")
    ens = ensemble_metrics(I, in_dir, steps, burn)
    slas = I.metrics(I.slas_station)
    loads = I.loads(I.slas_station)
    eff, nominal = outside(loads, eps)
    acc = list(e["acceptance_rate"].values())
    pb = e["post_burn_in"]
    return {"name": name, "eps": eps, "chains": e["n_chains"], "steps": steps, "plans": int(len(ens)),
            "acceptance": [min(acc), max(acc)],
            "ks_E1": pb["E1"]["ks_max"], "ks_P1": pb["P1"]["ks_max"], "ks_cut": pb["cut_edges"]["ks_max"],
            "open_mean": e["stations"]["n_open_per_plan_mean"], "open_range": e["stations"]["n_open_per_plan_range"],
            "open_over_90": e["stations"]["stations_open_in_over_90pct"],
            "groups_mean": e["groups"]["n_superdistricts_mean"], "ari": e["groups"]["ari_vs_real_groups_mean"],
            "E1_mean": float(ens["E1"].mean()), "E1_q": [float(ens["E1"].quantile(.05)), float(ens["E1"].quantile(.95))],
            "E1_min": float(ens["E1"].min()),
            "T_mean": float(ens["T_mean"].mean()), "T_mean_q": [float(ens["T_mean"].quantile(.05)), float(ens["T_mean"].quantile(.95))],
            "T_mean_min": float(ens["T_mean"].min()),
            "T90": float(ens["T90"].mean()), "T90_min": float(ens["T90"].min()),
            "C8": float(ens["C8"].mean()), "C8_q": [float(ens["C8"].quantile(.05)), float(ens["C8"].quantile(.95))],
            "C8_max": float(ens["C8"].max()),
            "slas": slas, "slas_loads": [float(loads.min()), float(loads.max())],
            "slas_outside_effective": eff, "slas_outside_nominal": nominal,
            "slas_below_every_plan": bool(slas["E1"] < ens["E1"].min()),
            "best_plans_E1_min": b["sample_extremes"]["E1_min"]}, ens


def main():
    I = Instance()
    out = {}
    out["maps"] = fig_maps(I)
    out["loads"] = fig_loads(I)
    base, ens = variant(I, "baseline", DER / "real_stations_u97", 80000, 16000, 0.15)
    out["baseline"] = base
    slas = base["slas"]
    opt = J(DER / "real_stations_u97_opt/optimizer_summary_T50000.json")
    best = opt["best"]
    B = np.load(DER / f"real_stations_u97_opt/best_opt_s{best['seed']}_T50000.npz")
    ann = I.metrics(B["station"][0])
    ann["n_open"] = int(len(np.unique(B["station"][0])))
    out["annealed"] = {**ann, "seed": best["seed"], "best_E1_all_seeds": [r["best_E1"] for r in opt["runs"]],
                       "runtime_min": [r["runtime_seconds"] / 60 for r in opt["runs"]]}
    i_best = int(ens["E1"].idxmin())
    bs = ens.loc[i_best].to_dict()
    out["best_sampled"] = bs
    fig_layout(I, ens, slas, ann)

    # table: operational layout against the ensemble (station leg only)
    def row(label, m, n_open, extra=""):
        return (f"{label} & ${n_open}$ & ${m['E1'] / 1e6:.2f}$ & ${m['T_mean']:.2f}$ & ${m['T90']:.2f}$ & "
                f"${100 * m['C8']:.1f}$ {extra}\\\\")
    q = lambda k, s=1: f"(${ens[k].quantile(.05) * s:.2f}$--${ens[k].quantile(.95) * s:.2f}$)"  # noqa: E731
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & open & station-leg energy & mean drive & $90$th pct. & within \\",
             r"plan & stations & [million min] & time [min] & [min] & $8$ min [\%] \\", r"\midrule",
             row("operational layout $s_{\\mathrm{LAS}}$", slas, 66),
             r"\midrule",
             f"ensemble, mean & ${ens['n_open'].mean():.1f}$ & ${ens['E1'].mean() / 1e6:.2f}$ & ${ens['T_mean'].mean():.2f}$ & "
             f"${ens['T90'].mean():.2f}$ & ${100 * ens['C8'].mean():.1f}$ \\\\",
             f"\\quad central $90\\%$ & ${ens['n_open'].quantile(.05):.0f}$--${ens['n_open'].quantile(.95):.0f}$ & "
             f"${ens['E1'].quantile(.05) / 1e6:.2f}$--${ens['E1'].quantile(.95) / 1e6:.2f}$ & "
             f"${ens['T_mean'].quantile(.05):.2f}$--${ens['T_mean'].quantile(.95):.2f}$ & "
             f"${ens['T90'].quantile(.05):.2f}$--${ens['T90'].quantile(.95):.2f}$ & "
             f"${100 * ens['C8'].quantile(.05):.1f}$--${100 * ens['C8'].quantile(.95):.1f}$ \\\\",
             row("best sampled plan", bs, int(bs["n_open"])),
             row("best annealed plan", ann, ann["n_open"]),
             r"\midrule",
             f"bound: annealed $-$ layout & & ${(ann['E1'] - slas['E1']) / 1e6:.2f}$ & ${ann['T_mean'] - slas['T_mean']:.2f}$ & "
             f"${ann['T90'] - slas['T90']:.2f}$ & ${100 * (ann['C8'] - slas['C8']):.1f}$ \\\\",
             r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_london_layout.tex").write_text("\n".join(lines) + "\n")

    # sensitivity table
    uni, _ = variant(Instance(str(DER / "demand_lsoa_uniform.csv")), "uniform", DER / "real_stations_u97_uniform", 80000, 16000, 0.15)
    e25, _ = variant(I, "eps25", DER / "real_stations_u97_eps25", 40000, 8000, 0.25)
    out["uniform"], out["eps25"] = uni, e25
    V = [base, uni, e25]

    def line(label, f):
        return label + " & " + " & ".join(f(v) for v in V) + r" \\"
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r" & baseline & uniform demand proxy & $\varepsilon^1=\varepsilon^2=0.25$ \\", r"\midrule",
             line("chains $\\times$ steps", lambda v: f"${v['chains']}\\times{num(v['steps'])}$"),
             line("acceptance", lambda v: f"{100 * v['acceptance'][0]:.1f}--{100 * v['acceptance'][1]:.1f}\\%"),
             line("largest KS between chains (access cost, $|\\Pc^1|$, cut edges)",
                  lambda v: f"${v['ks_E1']:.3f}$, ${v['ks_P1']:.3f}$, ${v['ks_cut']:.3f}$"),
             r"\midrule",
             line("open stations, mean (range)", lambda v: f"${v['open_mean']:.1f}$ (${v['open_range'][0]}$--${v['open_range'][1]}$)"),
             line("stations open in more than $90\\%$ of plans", lambda v: f"${v['open_over_90']}$"),
             line("Groups per plan, mean", lambda v: f"${v['groups_mean']:.1f}$"),
             line("adjusted Rand index against the real Groups", lambda v: f"${v['ari']:.2f}$"),
             r"\midrule",
             line("station-leg energy, ensemble mean [million min]", lambda v: f"${v['E1_mean'] / 1e6:.2f}$"),
             line("\\quad lowest stored plan", lambda v: f"${v['E1_min'] / 1e6:.2f}$"),
             line("\\quad operational layout", lambda v: f"${v['slas']['E1'] / 1e6:.2f}$"),
             line("mean drive time, ensemble / layout [min]", lambda v: f"${v['T_mean']:.1f}$ / ${v['slas']['T_mean']:.1f}$"),
             line("demand within $8$ min, ensemble / layout", lambda v: f"${100 * v['C8']:.1f}\\%$ / ${100 * v['slas']['C8']:.1f}\\%$"),
             r"\midrule",
             line("layout stations outside every window (of $66$)", lambda v: f"${v['slas_outside_effective']}$"),
             line("\\quad outside the nominal $\\pm\\varepsilon$ windows", lambda v: f"${v['slas_outside_nominal']}$"),
             line("layout below every sampled plan", lambda v: "yes" if v["slas_below_every_plan"] else "no"),
             r"\bottomrule", r"\end{tabular}"]
    (TEX / "tab_london_sens.tex").write_text("\n".join(lines) + "\n")
    (TEX / "london_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k in ("maps", "loads", "annealed", "best_sampled")}, indent=1, default=float))
    for k in ("baseline", "uniform", "eps25"):
        v = out[k]
        print(k, {kk: v[kk] for kk in ("acceptance", "open_mean", "E1_mean", "E1_min", "T_mean", "C8", "slas_outside_effective",
                                       "slas_outside_nominal", "slas_below_every_plan", "best_plans_E1_min")}, v["slas"])


if __name__ == "__main__":
    main()
