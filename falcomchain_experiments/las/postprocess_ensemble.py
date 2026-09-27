"""
Post-process the LAS ensemble + optimizer runs into the paper's numbers
and figures.

Inputs (data/derived/chain_v3/):
  chain_v3_sample_s{1..4}_T10000.json            diagnostics + final states
  chain_v3_snapshots_sample_s{1..4}_T10000.json  thinned assignment snapshots
  chain_v3_opt_b0.05_T10000.json                 optimizer run (best_state)

Outputs:
  ensemble_results.json    all headline numbers
  fig_convergence.pdf/png  energy traces + post-burn-in densities
  fig_boundary_freq.png    L1/L2 cut-frequency maps on the dual graph
  fig_plans.png            best optimizer plan vs s_LAS (two panels)
  las_results_review.html  browser gallery of the above + numbers

E(s_LAS) is computed with the same objective the chains logged
(falcomchain.markovchain.energy.compute_energy convention: both levels
demand-weighted; no artificial-candidate penalty).

Run:
    python -m falcomchain_experiments.las.postprocess_ensemble
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "data/derived/chain_v3"
GRAPH = REPO / "data/raw/london_graph.json"
STATIONS = REPO / "data/raw/LAS_stations.csv"
L2CSV = REPO / "data/raw/LAS_L2_facilities.csv"
LSOA_STATION = REPO / "data/derived/lsoa_to_station.csv"
LSOA_GROUP = REPO / "data/derived/lsoa_to_group.csv"
TRAVEL = REPO / "data/derived/cdba_travel_times_w10887_eps15_cmin1_dmax12.parquet"

SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
SEEDS = [1, 2, 3, 4]
T = 10_000
BURNIN = 1_000


def load_travel():
    df = pd.read_parquet(TRAVEL)
    return {(o, d): s / 60.0 for o, d, s in
            zip(df["origin"], df["dest"], df["seconds"])}


def main() -> int:
    graph = json.loads(GRAPH.read_text())
    node_info = {n["id"]: n for n in graph["nodes"]}
    code_of = {n["id"]: n["LSOA21CD"] for n in graph["nodes"]}
    id_of = {v: k for k, v in code_of.items()}
    demand = {n["id"]: n["demand"] for n in graph["nodes"]}

    las_codes = set(pd.read_csv(LSOA_GROUP)["LSOA21CD"])
    las_nodes = [n for n in code_of if code_of[n] in las_codes]
    las_set = set(las_nodes)
    edges = [(a["id"] if False else src, nb["id"])
             for src, adj in ((n["id"], graph["adjacency"][i])
                              for i, n in enumerate(graph["nodes"]))
             for nb in adj if src < nb["id"]]
    edges = [(u, v) for u, v in edges if u in las_set and v in las_set]

    tt = load_travel()

    def t_min(center_node, node):
        v = tt.get((code_of[center_node], code_of[node]))
        if v is None:
            v = tt.get((code_of[node], code_of[center_node]))
        return v if v is not None else 45.0  # unreachable fallback

    # ---------- E(s_LAS) under the implemented objective ----------
    st_map = pd.read_csv(LSOA_STATION)
    gr_map = pd.read_csv(LSOA_GROUP)
    stations = pd.read_csv(STATIONS)
    station_node = {r["station_code"]: id_of[r["LSOA21CD"]]
                    for _, r in stations.iterrows() if r["LSOA21CD"] in id_of}
    l2df = pd.read_csv(L2CSV)
    l2df = l2df[l2df["facility_type"] != "eoc"]

    lsoa_station = dict(zip(st_map["LSOA21CD"], st_map["station_code"]))
    lsoa_group = dict(zip(gr_map["LSOA21CD"], gr_map["group"]))
    # Group -> HQ node: facility_id is "GH-<group name>"
    hq_by_name = {}
    for _, r in l2df.iterrows():
        if r["LSOA21CD"] in id_of:
            gname = str(r["facility_id"]).removeprefix("GH-").strip()
            hq_by_name[gname] = id_of[r["LSOA21CD"]]

    E_las_l1 = E_las_l2 = 0.0
    slas_t = []  # (t_minutes, demand) for EMS metrics, L1 leg
    missing_hq = set()
    for n in las_nodes:
        code = code_of[n]
        sc = lsoa_station.get(code)
        f1 = station_node.get(sc)
        grp = lsoa_group.get(code)
        f2 = hq_by_name.get(str(grp).strip())
        d = demand[n]
        tv = t_min(f1, n)
        E_las_l1 += d * tv
        slas_t.append((tv, d))
        if f2 is None:
            missing_hq.add(grp)
            f2 = f1
        E_las_l2 += d * t_min(f2, n)
    E_las = E_las_l1 + E_las_l2
    if missing_hq:
        print(f"NOTE: no HQ node matched for groups {sorted(missing_hq)}; "
              f"used station as fallback for those nodes")

    def ems(pairs):
        pairs = sorted(pairs)
        tot = sum(d for _, d in pairs)
        tmean = sum(t * d for t, d in pairs) / tot
        acc = 0.0
        t90 = pairs[-1][0]
        for t_, d in pairs:
            acc += d
            if acc >= 0.9 * tot:
                t90 = t_
                break
        c8 = sum(d for t_, d in pairs if t_ <= 8.0) / tot
        c15 = sum(d for t_, d in pairs if t_ <= 15.0) / tot
        return {"T_mean": tmean, "T90": t90, "C8": c8, "C15": c15}
    ems_slas = ems(slas_t)

    # ---------- best state across all search runs + gap ----------
    # The fair comparator is the matched-count optimum (at most 66 open
    # sites, the number of real stations); the unconstrained optimum is
    # confounded by site count and reported only as context.
    best = None
    best_unc = None
    for p in sorted(DIR.glob(f"chain_v3_opt_*_T{T}.json")):
        d_ = json.loads(p.read_text())
        c66 = d_.get("best_state_66")
        if c66 and (best is None or c66["E"] < best["E"]):
            best = c66
            print(f"s*_66 candidate from {p.name}: E = {c66['E']:,.0f}")
        cu = d_.get("best_state")
        if cu and (best_unc is None or cu["E"] < best_unc["E"]):
            best_unc = cu
    E_star = best["E"]
    E_star_unconstrained = best_unc["E"] if best_unc else None
    gap = E_las - E_star
    # EMS for best plan
    best_assign = {int(k): int(v) for k, v in best["l1_assignment"].items()}
    best_centers = {int(k): v for k, v in best["centers"].items()}
    best_t = [(t_min(best_centers[dist], n), demand[n])
              for n, dist in best_assign.items()
              if best_centers.get(dist) is not None]
    ems_best = ems(best_t)

    # ---------- ensemble snapshots ----------
    snaps_all = []   # (chain_idx, snap)
    for ci, seed in enumerate(SEEDS):
        blob = json.loads(
            (DIR / f"chain_v3_snapshots_sample_s{seed}_T{T}.json").read_text())
        for s in blob["snapshots"]:
            if s["step"] > BURNIN:
                snaps_all.append((ci, s))
    n_snap = len(snaps_all)
    print(f"{n_snap} post-burn-in snapshots across {len(SEEDS)} chains")

    # boundary frequencies
    cut1 = np.zeros(len(edges))
    cut2 = np.zeros(len(edges))
    # facility usage
    station_nodes = set(station_node.values())
    real_usage = {n: 0 for n in station_nodes}
    art_center_count = 0
    center_count = 0
    ems_rows = []
    for ci, s in snaps_all:
        a = {int(k): int(v) for k, v in s["l1_assignment"].items()}
        sup = {int(k): int(v) for k, v in s["super_assignment"].items()}
        cen = {int(k): v for k, v in s["centers"].items()}
        for j, (u, v) in enumerate(edges):
            du, dv = a.get(u), a.get(v)
            if du != dv:
                cut1[j] += 1
                if sup.get(du) != sup.get(dv):
                    cut2[j] += 1
        for dist, cnode in cen.items():
            if cnode is None:
                continue
            center_count += 1
            if cnode in real_usage:
                real_usage[cnode] += 1
            else:
                art_center_count += 1
        pairs = [(t_min(cen[dist], n), demand[n])
                 for n, dist in a.items() if cen.get(dist) is not None]
        ems_rows.append(ems(pairs))
    cut1 /= n_snap
    cut2 /= n_snap

    def band(x):
        return {"robust": float((x > 0.9).mean()),
                "contested": float(((x >= 0.1) & (x <= 0.9)).mean()),
                "fixed": float((x < 0.1).mean())}
    b1, b2 = band(cut1), band(cut2)

    usage_frac = {n: c / n_snap for n, c in real_usage.items()}
    n_essential = sum(1 for f in usage_frac.values() if f > 0.9)
    n_regular = sum(1 for f in usage_frac.values() if 0.1 <= f <= 0.9)
    n_rare = sum(1 for f in usage_frac.values() if f < 0.1)
    real_center_share = sum(real_usage.values()) / max(1, center_count)

    ems_df = pd.DataFrame(ems_rows)
    ems_ens = {k: {"mean": float(ems_df[k].mean()),
                   "lo": float(ems_df[k].quantile(0.025)),
                   "hi": float(ems_df[k].quantile(0.975))}
               for k in ems_df.columns}

    # Rhat on E from the run diagnostics
    runs = [json.loads((DIR / f"chain_v3_sample_s{s}_T{T}.json").read_text())
            for s in SEEDS]
    post = [[r["E"] for r in d["diagnostics"] if r["step"] > BURNIN]
            for d in runs]
    half = min(len(c) for c in post) // 2
    seqs = []
    for c in post:
        seqs.append(c[:half]); seqs.append(c[half:2 * half])
    m, n = len(seqs), half
    means = [sum(s) / n for s in seqs]
    grand = sum(means) / m
    B = n / (m - 1) * sum((mu - grand) ** 2 for mu in means)
    Wv = sum(sum((x - mu) ** 2 for x in s) / (n - 1)
             for s, mu in zip(seqs, means)) / m
    rhat = math.sqrt(((n - 1) / n * Wv + B / n) / Wv)

    results = {
        "E_las": E_las, "E_las_l1": E_las_l1, "E_las_l2": E_las_l2,
        "E_star": E_star, "E_star_step": best["step"],
        "gap_abs": gap, "gap_rel_to_star": gap / E_star,
        "rhat_E": rhat,
        "acceptance": [d["acceptance_rate"] for d in runs],
        "boundary_l1": b1, "boundary_l2": b2, "n_edges": len(edges),
        "stations_essential_gt90": n_essential,
        "stations_intermittent": n_regular,
        "stations_rare_lt10": n_rare,
        "real_center_share": real_center_share,
        "ems_ensemble": ems_ens, "ems_slas": ems_slas, "ems_best": ems_best,
        "n_snapshots": n_snap,
    }
    (DIR / "ensemble_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # ---------- figures ----------
    # 1. convergence panel
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4),
                             gridspec_kw={"width_ratios": [1.9, 1]})
    ax = axes[0]
    for ci, d in enumerate(runs):
        xs = [r["step"] for r in d["diagnostics"]]
        ys = [r["E"] / 1e6 for r in d["diagnostics"]]
        ax.plot(xs, ys, color=SERIES[ci], lw=1.4, label=f"chain {SEEDS[ci]}")
    ax.axvspan(0, BURNIN, color="#e1e0d9", alpha=0.5, lw=0)
    ax.text(BURNIN * 0.5, ax.get_ylim()[1], "burn-in", fontsize=8,
            color="#898781", ha="center", va="top")
    ax.set_xlabel("step"); ax.set_ylabel("E(s) (millions)")
    ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    allp = [e / 1e6 for c in post for e in c]
    bins = np.linspace(min(allp), max(allp), 16)
    for ci, c in enumerate(post):
        hist, be = np.histogram([e / 1e6 for e in c], bins=bins)
        ax.plot((be[:-1] + be[1:]) / 2, hist, color=SERIES[ci], lw=1.4)
    ax.set_xlabel("E(s) post burn-in (millions)"); ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DIR / "fig_convergence.pdf")
    fig.savefig(DIR / "fig_convergence.png", dpi=180)
    plt.close(fig)

    # 2. boundary-frequency maps (edge segments between centroids).
    # Warm single-hue ramp with a raised floor so mid frequencies stay
    # visible at thin line widths; faint gray backdrop of all edges for
    # geographic context.
    RAMP = plt.cm.YlOrRd
    def coords(n):
        return node_info[n]["LONG"], node_info[n]["LAT"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, cut, ttl in ((axes[0], cut1, "level-1 district boundaries"),
                         (axes[1], cut2, "level-2 super-district boundaries")):
        for u, v in edges:
            x1, y1 = coords(u); x2, y2 = coords(v)
            ax.plot([x1, x2], [y1, y2], color="#d9d8d2", lw=0.3, zorder=1)
        order = np.argsort(cut)
        for j in order:
            u, v = edges[j]
            f = cut[j]
            if f < 0.05:
                continue
            x1, y1 = coords(u); x2, y2 = coords(v)
            ax.plot([x1, x2], [y1, y2],
                    color=RAMP(0.35 + 0.65 * f),
                    lw=0.7 + 2.0 * f, alpha=1.0,
                    solid_capstyle="round", zorder=2 + f)
        ax.set_title(ttl, fontsize=11)
        ax.set_aspect(1.6); ax.axis("off")
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "cutfreq", [RAMP(0.35 + 0.65 * x) for x in np.linspace(0, 1, 64)]),
        norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("cut frequency across ensemble", fontsize=10)
    fig.savefig(DIR / "fig_boundary_freq.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. best plan vs s_LAS (polygon maps)
    try:
        import geopandas as gpd
        gdf = gpd.read_file(REPO / "data/raw/LSOA_2021_London.gpkg")[
            ["LSOA21CD", "geometry"]]
        QUAL = SERIES + ["#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
                         "#67a9cf", "#7fbc41", "#c98500", "#8073ac",
                         "#d6604d", "#35978f", "#bf812d", "#de77ae",
                         "#4393c3", "#5aae61", "#9970ab", "#f46d43"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        # panel A: optimizer best plan, colored by super-district
        bsup = {int(k): int(v) for k, v in best["super_assignment"].items()}
        rows = [{"LSOA21CD": code_of[n], "super": bsup.get(d, -1),
                 "district": d} for n, d in best_assign.items()]
        gg = gdf.merge(pd.DataFrame(rows), on="LSOA21CD")
        supers = sorted(gg["super"].unique())
        cmap = {s: QUAL[i % len(QUAL)] for i, s in enumerate(supers)}
        gg["color"] = gg["super"].map(cmap)
        gg.plot(ax=axes[0], color=gg["color"], edgecolor="none")
        gg.dissolve("district").boundary.plot(ax=axes[0], color="white",
                                              linewidth=0.4)
        gg.dissolve("super").boundary.plot(ax=axes[0], color="#333333",
                                           linewidth=1.1)
        axes[0].set_title(
            f"matched-count optimum s*_66  E = {E_star/1e6:.2f}M  "
            f"(|P1| = {len(set(best_assign.values()))}, "
            f"|P2| = {len(supers)})", fontsize=10)
        # panel B: s_LAS
        rows = [{"LSOA21CD": c, "grp": g_} for c, g_ in lsoa_group.items()]
        gg2 = gdf.merge(pd.DataFrame(rows), on="LSOA21CD")
        grps = sorted(gg2["grp"].unique())
        cmap2 = {g_: QUAL[i % len(QUAL)] for i, g_ in enumerate(grps)}
        gg2["color"] = gg2["grp"].map(cmap2)
        gg2.plot(ax=axes[1], color=gg2["color"], edgecolor="none")
        gg2.dissolve("grp").boundary.plot(ax=axes[1], color="#333333",
                                          linewidth=1.1)
        axes[1].set_title(
            f"s_LAS (21 Groups)  E = {E_las/1e6:.2f}M", fontsize=10)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(DIR / "fig_plans.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"plan map failed: {e}")

    # 4. review gallery
    html = f"""<title>LAS results review</title>
<style>body{{font-family:system-ui;max-width:1200px;margin:24px auto;
background:#f9f9f7;color:#0b0b0b;padding:0 16px}}
img{{max-width:100%;border:1px solid rgba(11,11,11,.1);border-radius:8px;
background:#fff}}
table{{border-collapse:collapse;font-size:14px}}
td,th{{border:1px solid #e1e0d9;padding:6px 12px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}</style>
<h1>LAS ensemble + optimizer results</h1>
<table>
<tr><th></th><th>E (M)</th><th>T_mean (min)</th><th>T90 (min)</th>
<th>C8</th><th>C15</th></tr>
<tr><td>ensemble mean (95% band)</td><td>&ndash;</td>
<td>{ems_ens['T_mean']['mean']:.2f}
 ({ems_ens['T_mean']['lo']:.2f}&ndash;{ems_ens['T_mean']['hi']:.2f})</td>
<td>{ems_ens['T90']['mean']:.2f}</td>
<td>{ems_ens['C8']['mean']:.1%}</td><td>{ems_ens['C15']['mean']:.1%}</td></tr>
<tr><td>matched-count optimum s*_66 (&le; 66 sites)</td>
<td>{E_star/1e6:.3f}</td>
<td>{ems_best['T_mean']:.2f}</td><td>{ems_best['T90']:.2f}</td>
<td>{ems_best['C8']:.1%}</td><td>{ems_best['C15']:.1%}</td></tr>
<tr><td>s_LAS</td><td>{E_las/1e6:.3f}</td>
<td>{ems_slas['T_mean']:.2f}</td><td>{ems_slas['T90']:.2f}</td>
<td>{ems_slas['C8']:.1%}</td><td>{ems_slas['C15']:.1%}</td></tr>
</table>
<p><b>Energy gap</b> E(s_LAS) &minus; E(s*) = {gap/1e6:.3f}M
 ({gap/E_star:.1%} of E(s*)) &middot; R&#770;(E) = {rhat:.3f} &middot;
 boundary edges L1: {b1['robust']:.1%} robust / {b1['contested']:.1%}
 contested / {b1['fixed']:.1%} fixed &middot;
 stations used &gt;90%: {n_essential}, 10&ndash;90%: {n_regular},
 &lt;10%: {n_rare} &middot; real-station center share:
 {real_center_share:.1%}</p>
<h2>Convergence</h2><img src="fig_convergence.png">
<h2>Boundary frequency</h2><img src="fig_boundary_freq.png">
<h2>Optimizer plan vs s_LAS</h2><img src="fig_plans.png">
"""
    (DIR / "las_results_review.html").write_text(html)
    print(f"wrote {DIR / 'las_results_review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
