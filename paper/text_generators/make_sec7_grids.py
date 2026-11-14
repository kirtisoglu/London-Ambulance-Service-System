"""
Tables and figures of the restructured experiments section (synthetic part).

Reads only stored outputs and writes LaTeX tables to ``paper/sources/tex/``
and figures to ``paper/sources/figures/sec7/``:

    tab_enum.tex        exact enumeration, one row per instance
    tab_stability.tex   multi-start agreement: grid_10000, grid_50000, London
    tab_rejections.tex  rejected proposals by cause, every instance
    tab_scal.tex        per-step cost on the five grids
    tab_fixed.tex       fixed graph size, growing team count
    tab_ablation.tex    component ablations (grid_1000, grid_10000, London)
    tab_budget.tex      sensitivity of the readouts to the retry budget M
    fig_forgetting.pdf  share of the initial boundary still in place
    fig_placement.pdf   sampled objective values against the exact optimum

    python paper/text_generators/make_sec7_grids.py
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

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "falcomchain_experiments"
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
    """1234.5 -> 1{,}234.5 (LaTeX thousands separator)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "{,}")


def rng(a, b, d=3):
    """Range in math mode pieces: $a$--$b$."""
    return f"${a:.{d}f}$" if round(a, d) == round(b, d) else f"${a:.{d}f}$--${b:.{d}f}$"


def prng(a, b, d=1):
    """Percent range: 89.4--89.5\\%."""
    return pct(a, d) if round(100 * a, d) == round(100 * b, d) else f"{100 * a:.{d}f}--{pct(b, d)}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def write(name, text):
    (TEX / name).write_text(text)
    print("wrote", name)


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, allv, side="right") / len(a)
                               - np.searchsorted(b, allv, side="right") / len(b))))


# ---------------------------------------------------------------------------
# 7.2 exact enumeration
# ---------------------------------------------------------------------------

def tab_enum():
    rows = []
    for name, label in [("4x4", r"$4\times4$"), ("4x6", r"$4\times6$")]:   # the 3x4 run is stored but no longer reported
        d = J(DER / f"validation_v2/enumeration_{name}.json")
        steps = next(iter(d["runs"].values()))["steps"]
        fl, fj = d["iid_noise_floor"]["level1"], d["iid_noise_floor"]["joint"]
        halves = max(r["tv_first_vs_second_half"] for r in d["runs"].values())
        pooled = np.array([r["pooled_empirical"] for r in J(DER / f"validation_v2/enumeration_{name}_states.json")])
        exp_unvisited = float(((1.0 - pooled) ** (3 * steps)).sum())      # iid sample of the pooled length
        rows.append((label, "$100$", d["feasible_level1"], d["feasible_joint"], steps,
                     d["unvisited_level1_states"], exp_unvisited,
                     max(d["tv_between_starts"].values()), fl["tv_between_two_samples"],
                     halves, fl["tv_between_halves"],
                     max(d["joint"]["tv_between_starts"].values()), fj["tv_between_two_samples"],
                     d["pooled_tv_to_uniform"], d["pooled_tv_to_spanning_tree_law"],
                     d["max_over_min_pooled_frequency"]))
    p = DER / "validation_nonuniform/enumeration_nonuniform_4x4.json"
    nonuni = None
    if p.exists():
        nonuni = J(p)
    lines = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{feasible states} & & \multicolumn{3}{c}{TV between chains (noise floor)} "
             r"& \multicolumn{2}{c}{TV of pooled law to} & \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
             r"grid & demand & level 1 & joint & unvisited & level 1 & halves & joint & uniform & tree & max/min \\",
             r"\midrule"]
    for (label, dem, n1, n2, steps, unv, eunv, tv1, f1, hv, fh, tvj, fjn, tu, tt, mm) in rows:
        unv_s = f"{unv}" if unv == 0 else f"{unv} ({eunv:.0f})"
        lines.append(f"{label} & {dem} & ${num(n1)}$ & ${num(n2)}$ & {unv_s} & "
                     f"${tv1:.3f}$ (${f1:.3f}$) & ${hv:.3f}$ (${fh:.3f}$) & ${tvj:.3f}$ (${fjn:.3f}$) & "
                     f"${tu:.2f}$ & ${tt:.2f}$ & ${num(mm)}$ \\\\")
    if nonuni:
        d = nonuni
        cl = J(DER / "validation_nonuniform/enumeration_closure_4x4.json")
        fl, fj = d["iid_noise_floor"]["level1"], d["iid_noise_floor"]["joint"]
        halves = max(r["tv_first_vs_second_half"] for r in d["runs"].values())
        n_un = cl["reachable_level1_open_windows"] - d["visited_level1_union"]
        lines.append(r"\midrule")
        lines.append(f"$4\\times4$ & $80$--$120$ & ${num(cl['reachable_level1_open_windows'])}$ & "
                     f"${num(cl['reachable_joint_open_windows'])}$ & "
                     f"{n_un} & ${max(d['tv_between_starts_level1'].values()):.3f}$ (${fl['tv_between_two_samples']:.3f}$) & "
                     f"${halves:.3f}$ (${fl['tv_between_halves']:.3f}$) & "
                     f"${max(d['tv_between_starts_joint'].values()):.3f}$ (${fj['tv_between_two_samples']:.3f}$) & "
                     f"${d['pooled_tv_to_uniform_on_support']:.2f}$ & ${d['pooled_tv_to_spanning_tree_law_on_support']:.2f}$ & "
                     f"${num(d['max_over_min_pooled_frequency'])}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_enum.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 7.3 multi-start agreement (grids and London in one layout)
# ---------------------------------------------------------------------------

def london_forgetting(in_dirs, steps_by_dir, max_step=4000):
    """Share of each chain's initial station boundary still cut, and the share of one
    chain's boundary that a plan of another chain cuts (the independent-plan value)."""
    curves, finals = {}, []
    for in_dir, steps in zip(in_dirs, steps_by_dir):
        for f in sorted(Path(in_dir).glob(f"snap_real_s*_T{steps}.npz")):
            S = np.load(f)
            e = S["edges"]
            st = S["station"]
            m0 = st[0][e[:, 0]] != st[0][e[:, 1]]
            keep = S["step"] <= max_step
            share = [float(((st[i][e[:, 0]] != st[i][e[:, 1]]) & m0).sum() / m0.sum()) for i in np.where(keep)[0]]
            seed = int(f.name.split("_s")[1].split("_")[0])
            curves[seed] = (S["step"][keep], np.array(share))
            finals.append((seed, st[-1], e))
    base = []
    for i, (si, a, e) in enumerate(finals):
        for sj, b, _ in finals[i + 1:]:
            ma, mb = a[e[:, 0]] != a[e[:, 1]], b[e[:, 0]] != b[e[:, 1]]
            base.append(float((ma & mb).sum() / ma.sum()))
    return curves, float(np.mean(base))


def london_series(in_dir, steps, tag="real"):
    out = {}
    for f in sorted(Path(in_dir).glob(f"trace_{tag}_s*_T{steps}.csv")):
        seed = int(f.name.split("_s")[1].split("_")[0])
        out[seed] = pd.read_csv(f)
    return out


def split_rhat(chains):
    halves = []
    for c in chains:
        h = len(c) // 2
        halves += [c[:h], c[h:2 * h]]
    n = len(halves[0])
    m = np.array([h.mean() for h in halves])
    v = np.array([h.var(ddof=1) for h in halves])
    W, B = v.mean(), n * m.var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def ess(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n // 2):
        pair = acf[2 * k - 1] + acf[2 * k]
        if pair < 0:
            break
        tau += 2 * pair
    return n / tau


def london_diag(seeds_dirs, burn=16000, thin=20):
    """KS (between chains, within halves), split R-hat and ESS per observable."""
    traces = {}
    for in_dir, steps in seeds_dirs:
        traces.update(london_series(in_dir, steps))
    cols = {"E1": "E1", "P1": "P1", "P2": "P2", "cut": "cut_edges"}
    out = {"seeds": sorted(traces), "acceptance": {s: float(t["accepted"][1:].mean()) for s, t in traces.items()}}
    for key, col in cols.items():
        post = {s: t[col].to_numpy()[burn:] for s, t in traces.items()}
        seeds = sorted(post)
        pair = {(a, b): ks(post[a][::thin], post[b][::thin]) for i, a in enumerate(seeds) for b in seeds[i + 1:]}
        halves = {s: ks(v[:len(v) // 2][::thin], v[len(v) // 2:][::thin]) for s, v in post.items()}
        out[key] = {"pair": pair, "halves": halves, "rhat": split_rhat(list(post.values())),
                    "ess": {s: ess(v) for s, v in post.items()}}
    return out


def tab_stability():
    rows = []
    forget = {}
    for n in (10000, 50000):
        m = J(EXP / f"scalability/results/multistart/multistart_{n}.json")
        acc = list(m["acceptance"].values())
        f = m["forgetting"]
        forget[f"grid\\_{n}"] = (np.array(f["steps"]), {int(s): np.array(v) for s, v in f["share"].items()},
                                 f["independent_baseline"])
        stf = [v for v in f["steps_to_baseline"].values() if v is not None]
        cols = ["R1", "n_districts", "n_superdistricts", "cut_edges"]
        rows.append((f"grid\\_{n}", "4", prng(min(acc), max(acc)),
                     [f"${m[c]['max_pairwise_ks']:.3f}$" for c in cols],
                     [rng(*m[c]['within_chain_halves_ks_range']) for c in cols],
                     f"${min(stf)}$--${max(stf)}$",
                     f"${m['R1']['split_rhat']:.3f}$", f"${num(m['R1']['ess'])}$"))
    base_dir, grp_dir = DER / "real_stations_u97", DER / "real_stations_u97_groupstart"
    curves, baseline = london_forgetting([base_dir, grp_dir], [80000, 80000])
    forget["London"] = (None, curves, baseline)

    def steps_to(curve):
        st, sh = curve
        hit = st[sh <= 1.1 * baseline]
        return int(hit[0]) if len(hit) else None

    d4 = london_diag([(base_dir, 80000)])
    d5 = london_diag([(base_dir, 80000), (grp_dir, 80000)])
    keys = ["E1", "P1", "P2", "cut"]
    acc = list(d4["acceptance"].values())
    stf = [steps_to(curves[s]) for s in (1, 2, 3, 4)]
    rows.append(("London", "4", prng(min(acc), max(acc)),
                 [f"${max(d4[k]['pair'].values()):.3f}$" for k in keys],
                 [rng(min(d4[k]['halves'].values()), max(d4[k]['halves'].values())) for k in keys],
                 f"${min(stf)}$--${max(stf)}$",
                 f"${d4['E1']['rhat']:.3f}$", f"${num(sum(d4['E1']['ess'].values()))}$"))
    five = [max(v for (a, b), v in d5[k]["pair"].items() if 5 in (a, b)) for k in keys]
    rows.append(("London, fifth chain", "1", pct(d5["acceptance"][5]),
                 [f"${v:.3f}$" for v in five], None, f"${steps_to(curves[5])}$",
                 f"${d5['E1']['rhat']:.3f}$", f"${num(d5['E1']['ess'][5])}$"))
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{4}{c}{Kolmogorov--Smirnov distance} & & & \\", r"\cmidrule(lr){4-7}",
             r"instance & & acceptance & access cost & $|\Pc^1|$ & $|\Pc^2|$ & cut edges & steps to forget & $\hat R$ & ESS \\",
             r"\midrule"]
    for name, nch, acc_s, between, within, stf_s, rhat_s, ess_s in rows:
        label = "between chains" if within is not None else "to the four chains"
        lines.append(f"{name} & {label} & {acc_s} & " + " & ".join(between) + f" & {stf_s} & {rhat_s} & {ess_s} \\\\")
        if within is not None:
            lines.append(" & within a chain & & " + " & ".join(within) + r" & & & \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_stability.tex", "\n".join(lines) + "\n")
    (TEX / "stability_numbers.json").write_text(json.dumps({
        "london_baseline": baseline, "london_steps_to_forget": {str(s): steps_to(c) for s, c in curves.items()},
        "london4": {k: {"pair_max": max(d4[k]["pair"].values()), "halves": list(d4[k]["halves"].values()),
                        "rhat": d4[k]["rhat"], "ess": d4[k]["ess"]} for k in keys},
        "london5_to_four": dict(zip(keys, five)), "london5_rhat": {k: d5[k]["rhat"] for k in keys}}, indent=1, default=float))

    # figure: forgetting curves
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, (name, (steps, shares, base)) in zip(axes, forget.items()):
        for i, (seed, v) in enumerate(sorted(shares.items())):
            if steps is None:
                st, sh = v
            else:
                st, sh = steps, v
            keep = st <= 3000
            ax.plot(st[keep], sh[keep], lw=1.0, color=C[i % len(C)],
                    label=("Group-based start" if (name == "London" and seed == 5) else None),
                    ls="--" if (name == "London" and seed == 5) else "-")
        ax.axhline(base, color="k", lw=0.7, ls=":")
        ax.text(3000, base + 0.03, "independent plans", ha="right", fontsize=7)
        ax.set_title(name.replace("\\_", "_"), fontsize=9)
        ax.set_xlabel("step")
        ax.set_xlim(0, 3000)
    axes[0].set_ylabel("share of initial boundary\nstill in place")
    axes[2].legend(fontsize=7, loc="upper right")
    fig.savefig(FIG / "fig_forgetting.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# rejections by cause
# ---------------------------------------------------------------------------

def cause_shares(detail, steps, causes=None):
    detail = {**(causes or {}), **detail}          # the below-kappa count lives in the coarse dict on the grids
    g = lambda pre: sum(v for k, v in detail.items() if k.startswith(pre)) / steps   # noqa: E731
    return {"super_tree": g("cut_search_exhausted:super:tree"), "super_demand": g("cut_search_exhausted:super:demand"),
            "kappa": g("super_district_below_kappa"), "base": g("cut_search_exhausted:base"),
            "other": 0.0}


def tab_rejections():
    rows = []
    for n in (100, 400, 1000, 10000, 50000):
        d = J(EXP / f"scalability/results/grid_{n}_seed42.json")
        r = d["rejection_report"]
        rows.append((f"grid\\_{n}", r["steps"], r["acceptance_rate"], cause_shares(r["causes_detail"], r["steps"], {"super_district_below_kappa": r["causes"].get("super_district_below_kappa", 0)}),
                     r["causes"].get("cut_search_exhausted:base", 0) / r["steps"] * 1e4))
    tot, det, acc = 0, {}, 0
    for s in (1, 2, 3, 4):
        d = J(DER / f"real_stations_u97/summary_real_s{s}_T80000.json")
        tot += d["steps"] - 1
        for k, v in d["rejection_causes_detail"].items():
            det[k] = det.get(k, 0) + v
    rej = sum(det.values())
    rows.append(("London", tot, 1 - rej / tot, cause_shares(det, tot),
                 sum(v for k, v in det.items() if k.startswith("cut_search_exhausted:base")) / tot * 1e4))
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{supergraph recursion} & superdistrict & base-level \\", r"\cmidrule(lr){4-5}",
             r"instance & proposals & accepted & tree failure & demand failure & below $\kappa^2_{\min}$ & failures per $10^4$ \\",
             r"\midrule"]
    for name, steps, a, c, base in rows:
        lines.append(f"{name} & ${num(steps)}$ & {pct(a)} & {pct(c['super_tree'])} & {pct(c['super_demand'])} & "
                     f"{pct(c['kappa'])} & ${base:.1f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_rejections.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 7.5 scalability
# ---------------------------------------------------------------------------

def step_stats(d):
    recs = d["steps"]
    acc = [r for r in recs if r["accepted"]]
    t = np.array([r["t_step"] for r in acc]) * 1e3
    tall = np.array([r["t_step"] for r in recs]) * 1e3
    sup = sum(r["t_super"] for r in recs) / max(1e-12, sum(r["t_step"] for r in recs))
    sup_acc = sum(r["t_super"] for r in acc) / max(1e-12, sum(r["t_step"] for r in acc))
    return {"median": float(np.median(t)), "p95": float(np.percentile(t, 95)), "mean_all": float(tall.mean()),
            "H": float(np.mean([r["H_size"] for r in acc])), "districts": float(np.median([r["n_districts"] for r in recs])),
            "super_share": float(sup_acc), "super_share_all": float(sup), "rejected": 1 - len(acc) / len(recs), "total_min": float(tall.sum() / 6e4)}


def tab_scal():
    sweep = {}
    lines = [r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
             r"$|V|$ & $k$ & $|\Pc^1|$ & mean $|H|$ & $|H|/|V|$ & median step & mean proposal & $95$th pct & "
             r"supergraph share & $10^4$ proposals \\", r"\midrule"]
    for n in (100, 400, 1000, 10000, 50000):
        d = J(EXP / f"scalability/results/grid_{n}_seed42.json")
        s = step_stats(d)
        sweep[n] = (s, d["k_teams_coverage"])
        lines.append(f"${num(n)}$ & ${d['k_teams_coverage']}$ & ${s['districts']:.0f}$ & ${num(s['H'])}$ & ${s['H'] / n:.3f}$ & "
                     f"${s['median']:.1f}$\\,ms & ${s['mean_all']:.0f}$\\,ms & ${s['p95']:.0f}$\\,ms & "
                     f"{pct(s['super_share'], 0)} & ${s['total_min']:.1f}$\\,min \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_scal.tex", "\n".join(lines) + "\n")

    fixed = {}
    lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
             r"$w$ & $k$ & $|F^1|$ & $|\Pc^1|$ & initial plan & median step & mean $|H|$ & supergraph share & rejected \\",
             r"\midrule"]
    d = J(EXP / "scalability/results/grid_10000_seed42.json")
    s = step_stats(d)
    fixed[d["k_teams_coverage"]] = s
    lines.append(f"$20{{,}}000$ & ${d['k_teams_coverage']}$ & ${d['n_l1_candidates']}$ & ${s['districts']:.0f}$ & "
                 f"${d['init_partition_time_s']:.0f}$\\,s & ${s['median']:.0f}$\\,ms & ${num(s['H'])}$ & "
                 f"{pct(s['super_share'], 0)} & {pct(s['rejected'], 0)} \\\\")
    for w in (10000, 5000):
        runs = [J(EXP / f"scalability/results/fixed_size/grid_10000_w{w}_seed{sd}.json") for sd in (42, 43)]
        ss = [step_stats(r) for r in runs]
        fixed[runs[0]["k_teams_coverage"]] = ss[0]
        init = sorted(r["init_partition_time_s"] for r in runs)
        lines.append(f"${num(w)}$ & ${runs[0]['k_teams_coverage']}$ & ${runs[0]['n_l1_candidates']}$ & "
                     f"{rng(min(s_['districts'] for s_ in ss), max(s_['districts'] for s_ in ss), 0)} & "
                     f"{rng(init[0], init[1], 0)}\\,s & {rng(min(s_['median'] for s_ in ss), max(s_['median'] for s_ in ss), 0)}\\,ms & "
                     f"${np.mean([s_['H'] for s_ in ss]):.0f}$ & "
                     f"{prng(min(s_['super_share'] for s_ in ss), max(s_['super_share'] for s_ in ss), 0)} & "
                     f"{prng(min(s_['rejected'] for s_ in ss), max(s_['rejected'] for s_ in ss), 0)} \\\\")
    lines.append(r"$2{,}500$ & $401$ & $602$ & \multicolumn{6}{l}{initial plan not built in twelve attempts ($19$\,min)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_fixed.tex", "\n".join(lines) + "\n")

    ns = np.array(sorted(sweep))
    med = np.array([sweep[n][0]["median"] for n in ns])
    slope = np.polyfit(np.log(ns), np.log(med), 1)        # exponent of the step time in |V|, quoted in the text
    (TEX / "scaling_numbers.json").write_text(json.dumps({"slope": float(slope[0])}))


# ---------------------------------------------------------------------------
# placement of the ensemble relative to the optimum
# ---------------------------------------------------------------------------

def fig_placement():
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, n in zip(axes, (100, 400)):
        p = EXP / f"gurobi/results_sparse/sample_energy_{n}.json"
        if not p.exists():
            return
        d = J(p)
        opt = d["optimum"]
        burn = d["chains"][0]["steps"] // 5
        tot = np.concatenate([np.array(c["E1"][burn:]) + np.array(c["E2"][burn:]) for c in d["chains"]])
        rel = tot / opt
        ax.hist(rel, bins=50, color=C[0], alpha=0.75)
        ax.axvline(1.0, color=C[1], lw=1.5)
        ax.text(1.0, ax.get_ylim()[1] * 0.97, " optimum", color=C[1], va="top", fontsize=8)
        ms = J(EXP / f"gurobi/solution_{n}_falcom_multistart_summary.json")
        best = min(r["obj_value"] for r in ms["results"] if r.get("ok"))
        ax.axvline(best / opt, color=C[2], lw=1.2, ls="--")
        ax.text(best / opt, ax.get_ylim()[1] * 0.80, " annealed", color=C[2], va="top", fontsize=8)
        ax.set_xlabel("objective / optimum"); ax.set_yticks([])
        ax.set_title(f"grid_{n}", fontsize=9)
        ax.set_xlim(left=0.97)
        chain_means = [float(np.mean(np.array(c["E1"][burn:]) + np.array(c["E2"][burn:]))) / opt for c in d["chains"]]
        out[n] = {"optimum": opt, "seconds": d["optimum_seconds"], "min": float(rel.min()), "q05": float(np.quantile(rel, .05)),
                  "median": float(np.median(rel)), "q95": float(np.quantile(rel, .95)), "max": float(rel.max()),
                  "share_within_10pct": float((rel <= 1.10).mean()), "annealed_best": best / opt,
                  "chain_means": chain_means, "acceptance": [c["acceptance"] for c in d["chains"]],
                  "n": int(len(rel))}
    axes[0].set_ylabel("sampled plans")
    fig.tight_layout()
    fig.savefig(FIG / "fig_placement.pdf")
    plt.close(fig)
    (TEX / "placement_numbers.json").write_text(json.dumps(out, indent=1))
    milp = {r["instance"]: r for r in J(EXP / "gurobi/results_sparse/milp_summary.json")["rows"]}
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{exact optimum} & \multicolumn{2}{c}{sampled ensemble / optimum} & annealed \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"instance & sites / units & objective & time & no contiguity & median & $5$--$95\%$ & / optimum \\", r"\midrule"]
    for n in (100, 400):
        m, o = milp[str(n)], out[n]
        lines.append(f"grid\\_{n} & ${m['n_l1_candidates']}$ / ${m['k_teams']}$ & ${num(m['milp']['shir']['obj'])}$ & "
                     f"${num(m['milp']['shir']['wall_time_s'])}$\\,s & ${num(m['milp']['noshir']['obj'])}$ & "
                     f"${o['median']:.2f}$ & ${o['q05']:.2f}$--${o['q95']:.2f}$ & ${o['annealed_best']:.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_milp.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# ablations
# ---------------------------------------------------------------------------

ABL = EXP / "ablation/results"
ABL_L = DER / "ablation_london"
LABEL = {"default": "FalCom (all components)", "nocount": "no counting predicate", "noresid": "no residual predicate",
         "scaled": "capacity-scaled debt window", "global": "global demand target at level 1"}


def grid_run(variant, n, seed):
    p = ABL / variant / f"grid_{n}_seed{seed}.json"
    return J(p) if p.exists() else None


def london_run(variant, seed, steps=5000):
    p = ABL_L / variant / f"summary_abl_s{seed}_T{steps}.json"
    return J(p) if p.exists() else None


def grid_cell(d):
    if d is None:
        return None
    r = d["rejection_report"]
    n = len(d["steps"])                                  # proposals actually made (a variant may spend its budget first)
    c = cause_shares(r.get("causes_detail", {}), n,
                     {"super_district_below_kappa": r["causes"].get("super_district_below_kappa", 0)})
    rej = sum(1 for x in d["steps"] if not x["accepted"])
    return {"acc": 1 - rej / n, "super": c["super_tree"] + c["super_demand"], "kappa": c["kappa"], "base": c["base"],
            "steps": n, "asked": d["T_steps"], "ms": 1e3 * sum(x["t_step"] for x in d["steps"]) / n}


def london_cell(variant, seed, steps=5000):
    f = ABL_L / variant / f"trace_abl_s{seed}_T{steps}.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f).iloc[1:]                          # row 0 is the initial state
    n = len(t)
    cause = t["cause"].fillna("")
    d = J(ABL_L / variant / f"summary_abl_s{seed}_T{steps}.json")
    c = cause_shares(d["rejection_causes_detail"], n)
    return {"acc": float(t["accepted"].mean()), "super": float(cause.str.startswith("cut_search_exhausted:super").mean()),
            "kappa": float(cause.str.startswith("super_district_below_kappa").mean()),
            "base": float(cause.str.startswith("cut_search_exhausted:base").mean()),
            "super_tree": c["super_tree"], "super_demand": c["super_demand"],
            "steps": n, "asked": steps - 1, "ms": 1e3 * float(t["seconds"].sum()) / n}


def tab_ablation():
    if not ABL.exists():
        return
    insts = [("grid\\_1000", lambda v: grid_cell(grid_run(v, 1000, 42))),
             ("grid\\_10000", lambda v: grid_cell(grid_run(v, 10000, 42))),
             ("London", lambda v: london_cell(v, 1))]
    lines = [r"\begin{tabular}{l" + "rrr" * len(insts) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n, _ in insts) + r" \\",
             "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(insts))),
             "variant" + r" & accepted & base & supergr." * len(insts) + r" \\", r"\midrule"]
    dump = {}
    base_ms = {}
    for v in ("default", "nocount", "noresid", "scaled", "global"):
        cells = []
        for name, get in insts:
            c = get(v)
            dump[f"{v}|{name}"] = c
            if c is None:
                cells.append(r"\multicolumn{3}{c}{--}")
                continue
            if v == "default":
                base_ms[name] = c["ms"]
            rel = c["ms"] / base_ms[name] if name in base_ms else float("nan")
            c["time_rel"] = rel
            cells.append(f"{pct(c['acc'])} & {pct(c['base'], 1 if c['base'] >= 0.001 else 2)} & {pct(c['super'])}")
        lines.append(f"{LABEL[v]} & " + " & ".join(cells) + r" \\")
        if v == "default":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation.tex", "\n".join(lines) + "\n")
    (TEX / "ablation_numbers.json").write_text(json.dumps(dump, indent=1))

    # retry budget: acceptance and readouts under M = 200, 1000, 5000
    def grid_obs(d, burn=0.2):
        recs = d["steps"]
        b = int(len(recs) * burn)
        return {"R1": np.array([r["R1"] for r in recs[b:] if "R1" in r]),
                "P1": np.array([r["n_districts"] for r in recs[b:]], float)}

    def london_obs(variant, seed, steps=5000, burn=0.2):
        f = ABL_L / variant / f"trace_abl_s{seed}_T{steps}.csv"
        if not f.exists():
            return None
        t = pd.read_csv(f)
        b = int(len(t) * burn)
        return {"R1": t["E1"].to_numpy()[b:], "P1": t["P1"].to_numpy(float)[b:]}

    def long_reference(n):
        """Long baseline chains of the instance, where the paper has them."""
        if n is None:
            tr = london_series(DER / "real_stations_u97", 80000)
            return {"R1": np.concatenate([t["E1"].to_numpy()[16000:] for t in tr.values()]),
                    "P1": np.concatenate([t["P1"].to_numpy(float)[16000:] for t in tr.values()])}
        files = sorted((EXP / "scalability/results/multistart").glob(f"grid_{n}_seed*.json"))
        if not files:
            return None
        obs = [grid_obs(J(f)) for f in files]
        return {k: np.concatenate([o[k] for o in obs]) for k in ("R1", "P1")}

    rows = []
    for name, n, seeds, variants in [("grid\\_1000", 1000, (42, 43), ("M200", "M1000", "default", "M5000")),
                                     ("grid\\_10000", 10000, (42, 43), ("M200", "M1000", "default", "M5000")),
                                     ("London", None, (1, 2), ("M200", "default", "M5000"))]:
        def observe(v):
            if n is None:
                return [london_obs(v, sd) for sd in seeds]
            return [grid_obs(d) if (d := grid_run(v, n, sd)) else None for sd in seeds]
        default = observe("default")
        if any(o is None for o in default):
            continue
        ref = long_reference(n)
        against_long = ref is not None
        if not against_long:
            ref = {k: np.concatenate([o[k] for o in default]) for k in ("R1", "P1")}
        for v in variants:
            obs = default if v == "default" else observe(v)
            if any(o is None for o in obs):
                continue
            if n is None:
                accs = [c["acc"] for sd in seeds if (c := london_cell(v, sd))]
                mb, ms_ = (1000, 1000) if v == "default" else (int(v[1:]), int(v[1:]))
            else:
                accs = [grid_cell(grid_run(v, n, sd))["acc"] for sd in seeds]
                mb, ms_ = (5000, 1000) if v == "default" else (int(v[1:]), int(v[1:]))
            pool = {k: np.concatenate([o[k] for o in obs]) for k in ("R1", "P1")}
            if against_long:        # each short run against the long baseline chains; the default rows are the noise
                dist = {k: max(ks(o[k][::20], ref[k][::20]) for o in obs) for k in ("R1", "P1")}
            elif v == "default":    # no long chains: the two default seeds against each other
                dist = {k: ks(obs[0][k][::20], obs[1][k][::20]) for k in ("R1", "P1")}
            else:
                dist = {k: ks(pool[k][::20], ref[k][::20]) for k in ("R1", "P1")}
            rows.append({"instance": name, "M_base": mb, "M_super": ms_, "acc": [min(accs), max(accs)],
                         "R1_mean_rel": float(pool["R1"].mean() / ref["R1"].mean()), "P1_mean": float(pool["P1"].mean()),
                         "ks_R1": dist["R1"], "ks_P1": dist["P1"], "default": v == "default",
                         "against_long": against_long})
    if rows:
        lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                 r" & \multicolumn{2}{c}{retry budget $M$} & & \multicolumn{2}{c}{access cost} & \multicolumn{2}{c}{$|\Pc^1|$} \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
                 r"instance & base & supergraph & accepted & mean (rel.) & KS & mean & KS \\", r"\midrule"]
        last = None
        for r in rows:
            if last is not None and r["instance"] != last:
                lines.append(r"\midrule")
            last = r["instance"]
            ks1 = f"({r['ks_R1']:.3f})" if r["default"] else f"{r['ks_R1']:.3f}"
            ks2 = f"({r['ks_P1']:.3f})" if r["default"] else f"{r['ks_P1']:.3f}"
            lines.append(f"{r['instance'] if r['default'] or True else ''} & ${num(r['M_base'])}$ & ${num(r['M_super'])}$ & "
                         f"{100 * r['acc'][0]:.1f}--{pct(r['acc'][1])} & ${r['R1_mean_rel']:.3f}$ & ${ks1}$ & "
                         f"${r['P1_mean']:.1f}$ & ${ks2}$ \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write("tab_budget.tex", "\n".join(lines) + "\n")
        (TEX / "budget_numbers.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    tab_enum()
    tab_stability()
    tab_rejections()
    tab_scal()
    fig_placement()
    tab_ablation()
