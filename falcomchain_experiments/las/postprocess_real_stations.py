"""
Ensemble analysis of the real-station London chains produced by
:mod:`run_real_stations`.

Reads ``summary_<tag>_s<seed>_T<steps>.json``, ``trace_...csv`` and
``snap_...npz`` for every seed found and writes, to ``--out-dir``:

- ``ensemble_real_stations.json`` -- every number quoted in the paper;
- ``fig_traces.png``        -- energy / district-count / cut-edge traces of all
  chains with the burn-in shaded, and their post-burn-in energy histograms;
- ``fig_forgetting.png``    -- how fast each chain forgets its initial plan
  (share of the initial boundary edges still cut);
- ``fig_boundary_freq.png`` -- level-1 and level-2 boundary frequency maps;
- ``fig_contested.png``     -- choropleth of LSOA contestedness
  (1 - share of samples in which the LSOA is served by its modal station);
- ``fig_stations.png``      -- per-station opening frequency and capacity mix;
- ``fig_groups.png``        -- station-pair co-membership in super-districts
  versus the 21 real Groups;
- ``fig_slas.png``          -- the operational layout s_LAS against the
  ensemble on energy and drive-time metrics.

Convergence is assessed the way the redistricting literature does it:
agreement of post-burn-in summary distributions across independently
started chains (two-sample Kolmogorov-Smirnov statistics) and the loss of
memory of the start. Split-R-hat and effective sample sizes are reported as
secondary numbers.

Run from the repository root::

    python -m falcomchain_experiments.las.postprocess_real_stations --tag real --steps 40000
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
STATIONS = REPO / "data/raw/LAS_stations.csv"
L2CSV = REPO / "data/raw/LAS_L2_facilities.csv"
LSOA_STATION = REPO / "data/derived/lsoa_to_station.csv"
LSOA_GROUP = REPO / "data/derived/lsoa_to_group.csv"
LSOA_GPKG = REPO / "data/raw/LSOA_2021_London.gpkg"
REAL_TRAVEL = REPO / "data/derived/real_station_travel_times.parquet"
GRAPH = REPO / "data/raw/london_graph.json"
DEFAULT_IN = REPO / "data/derived/real_stations"

COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#d6402a", "#7a4fd6", "#008080"]


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------

def split_rhat(chains):
    """Split-R-hat (Gelman et al. 2013) of equally long scalar traces."""
    m = min(len(c) for c in chains)
    halves = []
    for c in chains:
        c = np.asarray(c[:m], dtype=float)
        halves += [c[: m // 2], c[m // 2: 2 * (m // 2)]]
    n = len(halves[0])
    means = np.array([h.mean() for h in halves])
    variances = np.array([h.var(ddof=1) for h in halves])
    W = variances.mean()
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else float("nan")


def ess(x):
    """Effective sample size via the initial positive sequence estimator."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    if n < 10 or x.var() == 0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conjugate(f))[:n] / n
    rho = acov / acov[0]
    tau, k = 1.0, 1
    while k + 1 < n:
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
        k += 2
    return float(n / tau)


def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def adjusted_rand(labels_a, labels_b):
    """Adjusted Rand index of two labelings of the same items."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    n = len(a)
    if n < 2:
        return float("nan")
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = comb(table).sum()
    sum_a = comb(table.sum(axis=1)).sum()
    sum_b = comb(table.sum(axis=0)).sum()
    total = comb(n)
    expected = sum_a * sum_b / total
    max_index = 0.5 * (sum_a + sum_b)
    return float((sum_ij - expected) / (max_index - expected)) if max_index != expected else 1.0


def ems_metrics(times, demands):
    """Demand-weighted mean, 90th percentile and 8/15-minute coverage."""
    order = np.argsort(times)
    t, d = np.asarray(times)[order], np.asarray(demands)[order]
    tot = d.sum()
    cum = np.cumsum(d)
    return {
        "T_mean": float((t * d).sum() / tot),
        "T90": float(t[np.searchsorted(cum, 0.9 * tot)]),
        "C8": float(d[t <= 8.0].sum() / tot),
        "C15": float(d[t <= 15.0].sum() / tot),
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_runs(in_dir: Path, tag: str, steps: int):
    runs = []
    for summ in sorted(in_dir.glob(f"summary_{tag}_s*_T{steps}.json")):
        stem = summ.name[len("summary_"):-len(".json")]
        s = json.loads(summ.read_text())
        trace = pd.read_csv(in_dir / f"trace_{stem}.csv")
        snap = np.load(in_dir / f"snap_{stem}.npz", allow_pickle=False)
        runs.append({"summary": s, "trace": trace, "snap": snap, "seed": s["seed"]})
    if not runs:
        raise SystemExit(f"no runs matching {tag}/T{steps} in {in_dir}")
    return runs


def load_instance(node_ids):
    graph = json.loads(GRAPH.read_text())
    info = {n["id"]: n for n in graph["nodes"]}
    lsoa_of = {n: info[n]["LSOA21CD"] for n in node_ids}
    demand = np.array([float(info[n]["demand"]) for n in node_ids])
    id_of = {v: k for k, v in lsoa_of.items()}
    stations = pd.read_csv(STATIONS)
    station_node = {r.station_code: id_of[r.LSOA21CD] for r in stations.itertuples()}
    station_name = dict(zip(stations.station_code, stations.station_name))
    # LSOA -> station (s_LAS) and LSOA -> Group
    st_map = pd.read_csv(LSOA_STATION)
    gr_map = pd.read_csv(LSOA_GROUP)
    lsoa_station = dict(zip(st_map.LSOA21CD, st_map.station_code))
    lsoa_group = dict(zip(gr_map.LSOA21CD, gr_map.group))
    l2 = pd.read_csv(L2CSV)
    l2 = l2[l2.facility_type != "eoc"]
    hq_node = {str(r.facility_id).removeprefix("GH-").strip(): id_of[r.LSOA21CD]
               for r in l2.itertuples() if r.LSOA21CD in id_of}
    # travel minutes station_node -> node
    tt_df = pd.read_parquet(REAL_TRAVEL)
    tt = {(id_of[o], id_of[d]): s / 60.0 for o, d, s in
          zip(tt_df.origin, tt_df.dest, tt_df.seconds) if o in id_of and d in id_of}
    xy = {n: (float(info[n]["BNG_E"]), float(info[n]["BNG_N"])) for n in node_ids}

    def minutes(u, v):
        t = tt.get((u, v))
        if t is None:
            (x1, y1), (x2, y2) = xy[u], xy[v]
            t = np.hypot(x1 - x2, y1 - y2) / 1000.0 / 25.0 * 60.0
        return t

    return dict(lsoa_of=lsoa_of, demand=demand, id_of=id_of, station_node=station_node,
                station_name=station_name, lsoa_station=lsoa_station, lsoa_group=lsoa_group,
                hq_node=hq_node, minutes=minutes, xy=xy)


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="real")
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--burn-in", type=int, default=None, help="steps; default 20%% of --steps")
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()
    in_dir = Path(a.in_dir)
    out_dir = Path(a.out_dir) if a.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    burn_in = a.burn_in if a.burn_in is not None else a.steps // 5

    runs = load_runs(in_dir, a.tag, a.steps)
    snap0 = runs[0]["snap"]
    node_ids = [int(x) for x in snap0["node_ids"]]
    idx = {n: i for i, n in enumerate(node_ids)}
    station_nodes = [int(x) for x in snap0["station_node_ids"]]
    n_st = len(station_nodes)
    edges = snap0["edges"]
    inst = load_instance(node_ids)
    demand = inst["demand"]
    st_idx_of_node = {n: i for i, n in enumerate(station_nodes)}
    code_of_station_idx = {}
    for code, node in inst["station_node"].items():
        if node in st_idx_of_node:
            code_of_station_idx[st_idx_of_node[node]] = code

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {"tag": a.tag, "steps": a.steps, "burn_in": burn_in, "n_chains": len(runs),
           "seeds": [r["seed"] for r in runs]}

    # ---------------- 1. traces and cross-chain agreement ----------------
    post = {}
    for r in runs:
        t = r["trace"]
        post[r["seed"]] = t[t.step >= burn_in]
    metrics = ["E", "E1", "P1", "P2", "cut_edges"]
    out["acceptance_rate"] = {r["seed"]: r["summary"]["acceptance_rate"] for r in runs}
    out["rejection_causes"] = {r["seed"]: r["summary"]["rejection_causes"] for r in runs}
    out["steps_per_second"] = {r["seed"]: r["summary"]["steps_per_second"] for r in runs}
    out["post_burn_in"] = {}
    for m in metrics:
        per_chain = {s: p[m].to_numpy(dtype=float) for s, p in post.items()}
        thin = {s: v[::20] for s, v in per_chain.items()}   # thinned for KS
        seeds = list(per_chain)
        ks = {f"{seeds[i]}-{seeds[j]}": ks_statistic(thin[seeds[i]], thin[seeds[j]])
              for i in range(len(seeds)) for j in range(i + 1, len(seeds))}
        out["post_burn_in"][m] = {
            "mean_by_chain": {s: float(v.mean()) for s, v in per_chain.items()},
            "sd_by_chain": {s: float(v.std()) for s, v in per_chain.items()},
            "pooled_mean": float(np.concatenate(list(per_chain.values())).mean()),
            "pooled_q05_q95": [float(np.quantile(np.concatenate(list(per_chain.values())), q))
                               for q in (0.05, 0.95)],
            "ks_pairwise_thinned": ks,
            "ks_max": max(ks.values()) if ks else None,
            "split_rhat": split_rhat(list(per_chain.values())) if len(per_chain) > 1 else None,
            "ess_by_chain": {s: ess(v) for s, v in per_chain.items()},
        }

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [3, 1]})
    for k, (m, label) in enumerate([("E", "energy E(s) [demand-weighted minutes]"),
                                    ("P1", "number of districts |P1|"),
                                    ("cut_edges", "cut edges")]):
        ax = axes[k, 0]
        for c, r in enumerate(runs):
            t = r["trace"]
            ax.plot(t.step, t[m], lw=0.6, color=COLORS[c % len(COLORS)], label=f"seed {r['seed']}")
        ax.axvspan(0, burn_in, color="0.85", zorder=0)
        ax.set_ylabel(label)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=len(runs))
        if k == 2:
            ax.set_xlabel("step")
        axh = axes[k, 1]
        for c, r in enumerate(runs):
            axh.hist(post[r["seed"]][m], bins=40, histtype="step", density=True,
                     color=COLORS[c % len(COLORS)])
        axh.set_title("post-burn-in", fontsize=9)
    fig.suptitle(f"Real-station London chains, T={a.steps:,}, burn-in {burn_in:,} (shaded)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_traces.png", dpi=140)
    plt.close(fig)

    # ---------------- 2. forgetting the initial plan ----------------
    forgetting = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, r in enumerate(runs):
        S = r["snap"]
        st = S["station"]
        cut0 = st[0][edges[:, 0]] != st[0][edges[:, 1]]
        share = []
        for k in range(st.shape[0]):
            cut = st[k][edges[:, 0]] != st[k][edges[:, 1]]
            share.append(float((cut & cut0).sum() / max(1, cut0.sum())))
        forgetting[r["seed"]] = {"step": [int(x) for x in S["step"]], "share": share}
        ax.plot(S["step"], share, color=COLORS[c % len(COLORS)], label=f"seed {r['seed']}")
    ax.set_xlabel("step")
    ax.set_ylabel("share of initial boundary edges still cut")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_forgetting.png", dpi=140)
    plt.close(fig)
    out["forgetting_share_at_end"] = {s: v["share"][-1] for s, v in forgetting.items()}
    # an independent random plan of the same family shares about this much:
    # the level-1 boundary density
    pooled_cut_density = float(np.mean([p["cut_edges"].mean() for p in post.values()]) / len(edges))
    out["boundary_density"] = pooled_cut_density

    # ---------------- 3. pooled post-burn-in snapshots ----------------
    keep = []
    for r in runs:
        S = r["snap"]
        mask = S["step"] >= burn_in
        keep.append((S["station"][mask], S["super"][mask], S["capacity"][mask]))
    station = np.concatenate([k[0] for k in keep])     # (S, N)
    sup = np.concatenate([k[1] for k in keep])
    cap = np.concatenate([k[2] for k in keep])
    n_snap = station.shape[0]
    out["n_snapshots"] = int(n_snap)

    # boundary frequencies
    f1 = (station[:, edges[:, 0]] != station[:, edges[:, 1]]).mean(axis=0)
    f2 = (sup[:, edges[:, 0]] != sup[:, edges[:, 1]]).mean(axis=0)
    def bands(f):
        return {"never_cut_(<0.10)": float((f < 0.10).mean()),
                "contested_(0.10-0.90)": float(((f >= 0.10) & (f <= 0.90)).mean()),
                "always_cut_(>0.90)": float((f > 0.90).mean()),
                "max": float(f.max())}
    out["boundary_level1"] = bands(f1)
    out["boundary_level2"] = bands(f2)

    # contestedness of LSOAs
    modal_share = np.empty(len(node_ids))
    modal_station = np.empty(len(node_ids), dtype=int)
    for i in range(len(node_ids)):
        col = station[:, i]
        vals, counts = np.unique(col[col >= 0], return_counts=True)
        j = counts.argmax()
        modal_share[i] = counts[j] / n_snap
        modal_station[i] = vals[j]
    out["lsoa_modal_station_share"] = {
        "mean": float(modal_share.mean()),
        "share_of_lsoas_with_modal_share_above_0.9": float((modal_share > 0.9).mean()),
        "share_of_lsoas_with_modal_share_below_0.5": float((modal_share < 0.5).mean()),
        "demand_weighted_mean": float((modal_share * demand).sum() / demand.sum()),
    }

    # stations: opening frequency and capacity mix
    st_own = np.array([idx[n] for n in station_nodes])
    open_mask = station[:, st_own] == np.arange(n_st)[None, :]      # (S, 66)
    open_freq = open_mask.mean(axis=0)
    cap_mix = {}
    for i in range(n_st):
        c = cap[open_mask[:, i], st_own[i]]
        cap_mix[i] = {int(k): float(v / max(1, len(c))) for k, v in Counter(c.tolist()).items()}
    n_open = open_mask.sum(axis=1)
    out["stations"] = {
        "open_frequency_by_station": {code_of_station_idx.get(i, str(i)): float(open_freq[i]) for i in range(n_st)},
        "capacity_mix_by_station": {code_of_station_idx.get(i, str(i)): cap_mix[i] for i in range(n_st)},
        "n_open_per_plan_mean": float(n_open.mean()),
        "n_open_per_plan_range": [int(n_open.min()), int(n_open.max())],
        "stations_open_in_over_90pct": int((open_freq > 0.9).sum()),
        "stations_open_in_under_10pct": int((open_freq < 0.1).sum()),
        "capacity_share_pooled": {int(k): float(v / n_open.sum()) for k, v in
                                  Counter(cap[open_mask.nonzero()[0], st_own[open_mask.nonzero()[1]]].tolist()).items()},
    }
    order = np.argsort(-open_freq)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    names = [inst["station_name"].get(code_of_station_idx.get(i, ""), str(i)) for i in order]
    axes[0].bar(range(n_st), open_freq[order], color="#2a78d6")
    axes[0].set_ylabel("opened in share of plans")
    bottom = np.zeros(n_st)
    for cval, colr in [(1, "#c6dbef"), (2, "#6baed6"), (3, "#08519c")]:
        h = np.array([cap_mix[i].get(cval, 0.0) for i in order])
        axes[1].bar(range(n_st), h, bottom=bottom, color=colr, label=f"{cval} unit{'s' if cval > 1 else ''}")
        bottom += h
    axes[1].set_ylabel("capacity mix when open")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].set_xticks(range(n_st))
    axes[1].set_xticklabels(names, rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_stations.png", dpi=140)
    plt.close(fig)

    # groups: co-membership vs the 21 real Groups
    group_of_station = {}
    for i in range(n_st):
        code = code_of_station_idx.get(i)
        lsoa = inst["lsoa_of"][station_nodes[i]]
        group_of_station[i] = inst["lsoa_group"].get(lsoa, "?")
    co = np.zeros((n_st, n_st))
    both = np.zeros((n_st, n_st))
    ari = []
    same_group = np.array([[group_of_station[i] == group_of_station[j] for j in range(n_st)] for i in range(n_st)])
    for k in range(n_snap):
        op = np.where(open_mask[k])[0]
        s_lab = sup[k, st_own[op]]
        eq = s_lab[:, None] == s_lab[None, :]
        co[np.ix_(op, op)] += eq
        both[np.ix_(op, op)] += 1
        if len(op) > 2:
            ari.append(adjusted_rand(s_lab, [group_of_station[i] for i in op]))
    co_freq = np.divide(co, both, out=np.zeros_like(co), where=both > 0)
    off = ~np.eye(n_st, dtype=bool)
    out["groups"] = {
        "ari_vs_real_groups_mean": float(np.mean(ari)) if ari else None,
        "ari_vs_real_groups_q05_q95": [float(np.quantile(ari, q)) for q in (0.05, 0.95)] if ari else None,
        "co_membership_mean_same_real_group": float(co_freq[same_group & off & (both > 0)].mean()),
        "co_membership_mean_different_real_group": float(co_freq[~same_group & off & (both > 0)].mean()),
        "n_superdistricts_mean": float(np.mean([p["P2"].mean() for p in post.values()])),
    }
    grp_order = sorted(range(n_st), key=lambda i: (group_of_station[i], i))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(co_freq[np.ix_(grp_order, grp_order)], cmap="viridis", vmin=0, vmax=1)
    labels = [group_of_station[i] for i in grp_order]
    ticks = [k for k in range(n_st) if k == 0 or labels[k] != labels[k - 1]]
    ax.set_xticks(ticks); ax.set_xticklabels([labels[k] for k in ticks], rotation=90, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels([labels[k] for k in ticks], fontsize=7)
    for k in ticks[1:]:
        ax.axhline(k - 0.5, color="w", lw=0.4); ax.axvline(k - 0.5, color="w", lw=0.4)
    fig.colorbar(im, ax=ax, label="share of plans with both stations in one super-district")
    ax.set_title("Station co-membership; stations ordered by their real LAS Group")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_groups.png", dpi=140)
    plt.close(fig)

    # ---------------- 4. s_LAS against the ensemble ----------------
    minutes = inst["minutes"]
    slas_station_node = np.array([inst["station_node"].get(inst["lsoa_station"].get(inst["lsoa_of"][n]), -1)
                                  for n in node_ids])
    slas_t = np.array([minutes(int(s), n) if s >= 0 else np.nan for s, n in zip(slas_station_node, node_ids)])
    ok = ~np.isnan(slas_t)
    E1_slas = float((demand[ok] * slas_t[ok]).sum())
    hq_t = []
    for n in node_ids:
        grp = inst["lsoa_group"].get(inst["lsoa_of"][n])
        hq = inst["hq_node"].get(str(grp).strip())
        hq_t.append(minutes(hq, n) if hq is not None else np.nan)
    hq_t = np.array(hq_t)
    ok2 = ~np.isnan(hq_t)
    E2_slas = float((demand[ok2] * hq_t[ok2]).sum())
    slas_ems = ems_metrics(slas_t[ok], demand[ok])
    slas_load = Counter()
    for s, n in zip(slas_station_node, node_ids):
        if s >= 0:
            slas_load[int(s)] += demand[idx[n]]
    W = runs[0]["summary"]["calibration"]["w"]
    slas_units = {inst["lsoa_station"].get(inst["lsoa_of"][s]): v / W for s, v in slas_load.items()}
    # ensemble drive-time metrics per snapshot (level-1 leg)
    tt_matrix = np.full((n_st, len(node_ids)), np.nan)
    for i, sn in enumerate(station_nodes):
        for j, n in enumerate(node_ids):
            tt_matrix[i, j] = minutes(sn, n)
    ens = defaultdict(list)
    for k in range(n_snap):
        t = tt_matrix[station[k], np.arange(len(node_ids))]
        m = ems_metrics(t, demand)
        for key, v in m.items():
            ens[key].append(v)
    E_pooled = np.concatenate([p["E"].to_numpy(dtype=float) for p in post.values()])
    E1_pooled = np.concatenate([p["E1"].to_numpy(dtype=float) for p in post.values()])
    def pct(x, sample):
        return float((np.asarray(sample) <= x).mean())
    out["s_LAS"] = {
        "E": E1_slas + E2_slas, "E1": E1_slas, "E2": E2_slas,
        "percentile_of_E_in_ensemble": pct(E1_slas + E2_slas, E_pooled),
        "percentile_of_E1_in_ensemble": pct(E1_slas, E1_pooled),
        "ems": slas_ems,
        "ems_percentile_in_ensemble": {k: pct(v, ens[k]) for k, v in slas_ems.items()},
        "station_load_in_units_range": [float(min(slas_units.values())), float(max(slas_units.values()))],
        "stations_outside_unit_window_share": float(np.mean([
            not any(abs(u - c) <= 0.15 * c for c in (1, 2, 3)) for u in slas_units.values()])),
    }
    out["ensemble_ems"] = {k: {"mean": float(np.mean(v)), "q05": float(np.quantile(v, 0.05)),
                               "q95": float(np.quantile(v, 0.95))} for k, v in ens.items()}
    out["ensemble_E"] = {"mean": float(E_pooled.mean()), "q05": float(np.quantile(E_pooled, 0.05)),
                         "q95": float(np.quantile(E_pooled, 0.95)), "min": float(E_pooled.min())}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    panels = [("E", E_pooled, E1_slas + E2_slas, "energy E(s)"),
              ("T_mean", np.array(ens["T_mean"]), slas_ems["T_mean"], "mean drive time [min]"),
              ("T90", np.array(ens["T90"]), slas_ems["T90"], "90th pct drive time [min]"),
              ("C8", np.array(ens["C8"]), slas_ems["C8"], "8-minute coverage")]
    for ax, (key, sample, val, label) in zip(axes, panels):
        ax.hist(sample, bins=40, color="#9ecae1")
        ax.axvline(val, color="#d6402a", lw=2, label="s_LAS")
        ax.set_xlabel(label)
        ax.set_yticks([])
    axes[0].legend()
    fig.suptitle("The operational layout s_LAS against the real-station ensemble")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_slas.png", dpi=140)
    plt.close(fig)

    # ---------------- 5. maps ----------------
    if not a.no_maps:
        try:
            import geopandas as gpd
            from matplotlib.collections import LineCollection
            gdf = gpd.read_file(LSOA_GPKG)
            gdf = gdf[gdf.LSOA21CD.isin(set(inst["lsoa_of"].values()))].copy()
            gdf["contested"] = gdf.LSOA21CD.map({inst["lsoa_of"][n]: 1 - modal_share[i] for i, n in enumerate(node_ids)})
            fig, ax = plt.subplots(figsize=(9, 7))
            gdf.plot(column="contested", cmap="magma_r", vmin=0, vmax=1, ax=ax, linewidth=0,
                     legend=True, legend_kwds={"label": "1 - share of plans served by the modal station", "shrink": 0.6})
            sx = [inst["xy"][n][0] for n in station_nodes]; sy = [inst["xy"][n][1] for n in station_nodes]
            ax.scatter(sx, sy, s=12, c="cyan", edgecolors="k", linewidths=0.4, zorder=5)
            ax.set_axis_off(); ax.set_title("Contested LSOAs (stations in cyan)")
            fig.tight_layout(); fig.savefig(out_dir / "fig_contested.png", dpi=150); plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
            segs = np.array([[inst["xy"][node_ids[u]], inst["xy"][node_ids[v]]] for u, v in edges])
            for ax, f, title in [(axes[0], f1, "level-1 boundary frequency"), (axes[1], f2, "level-2 boundary frequency")]:
                gdf.plot(ax=ax, color="0.95", linewidth=0)
                order_e = np.argsort(f)
                lc = LineCollection(segs[order_e], cmap="magma_r", linewidths=0.3 + 1.5 * f[order_e])
                lc.set_array(f[order_e]); lc.set_clim(0, 1)
                ax.add_collection(lc)
                ax.set_axis_off(); ax.set_title(title)
            fig.colorbar(lc, ax=axes, shrink=0.6, label="share of plans in which the edge is cut")
            fig.savefig(out_dir / "fig_boundary_freq.png", dpi=150); plt.close(fig)
        except Exception as exc:   # maps are optional
            out["map_error"] = f"{type(exc).__name__}: {exc}"

    (out_dir / "ensemble_real_stations.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("acceptance_rate", "boundary_level1", "boundary_level2",
                                          "lsoa_modal_station_share", "groups", "s_LAS", "ensemble_E")},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
