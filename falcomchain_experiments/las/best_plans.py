"""Companion readouts for the London case study (Section 7.4).

For a post-processed ensemble directory this script reports
  * the best sampled plans (lowest energy, lowest station-leg energy, best of each
    station-leg metric) and the sample extremes of the station-leg metrics, which
    bound the price of workload balance from above;
  * the distribution of the number of superdistricts, and the adjusted Rand index of
    the real Groups against the plans with exactly 21 superdistricts;
  * the strongest co-membership pairs within and across the real Groups, under both
    the conditional definition (among plans in which both stations are open) and the
    unconditional one (over all plans);
  * the LSOAs the road matrix cannot reach (the straight-line fallback);
  * per-chain values and Monte Carlo standard errors of the headline frequencies.

Usage: python -m falcomchain_experiments.las.best_plans --in-dir data/derived/real_stations_u97 --steps 80000
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from falcomchain_experiments.las.postprocess_real_stations import (
    LSOA_GROUP, STATIONS, adjusted_rand, ems_metrics, load_instance, load_runs)
from falcomchain_experiments.las.run_real_stations import REAL_TRAVEL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--tag", default="real")
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--burn-in", type=int, default=None)
    ap.add_argument("--demand-csv", default=None)
    a = ap.parse_args()
    in_dir = Path(a.in_dir)
    burn_in = a.burn_in if a.burn_in is not None else a.steps // 5

    runs = load_runs(in_dir, a.tag, a.steps)
    snap0 = runs[0]["snap"]
    node_ids = [int(x) for x in snap0["node_ids"]]
    idx = {n: i for i, n in enumerate(node_ids)}
    station_nodes = [int(x) for x in snap0["station_node_ids"]]
    n_st = len(station_nodes)
    edges = snap0["edges"]
    inst = load_instance(node_ids, a.demand_csv)
    demand = inst["demand"]
    minutes = inst["minutes"]
    st_idx_of_node = {n: i for i, n in enumerate(station_nodes)}
    code_of_idx = {st_idx_of_node[node]: code for code, node in inst["station_node"].items()
                   if node in st_idx_of_node}
    name_of_idx = {i: inst["station_name"].get(code_of_idx.get(i, ""), str(i)) for i in range(n_st)}
    st_own = np.array([idx[n] for n in station_nodes])
    group_of_station = {i: str(inst["lsoa_group"].get(inst["lsoa_of"][station_nodes[i]], "?")).strip()
                        for i in range(n_st)}

    tt = np.full((n_st, len(node_ids)), np.nan)
    for i, sn in enumerate(station_nodes):
        for j, n in enumerate(node_ids):
            tt[i, j] = minutes(sn, n)

    chains = []
    for r in runs:
        S = r["snap"]
        mask = S["step"] >= burn_in
        steps = S["step"][mask]
        t = r["trace"].drop_duplicates("step").set_index("step")
        chains.append(dict(seed=int(r["seed"]), step=steps, station=S["station"][mask], sup=S["super"][mask],
                           cap=S["capacity"][mask], E=t.loc[steps, "E"].to_numpy(float),
                           E1=t.loc[steps, "E1"].to_numpy(float)))
    station = np.concatenate([c["station"] for c in chains])
    sup = np.concatenate([c["sup"] for c in chains])
    cap = np.concatenate([c["cap"] for c in chains])
    E = np.concatenate([c["E"] for c in chains])
    E1 = np.concatenate([c["E1"] for c in chains])
    seed_of = np.concatenate([np.full(len(c["E"]), c["seed"]) for c in chains])
    step_of = np.concatenate([c["step"] for c in chains])
    chain_of = np.concatenate([np.full(len(c["E"]), k) for k, c in enumerate(chains)])
    n_snap = station.shape[0]
    open_mask = station[:, st_own] == np.arange(n_st)[None, :]
    n_open = open_mask.sum(axis=1)
    n_groups = np.array([len(np.unique(sup[k])) for k in range(n_snap)])

    ar = np.arange(len(node_ids))
    T = np.empty((n_snap, 4))
    E1_check = np.empty(n_snap)
    for k in range(n_snap):
        tk = tt[station[k], ar]
        m = ems_metrics(tk, demand)
        T[k] = [m["T_mean"], m["T90"], m["C8"], m["C15"]]
        E1_check[k] = float((demand * tk).sum())
    out = {"in_dir": str(in_dir), "steps": a.steps, "burn_in": burn_in, "n_plans": int(n_snap),
           "E1_trace_vs_recomputed_max_rel_diff": float(np.max(np.abs(E1_check - E1) / E1))}

    def plan_record(k: int) -> dict:
        return {"chain_seed": int(seed_of[k]), "step": int(step_of[k]), "E": float(E[k]), "E1": float(E1[k]),
                "E2": float(E[k] - E1[k]), "T_mean": float(T[k, 0]), "T90": float(T[k, 1]),
                "C8": float(T[k, 2]), "n_open": int(n_open[k]), "n_groups": int(n_groups[k]),
                "capacity_counts": {int(c): int(v) for c, v in
                                    Counter(cap[k, st_own[open_mask[k]]].tolist()).items()}}

    out["best_by_E"] = plan_record(int(np.argmin(E)))
    out["best_by_E1"] = plan_record(int(np.argmin(E1)))
    out["best_by_T_mean"] = plan_record(int(np.argmin(T[:, 0])))
    out["best_by_T90"] = plan_record(int(np.argmin(T[:, 1])))
    out["best_by_C8"] = plan_record(int(np.argmax(T[:, 2])))
    out["sample_extremes"] = {
        "E_min": float(E.min()), "E_q01": float(np.quantile(E, 0.01)),
        "E1_min": float(E1.min()), "E1_q01": float(np.quantile(E1, 0.01)),
        "T_mean_min": float(T[:, 0].min()), "T_mean_q01": float(np.quantile(T[:, 0], 0.01)),
        "T90_min": float(T[:, 1].min()), "C8_max": float(T[:, 2].max()), "C8_q99": float(np.quantile(T[:, 2], 0.99)),
        "T_mean_mean": float(T[:, 0].mean()), "T90_mean": float(T[:, 1].mean()), "C8_mean": float(T[:, 2].mean()),
        "corr_E_T_mean": float(np.corrcoef(E, T[:, 0])[0, 1]), "corr_E1_T_mean": float(np.corrcoef(E1, T[:, 0])[0, 1]),
        "corr_E_n_open": float(np.corrcoef(E, n_open)[0, 1]),
    }
    out["n_groups_distribution"] = {int(k): int(v) for k, v in sorted(Counter(n_groups.tolist()).items())}
    out["n_open_distribution"] = {int(k): int(v) for k, v in sorted(Counter(n_open.tolist()).items())}

    # adjusted Rand index against the real Groups, all plans and plans with exactly 21 superdistricts
    ari_all, ari21 = [], []
    for k in range(n_snap):
        op = np.where(open_mask[k])[0]
        if len(op) > 2:
            v = adjusted_rand(sup[k, st_own[op]], [group_of_station[i] for i in op])
            ari_all.append(v)
            if n_groups[k] == 21:
                ari21.append(v)
    out["ari"] = {"all_plans_mean": float(np.mean(ari_all)),
                  "all_plans_q05_q95": [float(np.quantile(ari_all, q)) for q in (0.05, 0.95)],
                  "all_plans_min_max": [float(min(ari_all)), float(max(ari_all))],
                  "plans_with_21_groups_n": len(ari21),
                  "plans_with_21_groups_mean": float(np.mean(ari21)) if ari21 else None}

    # co-membership pairs
    co = np.zeros((n_st, n_st))
    both = np.zeros((n_st, n_st))
    for k in range(n_snap):
        op = np.where(open_mask[k])[0]
        s_lab = sup[k, st_own[op]]
        eq = s_lab[:, None] == s_lab[None, :]
        co[np.ix_(op, op)] += eq
        both[np.ix_(op, op)] += 1
    co_cond = np.divide(co, both, out=np.zeros_like(co), where=both > 0)
    co_uncond = co / n_snap
    pairs = []
    for i in range(n_st):
        for j in range(i + 1, n_st):
            pairs.append({"conditional": float(co_cond[i, j]), "unconditional": float(co_uncond[i, j]),
                          "both_open_share": float(both[i, j] / n_snap),
                          "a": name_of_idx[i], "b": name_of_idx[j],
                          "group_a": group_of_station[i], "group_b": group_of_station[j],
                          "same_group": group_of_station[i] == group_of_station[j]})
    by_cond = sorted(pairs, key=lambda p: -p["conditional"])
    by_uncond = sorted(pairs, key=lambda p: -p["unconditional"])
    out["pairs"] = {
        "top_conditional_same_group": [p for p in by_cond if p["same_group"]][:12],
        "top_conditional_cross_group": [p for p in by_cond if not p["same_group"]][:12],
        "top_unconditional_same_group": [p for p in by_uncond if p["same_group"]][:12],
        "top_unconditional_cross_group": [p for p in by_uncond if not p["same_group"]][:12],
        "mean_conditional_same": float(np.mean([p["conditional"] for p in pairs if p["same_group"]])),
        "mean_conditional_cross": float(np.mean([p["conditional"] for p in pairs if not p["same_group"]])),
        "mean_unconditional_same": float(np.mean([p["unconditional"] for p in pairs if p["same_group"]])),
        "mean_unconditional_cross": float(np.mean([p["unconditional"] for p in pairs if not p["same_group"]])),
        "n_cross_group_pairs_conditional_above_0.3": int(sum(1 for p in pairs if not p["same_group"] and p["conditional"] > 0.3)),
        "n_same_group_pairs_conditional_above_0.3": int(sum(1 for p in pairs if p["same_group"] and p["conditional"] > 0.3)),
    }

    # the LSOAs the road matrix cannot reach
    df = pd.read_parquet(REAL_TRAVEL)
    st_codes = set(pd.read_csv(STATIONS)["LSOA21CD"])
    lsoas = set(inst["lsoa_of"].values())
    have = df[df["origin"].isin(st_codes)].groupby("dest")["origin"].nunique()
    lg = pd.read_csv(LSOA_GROUP)
    name_col = next((c for c in lg.columns if "NM" in c.upper() or "name" in c.lower()), None)
    lsoa_name = dict(zip(lg["LSOA21CD"], lg[name_col])) if name_col else {}
    host = {inst["lsoa_of"][n] for n in station_nodes}
    dem_of_lsoa = {inst["lsoa_of"][n]: float(demand[idx[n]]) for n in node_ids}
    missing = sorted(l for l in lsoas if int(have.get(l, 0)) < len(st_codes))
    out["fallback"] = {"n_lsoas": len(missing), "n_pairs": int(sum(len(st_codes) - int(have.get(l, 0)) for l in missing)),
                       "lsoas": [{"code": l, "name": lsoa_name.get(l), "stations_missing": int(len(st_codes) - int(have.get(l, 0))),
                                  "is_station_host": l in host, "demand": dem_of_lsoa.get(l)} for l in missing],
                       "share_of_demand": float(sum(dem_of_lsoa.get(l, 0) for l in missing) / demand.sum())}

    # per-chain readouts and Monte Carlo standard errors
    def readouts(rows):
        st, su, om, cp = station[rows], sup[rows], open_mask[rows], cap[rows]
        f1 = (st[:, edges[:, 0]] != st[:, edges[:, 1]]).mean(axis=0)
        f2 = (su[:, edges[:, 0]] != su[:, edges[:, 1]]).mean(axis=0)
        modal = np.array([np.unique(st[:, i], return_counts=True)[1].max() / len(st) for i in range(st.shape[1])])
        c = np.zeros((n_st, n_st)); b = np.zeros((n_st, n_st)); ar_ = []
        for k in range(len(st)):
            op = np.where(om[k])[0]
            s_lab = su[k, st_own[op]]
            c[np.ix_(op, op)] += s_lab[:, None] == s_lab[None, :]
            b[np.ix_(op, op)] += 1
            if len(op) > 2:
                ar_.append(adjusted_rand(s_lab, [group_of_station[i] for i in op]))
        cf = np.divide(c, b, out=np.zeros_like(c), where=b > 0)
        same = np.array([[group_of_station[i] == group_of_station[j] for j in range(n_st)] for i in range(n_st)])
        off = ~np.eye(n_st, dtype=bool)
        opened = cp[om.nonzero()[0], st_own[om.nonzero()[1]]]
        cnt = Counter(opened.tolist())
        tot = sum(cnt.values())
        return {"l1_interior_share": float((f1 < 0.1).mean()), "l1_max_cut_freq": float(f1.max()),
                "l2_interior_share": float((f2 < 0.1).mean()), "l2_max_cut_freq": float(f2.max()),
                "modal_share_mean": float(modal.mean()), "lsoas_modal_below_half": float((modal < 0.5).mean()),
                "lsoas_modal_above_0.9": float((modal > 0.9).mean()),
                "n_open_mean": float(om.sum(1).mean()), "n_groups_mean": float(n_groups[rows].mean()),
                "cap_share_1": cnt.get(1, 0) / tot, "cap_share_2": cnt.get(2, 0) / tot, "cap_share_3": cnt.get(3, 0) / tot,
                "co_same_group": float(cf[same & off & (b > 0)].mean()), "co_cross_group": float(cf[~same & off & (b > 0)].mean()),
                "ari_mean": float(np.mean(ar_)), "E_mean": float(E[rows].mean()), "E1_mean": float(E1[rows].mean()),
                "T_mean_mean": float(T[rows, 0].mean()), "C8_mean": float(T[rows, 2].mean())}
    per_chain = {int(c["seed"]): readouts(np.where(chain_of == k)[0]) for k, c in enumerate(chains)}
    keys = list(next(iter(per_chain.values())))
    vals = {key: np.array([per_chain[s][key] for s in per_chain]) for key in keys}
    out["per_chain_readouts"] = per_chain
    out["mc_standard_error"] = {key: {"mean_of_chains": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                                      "range": [float(v.min()), float(v.max())]} for key, v in vals.items()}

    path = in_dir / "best_plans.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("best_by_E", "best_by_E1", "best_by_T_mean", "best_by_C8", "sample_extremes",
                                          "n_groups_distribution", "ari", "fallback")}, indent=1))
    print("pairs:", json.dumps({k: v for k, v in out["pairs"].items() if not isinstance(v, list)}))
    for key in ("top_conditional_same_group", "top_conditional_cross_group", "top_unconditional_same_group", "top_unconditional_cross_group"):
        print(key)
        for p in out["pairs"][key][:8]:
            print(f"   {p['a']} & {p['b']}: cond {p['conditional']:.2f} uncond {p['unconditional']:.2f} both-open {p['both_open_share']:.2f} ({p['group_a']} / {p['group_b']})")
    print("MC SE:")
    for key, v in out["mc_standard_error"].items():
        print(f"   {key}: {v['mean_of_chains']:.4f} +- {v['se']:.4f}  range {v['range'][0]:.4f}-{v['range'][1]:.4f}")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
