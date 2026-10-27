"""
Print every number that Section 7.4 of the paper quotes, from the stored
outputs of a London ensemble directory (post-processed with
postprocess_real_stations.py and london_extras.py), with station names.

    python -m falcomchain_experiments.las.london_digest --in-dir data/derived/real_stations_u97 --steps 80000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(REPO / "data/derived/real_stations_u97"))
    ap.add_argument("--tag", default="real")
    ap.add_argument("--steps", type=int, default=80_000)
    a = ap.parse_args()
    d = Path(a.in_dir)
    ens = json.loads((d / "ensemble_real_stations.json").read_text())
    extras = json.loads((d / "london_extras.json").read_text()) if (d / "london_extras.json").exists() else {}
    names = dict(pd.read_csv(REPO / "data/raw/LAS_stations.csv")[["station_code", "station_name"]].values)
    sums = [json.loads(p.read_text()) for p in sorted(d.glob(f"summary_{a.tag}_s*_T{a.steps}.json"))]
    print(f"=== {d.name}: {len(sums)} chains x {a.steps} steps, burn-in {ens['burn_in']}, {ens['n_snapshots']} plans ===")
    print("units", [s.get("units") for s in sums], "initial capacity", [s.get("initial_total_capacity") for s in sums],
          "d_bar", round(sums[0].get("d_bar", 0), 1), "level-2 centre", sums[0].get("level2_target"))
    print("init attempts", [s["init_attempts"] for s in sums], "init s", [s["init_seconds"] for s in sums],
          "runtime min", [round(s["runtime_seconds"] / 60, 1) for s in sums], "steps/s", [s["steps_per_second"] for s in sums])
    print("acceptance", [s["acceptance_rate"] for s in sums])
    tot = {}
    for s in sums:
        for k, v in s.get("rejection_causes_detail", s["rejection_causes"]).items():
            tot.setdefault(k, []).append(v)
    for k, v in sorted(tot.items()):
        print(f"  cause {k}: per chain {v}  share {[round(x / a.steps, 4) for x in v]}")
    pb = ens["post_burn_in"]
    for m in ["E", "E1", "P1", "P2", "cut_edges"]:
        x = pb[m]
        print(f"{m}: chain means {[round(v) for v in x['mean_by_chain'].values()]}  sd {[round(v) for v in x['sd_by_chain'].values()]}  "
              f"pooled mean {round(x['pooled_mean'], 1)}  q05-q95 {[round(q, 1) for q in x['pooled_q05_q95']]}  KS max {round(x['ks_max'], 3)}  "
              f"halves KS {[round(v, 3) for v in x.get('ks_within_chain_halves_thinned', {}).values()]}  Rhat {round(x['split_rhat'], 4) if x['split_rhat'] else None}  "
              f"ESS {[round(v) for v in x['ess_by_chain'].values()]}")
    print("forgetting share at end", {k: round(v, 3) for k, v in ens["forgetting_share_at_end"].items()}, "boundary density", round(ens["boundary_density"], 3))
    print("boundary level1", ens["boundary_level1"]); print("boundary level2", ens["boundary_level2"])
    print("modal station share", ens["lsoa_modal_station_share"])
    st = ens["stations"]
    of = sorted(st["open_frequency_by_station"].items(), key=lambda kv: kv[1])
    print("open per plan mean", round(st["n_open_per_plan_mean"], 2), "range", st["n_open_per_plan_range"],
          "capacity share pooled", {k: round(v, 3) for k, v in st["capacity_share_pooled"].items()})
    print(">90%:", st["stations_open_in_over_90pct"], [names.get(c, c) for c, v in of if v > 0.9])
    print("<10%:", st["stations_open_in_under_10pct"])
    print("least open:", [(names.get(c, c), round(v, 2)) for c, v in of[:8]])
    print("most open:", [(names.get(c, c), round(v, 2)) for c, v in of[-6:]])
    print("median open frequency", round(float(np.median([v for _, v in of])), 3))
    g = ens["groups"]; print("groups:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()})
    sl = ens["s_LAS"]; print("s_LAS:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sl.items() if k not in ("ems", "ems_percentile_in_ensemble")})
    print("s_LAS ems", {k: round(v, 3) for k, v in sl["ems"].items()}, "percentiles", {k: round(v, 3) for k, v in sl["ems_percentile_in_ensemble"].items()})
    print("ensemble ems", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in ens["ensemble_ems"].items()})
    print("ensemble E", {k: round(v) for k, v in ens["ensemble_E"].items()})
    if extras:
        print("extras:", json.dumps({k: v for k, v in extras.items() if k not in ("chains", "in_dir")}, indent=None)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
