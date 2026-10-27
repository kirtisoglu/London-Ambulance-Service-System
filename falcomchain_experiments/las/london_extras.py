"""
Extra London statistics for the paper's revision (run after run_real_stations):

- realized per-unit station loads in the sampled plans (range; share outside the
  nominal (1 +/- eps) window; share of plans with every station inside it);
- the effect of holding the state on rejection: readouts under the recorded
  (time) law against the jump-chain law that weights each visit once
  (weights 1 / visit duration), for the station open frequencies, the mean
  number of open stations and the capacity shares;
- the effective number of plans from the per-chain ESS of the energy and of
  the cut-edge count, and Monte Carlo standard errors of the open
  frequencies from the four chains.

    python -m falcomchain_experiments.las.london_extras --in-dir data/derived/real_stations_u97 --steps 80000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .postprocess_real_stations import ess

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / "data/raw/london_graph.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(REPO / "data/derived/real_stations_u97"))
    ap.add_argument("--tag", default="real")
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--burn-in", type=int, default=None)
    a = ap.parse_args()
    in_dir = Path(a.in_dir); burn = a.burn_in if a.burn_in is not None else a.steps // 5
    graph = json.loads(GRAPH.read_text()); info = {n["id"]: n for n in graph["nodes"]}
    out = {"in_dir": str(in_dir), "steps": a.steps, "burn_in": burn, "chains": {}}
    all_loads, open_freq_time, open_freq_jump, nopen_time, nopen_jump = [], [], [], [], []
    cap_time, cap_jump = [], []
    viol_station, viol_plan, n_station_plans, n_plans = 0, 0, 0, 0
    ess_E, ess_cut = [], []
    for summ in sorted(in_dir.glob(f"summary_{a.tag}_s*_T{a.steps}.json")):
        s = json.loads(summ.read_text()); stem = summ.name[len("summary_"):-len(".json")]
        trace = pd.read_csv(in_dir / f"trace_{stem}.csv")
        snap = np.load(in_dir / f"snap_{stem}.npz", allow_pickle=False)
        node_ids = snap["node_ids"]; demand = np.array([float(info[int(n)]["demand"]) for n in node_ids])
        units = s.get("units") or int(np.ceil(demand.sum() / s["calibration"]["w"]))
        d_bar = demand.sum() / units
        steps = snap["step"]; keep = steps >= burn
        st, cap = snap["station"][keep], snap["capacity"][keep]; st_steps = steps[keep]
        # visit durations from the trace: a visit is a maximal run of steps between accepted steps
        acc = trace["accepted"].to_numpy().astype(bool)
        visit_id = np.cumsum(acc)                       # accepted step opens a new visit
        durations = np.bincount(visit_id)
        w_jump = 1.0 / durations[visit_id[st_steps]]   # weight of the visit the snapshot belongs to
        w_jump = w_jump / w_jump.sum()
        n_st = int(snap["station_node_ids"].shape[0])
        open_t = np.zeros(n_st); open_j = np.zeros(n_st); capsh_t = np.zeros(4); capsh_j = np.zeros(4)
        nop_t, nop_j = 0.0, 0.0
        for k in range(st.shape[0]):
            loads = {}
            for si, c, dv in zip(st[k], cap[k], demand):
                if si < 0: continue
                loads.setdefault(int(si), [int(c), 0.0]); loads[int(si)][1] += dv
            opened = np.array(sorted(loads)); nop_t += len(opened) / st.shape[0]; nop_j += len(opened) * w_jump[k]
            open_t[opened] += 1.0 / st.shape[0]; open_j[opened] += w_jump[k]
            plan_ok = True
            for si, (c, dsum) in loads.items():
                per_unit = dsum / (c * d_bar); all_loads.append(per_unit)
                capsh_t[c] += 1.0 / st.shape[0]; capsh_j[c] += w_jump[k]
                n_station_plans += 1
                if abs(per_unit - 1.0) > a.eps: viol_station += 1; plan_ok = False
            n_plans += 1; viol_plan += (not plan_ok)
        open_freq_time.append(open_t); open_freq_jump.append(open_j); nopen_time.append(nop_t); nopen_jump.append(nop_j)
        cap_time.append(capsh_t / max(1e-12, capsh_t.sum())); cap_jump.append(capsh_j / max(1e-12, capsh_j.sum()))
        post = trace[trace["step"] >= burn]
        ess_E.append(ess(post["E"].to_numpy(dtype=float))); ess_cut.append(ess(post["cut_edges"].to_numpy(dtype=float)))
        out["chains"][s["seed"]] = {"n_plans": int(st.shape[0]), "mean_visit_duration": float(durations[1:].mean()),
                                    "mean_open_time_law": nop_t, "mean_open_jump_law": nop_j}
    loads = np.array(all_loads)
    oft, ofj = np.mean(open_freq_time, 0), np.mean(open_freq_jump, 0)
    out.update({
        "realized_per_unit_load_range": [float(loads.min()), float(loads.max())],
        "share_station_loads_outside_nominal_window": viol_station / max(1, n_station_plans),
        "share_plans_with_a_station_outside_nominal_window": viol_plan / max(1, n_plans),
        "jump_chain_reweighting": {
            "max_abs_change_open_frequency": float(np.max(np.abs(oft - ofj))),
            "mean_abs_change_open_frequency": float(np.mean(np.abs(oft - ofj))),
            "mean_open_stations_time_law": float(np.mean(nopen_time)),
            "mean_open_stations_jump_law": float(np.mean(nopen_jump)),
            "capacity_share_time_law": [float(x) for x in np.mean(cap_time, 0)[1:]],
            "capacity_share_jump_law": [float(x) for x in np.mean(cap_jump, 0)[1:]],
            "set_open_over_90pct_unchanged": bool(set(np.where(oft > 0.9)[0]) == set(np.where(ofj > 0.9)[0])),
        },
        "effective_plans": {"energy": float(sum(ess_E)), "cut_edges": float(sum(ess_cut)),
                            "ess_by_chain_energy": [float(x) for x in ess_E], "ess_by_chain_cut_edges": [float(x) for x in ess_cut]},
        "open_frequency_mc_se": {"max_over_stations": float(np.max(np.std(open_freq_time, 0, ddof=1) / np.sqrt(len(open_freq_time)))),
                                 "median_over_stations": float(np.median(np.std(open_freq_time, 0, ddof=1) / np.sqrt(len(open_freq_time))))},
        "stations_open_over_90pct": int((oft > 0.9).sum()),
    })
    (in_dir / "london_extras.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "chains"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
