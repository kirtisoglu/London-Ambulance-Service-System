"""
Summarize the London optimizer runs (``run_real_stations.py --optimizer``):
the best state of every seed, its energy split and its station-leg metrics,
next to the operational layout and the sampled ensemble, so that the price of
workload balance can be bounded by an annealed plan rather than by a sample
minimum. Run from the repository root::

    python -m falcomchain_experiments.las.summarize_optimizer_runs \
        --in-dir data/derived/real_stations_u97_opt --steps 50000 \
        --ensemble data/derived/real_stations_u97/ensemble_real_stations.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .postprocess_real_stations import ems_metrics, load_instance

REPO = Path(__file__).resolve().parents[2]


def station_leg_metrics(snap, inst):
    """Drive-time metrics of a stored plan: every LSOA to its own station."""
    node_ids = snap["node_ids"]
    station_nodes = snap["station_node_ids"]
    st = snap["station"][0]
    times = np.array([inst["minutes"](int(station_nodes[s]), int(n)) for s, n in zip(st, node_ids)])
    return ems_metrics(times, inst["demand"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(REPO / "data/derived/real_stations_u97_opt"))
    ap.add_argument("--tag", default="opt")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--ensemble", default=None, help="ensemble JSON of the sampling runs, for reference")
    a = ap.parse_args()
    in_dir = Path(a.in_dir)
    rows = []
    inst = None
    for summ in sorted(in_dir.glob(f"summary_{a.tag}_s*_T{a.steps}.json")):
        s = json.loads(summ.read_text())
        stem = summ.name[len("summary_"):-len(".json")]
        best = np.load(in_dir / f"best_{stem}.npz", allow_pickle=False)
        if inst is None:
            inst = load_instance(best["node_ids"])
        m = station_leg_metrics(best, inst)
        open_stations = int(len(set(int(x) for x in best["station"][0])))
        caps = {}
        for st_i, c in zip(best["station"][0], best["capacity"][0]):
            caps[int(st_i)] = int(c)
        rows.append({"seed": s["seed"], "best_E": s["best_E"], "best_E1": s["best_E1"],
                     "best_E2": s["best_E"] - s["best_E1"], "best_step": s["best_step"],
                     "open_stations": open_stations, "total_capacity": int(sum(caps.values())),
                     "sa": s.get("sa"), "runtime_seconds": s["runtime_seconds"], **m})
    rows.sort(key=lambda r: r["best_E"])
    out = {"runs": rows,
           "best": rows[0] if rows else None,
           "median_best_E": float(np.median([r["best_E"] for r in rows])) if rows else None}
    if a.ensemble:
        ens = json.loads(Path(a.ensemble).read_text())
        out["ensemble_reference"] = {k: ens.get(k) for k in ("ensemble_E", "ensemble_ems", "s_las", "slas_ems", "s_las_metrics")
                                     if k in ens}
    (in_dir / f"optimizer_summary_T{a.steps}.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"seed {r['seed']:3d}  best E {r['best_E']:,.0f} (E1 {r['best_E1']:,.0f}, E2 {r['best_E2']:,.0f}) "
              f"at step {r['best_step']:6d}  open {r['open_stations']}  T_mean {r['T_mean']:.2f}  "
              f"T90 {r['T90']:.2f}  C8 {r['C8']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
