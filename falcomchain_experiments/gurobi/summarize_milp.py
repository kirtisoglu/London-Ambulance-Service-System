#!/usr/bin/env python3
"""Summarize the MILP-versus-FalCom comparison on the sparse grids (paper Section 7.3).

Reads, for each instance in {100, 400, 400_dense}, whichever of these exist in
this directory:

  solution_{inst}_median_shir.json     MILP optimum with SHIR contiguity
  solution_{inst}_median_noshir.json   MILP optimum without contiguity
  solution_{inst}_median.json          MILP optimum (unnamed contiguity mode)
  solution_{inst}_falcom_final.json    best state of one FalCom optimizer run
  solution_{inst}_falcom_multistart_summary.json  best over several seeds

and writes results_sparse/milp_summary.json plus a Markdown table on stdout.
Run it after the steps of RUNBOOK.md.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_sparse"
INSTANCES = ["100", "400", "400_dense"]


def load(name):
    p = HERE / name
    return json.load(open(p)) if p.exists() else None


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for inst in INSTANCES:
        meta_p = HERE / "data" / f"grid_{inst}.meta.json"
        meta = json.load(open(meta_p)) if meta_p.exists() else {}
        milp = {}
        for tag in ("shir", "noshir", ""):
            fname = f"solution_{inst}_median{('_' + tag) if tag else ''}.json"
            d = load(fname)
            if d:
                milp[tag or "default"] = {
                    "file": fname, "obj": d.get("obj_value"),
                    "l1_cost": d.get("median_l1_cost"), "l2_cost": d.get("median_l2_cost"),
                    "mip_gap": d.get("mip_gap"), "wall_time_s": d.get("wall_time_s"),
                    "n_open_l1": sum(1 for v in (d.get("y1") or {}).values() if v),
                    "n_open_l2": sum(1 for v in (d.get("y2") or {}).values() if v),
                    "params": d.get("params"),
                }
        fal = load(f"solution_{inst}_falcom_final.json")
        ms = load(f"solution_{inst}_falcom_multistart_summary.json")
        falcom = None
        if fal:
            falcom = {"obj": fal.get("obj_value"), "R1": fal.get("R1"), "R2": fal.get("R2"),
                      "wall_time_s": fal.get("wall_time_s"), "params": fal.get("params")}
        if ms:
            objs = [r.get("best_obj", r.get("obj_value")) for r in ms.get("results", [])]
            objs = [o for o in objs if o is not None]
            if objs:
                falcom = falcom or {}
                falcom["multistart_best"] = min(objs)
                falcom["multistart_median"] = sorted(objs)[len(objs) // 2]
                falcom["multistart_n_seeds"] = len(objs)
                falcom["multistart_total_wall_s"] = ms.get("total_wall_s")
        ref = (milp.get("shir") or milp.get("default") or {}).get("obj")
        best = None
        if falcom:
            cands = [falcom.get("obj"), falcom.get("multistart_best")]
            cands = [c for c in cands if c is not None]
            best = min(cands) if cands else None
        rows.append({
            "instance": inst,
            "n_nodes": meta.get("n_nodes"), "k_teams": meta.get("k_teams"),
            "n_l1_candidates": meta.get("n_l1_candidates"), "regime": meta.get("regime"),
            "milp": milp, "falcom": falcom,
            "falcom_over_milp_pct": (100.0 * (best - ref) / ref) if (best is not None and ref) else None,
        })
    json.dump({"rows": rows}, open(OUT / "milp_summary.json", "w"), indent=2)
    print("| instance | k | sites | MILP (SHIR) obj | gap | time s | MILP (no contig.) obj | FalCom best | FalCom vs MILP |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["milp"].get("shir") or r["milp"].get("default") or {}
        n = r["milp"].get("noshir") or {}
        f = r["falcom"] or {}
        fb = min([x for x in (f.get("obj"), f.get("multistart_best")) if x is not None], default=None)
        pct = r["falcom_over_milp_pct"]
        gap = "" if m.get("mip_gap") is None else "%.2f%%" % (100 * m["mip_gap"])
        secs = "" if m.get("wall_time_s") is None else "%.0f" % m["wall_time_s"]
        pcts = "" if pct is None else "%+.2f%%" % pct
        print(f"| grid_{r['instance']} | {r['k_teams']} | {r['n_l1_candidates']} | {m.get('obj')} | "
              f"{gap} | {secs} | {n.get('obj', '')} | {fb if fb is not None else ''} | {pcts} |")
    print(f"\nwrote {OUT / 'milp_summary.json'}")


if __name__ == "__main__":
    main()
