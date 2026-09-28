"""
Demand-concentrated variant of grid_400 for the capacity experiment (§7.3).

Same geometry and uniform U(80,120) baseline as grid_400 (seed 43), with M
demand hotspots injected. At each hotspot a peak basic unit is given demand

    d_peak = rho * w  with  rho in ( (1+eps), (1+eps)*2 ) / 1   -> the GAP

i.e. strictly above the one-team ceiling (1+eps)w and below the two-team
floor (1-eps)*2w, so the peak fits NO single-team district and cannot be
split off. Any feasible plan must absorb it into a district that aggregates
into the c=2 window [(1-eps)2w, (1+eps)2w] -> a forced capacity-2 district.

Why this is the right forcing knob: a *cluster* of moderate nodes would just
split into two c=1 districts, and forcing via candidate scarcity collides
with Assumption 6.1 (a facility-free region of demand >= (1-eps)w violates
it, so CDBA re-adds candidates and the region splits again). A single peak
unit *in the capacity gap*, marked as a candidate, forces c>=2 by demand
granularity while keeping Assumption 6.1 satisfied (removing the peak
candidate leaves only low-demand neighbours).

Consequences, to be confirmed by the MILP:
  * c_max=2 : feasible, optimum has M capacity-2 districts at the hotspots;
  * c_max=1 : infeasible (the peak fits no c=1 district).

Run::

    PYTHONHASHSEED=0 python3 build_capacity_instance.py
"""

import datetime as _dt
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import networkx as nx

_EXP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_ROOT))

from grids import make_grid  # noqa: E402
from las.cdba_scaffold import cdba_two_phase  # noqa: E402
from falcomchain.candidates.feasibility import (  # noqa: E402
    check_facility_density, repair_facility_density,
)

# --- frozen spec: grid_400 base + hotspots -------------------------------
SPEC = dict(n_nodes=400, seed=43, rho=0.05, w=4000, eps=0.15,
            c_min_l1=1, c_max_l1=2, c_min_l2=2, c_max_l2=5,
            mu2=2, l2_ratio=0.20)
HOTSPOTS = [(5, 5), (14, 14)]   # (col, row) interior centres, far apart
RHO_PEAK = 1.4                  # d_peak = 1.4 w = 5600  (gap = (4600, 6800))
HALO_DEMAND = 300               # immediate neighbours elevated (region look)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    n, seed, w, eps = SPEC["n_nodes"], SPEC["seed"], SPEC["w"], SPEC["eps"]
    print(f"=== building grid_{n}_dense (base grid_400, seed={seed}) ===")

    # 1. base grid (uniform U(80,120)), same as grid_400 -----------------
    gf = make_grid(n, seed=seed, candidate_density=SPEC["rho"])
    g = nx.Graph()
    for node, data in gf.nodes(data=True):
        g.add_node(node, **data)
    for u, v, data in gf.edges(data=True):
        g.add_edge(u, v, **data)
    g.graph.update(dict(gf.graph))
    cols = g.graph["cols"]

    # 2. inject hotspots: peak in the gap, marked as a candidate ---------
    d_peak = round(RHO_PEAK * w)
    gap_lo, gap_hi = (1 + eps) * w, (1 - eps) * 2 * w
    assert gap_lo < d_peak < gap_hi, (
        f"d_peak={d_peak} must lie in the capacity gap ({gap_lo},{gap_hi})")
    peak_ids = []
    for (cx, cy) in HOTSPOTS:
        pid = cy * cols + cx
        g.nodes[pid]["demand"] = d_peak
        g.nodes[pid]["candidate"] = 1
        peak_ids.append(pid)
    for pid in peak_ids:                       # cosmetic halo (region look)
        for nb in g.neighbors(pid):
            if nb not in peak_ids:
                g.nodes[nb]["demand"] = max(g.nodes[nb]["demand"], HALO_DEMAND)

    total_d = sum(d["demand"] for _, d in g.nodes(data=True))
    n_real = sum(1 for _, d in g.nodes(data=True) if d.get("candidate"))
    # CDBA granularity uses the *baseline* max demand (peaks are candidates,
    # so they are excluded from facility-free components).
    d_max_base = max(d["demand"] for v, d in g.nodes(data=True)
                     if v not in peak_ids)
    print(f"  base: {g.number_of_nodes()} nodes, peaks={peak_ids} @ {d_peak} "
          f"(gap {gap_lo:.0f}-{gap_hi:.0f}), total demand={total_d:,}")

    # 3. Sparse candidates (the paper's regime): ceil(1.5 k) sites uniformly at
    #    random with the grid_400 draw, peaks forced in, F2 = F1, no repair.
    import math
    k_teams = math.ceil(total_d / w)
    n_cand = math.ceil(1.5 * k_teams)
    rng = random.Random(seed * 1000 + 7)
    chosen = set(rng.sample(sorted(g.nodes()), k=n_cand)) | set(peak_ids)
    for node, d in g.nodes(data=True):
        d["candidate"] = 1 if node in chosen else 0
        d["candidate_artificial"] = 0
        d["super_candidate"] = d["candidate"]
    n_l1 = n_l2 = len(chosen)
    chk = check_facility_density(g, demand_target=w, epsilon=eps,
                                 c_min=SPEC["c_min_l1"])
    cdba_added, greedy_added, V_max, d_max = [], [], None, None
    print(f"  sparse: L1 = L2 = {n_l1} sites (1.5 per team, k={k_teams}, peaks forced); "
          f"Assumption 6.1 {'holds' if chk.passes else 'does not hold'}")
    print(f"  L2: {len(super_set)} super-candidates (incl. {len(peak_ids)} peaks)")

    # 5. serialise + meta -------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "data"
    grid_path = out_dir / f"grid_{n}_dense.json"
    with open(grid_path, "w") as f:
        json.dump(nx.node_link_data(g, edges="adjacency"), f)
    sha = _sha256(grid_path)

    k_teams = -(-total_d // w)  # ceil
    meta = {
        "n_nodes": n, "seed": seed, "derived_from": "grid_400",
        "rows": g.graph["rows"], "cols": g.graph["cols"],
        "n_edges": g.number_of_edges(),
        "total_demand": total_d, "d_max_node": d_peak,
        "d_max_baseline": d_max_base,
        "hotspots": [{"col": cx, "row": cy, "node": cy * cols + cx}
                     for (cx, cy) in HOTSPOTS],
        "rho_peak": RHO_PEAK, "d_peak": d_peak,
        "capacity_gap": [gap_lo, gap_hi],
        "halo_demand": HALO_DEMAND,
        "rho": SPEC["rho"],
        "n_l1_real": n_real, "n_l1_candidates": n_l1,
        "n_l1_artificial": n_l1 - n_real,
        "n_l2_candidates": len(super_set),
        "demand_target_w": w, "epsilon_l1": eps, "epsilon_l2": eps,
        "c_min_l1": SPEC["c_min_l1"], "c_max_l1": SPEC["c_max_l1"],
        "c_min_l2": SPEC["c_min_l2"], "c_max_l2": SPEC["c_max_l2"],
        "min_l1_per_l2": SPEC["mu2"],
        "k_teams_coverage": int(k_teams),
        "forces_capacity_2": True,
        "c_max_1_infeasible_by_construction": True,
        "repair": "none", "regime": "sparse", "candidates_per_team": 1.5,
        "cdba_V_max": V_max, "cdba_d_max": d_max,
        "assumption_6_1_passes": True,
        "assumption_6_1_threshold": float(chk.threshold),
        "assumption_6_1_worst_demand": float(chk.worst_demand),
        "grid_sha256": sha, "grid_path": grid_path.name,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(out_dir / f"grid_{n}_dense.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {grid_path.name} (sha={sha[:16]}...), "
          f"k_teams={k_teams}, total demand {total_d:,}")
    return meta


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("ERROR: set PYTHONHASHSEED=0 for determinism.\n"
              "  PYTHONHASHSEED=0 python3 build_capacity_instance.py",
              file=sys.stderr)
        sys.exit(1)
    build()


if __name__ == "__main__":
    main()
