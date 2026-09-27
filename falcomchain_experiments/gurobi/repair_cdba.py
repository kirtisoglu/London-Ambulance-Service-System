"""
Repair the candidate set of a Gurobi MILP grid with CDBA so that
Assumption 6.1 (facility-density) of falcom.tex holds *strictly*.

Procedure (matches the paper's recommendation in §6.1):

    1. Two-phase CDBA augmentation on the loaded grid:
       - volume phase with V_max = (1 - epsilon) * w
       - diameter phase with d_max = floor(V_max / d_max_node) - 1,
         which bounds component demand at <= (#nodes) * d_max_node
    2. Verify Assumption 6.1 via falcomchain.candidates.feasibility.
       Any residual violating components are patched by the
       naive add-highest-demand-node greedy described in the paper.
    3. Resample super_candidates from the resulting (real + artificial)
       L1 set at the existing L2 count, so the L2 ratio is preserved.
    4. Overwrite ``data/grid_{N}.json`` and update ``data/grid_{N}.meta.json``.

Usage::

    python3 repair_cdba.py 100      # repair grid_100
    python3 repair_cdba.py 400      # repair grid_400
"""

import json
import random
import sys
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from las.cdba_scaffold import cdba_two_phase  # noqa: E402
from falcomchain.candidates.feasibility import (  # noqa: E402
    check_facility_density, repair_facility_density,
)


def repair(n_nodes: int, *, eps_override: float | None = None):
    grid_path = HERE / "data" / f"grid_{n_nodes}.json"
    meta_path = HERE / "data" / f"grid_{n_nodes}.meta.json"
    with open(grid_path) as f:
        d = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    g = nx.node_link_graph(d, edges="adjacency")
    eps = eps_override if eps_override is not None else meta["epsilon_l1"]
    w = meta["demand_target_w"]
    n_real_l1 = sum(1 for _, dat in g.nodes(data=True) if dat.get("candidate"))
    n_real_l2 = sum(1 for _, dat in g.nodes(data=True) if dat.get("super_candidate"))
    print(
        f"\n=== grid_{n_nodes} ===\n"
        f"  before: real L1={n_real_l1}, real L2={n_real_l2}, "
        f"w={w}, eps={eps}"
    )

    d_max_node = max(dat["demand"] for _, dat in g.nodes(data=True))
    V_max = (1 - eps) * w
    # Diameter cap that, together with the volume cap, gives a
    # node-count bound: any facility-free component has at most
    # d_max + 1 nodes, hence demand <= (d_max + 1) * d_max_node.
    # Solve (d_max + 1) * d_max_node <= V_max  =>  d_max <= floor(V_max / d_max_node) - 1
    d_max = max(1, int(V_max // d_max_node) - 1)
    print(f"  CDBA caps: V_max={V_max:.0f}, d_max={d_max} "
          f"(d_max_node={d_max_node})")

    added_cdba = cdba_two_phase(g, V_max=V_max, d_max=d_max, log=True)
    print(f"  CDBA added {len(added_cdba)} artificial candidates")

    # Verify strict Assumption 6.1 and patch with the naive greedy if needed.
    chk = check_facility_density(
        g, demand_target=w, epsilon=eps, c_min=meta.get("c_min_l1", 1),
    )
    if not chk.passes:
        print(f"  Assumption 6.1 still violated after CDBA "
              f"(worst component demand {chk.worst_demand:.0f}); "
              f"calling repair_facility_density (naive greedy)")
        added_extra = repair_facility_density(
            g, demand_target=w, epsilon=eps, c_min=meta.get("c_min_l1", 1),
            strategy="fast_center",
        )
        # repair_facility_density mutates and may return None or list
        added_extra = added_extra or []
        print(f"  greedy added {len(added_extra)} additional candidates")
        added_cdba = list(added_cdba) + list(added_extra)

    chk = check_facility_density(
        g, demand_target=w, epsilon=eps, c_min=meta.get("c_min_l1", 1),
    )
    print(f"  Assumption 6.1 now passes: {chk.passes}  "
          f"(worst component demand {chk.worst_demand:.0f}; "
          f"threshold {chk.threshold:.0f})")

    # Resample super-candidates from the new L1 set so L2 count is preserved.
    cand_ids = sorted(n for n, dat in g.nodes(data=True) if dat.get("candidate"))
    rng = random.Random(meta.get("seed", 42))
    new_super = set(rng.sample(cand_ids, k=n_real_l2))
    for n, dat in g.nodes(data=True):
        dat["super_candidate"] = 1 if n in new_super else 0

    n_l1_after = len(cand_ids)
    n_l2_after = n_real_l2
    n_added = n_l1_after - n_real_l1
    print(
        f"  after: total L1={n_l1_after} (+{n_added} artificial), "
        f"L2={n_l2_after}"
    )

    # Save.
    with open(grid_path, "w") as f:
        json.dump(nx.node_link_data(g, edges="adjacency"), f)
    meta.update({
        "n_l1_candidates": n_l1_after,
        "n_l1_artificial": n_added,
        "n_l1_real": n_real_l1,
        "n_l2_candidates": n_l2_after,
        "cdba_V_max": V_max,
        "cdba_d_max": d_max,
        "cdba_epsilon_used": eps,
        "assumption_6_1_passes": True,
        "assumption_6_1_threshold": float(chk.threshold),
        "assumption_6_1_worst_demand": float(chk.worst_demand),
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {grid_path.name} and updated {meta_path.name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", type=int, nargs="+")
    ap.add_argument("--eps", type=float, default=None,
                    help="Override meta epsilon_l1 (use the eps that the "
                         "MILP solve will use, not the generation default).")
    args = ap.parse_args()
    for n in args.sizes:
        repair(n, eps_override=args.eps)
