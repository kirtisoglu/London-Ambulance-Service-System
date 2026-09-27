"""
Repair the level-2 super_assignment in an existing FalCom initial-solution
JSON, without touching the level-1 districts.

Reads ``solution_{INSTANCE}_falcom_initial.json``, builds a deterministic
super-partition from the existing L1 partition via
:func:`deterministic_super.deterministic_super_partition`, replaces the
``x2 / y2 / c2`` MILP-format L2 fields with the new grouping, and re-renders
the plotly figure (HTML + PNG) via ``plot_falcom_solution.plot_falcom_solution``.

Use case: ``Partition.from_random_assignment(init_super_partition=True)``
can silently fall back to the identity grouping when the recursive
spanning-tree heuristic stalls on a small supergraph (e.g. the 8-supernode
supergraph for grid_400 at the initial seed). The identity grouping
violates mu^2 = 2, which makes the seed state infeasible w.r.t. the §3 MILP
formulation. This script repairs the saved solution to a mu^2-feasible
grouping deterministically.

Run::

    python3 repair_super_init.py 400      # repair solution_400_falcom_initial.json
    python3 repair_super_init.py 100      # likewise
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from deterministic_super import deterministic_super_partition  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=str)
    args = ap.parse_args()
    inst = args.instance

    sol_path = HERE / f"solution_{inst}_falcom_initial.json"
    meta = json.load(open(HERE / "data" / f"grid_{inst}.meta.json"))
    g = nx.node_link_graph(
        json.load(open(HERE / "data" / meta["grid_path"])), edges="adjacency")
    graph_nodes = g.nodes

    sol = json.load(open(sol_path))
    x1 = {int(n): int(d) for n, d in sol["x1"].items()}
    c1 = {int(d): int(round(t)) for d, t in sol["c1"].items() if t > 0.5}
    total_teams = sum(c1.values())
    print(f"L1: {len(c1)} districts, total teams {total_teams}")

    # Deterministic super-partition over the L1 districts.
    super_of_district = deterministic_super_partition(
        g, x1, c1,
        c_min_super=meta["c_min_l2"],
        c_max_super=meta["c_max_l2"],
        min_districts_super=meta["min_l1_per_l2"],
    )
    districts_in_super = {}
    for d, sid in super_of_district.items():
        districts_in_super.setdefault(sid, []).append(d)

    # For each super-district choose a super-facility (an L2 super-candidate
    # that lies in its node footprint). Deterministic: lowest-id candidate.
    super_centers = {}
    for sid, ds in sorted(districts_in_super.items()):
        nodes_in_super = sorted(n for n, dd in x1.items() if dd in ds)
        cands = [n for n in nodes_in_super
                 if graph_nodes[n].get("super_candidate")]
        if not cands:
            raise RuntimeError(
                f"super {sid} (districts {ds}) contains no super-candidate; "
                "cannot place an L2 facility — re-generate the instance with "
                "more L2 candidates.")
        super_centers[sid] = cands[0]

    # Rebuild the MILP-format L2 fields x2/y2/c2.
    x2 = {n: super_centers[super_of_district[x1[n]]] for n in x1}
    open_centers = set(super_centers.values())
    y2 = {n: (1 if n in open_centers else 0) for n in x1}
    c2 = {n: 0 for n in x1}
    for sid, sc in super_centers.items():
        c2[sc] = sum(c1[d] for d in districts_in_super[sid])

    sol["x2"] = {str(n): int(s) for n, s in x2.items()}
    sol["y2"] = {str(n): int(v) for n, v in y2.items()}
    sol["c2"] = {str(n): int(v) for n, v in c2.items()}
    sol["super_assignment_method"] = "deterministic_repair"

    # Recompute R^2 (level-2 demand-weighted access cost) with the new x2.
    demand = {n: g.nodes[n]["demand"] for n in g.nodes}
    def md(u, v):
        ux, uy = g.nodes[u]["C_X"], g.nodes[u]["C_Y"]
        vx, vy = g.nodes[v]["C_X"], g.nodes[v]["C_Y"]
        return abs(ux - vx) + abs(uy - vy)
    r2 = sum(demand[n] * md(n, x2[n]) for n in x1)
    if "R2" in sol:
        sol["R2"] = float(r2)
    if "median_l2_cost" in sol:
        sol["median_l2_cost"] = float(r2)
    if "obj_value" in sol and "R1" in sol:
        sol["obj_value"] = float(sol["R1"]) + float(r2)

    json.dump(sol, open(sol_path, "w"), indent=2)
    print(f"Wrote {sol_path}")
    for sid, ds in sorted(districts_in_super.items()):
        teams = sum(c1[d] for d in ds)
        print(f"  super {sid}: {len(ds)} districts {sorted(ds)}, "
              f"teams={teams}, centre={super_centers[sid]}")

    # Re-render the figure via plot_falcom_solution.
    import plot_falcom_solution as P
    fig = P.plot_falcom_solution(g, sol)
    out_html = HERE / "figures" / f"solution_{inst}_falcom_initial.html"
    out_html.parent.mkdir(exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    out_png = HERE / "figures" / f"solution_{inst}_falcom_initial.png"
    fig.write_image(str(out_png), width=900, height=850, scale=2)
    print(f"Wrote {out_html} and {out_png}")


if __name__ == "__main__":
    main()
