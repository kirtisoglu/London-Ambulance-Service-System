"""
Minimum candidate set that satisfies Assumption 6.1 — exact ILP.

Solves: min |F|  s.t.  every connected component C of G[V \ F]
has demand sum strictly less than (1 - eps) * w.

Approach: lazy CUT generation. Start with no subgraph constraints,
solve the LP/IP, find any violating connected component, add the
constraint sum_{v in C} y_v >= 1, repeat until no violation.

Usage::

    python3 find_min_cover.py 100 --eps 0.15 --w 2000
"""

import argparse
import json
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import networkx as nx


HERE = Path(__file__).resolve().parent


def solve_min_cover(graph: nx.Graph, *, threshold: float,
                    fixed_real: set | None = None,
                    warm_start: set | None = None,
                    time_limit: float = 300.0) -> dict:
    """Return the minimum candidate set; fixed_real (optional) are nodes
    that must be in F (e.g. existing real candidates); warm_start
    (optional) is a known-feasible set used as a MIP-start to give
    Gurobi an upper bound from the first node onward."""
    nodes = list(graph.nodes)
    demand = {n: graph.nodes[n]["demand"] for n in nodes}
    fixed_real = fixed_real or set()

    m = gp.Model("min_cover")
    m.Params.LazyConstraints = 1
    m.Params.MIPGap = 1e-9
    m.Params.TimeLimit = time_limit

    y = m.addVars(nodes, vtype=GRB.BINARY, name="y")
    for v in fixed_real:
        m.addConstr(y[v] == 1, f"fixed[{v}]")
    m.setObjective(gp.quicksum(y[v] for v in nodes), GRB.MINIMIZE)

    if warm_start:
        for v in nodes:
            y[v].Start = 1 if v in warm_start else 0
        print(f"  Warm-starting with |F| = {len(warm_start)}")

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        y_vals = model.cbGetSolution(y)
        F = {v for v in nodes if y_vals[v] > 0.5}
        sub = graph.subgraph(set(nodes) - F)
        for comp in nx.connected_components(sub):
            d_sum = sum(demand[v] for v in comp)
            if d_sum >= threshold:
                # Violating component: at least one of its nodes must be a candidate.
                model.cbLazy(gp.quicksum(y[v] for v in comp) >= 1)
                model._lazy += 1

    m._lazy = 0
    t0 = time.perf_counter()
    m.optimize(callback)
    dt = time.perf_counter() - t0

    if m.SolCount == 0:
        return {
            "status": int(m.Status), "wall_time_s": dt,
            "lazy_cuts": m._lazy, "F_size": None, "F": None,
            "mip_gap": None, "obj": None,
            "note": "No solution found within time limit.",
        }
    F_opt = sorted(v for v in nodes if y[v].X > 0.5)
    return {
        "status": int(m.Status),
        "wall_time_s": dt,
        "lazy_cuts": m._lazy,
        "F_size": len(F_opt),
        "F": F_opt,
        "mip_gap": float(m.MIPGap),
        "obj": float(m.ObjVal),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=int, default=100, nargs="?")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--w", type=float, default=2000)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--include-real", action="store_true",
                    help="Force the 5 real Bernoulli candidates into F.")
    ap.add_argument("--warm-start-existing", action="store_true",
                    help="Use the existing candidate set in the JSON as MIP-start.")
    args = ap.parse_args()

    inst_dir = HERE / "data"
    with open(inst_dir / f"grid_{args.instance}.json") as f:
        graph = nx.node_link_graph(json.load(f), edges="adjacency")
    all_existing = {v for v, d in graph.nodes(data=True) if d.get("candidate")}
    fixed = all_existing if args.include_real else set()
    warm = all_existing if args.warm_start_existing else None
    T = (1.0 - args.eps) * args.w
    print(f"=== Minimum candidate set for grid_{args.instance} ===")
    print(f"  |V|={graph.number_of_nodes()}, |E|={graph.number_of_edges()}")
    print(f"  threshold = (1-{args.eps})*{args.w} = {T}")
    print(f"  fixed real candidates: {len(fixed)} "
          f"({'forced in F' if fixed else 'none — solving from scratch'})")

    res = solve_min_cover(
        graph, threshold=T, fixed_real=fixed,
        warm_start=warm, time_limit=args.time_limit,
    )
    gap_str = f"{res['mip_gap']:g}" if res['mip_gap'] is not None else "—"
    print(f"\n  status: {res['status']}, gap: {gap_str}, "
          f"wall: {res['wall_time_s']:.2f}s, lazy cuts: {res['lazy_cuts']}")
    if res["F_size"] is None:
        print(f"  No solution found; {res.get('note', '')}")
    else:
        print(f"  Minimum |F| = {res['F_size']}")
        print(f"  F = {res['F']}")


if __name__ == "__main__":
    main()
