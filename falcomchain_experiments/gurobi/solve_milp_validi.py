"""
Validi-style lazy CUT contiguity for the §3 MILP.

Three improvements over the textbook lazy CUT in `solve_milp.py`:

1. **Minimum-cardinality separator.** When the lazy callback finds a
   disconnected component in G^1[S_j], compute the *minimum vertex
   cut* in the residual subgraph G^1[(V \ S_j) ∪ {i, j}] between the
   unreached node i and the facility j. This is the smallest set of
   nodes whose removal disconnects i from j among nodes that are
   *not currently assigned to j*. The resulting cut inequality is the
   tightest violated separator inequality and dominates the "boundary
   of unreached component" cut used in `solve_milp.py`.

2. **a-cuts (Fischetti-style preprocessing).** For every base node i
   and every candidate j, add the static lifted inequality
   `x^1_{i,j} ≤ sum_{v ∈ N_{G^1}(i)} x^1_{v,j}` for all i != j. This
   enforces "if i is in district j, at least one of i's neighbours
   in G^1 is also in district j" — a necessary but cheap implication
   of contiguity that tightens the LP relaxation upfront.

3. **Branching priority on `y` variables.** Force Gurobi to branch
   on the open/close decisions before the assignment decisions. Once
   `y` is fixed the contiguity callback has a stable facility set to
   check, dramatically reducing the number of distinct incumbents
   that need oracle checks.

These are the three concrete enhancements Validi, Buchanan & Lykhovyd
(Operations Research, 2022) report as the main drivers of CUT's
advantage over compact flow formulations on political-districting
instances.

Usage (mirrors solve_milp.py CLI)::

    python3 solve_milp_validi.py 100 --obj radius_minmax --eps 0.15 \\
        --c-max-l1 2 --c-max-l2 5 --c-min-l2 2 --min-l1-per-l2 2 \\
        --time-limit 600
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import networkx as nx

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Model build (mirrors solve_milp.py but without SHIR; contiguity is lazy)
# ---------------------------------------------------------------------------
def build_model(graph: nx.Graph, *, w: int, eps_l1: float, eps_l2: float,
                c_max_l1: int, c_max_l2: int,
                c_min_l1: int, c_min_l2: int,
                min_l1_per_l2: int,
                objective: str):
    nodes = list(graph.nodes)
    d = {i: graph.nodes[i]["demand"] for i in nodes}
    coords = {i: (graph.nodes[i]["C_X"], graph.nodes[i]["C_Y"]) for i in nodes}
    F1 = sorted(i for i in nodes if graph.nodes[i].get("candidate"))
    F2 = sorted(i for i in nodes if graph.nodes[i].get("super_candidate"))
    total_d = sum(d.values())
    print(f"  |V|={len(nodes)}, |E|={graph.number_of_edges()}, "
          f"|F^1|={len(F1)}, |F^2|={len(F2)}, total demand={total_d:,}")

    m = gp.Model("falcom_milp_validi")
    m.Params.LogToConsole = 1
    m.Params.LazyConstraints = 1
    m.Params.MIPGap = 1e-6

    # ---- vars ---------------------------------------------------------
    x1 = m.addVars(nodes, F1, vtype=GRB.BINARY, name="x1")
    x2 = m.addVars(nodes, F2, vtype=GRB.BINARY, name="x2")
    y1 = m.addVars(F1, vtype=GRB.BINARY, name="y1")
    y2 = m.addVars(F2, vtype=GRB.BINARY, name="y2")
    c1 = m.addVars(F1, lb=0, ub=c_max_l1, vtype=GRB.INTEGER, name="c1")
    c2 = m.addVars(F2, lb=0, ub=c_max_l2, vtype=GRB.INTEGER, name="c2")

    # ---- assignment ---------------------------------------------------
    for i in nodes:
        m.addConstr(gp.quicksum(x1[i, j] for j in F1) == 1, f"assign1[{i}]")
        m.addConstr(gp.quicksum(x2[i, j] for j in F2) == 1, f"assign2[{i}]")

    # ---- hierarchy ----------------------------------------------------
    for i in nodes:
        for j in F2:
            for jp in F1:
                m.addConstr(x2[i, j] + x1[i, jp] <= 1 + x2[jp, j],
                            f"hier[{i},{j},{jp}]")

    # ---- cap_link (with c_min) ----------------------------------------
    for j in F1:
        m.addConstr(c_min_l1 * y1[j] <= c1[j], f"capL1lo[{j}]")
        m.addConstr(c1[j] <= c_max_l1 * y1[j], f"capL1hi[{j}]")
    for j in F2:
        m.addConstr(c_min_l2 * y2[j] <= c2[j], f"capL2lo[{j}]")
        m.addConstr(c2[j] <= c_max_l2 * y2[j], f"capL2hi[{j}]")

    # ---- anchoring ----------------------------------------------------
    for j in F1:
        m.addConstr(x1[j, j] == y1[j], f"anchor1[{j}]")
    for j in F2:
        m.addConstr(x2[j, j] == y2[j], f"anchor2[{j}]")

    # ---- coverage (ceil) ----------------------------------------------
    cap_ub = math.ceil(total_d / w)
    m.addConstr(gp.quicksum(c1[j] for j in F1) <= cap_ub, "coverage1")
    m.addConstr(gp.quicksum(c2[j] for j in F2) <= cap_ub, "coverage2")

    # ---- balance ------------------------------------------------------
    for j in F1:
        lhs = gp.quicksum(d[i] * x1[i, j] for i in nodes)
        m.addConstr(lhs >= (1 - eps_l1) * w * c1[j], f"balL1lo[{j}]")
        m.addConstr(lhs <= (1 + eps_l1) * w * c1[j], f"balL1hi[{j}]")
    for j in F2:
        lhs = gp.quicksum(d[i] * x2[i, j] for i in nodes)
        m.addConstr(lhs >= (1 - eps_l2) * w * c2[j], f"balL2lo[{j}]")
        m.addConstr(lhs <= (1 + eps_l2) * w * c2[j], f"balL2hi[{j}]")

    # ---- min L1 per L2 (linearised product) ---------------------------
    if min_l1_per_l2 > 1:
        z = m.addVars(F1, F2, vtype=GRB.BINARY, name="z")
        for j1 in F1:
            for j2 in F2:
                m.addConstr(z[j1, j2] <= y1[j1], f"zUB_y[{j1},{j2}]")
                m.addConstr(z[j1, j2] <= x2[j1, j2], f"zUB_x[{j1},{j2}]")
                m.addConstr(z[j1, j2] >= y1[j1] + x2[j1, j2] - 1,
                            f"zLB[{j1},{j2}]")
        for j2 in F2:
            m.addConstr(
                gp.quicksum(z[j1, j2] for j1 in F1) >= min_l1_per_l2 * y2[j2],
                f"minL1perL2[{j2}]",
            )

    # ---- a-cuts (Validi preprocessing): lifted separator inequalities
    # x^1_{i,j} <= sum_{v in N(i)} x^1_{v,j}  for all i != j, j in F^1.
    # "If i is in district j, then some neighbour of i in G^1 is also
    # in district j." This is implied by contiguity but cheap to add
    # upfront — tightens the LP relaxation.
    n_acuts = 0
    for j in F1:
        for i in nodes:
            if i == j:
                continue
            nbrs = list(graph.neighbors(i))
            if not nbrs:
                continue
            m.addConstr(
                x1[i, j] <= gp.quicksum(x1[v, j] for v in nbrs),
                f"acut1[{i},{j}]",
            )
            n_acuts += 1
    print(f"  Added {n_acuts:,} a-cuts (Fischetti-style lifted separator).")

    # ---- branching priority: branch on y first --------------------------
    for j in F1:
        y1[j].BranchPriority = 10
    for j in F2:
        y2[j].BranchPriority = 10

    # ---- objective ----------------------------------------------------
    if objective == "feasibility":
        m.setObjective(0, GRB.MINIMIZE)
    elif objective == "open":
        m.setObjective(gp.quicksum(y1[j] for j in F1), GRB.MINIMIZE)
    elif objective in ("radius_sum", "radius_minmax"):
        # See solve_milp.py for full doc. radius_minmax = equity
        # (min max r1 + max r2). radius_sum = total cost (min sum).
        def md(i, j):
            return abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
        max_dist = max(md(i, j) for i in nodes for j in F1)
        r1 = m.addVars(F1, lb=0, ub=max_dist, name="r1")
        for j in F1:
            for i in nodes:
                d_ij = md(i, j)
                if d_ij == 0:
                    continue
                m.addConstr(r1[j] >= d_ij * x1[i, j], f"radL1[{j},{i}]")
        r2 = m.addVars(F2, lb=0, ub=max_dist, name="r2")
        M_big = max_dist
        for j2 in F2:
            for j1 in F1:
                d_12 = md(j1, j2)
                if d_12 == 0:
                    continue
                m.addConstr(
                    r2[j2] >= d_12 - M_big * (2 - y1[j1] - x2[j1, j2]),
                    f"radL2[{j2},{j1}]",
                )
        if objective == "radius_sum":
            m.setObjective(
                gp.quicksum(r1[j] for j in F1)
                + gp.quicksum(r2[j2] for j2 in F2),
                GRB.MINIMIZE,
            )
        else:  # radius_minmax
            R1 = m.addVar(lb=0, ub=max_dist, name="R1")
            R2 = m.addVar(lb=0, ub=max_dist, name="R2")
            for j in F1:
                m.addConstr(R1 >= r1[j])
            for j2 in F2:
                m.addConstr(R2 >= r2[j2])
            m.setObjective(R1 + R2, GRB.MINIMIZE)
        m._r1, m._r2 = r1, r2
    else:
        raise ValueError(objective)

    m._x1, m._x2, m._y1, m._y2, m._c1, m._c2 = x1, x2, y1, y2, c1, c2
    m._F1, m._F2 = F1, F2
    m._nodes = nodes
    m._graph = graph
    m._lazy_cuts_added = 0
    return m


# ---------------------------------------------------------------------------
# Lazy CUT callback using minimum vertex separator (Validi-style)
# ---------------------------------------------------------------------------
def lazy_min_cut_callback(model, where):
    if where != GRB.Callback.MIPSOL:
        return

    graph: nx.Graph = model._graph
    F1 = model._F1
    x_vals = model.cbGetSolution(model._x1)
    y_vals = model.cbGetSolution(model._y1)

    for j in F1:
        if y_vals[j] < 0.5:
            continue
        S = {i for i in model._nodes if x_vals[i, j] > 0.5}
        if j not in S:
            continue

        # BFS from j inside G^1[S]
        reached = {j}
        q = deque([j])
        while q:
            u = q.popleft()
            for v in graph.neighbors(u):
                if v in S and v not in reached:
                    reached.add(v)
                    q.append(v)

        unreached = S - reached
        if not unreached:
            continue

        # Pre-compute connected components of G^1[S] so we can group
        # unreached nodes and emit one boundary cut per component as a
        # fallback when the residual-subgraph min-cut is unavailable.
        S_sub = graph.subgraph(S)
        comp_of = {}
        for cc in nx.connected_components(S_sub):
            for n in cc:
                comp_of[n] = cc

        for i in unreached:
            # Try the Validi-style minimum vertex cut in the residual
            # subgraph H = G^1[(V \ S) ∪ {i, j}]. If H has an i–j path,
            # min_node_cut(H, i, j) returns the smallest set of nodes in
            # V \ S that separates them — the tightest violated cut.
            allowed = (set(graph.nodes) - S) | {i, j}
            H = graph.subgraph(allowed)
            C = None
            if nx.has_path(H, i, j):
                try:
                    C = set(nx.minimum_node_cut(H, s=i, t=j))
                except Exception:
                    C = None
            if not C:
                # Fallback: boundary of i's component in G^1[S]. Always a
                # valid (possibly not violated) separator. Never the
                # invalid "x = 0" cut, so feasibility is preserved.
                comp_i = comp_of.get(i, {i})
                C = set()
                for v in comp_i:
                    for u in graph.neighbors(v):
                        if u not in comp_i:
                            C.add(u)
                C.discard(j)
            if not C:
                continue
            model.cbLazy(
                model._x1[i, j] <= gp.quicksum(model._x1[c, j] for c in C)
            )
            model._lazy_cuts_added += 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", choices=["100", "400"], default="100", nargs="?")
    ap.add_argument("--obj", default="radius_minmax",
                    choices=["feasibility", "open", "radius_sum", "radius_minmax"])
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--c-max-l1", type=int, default=None)
    ap.add_argument("--c-max-l2", type=int, default=None)
    ap.add_argument("--c-min-l1", type=int, default=1)
    ap.add_argument("--c-min-l2", type=int, default=1)
    ap.add_argument("--min-l1-per-l2", type=int, default=1)
    args = ap.parse_args()

    inst_dir = HERE / "data"
    meta = json.load(open(inst_dir / f"grid_{args.instance}.meta.json"))
    with open(inst_dir / meta["grid_path"]) as f:
        graph = nx.node_link_graph(json.load(f), edges="adjacency")

    eps_l1 = args.eps if args.eps is not None else meta["epsilon_l1"]
    eps_l2 = args.eps if args.eps is not None else meta["epsilon_l2"]
    c_max_l1 = args.c_max_l1 if args.c_max_l1 is not None else meta["c_max_l1"]
    c_max_l2 = args.c_max_l2 if args.c_max_l2 is not None else meta["c_max_l2"]

    print(f"=== Validi CUT solve for grid_{args.instance}.json "
          f"(obj={args.obj}, eps={eps_l1}/{eps_l2}, "
          f"c1=[{args.c_min_l1},{c_max_l1}], c2=[{args.c_min_l2},{c_max_l2}], "
          f"min L1 per L2={args.min_l1_per_l2}) ===")
    m = build_model(
        graph, w=meta["demand_target_w"],
        eps_l1=eps_l1, eps_l2=eps_l2,
        c_max_l1=c_max_l1, c_max_l2=c_max_l2,
        c_min_l1=args.c_min_l1, c_min_l2=args.c_min_l2,
        min_l1_per_l2=args.min_l1_per_l2,
        objective=args.obj,
    )
    m.Params.TimeLimit = args.time_limit

    t0 = time.perf_counter()
    m.optimize(lazy_min_cut_callback)
    dt = time.perf_counter() - t0

    print(f"\nStatus: {m.Status}   wall: {dt:.2f}s   "
          f"lazy cuts added: {m._lazy_cuts_added}")
    if m.SolCount > 0:
        r1_vals = ({str(j): float(m._r1[j].X) for j in m._F1}
                   if hasattr(m, "_r1") else None)
        r2_vals = ({str(j): float(m._r2[j].X) for j in m._F2}
                   if hasattr(m, "_r2") else None)
        out = {
            "instance": meta["grid_path"],
            "method": "validi_lazy_cut",
            "objective": args.obj,
            "obj_value": float(m.ObjVal),
            "r1": r1_vals,
            "r2": r2_vals,
            "R1_sum": (sum(r1_vals.values()) if r1_vals else None),
            "R2_sum": (sum(r2_vals.values()) if r2_vals else None),
            "R1_max": (max(r1_vals.values()) if r1_vals else None),
            "R2_max": (max(r2_vals.values()) if r2_vals else None),
            "mip_gap": float(m.MIPGap),
            "wall_time_s": dt,
            "lazy_cuts_added": m._lazy_cuts_added,
            "params": {
                "w": meta["demand_target_w"], "eps_l1": eps_l1, "eps_l2": eps_l2,
                "c_min_l1": args.c_min_l1, "c_max_l1": c_max_l1,
                "c_min_l2": args.c_min_l2, "c_max_l2": c_max_l2,
                "min_l1_per_l2": args.min_l1_per_l2,
                "contiguity": "cut_validi",
            },
            "y1": {str(j): int(round(m._y1[j].X)) for j in m._F1},
            "c1": {str(j): int(round(m._c1[j].X)) for j in m._F1},
            "x1": {str(i): next(j for j in m._F1 if m._x1[i, j].X > 0.5)
                   for i in m._nodes},
            "y2": {str(j): int(round(m._y2[j].X)) for j in m._F2},
            "c2": {str(j): int(round(m._c2[j].X)) for j in m._F2},
            "x2": {str(i): next(j for j in m._F2 if m._x2[i, j].X > 0.5)
                   for i in m._nodes},
        }
        out_path = inst_dir.parent / f"solution_{args.instance}_{args.obj}_validi.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {out_path}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
