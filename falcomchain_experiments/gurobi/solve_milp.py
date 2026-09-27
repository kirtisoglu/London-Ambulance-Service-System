"""
Solve the §3 MILP of falcom.tex on a Gurobi backend with lazy CUT
contiguity (Validi, Buchanan, Lykhovyd 2022).

Model summary (paper §3):
- Two hierarchy levels L = {1, 2}.
- Decision vars
    x^l_{ij}  in {0,1}   assignment of base unit i to facility j at level l
    y^l_j     in {0,1}   facility j open at level l
    c^l_j     in [0..c_max^l]  capacity (#teams) of facility j at level l
- Constraints (in code names):
    assignment   sum_j x^l_{ij} = 1
    hierarchy    x^l_{ij} + x^{l-1}_{ij'} <= 1 + x^l_{j'j}     l >= 2
    contiguity   x^1_{ij} <= sum_{c in C} x^1_{cj}             l = 1, lazy
    cap_link     y^l_j <= c^l_j <= c_max^l * y^l_j
    coverage     sum_j c^l_j <= ceil(d / w)
    balance      (1 - eps^l) w c^l_j  <=  sum_i d_i x^l_{ij}  <=  (1 + eps^l) w c^l_j
- Objective: dummy 0 (paper §3 is feasibility-only; pass `--obj radius`
  to switch to demand-weighted Manhattan).
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import networkx as nx


HERE = Path(__file__).resolve().parent


def load_instance(path: Path):
    with open(path) as f:
        d = json.load(f)
    g = nx.node_link_graph(d, edges="adjacency")
    return g, d.get("graph", {})


def boundary_of_component(graph: nx.Graph, component: set) -> set:
    """All graph nodes adjacent to `component` but not in it."""
    boundary = set()
    for v in component:
        for u in graph.neighbors(v):
            if u not in component:
                boundary.add(u)
    return boundary


def add_shir_contiguity(m, graph, F1, x1, y1, *, big_M=None):
    """Single-commodity flow contiguity at level 1 (Shirabe / Validi 2022).

    For each candidate facility j in F1 and each undirected edge (u, v) in
    the base graph, introduce two non-negative flow variables f^j_{uv} and
    f^j_{vu}. Conservation requires every node assigned to j (with j as the
    source) to receive exactly one unit of flow. Capacity ties flow to the
    assignment: f^j_{uv} <= big_M * x^1_{vj}, so flow into v is only
    permitted if v is itself in district j.

    Adds O(|F^1| * |E|) variables and O(|F^1| * (|V| + |E|)) constraints.
    """
    nodes = list(graph.nodes)
    edges_undirected = list(graph.edges)
    if big_M is None:
        big_M = len(nodes)

    f = {}
    for j in F1:
        for (u, v) in edges_undirected:
            f[j, u, v] = m.addVar(lb=0, ub=big_M, name=f"f[{j},{u},{v}]")
            f[j, v, u] = m.addVar(lb=0, ub=big_M, name=f"f[{j},{v},{u}]")
    m.update()

    # Capacity: f^j_{uv} <= big_M * x^1_{vj}. (Flow into v allowed only
    # when v is in district j.)
    for j in F1:
        for (u, v) in edges_undirected:
            m.addConstr(f[j, u, v] <= big_M * x1[v, j], f"shir_cap[{j},{u},{v}]")
            m.addConstr(f[j, v, u] <= big_M * x1[u, j], f"shir_cap[{j},{v},{u}]")

    # Conservation: for every non-source node u, in - out = x^1_{uj}.
    # For the source j itself: out - in = sum_{u != j} x^1_{uj} = (#assigned - x_{jj}).
    # We enforce the per-node form and let Gurobi imply the source balance.
    for j in F1:
        for u in nodes:
            in_flow = gp.quicksum(f[j, v, u] for v in graph.neighbors(u))
            out_flow = gp.quicksum(f[j, u, v] for v in graph.neighbors(u))
            if u == j:
                # source: net out = (total assigned to j) - x_{jj}.
                # With anchoring x_{jj} = y_j, that's (assigned - y_j).
                total_assigned = gp.quicksum(x1[i, j] for i in nodes)
                m.addConstr(
                    out_flow - in_flow == total_assigned - x1[j, j],
                    f"shir_src[{j}]",
                )
            else:
                m.addConstr(
                    in_flow - out_flow == x1[u, j],
                    f"shir_cons[{j},{u}]",
                )


def build_model(graph: nx.Graph, *, w: int, demand_target: int,
                eps_l1: float, eps_l2: float,
                c_max_l1: int, c_max_l2: int,
                c_min_l1: int = 1, c_min_l2: int = 1,
                min_l1_per_l2: int = 1,
                budget_l1: int = None,
                objective: str,
                contiguity: str = "shir",
                threads: int = None):
    nodes = list(graph.nodes)
    d = {i: graph.nodes[i]["demand"] for i in nodes}
    coords = {i: (graph.nodes[i]["C_X"], graph.nodes[i]["C_Y"]) for i in nodes}
    F1 = sorted(i for i in nodes if graph.nodes[i].get("candidate"))
    F2 = sorted(i for i in nodes if graph.nodes[i].get("super_candidate"))
    total_d = sum(d.values())
    print(
        f"  |V|={len(nodes)}, |E|={graph.number_of_edges()}, "
        f"|F^1|={len(F1)}, |F^2|={len(F2)}, total demand={total_d:,}"
    )

    # Per paper §3: F^l are pairwise disjoint. The synthetic grids have
    # F^2 ⊂ F^1 — we relax this by making F^2 disjoint via removal
    # of F^2 nodes from F^1 for the MILP. (Not changing the JSON.)
    # Equivalently, allow overlap but the assignment vars are independent
    # per level, so disjointness is purely a paper convention.
    # We keep F^1 as-is (with F^2 inside it) since the assignment vars
    # are independent. The hierarchy constraint links them properly.

    m = gp.Model("falcom_milp")
    m.Params.LogToConsole = 1
    if contiguity == "cut":
        m.Params.LazyConstraints = 1
    m.Params.MIPGap = 1e-6
    m.Params.Seed = 0          # pin RNG for reproducibility
    if threads is not None:
        m.Params.Threads = threads   # Threads=1 => bit-reproducible

    # ---- vars ---------------------------------------------------------
    x1 = m.addVars(nodes, F1, vtype=GRB.BINARY, name="x1")
    x2 = m.addVars(nodes, F2, vtype=GRB.BINARY, name="x2")
    y1 = m.addVars(F1, vtype=GRB.BINARY, name="y1")
    y2 = m.addVars(F2, vtype=GRB.BINARY, name="y2")
    c1 = m.addVars(F1, lb=0, ub=c_max_l1, vtype=GRB.INTEGER, name="c1")
    c2 = m.addVars(F2, lb=0, ub=c_max_l2, vtype=GRB.INTEGER, name="c2")

    # ---- assignment (every node assigned to exactly one facility per level)
    for i in nodes:
        m.addConstr(gp.quicksum(x1[i, j] for j in F1) == 1, f"assign1[{i}]")
        m.addConstr(gp.quicksum(x2[i, j] for j in F2) == 1, f"assign2[{i}]")

    # ---- hierarchy: x^2_{ij} + x^1_{ij'} <= 1 + x^2_{j',j}
    # Reads: if base unit i is in L2 district j and also in L1 district j',
    #        then L1 facility j' must itself be assigned to L2 district j.
    for i in nodes:
        for j in F2:
            for jp in F1:
                m.addConstr(
                    x2[i, j] + x1[i, jp] <= 1 + x2[jp, j],
                    f"hier[{i},{j},{jp}]",
                )

    # ---- cap_link: c_min y <= c <= c_max y
    for j in F1:
        m.addConstr(c_min_l1 * y1[j] <= c1[j], f"capL1lo[{j}]")
        m.addConstr(c1[j] <= c_max_l1 * y1[j], f"capL1hi[{j}]")
    for j in F2:
        m.addConstr(c_min_l2 * y2[j] <= c2[j], f"capL2lo[{j}]")
        m.addConstr(c2[j] <= c_max_l2 * y2[j], f"capL2hi[{j}]")

    # ---- anchoring: an open facility serves its own unit
    # (paper §3 implicit; explicit cut accelerates Gurobi noticeably).
    for j in F1:
        m.addConstr(x1[j, j] == y1[j], f"anchor1[{j}]")
    for j in F2:
        m.addConstr(x2[j, j] == y2[j], f"anchor2[{j}]")

    # ---- min L1 districts per open L2 super-district
    # z[j1, j2] = 1  iff  L1 facility j1 is open AND its base unit j1
    # is assigned to L2 super-district j2 (linearised product).
    # Then for each open L2 super-district j2: sum_{j1} z[j1, j2] >= min_l1_per_l2.
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
                gp.quicksum(z[j1, j2] for j1 in F1)
                >= min_l1_per_l2 * y2[j2],
                f"minL1perL2[{j2}]",
            )

    # ---- coverage: sum c <= ceil(d / w)
    # Paper §3 writes "sum y_j c_j <= d/w". With sum c integer, treating
    # the RHS as floor(d/w) can collide with the demand-balance lower
    # bound ceil(d/((1+eps)w)) and create empty integer slack; using the
    # ceiling preserves coverage as a strict upper bound on the number
    # of teams while admitting integer-feasible (sum c) values.
    import math
    cap_ub = math.ceil(total_d / w)
    m.addConstr(gp.quicksum(c1[j] for j in F1) <= cap_ub, "coverage1")
    m.addConstr(gp.quicksum(c2[j] for j in F2) <= cap_ub, "coverage2")

    # ---- facility budget: sum_j y^1_j <= B (optional)
    # Caps the NUMBER of open level-1 facilities. With B < ceil(d/w) the
    # cap_ub teams cannot be spread one-per-facility, so some districts
    # must take c>=2 -- the realistic driver of capacity (limited stations),
    # compatible with Assumption 6.1 (candidates stay dense). B < cap_ub with
    # c_max_l1=1 is infeasible (cannot field cap_ub teams from <cap_ub
    # single-team facilities).
    if budget_l1 is not None:
        m.addConstr(gp.quicksum(y1[j] for j in F1) <= budget_l1, "budget1")

    # ---- balance: (1 - eps) w c <=  sum_i d_i x_{ij}  <= (1 + eps) w c
    for j in F1:
        lhs = gp.quicksum(d[i] * x1[i, j] for i in nodes)
        m.addConstr(lhs >= (1 - eps_l1) * w * c1[j], f"balL1lo[{j}]")
        m.addConstr(lhs <= (1 + eps_l1) * w * c1[j], f"balL1hi[{j}]")
    for j in F2:
        lhs = gp.quicksum(d[i] * x2[i, j] for i in nodes)
        m.addConstr(lhs >= (1 - eps_l2) * w * c2[j], f"balL2lo[{j}]")
        m.addConstr(lhs <= (1 + eps_l2) * w * c2[j], f"balL2hi[{j}]")

    # ---- objective ----------------------------------------------------
    if objective == "feasibility":
        m.setObjective(0, GRB.MINIMIZE)
    elif objective == "open":
        # Minimise number of open L1 facilities (a natural Validi-friendly
        # surrogate that picks a unique extremal point of the feasible set).
        m.setObjective(gp.quicksum(y1[j] for j in F1), GRB.MINIMIZE)
    elif objective == "median":
        # Disaggregated hierarchical median (successively-inclusive,
        # Narula 1984; Şahin & Süral 2007). Demand-weighted Manhattan
        # distance from every base unit to its L1 facility PLUS to its
        # L2 super-facility. Separable across levels: neither term pulls
        # an L1 facility toward an L2 facility, so each level's facility
        # centralises within its own (super)district. Linear in x.
        def md(i, j):
            return abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
        l1_cost = gp.quicksum(d[i] * md(i, j) * x1[i, j]
                              for i in nodes for j in F1)
        l2_cost = gp.quicksum(d[i] * md(i, j) * x2[i, j]
                              for i in nodes for j in F2)
        m.setObjective(l1_cost + l2_cost, GRB.MINIMIZE)
        m._median_l1 = l1_cost
        m._median_l2 = l2_cost
    elif objective in ("radius_sum", "radius_minmax"):
        # FalCom facility-assignment criterion at both levels.
        # Per paper §Facility Assignment:
        #   L1 r^1_j = max_{u in D^1_j} d(u, j)
        #   L2 r^2_j = max_{j1 in D^2_j, y^1_{j1}=1} d(j1, j)
        # Two objective shapes:
        #   radius_sum   = sum_j r^1_j + sum_j r^2_j  (total cost / p-median)
        #   radius_minmax = max_j r^1_j + max_j r^2_j (equity / p-center)
        # Per-district r1[j], r2[j2] are always created (lets us report
        # the per-district radii uniformly in the solution JSON).
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
                m.addConstr(R1 >= r1[j], f"R1ge_r1[{j}]")
            for j2 in F2:
                m.addConstr(R2 >= r2[j2], f"R2ge_r2[{j2}]")
            m.setObjective(R1 + R2, GRB.MINIMIZE)
            m._R1, m._R2 = R1, R2
        m._r1, m._r2 = r1, r2
    else:
        raise ValueError(f"unknown --obj {objective}")

    # ---- contiguity at level 1 -----------------------------------------
    if contiguity == "shir":
        print(f"  Adding SHIR contiguity (|F^1| x |E| x 2 = "
              f"{len(F1) * graph.number_of_edges() * 2:,} flow vars)")
        add_shir_contiguity(m, graph, F1, x1, y1)
    elif contiguity == "cut":
        print("  Using lazy CUT contiguity (separated by callback)")
    elif contiguity == "none":
        print("  WARNING: contiguity disabled")
    else:
        raise ValueError(f"unknown contiguity={contiguity}")

    m._x1 = x1
    m._x2 = x2
    m._y1 = y1
    m._y2 = y2
    m._c1 = c1
    m._c2 = c2
    m._F1 = F1
    m._F2 = F2
    m._nodes = nodes
    m._graph = graph
    m._lazy_cuts_added = 0
    return m


# ---------------------------------------------------------------------------
# Lazy CUT callback for level-1 contiguity
# ---------------------------------------------------------------------------
def lazy_contiguity_callback(model, where):
    if where != GRB.Callback.MIPSOL:
        return

    graph: nx.Graph = model._graph
    F1 = model._F1

    # Pull integer-incumbent values of x1 and y1.
    x_vals = model.cbGetSolution(model._x1)
    y_vals = model.cbGetSolution(model._y1)

    for j in F1:
        if y_vals[j] < 0.5:
            continue
        # S_j: base units assigned to j; must include j itself (since
        # F^1 ⊂ V^1 in the paper).
        S = {i for i in model._nodes if x_vals[i, j] > 0.5}
        if not S:
            continue
        if j not in S:
            # Pathological: y_j = 1 but x_jj = 0. Force x_jj on j's
            # location. (Not strictly required by §3 — facilities sit
            # at base units but the paper doesn't enforce x_jj = y_j.)
            continue

        # BFS from j in G^1[S].
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

        # Group unreached nodes by their connected component in G^1[S]
        # so we add one cut per disconnected piece.
        remaining = set(unreached)
        while remaining:
            seed = next(iter(remaining))
            comp = {seed}
            q = deque([seed])
            while q:
                u = q.popleft()
                for v in graph.neighbors(u):
                    if v in remaining and v not in comp:
                        comp.add(v)
                        q.append(v)
            remaining -= comp

            # Boundary of `comp` in G^1 — these are guaranteed not in S,
            # i.e., currently NOT assigned to j. They form an i–j vertex
            # separator for every i in comp.
            cut = boundary_of_component(graph, comp)
            if not cut:
                # comp is a connected component of G^1 itself — no cut
                # can separate. Falls through.
                continue

            # Validi-style fast cut: x^1_{ij} <= sum_{c in C} x^1_{cj}.
            # One inequality per i in comp.
            for i in comp:
                model.cbLazy(
                    model._x1[i, j]
                    <= gp.quicksum(model._x1[c, j] for c in cut)
                )
                model._lazy_cuts_added += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", choices=["100", "400", "400_dense"],
                    default="400", nargs="?")
    ap.add_argument(
        "--obj",
        choices=["feasibility", "open", "radius_sum", "radius_minmax",
                 "median"],
        default="median",
        help=("median = disaggregated hierarchical median (demand-weighted "
              "node->L1 + node->L2 distance); radius_minmax = equity "
              "(p-center); radius_sum = total radius (p-median over radii)."),
    )
    ap.add_argument("--c-max-l1", type=int, default=None,
                    help="Override c_max_l1 from meta.")
    ap.add_argument("--c-max-l2", type=int, default=None,
                    help="Override c_max_l2 from meta.")
    ap.add_argument("--c-min-l1", type=int, default=1,
                    help="Minimum teams when an L1 facility is open (default 1).")
    ap.add_argument("--c-min-l2", type=int, default=1,
                    help="Minimum teams when an L2 facility is open (default 1).")
    ap.add_argument("--min-l1-per-l2", type=int, default=1,
                    help="Minimum #open L1 districts inside each open L2 "
                         "super-district (default 1).")
    ap.add_argument("--budget-l1", type=int, default=None,
                    help="Max number of open L1 facilities (sum y1 <= B). "
                         "B < ceil(d/w) forces capacity-2 districts.")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--eps", type=float, default=None,
                    help="Override both eps_l1 and eps_l2 (default: meta).")
    ap.add_argument("--contiguity", choices=["shir", "cut", "none"],
                    default="shir")
    ap.add_argument("--warm-start", type=str, default=None,
                    help="Path to a solution JSON (e.g. a FalCom state) "
                         "to use as a Gurobi MIP-start.")
    ap.add_argument("--threads", type=int, default=None,
                    help="Gurobi thread count. Set 1 for bit-reproducible "
                         "runs (default: Gurobi auto / all cores).")
    args = ap.parse_args()

    inst_dir = HERE / "data"
    meta = json.load(open(inst_dir / f"grid_{args.instance}.meta.json"))
    graph, _ = load_instance(inst_dir / meta["grid_path"])

    eps_l1 = args.eps if args.eps is not None else meta["epsilon_l1"]
    eps_l2 = args.eps if args.eps is not None else meta["epsilon_l2"]
    c_max_l1 = args.c_max_l1 if args.c_max_l1 is not None else meta["c_max_l1"]
    c_max_l2 = args.c_max_l2 if args.c_max_l2 is not None else meta["c_max_l2"]
    print(f"=== Solving §3 MILP for grid_{args.instance}.json "
          f"(obj={args.obj}, eps_l1={eps_l1}, eps_l2={eps_l2}, "
          f"c_l1 in [{args.c_min_l1}, {c_max_l1}], "
          f"c_l2 in [{args.c_min_l2}, {c_max_l2}], "
          f"contiguity={args.contiguity}) ===")
    m = build_model(
        graph,
        w=meta["demand_target_w"],
        demand_target=meta["demand_target_w"],
        eps_l1=eps_l1,
        eps_l2=eps_l2,
        c_max_l1=c_max_l1,
        c_max_l2=c_max_l2,
        c_min_l1=args.c_min_l1,
        c_min_l2=args.c_min_l2,
        min_l1_per_l2=args.min_l1_per_l2,
        budget_l1=args.budget_l1,
        objective=args.obj,
        contiguity=args.contiguity,
        threads=args.threads,
    )
    m.Params.TimeLimit = args.time_limit

    # ---- MIP warm-start from a FalCom (or other) solution JSON --------
    if args.warm_start:
        with open(args.warm_start) as f:
            ws = json.load(f)
        ws_x1 = {int(k): int(v) for k, v in ws["x1"].items()}
        ws_y1 = {int(k): int(v) for k, v in ws["y1"].items()}
        ws_c1 = {int(k): int(v) for k, v in ws["c1"].items()}
        ws_x2 = {int(k): int(v) for k, v in ws["x2"].items()}
        ws_y2 = {int(k): int(v) for k, v in ws["y2"].items()}
        ws_c2 = {int(k): int(v) for k, v in ws["c2"].items()}
        n_set = 0
        for i in m._nodes:
            for j in m._F1:
                m._x1[i, j].Start = 1 if ws_x1.get(i) == j else 0
            for j in m._F2:
                m._x2[i, j].Start = 1 if ws_x2.get(i) == j else 0
        for j in m._F1:
            m._y1[j].Start = ws_y1.get(j, 0)
            m._c1[j].Start = ws_c1.get(j, 0)
            n_set += 1
        for j in m._F2:
            m._y2[j].Start = ws_y2.get(j, 0)
            m._c2[j].Start = ws_c2.get(j, 0)
        print(f"  MIP warm-start loaded from {args.warm_start} "
              f"(open L1={sum(ws_y1.values())}, open L2={sum(ws_y2.values())})")

    t0 = time.perf_counter()
    if args.contiguity == "cut":
        m.optimize(lazy_contiguity_callback)
    else:
        m.optimize()
    dt = time.perf_counter() - t0

    print(f"\nStatus: {m.Status}   wall: {dt:.2f}s   "
          f"lazy cuts added: {m._lazy_cuts_added}")
    if m.SolCount > 0:
        # Dump a JSON of the chosen assignment for downstream plotting.
        r1_vals = ({str(j): float(m._r1[j].X) for j in m._F1}
                   if hasattr(m, "_r1") else None)
        r2_vals = ({str(j): float(m._r2[j].X) for j in m._F2}
                   if hasattr(m, "_r2") else None)
        out = {
            "instance": meta["grid_path"],
            "objective": args.obj,
            "obj_value": float(m.ObjVal),
            "r1": r1_vals,
            "r2": r2_vals,
            "R1_sum": (sum(r1_vals.values()) if r1_vals else None),
            "R2_sum": (sum(r2_vals.values()) if r2_vals else None),
            "R1_max": (max(r1_vals.values()) if r1_vals else None),
            "R2_max": (max(r2_vals.values()) if r2_vals else None),
            "median_l1_cost": (float(m._median_l1.getValue())
                               if hasattr(m, "_median_l1") else None),
            "median_l2_cost": (float(m._median_l2.getValue())
                               if hasattr(m, "_median_l2") else None),
            "mip_gap": float(m.MIPGap),
            "wall_time_s": dt,
            "lazy_cuts_added": m._lazy_cuts_added,
            "params": {
                "w": meta["demand_target_w"],
                "eps_l1": eps_l1,
                "eps_l2": eps_l2,
                "c_min_l1": args.c_min_l1,
                "c_max_l1": c_max_l1,
                "c_min_l2": args.c_min_l2,
                "c_max_l2": c_max_l2,
                "min_l1_per_l2": args.min_l1_per_l2,
                "contiguity": args.contiguity,
            },
            "y1": {str(j): int(round(m._y1[j].X)) for j in m._F1},
            "c1": {str(j): int(round(m._c1[j].X)) for j in m._F1},
            "x1": {
                str(i): next(
                    j for j in m._F1 if m._x1[i, j].X > 0.5
                )
                for i in m._nodes
            },
            "y2": {str(j): int(round(m._y2[j].X)) for j in m._F2},
            "c2": {str(j): int(round(m._c2[j].X)) for j in m._F2},
            "x2": {
                str(i): next(
                    j for j in m._F2 if m._x2[i, j].X > 0.5
                )
                for i in m._nodes
            },
        }
        out_path = inst_dir.parent / f"solution_{args.instance}_{args.obj}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {out_path}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
