"""
Capacitated Diameter-Bounded Augmentation (CDBA) scaffolding for the
recursive bipartition step of FalCom's initial-state generation.

CDBA finds an augmentation set ``S \\subseteq V \\ F^1`` such that
every connected component ``C`` of ``G[V \\ (F^1 \\cup S)]`` satisfies:

  - graph diameter ``\\leq d_max``     (bounds tree-cut admissibility)
  - demand volume ``< V_max``         (Assumption 6.1)

with ``V_max = c_min * (1 - eps^1) * w``, where ``c_min`` is the
minimum capacity admitted at the base level.

Three-phase greedy heuristic:

  Phase 1 (volume cover): demand-bounded ball cover. Each candidate
  "owns" a BFS-grown ball whose accumulated demand stays at most
  ``V_max``. Add candidates greedily until every node is covered.

  Phase 2 (diameter): while some candidate-free component has
  approximate graph diameter ``> d_max``, place a candidate at the
  graph 1-center (midpoint of an approximate longest path, via two
  BFS sweeps).

  Phase 3 (strict demand cap): naive cleanup. While some
  candidate-free component still has demand ``\\geq V_max``, add the
  highest-demand node from that component. Guarantees Assumption 6.1.

Phases 1 and 2 reduce the work Phase 3 has to do; Phase 3 is the
correctness backbone for Assumption 6.1.

See Salazar-Aguilar et al. (2011) for compact-capacitated districting
and Cygan, Hajiaghayi & Khuller (2012) for the capacitated k-center
LP-rounding angle.
"""

from __future__ import annotations

from collections import deque

import networkx as nx


def _grow_ball(graph, source, demand_cap: float) -> set:
    """BFS-expand from ``source``, halting when the next layer would push
    the accumulated demand over ``demand_cap``. Returns the set of nodes
    inside the ball."""
    ball = {source}
    cum = graph.nodes[source]["demand"]
    frontier = [source]
    visited = {source}
    while frontier:
        next_frontier = []
        for v in frontier:
            for u in graph.neighbors(v):
                if u in visited:
                    continue
                if cum + graph.nodes[u]["demand"] > demand_cap:
                    visited.add(u)
                    continue
                visited.add(u)
                ball.add(u)
                cum += graph.nodes[u]["demand"]
                next_frontier.append(u)
        frontier = next_frontier
    return ball


def _ball_cover(graph, demand_cap: float) -> list:
    """Phase 1: greedy demand-bounded ball cover. Start from the existing
    real candidates' balls and add new candidates one at a time at the
    uncovered node whose ball covers the most uncovered nodes."""
    real_candidates = [
        v for v in graph.nodes if graph.nodes[v].get("candidate", 0) == 1
    ]
    covered = set()
    for f in real_candidates:
        covered |= _grow_ball(graph, f, demand_cap)

    all_nodes = set(graph.nodes)
    added = []
    while True:
        uncovered = all_nodes - covered
        if not uncovered:
            break
        # Sample-based greedy choice (bounded to keep iteration cheap)
        best_node, best_gain = next(iter(uncovered)), 0
        for cand in list(uncovered)[:100]:
            ball = _grow_ball(graph, cand, demand_cap)
            gain = len(ball & uncovered)
            if gain > best_gain:
                best_gain, best_node = gain, cand
        added.append(best_node)
        covered |= _grow_ball(graph, best_node, demand_cap)
    return added


def _bfs_farthest(graph, source, allowed_set):
    """BFS within ``allowed_set``. Returns (farthest_node, depth, parent_map)."""
    parent = {source: None}
    depth = {source: 0}
    queue = deque([source])
    farthest, far_d = source, 0
    while queue:
        v = queue.popleft()
        d = depth[v]
        for u in graph.neighbors(v):
            if u in allowed_set and u not in depth:
                depth[u] = d + 1
                parent[u] = v
                queue.append(u)
                if d + 1 > far_d:
                    far_d, farthest = d + 1, u
    return farthest, far_d, parent


def _two_bfs_diam_and_center(graph, component):
    """2-BFS approximation of (diameter, 1-center) on a graph component.

    Returns (approx_diameter, midpoint_node). The midpoint is the node
    halfway along an approximate longest path.
    """
    any_node = next(iter(component))
    u, _, _ = _bfs_farthest(graph, any_node, component)
    v, d_uv, parent_v = _bfs_farthest(graph, u, component)
    cur = v
    for _ in range(d_uv // 2):
        cur = parent_v[cur]
    return d_uv, cur


def cdba_two_phase(
    graph,
    *,
    V_max: float,
    d_max: int,
    mark_attr: str = "candidate_artificial",
    log: bool = False,
) -> list:
    """Run two-phase CDBA on ``graph`` and mark added nodes as candidates.

    Phases 1 (volume cover) and 2 (diameter cap) only. For the strict
    per-component demand cap that establishes Assumption 6.1, use
    :func:`cdba_three_phase`.

    Mutates the graph in place: sets ``graph.nodes[v]["candidate"] = 1``
    and ``graph.nodes[v][mark_attr] = 1`` for every added node.

    :returns: List of node IDs added (Phase 1 first, then Phase 2).
    """
    phase1 = _ball_cover(graph, V_max)
    for v in phase1:
        graph.nodes[v]["candidate"] = 1
        graph.nodes[v][mark_attr] = 1
    if log:
        print(f"  CDBA Phase 1 (volume cap {V_max:.0f}): +{len(phase1)} artificials")

    phase2 = []
    iter_count = 0
    while True:
        iter_count += 1
        ff_nodes = {
            v for v in graph.nodes if not graph.nodes[v].get("candidate", 0)
        }
        ff_sub = graph.subgraph(ff_nodes)
        worst_C, worst_d, worst_mid = None, 0, None
        for C in nx.connected_components(ff_sub):
            if len(C) < 2:
                continue
            d, mid = _two_bfs_diam_and_center(graph, C)
            if d > worst_d:
                worst_d, worst_C, worst_mid = d, C, mid
        if worst_d <= d_max:
            break
        graph.nodes[worst_mid]["candidate"] = 1
        graph.nodes[worst_mid][mark_attr] = 1
        phase2.append(worst_mid)
        if log and (iter_count <= 5 or iter_count % 100 == 0):
            print(f"    Phase 2 iter {iter_count}: worst diam={worst_d}, |C|={len(worst_C)}")
    if log:
        print(f"  CDBA Phase 2 (diameter cap {d_max}): +{len(phase2)} artificials")

    return phase1 + phase2


def _phase3_strict_demand_cap(
    graph,
    *,
    V_max: float,
    mark_attr: str = "candidate_artificial",
    log: bool = False,
) -> list:
    """Phase 3: naive strict per-component demand cap.

    While some candidate-free component has demand ``\\geq V_max``,
    promote the highest-demand node in the worst (largest-demand)
    component to a candidate. Terminates when every candidate-free
    component has demand ``< V_max``, establishing Assumption 6.1 at
    the chosen ``V_max = c_min * (1 - eps^1) * w``.
    """
    added = []
    while True:
        sub = graph.subgraph(
            v for v in graph.nodes if not graph.nodes[v].get("candidate", 0)
        )
        bad = []
        for C in nx.connected_components(sub):
            d = sum(graph.nodes[v]["demand"] for v in C)
            if d >= V_max:
                bad.append((C, d))
        if not bad:
            break
        worst_C, _ = max(bad, key=lambda cd: cd[1])
        node = max(worst_C, key=lambda v: graph.nodes[v]["demand"])
        graph.nodes[node]["candidate"] = 1
        graph.nodes[node][mark_attr] = 1
        added.append(node)
    if log:
        print(f"  CDBA Phase 3 (strict demand cap {V_max:.0f}): +{len(added)} artificials")
    return added


def cdba_three_phase(
    graph,
    *,
    V_max: float,
    d_max: int,
    mark_attr: str = "candidate_artificial",
    log: bool = False,
) -> list:
    """Run full three-phase CDBA: ball cover, diameter cap, strict
    demand cap. The third phase guarantees Assumption 6.1 at the chosen
    ``V_max = c_min * (1 - eps^1) * w``.

    :returns: List of node IDs added (Phase 1, then Phase 2, then Phase 3).
    """
    phase12 = cdba_two_phase(
        graph, V_max=V_max, d_max=d_max, mark_attr=mark_attr, log=log
    )
    phase3 = _phase3_strict_demand_cap(
        graph, V_max=V_max, mark_attr=mark_attr, log=log
    )
    return phase12 + phase3
