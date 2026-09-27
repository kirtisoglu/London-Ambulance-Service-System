"""
Deterministic level-2 super-partition for a given level-1 partition.

Given the L1 districts (each with a team count and a node set on the base
graph G^1), build the district-adjacency supergraph and group districts into
super-districts that satisfy

    mu^2 = min_districts_super  (>= districts per super-district)
    c^2  in [c_min_super, c_max_super]  (teams per super-district)
    each super-district is connected on the supergraph

This is the fallback we use when ``Partition.from_random_assignment``'s
RNG-based ``init_super_partition`` silently degrades to the identity
grouping (the recursive spanning-tree heuristic can stall on small
supergraphs of <~10 super-nodes). The algorithm is fully deterministic:
same L1 partition + same bounds => same super-assignment, every call.

The procedure (greedy BFS-grow + small-piece merge):
  1. Build the supergraph from district adjacencies.
  2. In sorted-district-id order, pick the next unassigned seed and BFS
     outward (sorted frontier), adding adjacent districts while the
     running team-sum stays <= c_max_super.
  3. Any super that ends up under-sized (too few districts or too few
     teams) is merged into the adjacent super with the smallest legal
     post-merge team count (tiebreak by lowest super id). The merge
     repeats until every super is well-sized.
"""
from typing import Dict, Mapping

import networkx as nx


def deterministic_super_partition(
    base_graph: nx.Graph,
    x1: Mapping[int, int],
    c1: Mapping[int, int],
    *,
    c_min_super: int = 2,
    c_max_super: int = 5,
    min_districts_super: int = 2,
) -> Dict[int, int]:
    """Return a dict mapping each L1 district id -> super-district index
    (contiguous 0..k-1).

    :param base_graph: the base graph $G^1$.
    :param x1: assignment dict node -> L1 district id (must cover every
        node of ``base_graph``).
    :param c1: dict L1 district id -> team count (c^1).
    :param c_min_super: minimum teams per super-district.
    :param c_max_super: maximum teams per super-district.
    :param min_districts_super: mu^2, the minimum L1 districts per super.
    """
    districts = sorted(c1)

    # Supergraph of district adjacencies.
    S = nx.Graph()
    S.add_nodes_from(districts)
    for u, v in base_graph.edges():
        du, dv = x1[u], x1[v]
        if du != dv:
            S.add_edge(du, dv)

    # Greedy BFS-grow super-districts in sorted-seed order.
    assigned: Dict[int, int] = {}
    supers = []   # each entry: {"districts": [..], "teams": int}
    for seed in districts:
        if seed in assigned:
            continue
        sup = [seed]
        team = c1[seed]
        frontier = sorted(S.neighbors(seed))
        while frontier:
            nxt = frontier.pop(0)
            if nxt in assigned or nxt in sup:
                continue
            if team + c1[nxt] > c_max_super:
                continue
            sup.append(nxt)
            team += c1[nxt]
            for nb in sorted(S.neighbors(nxt)):
                if nb not in assigned and nb not in sup and nb not in frontier:
                    frontier.append(nb)
        for d in sup:
            assigned[d] = len(supers)
        supers.append({"districts": sup, "teams": team})

    # Merge any super that doesn't meet (#districts >= mu^2 AND teams >= c_min)
    # into the adjacent super whose post-merge team count is smallest.
    while True:
        bad = [i for i, s in enumerate(supers)
               if len(s["districts"]) < min_districts_super
               or s["teams"] < c_min_super]
        if not bad:
            break
        i = bad[0]
        target = None
        best_team = None
        for d in supers[i]["districts"]:
            for nb in S.neighbors(d):
                j = assigned[nb]
                if j == i:
                    continue
                new_team = supers[j]["teams"] + supers[i]["teams"]
                if new_team > c_max_super:
                    continue
                if (target is None
                        or new_team < best_team
                        or (new_team == best_team and j < target)):
                    target = j
                    best_team = new_team
        if target is None:
            # No legal merge available; accept any adjacent super (overshoot
            # is preferable to a state with no valid super-partition at all).
            for d in supers[i]["districts"]:
                neigh = sorted(S.neighbors(d))
                for nb in neigh:
                    j = assigned[nb]
                    if j != i:
                        target = j
                        break
                if target is not None:
                    break
        if target is None:
            raise RuntimeError(
                f"could not merge under-sized super {i} (districts="
                f"{supers[i]['districts']}, teams={supers[i]['teams']}) — "
                "supergraph appears disconnected.")
        for d in supers[i]["districts"]:
            assigned[d] = target
            supers[target]["districts"].append(d)
            supers[target]["teams"] += c1[d]
        supers.pop(i)
        for d, k in assigned.items():
            if k > i:
                assigned[d] = k - 1

    return assigned
