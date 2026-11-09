"""
Exact reachable set of the FalCom proposal on the non-uniform 4x4 instance.

``enumeration_nonuniform.py`` lists the states that satisfy the balance rule of
the paper (an outer description) and records which of them the chains visit.
This script computes, independently of the sampler, the set of level-1 states
that the proposal can reach: starting from one state it applies every move the
proposal allows and closes the set under them. A move is

1. a level-2 grouping the supergraph recursion can return (here, with four
   capacity units and c2 in [2, 4]: the whole graph as one superdistrict, or two
   connected superdistricts of two units each inside their level-2 windows, each
   with at least kappa districts), then
2. a re-cut of one superdistrict by the level-1 recursion: sequential
   extraction of a connected district whose complement in the residual is
   connected (a pendant subtree of some spanning tree), admissible under the
   debt-corrected window of the paper at the current debt, with the candidate
   and counting predicates, one-sided while more than c_max units remain and
   two-sided afterwards; the re-cut is rejected if it leaves fewer than kappa
   districts.

No spanning tree is drawn: the rules are applied to every connected set, so
the result does not depend on the retry budget. The script reports whether the
closure equals the set the three chains visited.

    PYTHONHASHSEED=0 python -m falcomchain_experiments.validation.enumeration_closure
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from .enumeration_nonuniform import EPS, INSTANCES, OUT_DIR, build_graph, enumerate_states

NAME = "4x4"
spec = INSTANCES[NAME]
g, cand, demand = build_graph(spec)
N = g.number_of_nodes()
FULL = (1 << N) - 1
W = spec["w"]
C_MAX = spec["c_max"]
C2_MIN, C2_MAX = spec["c_super"]
KAPPA = spec["kappa"]
CAND = sum(1 << v for v in cand)
NBR = [sum(1 << u for u in g.neighbors(v)) for v in range(N)]
DSUM = [0.0] * (FULL + 1)
for m in range(1, FULL + 1):
    low = (m & -m).bit_length() - 1
    DSUM[m] = DSUM[m & (m - 1)] + float(demand[low])


def is_connected(m: int) -> bool:
    if m == 0:
        return False
    seen = m & -m
    frontier = seen
    while frontier:
        v = (frontier & -frontier).bit_length() - 1
        frontier &= frontier - 1
        new = NBR[v] & m & ~seen
        seen |= new
        frontier |= new
    return seen == m


CONN = [False] * (FULL + 1)
for m in range(1, FULL + 1):
    CONN[m] = is_connected(m)


def ncand(m: int) -> int:
    return bin(m & CAND).count("1")


def submasks(m: int):
    s = (m - 1) & m
    while s:
        yield s
        s = (s - 1) & m


def window(d_bar: float, debt: float):
    lo = max(d_bar * (1 - EPS), d_bar * (1 - EPS) - debt)
    hi = min(d_bar * (1 + EPS), d_bar * (1 + EPS) - debt)
    return (lo + hi) / 2.0, (hi - lo) / 2.0            # centre d', half-width h


TOL = 1e-9


def recurse(R: int, n: int, debt: float, d_bar: float):
    """All (district mask, capacity) partitions of R the level-1 recursion can produce."""
    centre, half = window(d_bar, debt)
    if half < -TOL:
        return set()

    def ok(mask, c):
        return abs(DSUM[mask] - c * centre) <= half + TOL

    out = set()
    if n <= C_MAX:                                       # two-sided mode
        if ok(R, n) and ncand(R) >= 1:
            out.add(frozenset([(R, n)]))
        for c in range(1, min(C_MAX, n) + 1):
            if n - c <= 0:
                continue
            for D in submasks(R):
                rest = R & ~D
                if not (CONN[D] and CONN[rest]) or ncand(D) < 1 or ncand(rest) < 1:
                    continue
                if ok(D, c) and ok(rest, n - c):
                    for tail in recurse(rest, n - c, debt + DSUM[D] - c * d_bar, d_bar):
                        out.add(tail | {(D, c)})
    else:                                                # one-sided mode
        for c in range(1, min(C_MAX, n) + 1):
            need = max(1, math.ceil((n - c) / C_MAX))
            for D in submasks(R):
                rest = R & ~D
                if not (CONN[D] and CONN[rest]) or ncand(D) < 1 or ncand(rest) < need:
                    continue
                if ok(D, c):
                    for tail in recurse(rest, n - c, debt + DSUM[D] - c * d_bar, d_bar):
                        out.add(tail | {(D, c)})
    return out


@lru_cache(maxsize=None)
def recut(S: int, n: int):
    """Outcomes of re-cutting the superdistrict S of capacity n (at least KAPPA districts)."""
    return frozenset(p for p in recurse(S, n, 0.0, DSUM[S] / n) if len(p) >= KAPPA)


def groupings(state):
    """Level-2 groupings the supergraph recursion can return, as lists of district lists."""
    districts = sorted(state)
    res = []
    if len(districts) >= KAPPA:
        res.append([districts])                           # the whole graph: capacity 4 <= c2_max
    k = len(districts)
    adj = {i: {j for j in range(k) if j != i and any(NBR[v] & districts[j][0] for v in range(N) if districts[i][0] >> v & 1)}
           for i in range(k)}

    def connected(idx):
        idx = set(idx)
        seen, stack = {next(iter(idx))}, [next(iter(idx))]
        while stack:
            i = stack.pop()
            for j in adj[i] & idx:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return seen == idx

    for bits in range(1, 1 << (k - 1)):                   # bipartitions, district 0 on side A... of the rest
        A = [i for i in range(k) if (bits >> i) & 1 or False]
        A = [i for i in range(k - 1) if (bits >> i) & 1]
        B = [i for i in range(k) if i not in A]
        sides = (A, B)
        good = True
        for side in sides:
            cap = sum(districts[i][1] for i in side)
            dem = sum(DSUM[districts[i][0]] for i in side)
            if (len(side) < KAPPA or cap < C2_MIN or cap > C2_MAX or not connected(side)
                    or abs(dem - cap * W) > EPS * cap * W + TOL):
                good = False
                break
        if good:
            res.append([[districts[i] for i in A], [districts[i] for i in B]])
    return res


def closure(start):
    seen = {start}
    stack = [start]
    while stack:
        state = stack.pop()
        for grouping in groupings(state):
            for block in grouping:
                S = 0
                n = 0
                for mask, cap in block:
                    S |= mask
                    n += cap
                keep = state - frozenset(block)
                for new_part in recut(S, n):
                    nxt = keep | new_part
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
    return seen


def to_mask_state(state, caps):
    return frozenset((sum(1 << v for v in d), caps[d]) for d in state)


def reachable(tol):
    """Recurrent reachable set under window tolerance ``tol`` and its joint (level-1, level-2) count."""
    global TOL
    TOL = tol
    recut.cache_clear()
    core = set(recut(FULL, spec["units"]))                # the whole graph is always an admissible superdistrict
    reach = closure(next(iter(core)))
    irreducible = all(closure(s) == reach for s in reach)
    joint = sum(len(groupings(s)) for s in reach)
    return reach, irreducible, joint


def main() -> int:
    states1, level2, caps = enumerate_states(g, cand, demand, spec)
    rows = json.loads((OUT_DIR / f"enumeration_nonuniform_{NAME}_states.json").read_text())
    summary_chain = json.loads((OUT_DIR / f"enumeration_nonuniform_{NAME}.json").read_text())
    assert len(rows) == len(states1)
    visited = {to_mask_state(s, caps) for s, r in zip(states1, rows) if r["pooled"] > 0}
    outer = {to_mask_state(s, caps) for s in states1}
    # closed windows (ties on a window boundary admitted) and open windows (ties excluded): a tie is
    # decided by floating-point rounding in the implementation, so the two bracket what the chain can do
    closed, irr_closed, joint_closed = reachable(1e-9)
    opened, irr_open, joint_open = reachable(-1e-9)
    summary = {
        "outer_set_level1": len(outer), "outer_set_joint": summary_chain["feasible_joint"],
        "visited_level1": len(visited), "visited_joint": summary_chain["visited_joint_union"],
        "reachable_level1_closed_windows": len(closed), "reachable_joint_closed_windows": joint_closed,
        "reachable_level1_open_windows": len(opened), "reachable_joint_open_windows": joint_open,
        "boundary_tie_states": len(closed - opened),
        "visited_equals_reachable_open_windows": visited == opened,
        "visited_subset_of_reachable_closed_windows": visited <= closed,
        "reachable_inside_outer_set": closed <= outer,
        "single_communicating_class_closed": irr_closed, "single_communicating_class_open": irr_open,
        "start_states_reachable": {name: to_mask_state(st, caps) in opened
                                   for name, st in (("first", states1[0]), ("middle", states1[len(states1) // 2]),
                                                    ("last", states1[-1]))},
    }
    (OUT_DIR / f"enumeration_closure_{NAME}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
