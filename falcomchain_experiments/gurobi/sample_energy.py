"""
Where the sampled ensemble sits relative to the exact optimum.

Runs the *sampler* (the configuration of ``scalability/run_scalability.py``:
uniform cuts, every closed proposal accepted, no objective) on ``grid_100``
and ``grid_400`` and evaluates the disaggregated hierarchical median, the
objective of the mixed-integer model, on every state of the chain. The
histogram of these values against the proven optimum is the placement figure
of the paper.

    PYTHONHASHSEED=0 python -m falcomchain_experiments.gurobi.sample_energy \
        --sizes 100 400 --steps 20000 --seeds 1 2 3 4

Output: ``gurobi/results_sparse/sample_energy_<N>.json`` (per-chain series of
the level-1 and level-2 terms after burn-in, acceptance, and the optimum read
from ``milp_summary.json``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scalability"))
sys.setrecursionlimit(50000)

import run_scalability as rs  # noqa: E402

from falcomchain import MarkovChain, Partition, always_accept, hierarchical_recom, set_seed  # noqa: E402
from falcomchain.markovchain.state import ChainState  # noqa: E402
from falcomchain.partition.assignment import Assignment  # noqa: E402


def median_objective(partition, cand1, cand2, dist, demand):
    """(level-1 term, level-2 term) of the disaggregated hierarchical median."""
    parts = partition.parts
    sup = partition.super_assignment
    base = 0.0
    for nodes in parts.values():
        base += min(sum(demand[v] * dist[(c, v)] for v in nodes) for c in cand1 if c in nodes)
    blocks = {}
    for did, nodes in parts.items():
        blocks.setdefault(sup.get(did, did), set()).update(nodes)
    coord = 0.0
    for nodes in blocks.values():
        cands = [c for c in cand2 if c in nodes] or [c for c in cand1 if c in nodes]
        coord += min(sum(demand[v] * dist[(c, v)] for v in nodes) for c in cands)
    return base, coord


def run(n: int, seed: int, steps: int):
    meta = json.loads((rs.GUROBI_DATA / f"grid_{n}.meta.json").read_text())
    g = rs.load_grid(rs.GUROBI_DATA / f"grid_{n}.json")
    set_seed(seed)
    dist = rs.GridManhattanDist(g)
    Assignment.travel_times = dist
    demand = {v: g.nodes[v]["demand"] for v in g.nodes}
    cand1 = {v for v in g.nodes if g.nodes[v].get("candidate")}
    cand2 = {v for v in g.nodes if g.nodes[v].get("super_candidate")}
    w = meta["demand_target_w"]
    total = sum(demand.values())
    k = max(1, math.ceil(total / w))
    partition = None
    for r in range(rs.N_INIT_RETRIES):
        try:
            partition = Partition.from_random_assignment(
                graph=g, epsilon=meta["epsilon_l1"], demand_target=total / k, assignment_class=None,
                capacity_level=meta["c_max_l1"], c_min=meta["c_min_l1"])
            break
        except Exception:  # noqa: BLE001
            set_seed(seed * 1000 + r + 1)
    if partition is None:
        raise RuntimeError(f"no initial plan for grid_{n}, seed {seed}")
    state = ChainState.initial(partition=partition, energy=0.0, beta=1.0)
    proposal = partial(hierarchical_recom, epsilon_base=meta["epsilon_l1"], epsilon_super=meta["epsilon_l2"],
                       demand_target=w, c_min_base=meta["c_min_l1"], c_min_super=meta["c_min_l2"],
                       c_max_super=meta["c_max_l2"], min_districts_super=int(meta.get("min_l1_per_l2", 1)))
    chain = MarkovChain(proposal=proposal, constraints=lambda p: True, accept=always_accept,
                        initial_state=state, total_steps=steps)
    e1, e2 = [], []
    last = None
    for i, st in enumerate(chain):
        if i == 0:
            continue                       # the start carries the identity level-2 grouping
        if chain.last_rejection is None or last is None:
            last = median_objective(st.partition, cand1, cand2, dist, demand)
        e1.append(last[0])
        e2.append(last[1])
    rep = chain.rejection_report()
    return {"seed": seed, "steps": steps, "acceptance": rep["acceptance_rate"], "E1": e1, "E2": e2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 400])
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()
    milp = {r["instance"]: r for r in json.loads((HERE / "results_sparse/milp_summary.json").read_text())["rows"]}
    for n in a.sizes:
        chains = [run(n, s, a.steps) for s in a.seeds]
        opt = milp[str(n)]["milp"]["shir"]
        out = {"instance": f"grid_{n}", "optimum": opt["obj"], "optimum_l1": opt["l1_cost"],
               "optimum_l2": opt["l2_cost"], "optimum_seconds": opt["wall_time_s"], "chains": chains}
        path = HERE / f"results_sparse/sample_energy_{n}.json"
        path.write_text(json.dumps(out))
        tot = [x + y for c in chains for x, y in zip(c["E1"][a.steps // 5:], c["E2"][a.steps // 5:])]
        tot.sort()
        print(f"grid_{n}: optimum {opt['obj']:.0f}; sampled min {tot[0]:.0f} median {tot[len(tot) // 2]:.0f} "
              f"max {tot[-1]:.0f}; acceptance {[round(c['acceptance'], 3) for c in chains]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
