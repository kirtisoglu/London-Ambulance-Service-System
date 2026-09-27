"""
Diagnostic: A/B test the per_team debt-correction divisor in
capacitated_recursive_tree on the LAS chain.

Background. The per_team rule corrects accumulated demand "debt" by
shifting the per-team admissibility window. The legacy formula subtracts
the FULL debt; the LAS branch experiments with spreading it over
remaining teams. This script measures, for a chosen debt mode, how often
the per-step base re-partition strands teams (exhausts nodes with teams
unplaced -> rejection) and the resulting chain acceptance rate.

The debt mode is selected by the CRT_DEBT_MODE env var, read inside
falcomchain/tree/tree.py:
    full   -> debt              (original legacy behaviour)
    rt     -> debt / max(1, remaining_teams)
    spread -> debt / max(1, remaining_teams - capacity_level)  (default)

Run:
    CRT_DEBT_MODE=full   python -m falcomchain_experiments.las.diag_debt_mode --steps 200
    CRT_DEBT_MODE=spread python -m falcomchain_experiments.las.diag_debt_mode --steps 200

Set CRT_DEBUG=1 to also print each empty-nodes (team-stranding) event.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, "/Users/kirtisoglu/GitHub/FalcomChain")

from falcomchain.graph import Graph
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.markovchain import MarkovChain
from falcomchain.markovchain.accept import always_accept
from falcomchain.markovchain.proposals import hierarchical_recom
from falcomchain.markovchain.state import ChainState
from falcomchain.markovchain.energy import compute_energy

from .build_s_las import load_graph, install_l2_candidates
from .run_chain_v3 import install_cdba_candidates, super_facility_open_l1
from .proxy_travel import ProxyTravelTimes

REPO = Path(__file__).resolve().parents[2]

# Defaults are the c in [3,8], w=3642 (1 ambulance per capacity level) setup.
# Override via CLI to test the "k ambulances per capacity level" reframing
# (e.g. k=2 -> w_unit=7284, c in [1,3]).
W, EPS, C_MIN, C_MAX = 3642, 0.15, 3, 8
C_MIN_SUPER, C_MAX_SUPER, MIN_DIS_SUPER = 6, 16, 2
CDBA_CSV = REPO / "data/derived/cdba_candidates_w3642_eps15_cmin3_dmax12.csv"


def build_graph(cdba_csv):
    g_full = load_graph()
    las = set(pd.read_csv(REPO / "data/derived/lsoa_to_group.csv")["LSOA21CD"])
    keep = [n for n in g_full.nodes if g_full.nodes[n].get("LSOA21CD") in las]
    g = Graph.from_networkx(nx.Graph(g_full.subgraph(keep)))
    install_cdba_candidates(g, cdba_csv)
    install_l2_candidates(g)
    Assignment.travel_times = ProxyTravelTimes(g, kmh=25.0, cache=True)
    return g


def build_initial(g, w, eps, c_min, c_max):
    """Retry seeds until an L1 (identity-L2) partition builds."""
    for seed in range(1, 80):
        random.seed(seed)
        try:
            part = Partition.from_random_assignment(
                graph=g, epsilon=eps, demand_target=w,
                assignment_class=Assignment, capacity_level=c_max,
                c_min=c_min, rule="per_team", init_super_partition=False)
            return part, seed
        except Exception:
            continue
    raise RuntimeError("no seed built an initial L1 partition")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--w", type=int, default=W)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--cmin", type=int, default=C_MIN)
    ap.add_argument("--cmax", type=int, default=C_MAX)
    ap.add_argument("--cdba", default=str(CDBA_CSV))
    args = ap.parse_args()
    mode = os.environ.get("CRT_DEBT_MODE", "spread")

    w, eps, c_min, c_max = args.w, args.eps, args.cmin, args.cmax
    cms, cxs = 2 * c_min, 2 * c_max
    print(f"[mode={mode}] params: w={w}, eps={eps}, c1=[{c_min},{c_max}], "
          f"c2=[{cms},{cxs}], min_dis_super={MIN_DIS_SUPER}, cdba={Path(args.cdba).name}",
          flush=True)

    g = build_graph(Path(args.cdba))
    part, seed = build_initial(g, w, eps, c_min, c_max)
    print(f"[mode={mode}] L1 built seed={seed}: |P1|={len(part.parts)}", flush=True)

    state0 = ChainState.initial(
        partition=part, energy=0.0, beta=0.0,
        energy_fn=compute_energy, super_facility_fn=super_facility_open_l1)
    proposal = partial(
        hierarchical_recom, epsilon_base=eps, epsilon_super=eps,
        demand_target=w, c_min_base=c_min, c_min_super=cms,
        c_max_super=cxs, min_districts_super=MIN_DIS_SUPER,
        gamma_base=0.0, gamma_super=0.0)

    # Manual chain loop replicating MarkovChain.__next__, classifying each
    # step's outcome. Rejection causes:
    #   accepted          - proposal succeeded, valid, accepted
    #   reject_super      - proposal raised RuntimeError in the supergraph
    #                       re-partition (resample_super_partition)
    #   reject_base       - proposal raised RuntimeError in the base
    #                       re-partition (incl. team-strand guard)
    #   reject_other_err  - proposal raised some other exception
    #   reject_invalid    - proposal returned a state failing is_valid
    counts = {"accepted": 0, "reject_super": 0, "reject_base": 0,
              "reject_other_err": 0, "reject_invalid": 0}
    state = state0
    t0 = time.perf_counter()
    for i in range(1, args.steps + 1):
        if state.partition is not None:
            state.partition.parent = None
        try:
            proposed = proposal(state)
        except RuntimeError as e:
            msg = str(e)
            if "Supergraph = True" in msg:
                counts["reject_super"] += 1
            else:  # "Supergraph = False" or the strand-guard message
                counts["reject_base"] += 1
            proposed = None
        except Exception:
            counts["reject_other_err"] += 1
            proposed = None

        if proposed is not None:
            # constraints=[] -> is_valid always True; keep explicit for clarity
            if always_accept(proposed, state):
                state = proposed
                counts["accepted"] += 1
            else:
                counts["reject_invalid"] += 1

        if i % 50 == 0:
            acc = counts["accepted"] / i
            print(f"  [mode={mode}] step {i}: accept={acc:.1%} "
                  f"super={counts['reject_super']} base={counts['reject_base']} "
                  f"other={counts['reject_other_err']}", flush=True)

    el = time.perf_counter() - t0
    n = args.steps
    print(f"\n[mode={mode}] === rejection-cause breakdown over {n} steps "
          f"({el:.0f}s) ===")
    for k, v in counts.items():
        print(f"  {k:<18} {v:>4}  ({100*v/n:.1f}%)")
    print(f"  acceptance rate: {counts['accepted']/n:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
