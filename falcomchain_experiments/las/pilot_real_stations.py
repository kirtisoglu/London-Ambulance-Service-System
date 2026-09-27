"""
Pilot: run the FalCom chain on the 66 REAL LAS stations only (no CDBA
augmentation), i.e. in rejection mode where Assumption 6.1 does not hold.

Measures, per configuration:
  * whether a feasible initial state can be built on real stations
  * acceptance rate and the split of rejection causes (base vs supergraph)
  * spanning-tree retry statistics per bipartition call
  * wall time per step
  * |P^1|, |P^2| and the number of districts without a real station

Run from the LAS repo root:
    python -m falcomchain_experiments.las.pilot_real_stations --steps 300 \
        --counting 1 --max-attempts-base 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from functools import partial
from pathlib import Path

import networkx as nx
import pandas as pd

import sys
# falcomchain must be installed (pip install -e ../FalcomChain); no hard-coded paths.

from falcomchain.graph import Graph
from falcomchain.markovchain import MarkovChain
from falcomchain.markovchain.accept import always_accept
from falcomchain.markovchain.proposals import hierarchical_recom
from falcomchain.markovchain.state import ChainState
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
import importlib
# ``falcomchain.tree`` re-exports a ``tree`` name that shadows the submodule
# (networkx.algorithms.tree), so fetch the submodule explicitly.
treemod = importlib.import_module("falcomchain.tree.tree")

from .cdba_travel import CdbaTravelTimes
from .build_s_las import load_graph, install_l2_candidates
from .run_chain_v3 import (
    REPO, DEFAULT_CDBA_CSV, DEFAULT_TRAVEL,
    W, EPS_BASE, EPS_SUPER, C_MIN_BASE, C_MAX_BASE, C_MIN_SUPER, C_MAX_SUPER,
    MIN_DISTRICTS_SUPER, GAMMA_BASE, GAMMA_SUPER,
    install_cdba_candidates, super_facility_open_l1,
)

OUT_DIR = REPO / "data/derived/pilot_real_stations"


def install_real_candidates(g, cdba_csv: Path) -> int:
    """Mark only the real stations (is_real == 1) as candidates."""
    df = pd.read_csv(cdba_csv)
    code_to_id = {g.nodes[n]["LSOA21CD"]: n for n in g.nodes}
    for n in g.nodes:
        g.nodes[n]["candidate"] = 0
        g.nodes[n]["candidate_artificial"] = 0
    n_real = 0
    for _, row in df[df["is_real"] == 1].iterrows():
        nid = code_to_id.get(row["LSOA21CD"])
        if nid is not None:
            g.nodes[nid]["candidate"] = 1
            n_real += 1
    return n_real


def load_las_graph(cdba_csv: Path, travel_path: Path):
    g_full = load_graph()
    las_lsoas = set(pd.read_csv(REPO / "data/derived/lsoa_to_group.csv")["LSOA21CD"])
    keep = [n for n in g_full.nodes if g_full.nodes[n].get("LSOA21CD") in las_lsoas]
    g = Graph.from_networkx(nx.Graph(g_full.subgraph(keep)))
    install_l2_candidates(g)
    Assignment.travel_times = CdbaTravelTimes(g, matrix_path=travel_path)
    return g


def districts_without_station(partition, g) -> int:
    return sum(
        1 for nodes in partition.parts.values()
        if not any(g.nodes[n]["candidate"] == 1 for n in nodes)
    )


def try_init_real(g, seed, max_attempts, counting, retries):
    """Recursive partitioner on the full graph with real stations only."""
    for r in range(retries):
        s = seed * 1000 + 17 * r + 1
        random.seed(s)
        treemod.rng.seed(s) if hasattr(treemod.rng, "seed") else None
        t0 = time.perf_counter()
        try:
            p = Partition.from_random_assignment(
                graph=g, epsilon=EPS_BASE, demand_target=W,
                assignment_class=Assignment, capacity_level=C_MAX_BASE,
                c_min=C_MIN_BASE, init_super_partition=False,
                max_attempts=max_attempts, count_candidates=counting,
            )
            return p, r + 1, time.perf_counter() - t0, "real"
        except Exception as e:
            print(f"    init attempt {r + 1} failed after {time.perf_counter() - t0:.0f}s: "
                  f"{type(e).__name__}: {str(e)[:80]}")
    return None, retries, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--counting", type=int, default=1)
    ap.add_argument("--max-attempts-base", type=int, default=200)
    ap.add_argument("--max-attempts-super", type=int, default=1000)
    ap.add_argument("--init-attempts", type=int, default=200)
    ap.add_argument("--init-retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cdba", default=str(DEFAULT_CDBA_CSV))
    ap.add_argument("--travel", default=str(DEFAULT_TRAVEL))
    ap.add_argument("--record", type=int, default=0,
                    help="record open stations and level-1 cut-edge count per step")
    args = ap.parse_args()
    counting = bool(args.counting)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"cnt{int(counting)}_M{args.max_attempts_base}_s{args.seed}_T{args.steps}"

    print(f"=== pilot real stations: {tag} ===")
    g = load_las_graph(Path(args.cdba), Path(args.travel))
    n_real = install_real_candidates(g, Path(args.cdba))
    total = sum(g.nodes[n]["demand"] for n in g.nodes)
    print(f"  |V|={g.number_of_nodes()} |E|={g.number_of_edges()} real candidates={n_real} "
          f"total demand={total:,} units={-(-total // W)}")

    # ---------------- initial state ----------------
    print(f"[init] recursive partitioner on real stations "
          f"(M={args.init_attempts}, counting={counting}, retries={args.init_retries})")
    partition, n_tries, t_init, init_mode = try_init_real(
        g, args.seed, args.init_attempts, counting, args.init_retries)
    if partition is None:
        print("[init] FAILED on real stations -> fallback: CDBA init, then switch to real-only")
        install_cdba_candidates(g, Path(args.cdba))
        random.seed(args.seed * 1000 + 1)
        t0 = time.perf_counter()
        partition = Partition.from_random_assignment(
            graph=g, epsilon=EPS_BASE, demand_target=W,
            assignment_class=Assignment, capacity_level=C_MAX_BASE,
            c_min=C_MIN_BASE, init_super_partition=False,
        )
        t_init = time.perf_counter() - t0
        install_real_candidates(g, Path(args.cdba))
        init_mode = "cdba_then_switch"
    n_bad0 = districts_without_station(partition, g)
    print(f"  init mode={init_mode} tries={n_tries} time={t_init:.0f}s "
          f"|P^1|={len(partition.parts)} teams={sum(partition.teams.values())} "
          f"districts without station={n_bad0}")

    state = ChainState.initial(
        partition=partition, energy=0.0, beta=0.0,
        energy_fn=None, super_facility_fn=super_facility_open_l1,
    )

    # ---------------- chain ----------------
    base_proposal = partial(
        hierarchical_recom,
        epsilon_base=EPS_BASE, epsilon_super=EPS_SUPER, demand_target=W,
        c_min_base=C_MIN_BASE, c_min_super=C_MIN_SUPER, c_max_super=C_MAX_SUPER,
        min_districts_super=MIN_DISTRICTS_SUPER,
        gamma_base=GAMMA_BASE, gamma_super=GAMMA_SUPER,
        max_attempts_super=args.max_attempts_super,
        max_attempts_base=args.max_attempts_base,
        count_candidates_base=counting,
    )
    causes = Counter()
    step_times = []

    def proposal(st):
        t0 = time.perf_counter()
        try:
            out = base_proposal(st)
            causes["proposed"] += 1
            return out
        except RuntimeError as e:
            msg = str(e)
            if "Supergraph = True" in msg:
                causes["reject_supergraph"] += 1
            elif "Supergraph = False" in msg:
                causes["reject_base"] += 1
            else:
                causes["reject_other_runtime"] += 1
            raise
        except Exception as e:
            causes[f"reject_{type(e).__name__}"] += 1
            raise RuntimeError(str(e)) from e
        finally:
            step_times.append(time.perf_counter() - t0)

    treemod.ATTEMPT_LOG_ENABLED = True
    treemod.ATTEMPT_LOG.clear()
    random.seed(args.seed * 7919 + 3)
    chain = MarkovChain(proposal=proposal, constraints=[], accept=always_accept,
                        initial_state=state, total_steps=args.steps)
    accepted = 0
    rows = []
    first_feasible = None if n_bad0 > 0 else 0
    t_chain = time.perf_counter()
    for i, st in enumerate(chain, start=1):
        acc = (st is not state)
        state = st
        if acc:
            accepted += 1
        n_bad = districts_without_station(st.partition, g)
        if first_feasible is None and n_bad == 0:
            first_feasible = i
        row = {"step": i, "accepted": acc, "P1": len(st.partition.parts),
               "P2": len(set(st.partition.super_assignment.values())),
               "no_station": n_bad}
        if args.record:
            # open level-1 facilities (real stations) and cut edges at level 1
            centers = [int(c) for c in st.facility.centers.values() if c is not None] \
                if st.facility else []
            row["open"] = sorted(centers)
            m = st.partition.assignment.mapping
            row["cut_edges"] = sum(1 for u, v in g.edges if m[u] != m[v])
        rows.append(row)
        if i % 50 == 0 or i == args.steps:
            el = time.perf_counter() - t_chain
            print(f"  step {i:5d}  acc={accepted / i:.3f}  |P1|={rows[-1]['P1']:3d} "
                  f"|P2|={rows[-1]['P2']:2d}  no_station={n_bad:2d}  "
                  f"{i / el:.2f} steps/s  causes={dict(causes)}")
    elapsed = time.perf_counter() - t_chain

    # ---------------- attempt statistics ----------------
    log = treemod.ATTEMPT_LOG
    base = [d for d in log if not d["supergraph"]]
    sup = [d for d in log if d["supergraph"]]

    def summ(L):
        if not L:
            return {}
        a = [d["attempts"] for d in L]
        return {"calls": len(L), "fail_calls": sum(1 for d in L if not d["success"]),
                "mean_attempts": round(statistics.mean(a), 1),
                "median_attempts": statistics.median(a), "max_attempts": max(a)}

    result = {
        "tag": tag, "counting": counting, "max_attempts_base": args.max_attempts_base,
        "max_attempts_super": args.max_attempts_super, "seed": args.seed,
        "n_real_candidates": n_real, "init_mode": init_mode, "init_tries": n_tries,
        "init_seconds": round(t_init, 1), "init_districts_without_station": n_bad0,
        "first_feasible_step": first_feasible,
        "steps": args.steps, "accepted": accepted,
        "acceptance_rate": round(accepted / args.steps, 4),
        "causes": dict(causes), "steps_per_second": round(args.steps / elapsed, 3),
        "median_step_seconds": round(statistics.median(step_times), 3),
        "max_step_seconds": round(max(step_times), 2),
        "attempts_base": summ(base), "attempts_super": summ(sup),
        "P1_range": [min(r["P1"] for r in rows), max(r["P1"] for r in rows)],
        "P2_range": [min(r["P2"] for r in rows), max(r["P2"] for r in rows)],
        "station_nodes": sorted(int(n) for n in g.nodes if g.nodes[n]["candidate"] == 1),
        "station_codes": {int(n): g.nodes[n]["LSOA21CD"] for n in g.nodes
                          if g.nodes[n]["candidate"] == 1},
        "final_l1_assignment": {int(n): int(p) for p, nodes in state.partition.parts.items()
                                for n in nodes},
        "rows": rows,
    }
    out = OUT_DIR / f"pilot_{tag}.json"
    out.write_text(json.dumps(result, indent=1, default=str))
    print("\nSUMMARY", json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
