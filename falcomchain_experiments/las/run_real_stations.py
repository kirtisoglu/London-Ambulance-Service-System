"""
FalCom on the 66 real LAS ambulance stations, no artificial candidates.

Builds the instance from the committed raw files (``data/raw/london_graph.json``,
``data/raw/LAS_stations.csv``) and the station rows of the road-network
travel-time matrix, then runs the sampler at the locked calibration
(:mod:`calibration`) with the counting predicate and the paper's debt rule
(both library defaults). Level-2 facilities follow the paper's dynamic-nested
rule: each super-district's facility is the demand-weighted 1-median among the
stations opened inside it.

Per step it records acceptance, the rejection cause, the energy (level-1
access cost + level-2 coordination cost, both demand-weighted minutes), the
district and super-district counts and the number of cut edges; every
``--snap-every`` steps it stores a compact snapshot (station of every LSOA,
super-district of every LSOA, capacity of every LSOA's district) for the
ensemble analysis in :mod:`postprocess_real_stations`.

Run from the repository root::

    python -m falcomchain_experiments.las.run_real_stations --seed 1 --steps 20000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from falcomchain.graph import Graph
from falcomchain.markovchain import MarkovChain
from falcomchain.markovchain.accept import always_accept
from falcomchain.markovchain.energy import compute_energy
from falcomchain.markovchain.facility import SuperFacilityAssignment
from falcomchain.markovchain.proposals import hierarchical_recom
from falcomchain.markovchain.state import ChainState
from falcomchain.partition import Partition
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed
from falcomchain.tree.errors import ProposalRejected
from falcomchain.tree.tree import CutParams

from .build_s_las import load_graph
from .calibration import (
    C_MAX_BASE, C_MAX_SUPER, C_MIN_BASE, C_MIN_SUPER, EPS_BASE, EPS_SUPER,
    GAMMA_BASE, GAMMA_SUPER, MIN_DISTRICTS_SUPER, W,
)

REPO = Path(__file__).resolve().parents[2]
STATIONS = REPO / "data/raw/LAS_stations.csv"
LSOA_GROUP = REPO / "data/derived/lsoa_to_group.csv"
LSOA_SECTOR = REPO / "data/derived/lsoa_to_sector_borough.csv"
CDBA_TRAVEL = REPO / "data/derived/cdba_travel_times_w10887_eps15_cmin1_dmax12.parquet"
REAL_TRAVEL = REPO / "data/derived/real_station_travel_times.parquet"
OUT_DIR = REPO / "data/derived/real_stations"

PROXY_KMH = 25.0   # straight-line fallback speed for the few pairs the road matrix lacks


# --------------------------------------------------------------------------
# Instance
# --------------------------------------------------------------------------

def load_las_graph() -> Graph:
    """The 4,994-node LAS dual graph with the 66 stations as the only candidates."""
    g_full = load_graph()
    las = set(pd.read_csv(LSOA_GROUP)["LSOA21CD"])
    keep = [n for n in g_full.nodes if g_full.nodes[n].get("LSOA21CD") in las]
    g = Graph.from_networkx(nx.Graph(g_full.subgraph(keep)))
    for n in g.nodes:
        g.nodes[n]["demand"] = float(g.nodes[n]["demand"])
        g.nodes[n]["candidate"] = 0
        g.nodes[n]["candidate_artificial"] = 0
    id_of = {g.nodes[n]["LSOA21CD"]: n for n in g.nodes}
    stations = pd.read_csv(STATIONS)
    for code in stations["LSOA21CD"]:
        g.nodes[id_of[code]]["candidate"] = 1
    return g


class StationTravelTimes(dict):
    """``(station_node, node) -> minutes`` with a straight-line fallback."""

    def __init__(self, graph, pairs):
        super().__init__(pairs)
        self._xy = {n: (float(graph.nodes[n]["BNG_E"]), float(graph.nodes[n]["BNG_N"]))
                    for n in graph.nodes}
        self.n_fallbacks = 0

    def __missing__(self, key):
        u, v = key
        (x1, y1), (x2, y2) = self._xy[u], self._xy[v]
        self.n_fallbacks += 1
        return math.hypot(x1 - x2, y1 - y2) / 1000.0 / PROXY_KMH * 60.0


def load_station_travel_times(g) -> StationTravelTimes:
    """Road-network minutes from every station to every LSOA.

    Extracted once from the CDBA matrix (whose origins include the 66
    stations) into ``data/derived/real_station_travel_times.parquet`` so the
    real-station study does not depend on the 87 MB augmented matrix.
    """
    station_codes = sorted(pd.read_csv(STATIONS)["LSOA21CD"])
    if not REAL_TRAVEL.exists():
        table = pq.read_table(CDBA_TRAVEL)
        mask = pc.is_in(table["origin"], value_set=pa.array(station_codes))
        pq.write_table(table.filter(mask), REAL_TRAVEL)
    df = pd.read_parquet(REAL_TRAVEL)
    id_of = {g.nodes[n]["LSOA21CD"]: n for n in g.nodes}
    pairs = {}
    for o, d, s in zip(df["origin"], df["dest"], df["seconds"]):
        u, v = id_of.get(o), id_of.get(d)
        if u is not None and v is not None:
            pairs[(u, v)] = float(s) / 60.0
    tt = StationTravelTimes(g, pairs)
    stations = [n for n in g.nodes if g.nodes[n]["candidate"] == 1]
    tt.n_missing_pairs = sum(1 for u in stations for v in g.nodes if (u, v) not in pairs)
    return tt


def super_facility_open_l1_median(state):
    """Paper's dynamic-nested level-2 rule: the level-2 facility of a
    super-district is the demand-weighted 1-median, over the super-district's
    LSOAs, among the stations opened inside it."""
    sfa = SuperFacilityAssignment()
    partition = state.partition
    tt = state.assignment.travel_times
    centers = state.facility.centers
    nodes = partition.graph.nodes
    for super_id, l1_ids in partition.super_parts.items():
        cands = [centers[d] for d in l1_ids if centers.get(d) is not None]
        base = set()
        for d in l1_ids:
            base |= partition.parts.get(d, frozenset())
        if not cands or not base:
            continue
        best, best_cost = None, float("inf")
        for c in cands:
            cost = sum(nodes[v]["demand"] * tt[(c, v)] for v in base)
            if cost < best_cost:
                best, best_cost = c, cost
        sfa._centers[super_id] = best
        sfa._radii[super_id] = best_cost
    return sfa


# --------------------------------------------------------------------------
# Chain
# --------------------------------------------------------------------------

def sector_zones(g) -> dict:
    """LSOA -> LAS operational sector (the five real sectors), used as the
    zones of the initial partition."""
    sec = pd.read_csv(LSOA_SECTOR)
    sector_of = dict(zip(sec["LSOA21CD"], sec["sector"]))
    return {n: sector_of[g.nodes[n]["LSOA21CD"]] for n in g.nodes}


def build_initial_state(g, seed, max_attempts, retries=30, log=print, init="sectors"):
    """Feasible initial state. ``init="sectors"`` (default) partitions each of
    the five real LAS sectors separately against its own per-team target
    (about 1,000 LSOAs and 11-17 stations each); the first accepted step
    replaces this level-2 grouping by a sampled one. ``init="global"`` runs
    the recursion on the whole graph, which is much slower on 66 candidates."""
    zones = sector_zones(g) if init == "sectors" else None
    for r in range(retries):
        s = seed * 1000 + 17 * r + 1
        set_seed(s)
        random.seed(s)
        t0 = time.perf_counter()
        try:
            partition = Partition.from_random_assignment(
                graph=g, epsilon=EPS_BASE, demand_target=W,
                assignment_class=Assignment, capacity_level=C_MAX_BASE,
                c_min=C_MIN_BASE, init_super_partition=False,
                max_attempts=max_attempts, super_assignment=zones,
            )
        except ProposalRejected as exc:
            log(f"  init attempt {r + 1} failed after {time.perf_counter() - t0:.0f}s: {exc}")
            continue
        state = ChainState.initial(
            partition=partition, energy=0.0, beta=0.0,
            energy_fn=compute_energy, super_facility_fn=super_facility_open_l1_median,
        )
        return state, r + 1, time.perf_counter() - t0
    raise RuntimeError(f"no feasible initial state after {retries} attempts (seed {seed})")


def run(seed: int, steps: int, snap_every: int, max_attempts: int, out_dir: Path,
        tag: str, log=print, init: str = "sectors") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.perf_counter()
    g = load_las_graph()
    Assignment.travel_times = load_station_travel_times(g)
    node_list = sorted(g.nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    station_index = {n: i for i, n in enumerate(
        sorted(n for n in g.nodes if g.nodes[n]["candidate"] == 1))}
    edges = np.array([(idx[u], idx[v]) for u, v in g.edges()], dtype=np.int32)
    total = sum(g.nodes[n]["demand"] for n in g.nodes)
    log(f"instance: |V|={g.number_of_nodes()} |E|={g.number_of_edges()} "
        f"stations={len(station_index)} demand={total:,.0f} units={math.ceil(total / W)} "
        f"travel pairs={len(Assignment.travel_times):,} ({time.perf_counter() - t_load:.0f}s)")

    state, init_tries, init_secs = build_initial_state(g, seed, max_attempts, log=log, init=init)
    log(f"initial state: |P1|={len(state.partition.parts)} "
        f"E={state.energy:,.0f} (attempt {init_tries}, {init_secs:.0f}s)")

    proposal = partial(
        hierarchical_recom,
        epsilon_base=EPS_BASE, epsilon_super=EPS_SUPER, demand_target=W,
        c_min_base=C_MIN_BASE, c_min_super=C_MIN_SUPER, c_max_super=C_MAX_SUPER,
        min_districts_super=MIN_DISTRICTS_SUPER,
        gamma_base=GAMMA_BASE, gamma_super=GAMMA_SUPER,
        max_attempts_base=max_attempts, max_attempts_super=1000,
    )
    chain = MarkovChain(proposal=proposal, constraints=[], accept=always_accept,
                        initial_state=state, total_steps=steps)

    def vectors(st):
        p = st.partition
        station = np.full(len(node_list), -1, dtype=np.int16)
        sup = np.zeros(len(node_list), dtype=np.int16)
        cap = np.zeros(len(node_list), dtype=np.int8)
        sup_label = {}
        for part, members in p.parts.items():
            c = st.facility.centers.get(part)
            si = station_index.get(c, -1)
            s_raw = p.super_assignment.get(part)
            s_lab = sup_label.setdefault(s_raw, len(sup_label))
            cap_p = int(p.teams.get(part, 0))
            for n in members:
                i = idx[n]
                station[i] = si
                sup[i] = s_lab
                cap[i] = cap_p
        return station, sup, cap

    def n_cut(st):
        a = np.empty(len(node_list), dtype=np.int32)
        m = st.partition.assignment.mapping
        for n, i in idx.items():
            a[i] = m[n]
        return int((a[edges[:, 0]] != a[edges[:, 1]]).sum())

    rows = []
    snaps = {"step": [], "station": [], "super": [], "capacity": []}
    causes = Counter()
    t0 = time.perf_counter()
    t_prev = t0
    last_print = t0
    for i, st in enumerate(chain):
        now = time.perf_counter()
        if i == 0:
            accepted, cause = True, None
        else:
            cause = chain.last_rejection
            accepted = cause is None
            if cause:
                causes[cause] += 1
        e1 = float(sum(st.facility.radii.values()))
        rows.append((i, int(accepted), cause or "", float(st.energy), e1,
                     len(st.partition.parts), len(st.partition.super_parts),
                     n_cut(st), now - t_prev))
        t_prev = now
        if i % snap_every == 0:
            station, sup, cap = vectors(st)
            snaps["step"].append(i)
            snaps["station"].append(station)
            snaps["super"].append(sup)
            snaps["capacity"].append(cap)
        if now - last_print >= 30 or i == steps - 1:
            acc = sum(r[1] for r in rows[1:]) / max(1, i)
            log(f"  step {i:6d}/{steps}  |P1|={len(st.partition.parts):3d} "
                f"|P2|={len(st.partition.super_parts):2d}  E={st.energy:,.0f}  "
                f"acc={acc:.2f}  {i / (now - t0):.1f} steps/s")
            last_print = now
    elapsed = time.perf_counter() - t0

    stem = f"{tag}_s{seed}_T{steps}"
    np.savez_compressed(
        out_dir / f"snap_{stem}.npz",
        step=np.array(snaps["step"], dtype=np.int32),
        station=np.stack(snaps["station"]), super=np.stack(snaps["super"]),
        capacity=np.stack(snaps["capacity"]),
        node_ids=np.array(node_list, dtype=np.int32),
        node_lsoa=np.array([g.nodes[n]["LSOA21CD"] for n in node_list]),
        station_node_ids=np.array(sorted(station_index, key=station_index.get), dtype=np.int32),
        edges=edges,
    )
    trace = pd.DataFrame(rows, columns=["step", "accepted", "cause", "E", "E1", "P1", "P2",
                                        "cut_edges", "seconds"])
    trace.to_csv(out_dir / f"trace_{stem}.csv", index=False)
    summary = {
        "tag": tag, "seed": seed, "steps": steps, "snap_every": snap_every,
        "max_attempts_base": max_attempts, "counting": True,
        "rule": CutParams(ideal_demand=1.0, epsilon=0.1, capacity_level=1, n_teams=1).rule,
        "falcomchain_version": __import__("importlib.metadata").metadata.version("falcomchain"),
        "calibration": {"w": W, "eps_base": EPS_BASE, "eps_super": EPS_SUPER,
                        "c_min_base": C_MIN_BASE, "c_max_base": C_MAX_BASE,
                        "c_min_super": C_MIN_SUPER, "c_max_super": C_MAX_SUPER,
                        "min_districts_super": MIN_DISTRICTS_SUPER,
                        "gamma_base": GAMMA_BASE, "gamma_super": GAMMA_SUPER},
        "init": init, "init_attempts": init_tries, "init_seconds": round(init_secs, 1),
        "runtime_seconds": round(elapsed, 1), "steps_per_second": round(steps / elapsed, 2),
        "acceptance_rate": round(float(trace["accepted"][1:].mean()), 4),
        "rejection_causes": dict(causes),
        "travel_pairs_missing_from_road_matrix": Assignment.travel_times.n_missing_pairs,
        "travel_fallback_lookups": Assignment.travel_times.n_fallbacks,
        "P1_range": [int(trace["P1"].min()), int(trace["P1"].max())],
        "P2_range": [int(trace["P2"].min()), int(trace["P2"].max())],
        "E_final": float(trace["E"].iloc[-1]),
    }
    (out_dir / f"summary_{stem}.json").write_text(json.dumps(summary, indent=2))
    log(f"done: {steps} steps in {elapsed:.0f}s ({steps / elapsed:.1f}/s), "
        f"acceptance {summary['acceptance_rate']:.3f}, causes {dict(causes)}")
    return out_dir / f"summary_{stem}.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--snap-every", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=1000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tag", default="real")
    ap.add_argument("--init", choices=["sectors", "global"], default="sectors")
    a = ap.parse_args()
    run(a.seed, a.steps, a.snap_every, a.max_attempts, Path(a.out_dir), a.tag, init=a.init)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
