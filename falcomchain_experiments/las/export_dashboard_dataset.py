"""
Export FalcomPlot dashboard datasets for akirtisoglu.me/research/falcom.

Two datasets:

* ``las``  — the paper's locked capacity-block calibration (reuses
  run_chain_v3's setup verbatim: CDBA candidates, w_unit = 10,887,
  eps = 0.15, c^1 in [1,3], c^2 in [2,6]).  Writes per-step supers +
  centers so the viewer can draw the two-level hierarchy, an
  ensemble.json for the boundary/facility overlays, and a blocks.json
  regenerated with topology-preserving simplification so shared
  borders match vertex-for-vertex (required by the viewer's
  super-boundary layer).
* ``grid`` — a 20x20 synthetic grid recorded WITH substeps, so the
  spanning-tree phase animation works on the public dashboard.

Run:
    python -m falcomchain_experiments.las.export_dashboard_dataset --dataset grid
    python -m falcomchain_experiments.las.export_dashboard_dataset --dataset las --steps 1000
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path

import sys
sys.path.insert(0, "/Users/kirtisoglu/GitHub/FalcomChain")

from falcomchain import MarkovChain, Partition, always_accept, hierarchical_recom
from falcomchain.ensemble import EnsembleStats
from falcomchain.markovchain.state import ChainState
from falcomchain.partition.assignment import Assignment
from falcomchain.random import set_seed

REPO = Path(__file__).resolve().parents[2]
SITE = Path("/Users/kirtisoglu/GitHub/website/static/falcomplot")


# --------------------------------------------------------------------------
# Shared writer
# --------------------------------------------------------------------------

def _assignment_blob(partition) -> dict:
    return {
        str(n): str(part)
        for part, nodes in partition.parts.items()
        for n in nodes
    }


def _districts_blob(graph, partition) -> dict:
    out = {}
    for part, nodes in partition.parts.items():
        out[str(part)] = {
            "nodes": len(nodes),
            "demand": float(sum(graph.nodes[v]["demand"] for v in nodes)),
            "teams": int(partition.teams.get(part, 0)),
        }
    return out


def _step_blob(graph, state, i, accepted, prev_assignment) -> tuple[dict, dict]:
    p = state.partition
    assignment = _assignment_blob(p)
    changed = {
        k: v for k, v in assignment.items()
        if prev_assignment.get(k) != v
    }
    supers = {str(d): str(s) for d, s in p.super_assignment.items()}
    centers = {
        str(d): (str(c) if c is not None else None)
        for d, c in (state.facility.centers if state.facility else {}).items()
    }
    blob = {
        "step": i,
        "accepted": bool(accepted),
        "energy": float(state.energy) if state.energy is not None else None,
        "log_proposal_ratio": float(getattr(state, "log_proposal_ratio", 0.0) or 0.0),
        "assignment": assignment,
        "changed_nodes": changed,
        "districts": _districts_blob(graph, p),
        "supers": supers,
        "centers": centers,
    }
    return blob, assignment


def _ensemble_blob(ensemble: EnsembleStats) -> dict:
    boundary = {
        f"{u}|{v}": round(f, 4)
        for (u, v), f in ensemble.boundary.frequencies().items()
    }
    facility = {
        str(n): round(f, 4)
        for n, f in ensemble.facility.frequencies().items()
    }
    return {
        "n_samples": ensemble.n_samples,
        "boundary_frequencies": boundary,
        "boundary_robust_count": len(ensemble.boundary.robust(0.9)),
        "facility_frequencies": facility,
        "facility_essential_count": len(ensemble.facility.essential(0.9)),
        "facility_substitutable_count": len(ensemble.facility.substitutable(0.5)),
        "capacity": ensemble.capacity.summary(),
    }


def run_and_write(
    *, graph, initial_state, proposal, total_steps, out_dir: Path,
    coordinates: dict, candidates: dict, params: dict,
    candidates_l2: dict | None = None,
    candidates_repaired: dict | None = None,
    burn_in: int, thin: int = 2,
    recorder=None, log_every: int = 50,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble = EnsembleStats(burn_in=burn_in, thin=thin)

    accepted_flags = []

    def _cb(st, accepted):
        accepted_flags.append(bool(accepted))
        ensemble.observe(st, accepted)

    chain = MarkovChain(
        proposal=proposal, constraints=[], accept=always_accept,
        initial_state=initial_state, total_steps=total_steps,
        recorder=recorder,
    )
    chain.callbacks = [_cb]

    energy_series = []
    district_series = []
    prev_assignment: dict = {}
    t0 = time.perf_counter()
    for i, st in enumerate(chain, start=1):
        accepted = accepted_flags[-1] if accepted_flags else True
        blob, prev_assignment = _step_blob(graph, st, i, accepted, prev_assignment)
        (out_dir / f"step_{i:04d}.json").write_text(
            json.dumps(blob, separators=(",", ":")))
        energy_series.append(blob["energy"])
        district_series.append(len(st.partition.parts))
        if i % log_every == 0 or i == total_steps:
            rate = i / max(1e-9, time.perf_counter() - t0)
            print(f"  step {i:5d}/{total_steps}  |P1|={district_series[-1]:>3}  "
                  f"E={blob['energy'] or 0:>14,.0f}  ({rate:.2f} steps/s)",
                  flush=True)

    manifest = {
        "total_steps": total_steps,
        "graph_nodes": graph.number_of_nodes(),
        "parameters": params,
        "node_coordinates": coordinates,
        "node_candidates": candidates,
        "energy_series": energy_series,
        "energy_series_label": "demand-weighted travel time",
        "n_districts_series": district_series,
    }
    if candidates_l2:
        manifest["node_candidates_l2"] = candidates_l2
    if candidates_repaired:
        manifest["node_candidates_repaired"] = candidates_repaired
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")))
    (out_dir / "ensemble.json").write_text(
        json.dumps(_ensemble_blob(ensemble), separators=(",", ":")))
    (out_dir / "adjacency.json").write_text(json.dumps(
        [[str(u), str(v)] for u, v in graph.edges()], separators=(",", ":")))
    print(f"  wrote manifest, ensemble ({ensemble.n_samples} samples), adjacency")


# --------------------------------------------------------------------------
# LAS dataset (locked calibration, reusing run_chain_v3's setup)
# --------------------------------------------------------------------------

def export_las(steps: int, seed: int, out_dir: Path) -> None:
    from .run_chain_v3 import (
        C_MAX_BASE, C_MAX_SUPER, C_MIN_BASE, C_MIN_SUPER, DEFAULT_CDBA_CSV,
        DEFAULT_TRAVEL, EPS_BASE, EPS_SUPER, GAMMA_BASE, GAMMA_SUPER,
        MIN_DISTRICTS_SUPER, W, build_initial_state,
    )
    import geopandas as gpd
    import pandas as pd
    import shapely

    state, g = build_initial_state(DEFAULT_CDBA_CSV, DEFAULT_TRAVEL, seed=seed)

    base_proposal = partial(
        hierarchical_recom,
        epsilon_base=EPS_BASE, epsilon_super=EPS_SUPER, demand_target=W,
        c_min_base=C_MIN_BASE, c_min_super=C_MIN_SUPER, c_max_super=C_MAX_SUPER,
        min_districts_super=MIN_DISTRICTS_SUPER,
        gamma_base=GAMMA_BASE, gamma_super=GAMMA_SUPER,
    )

    def proposal(st):
        try:
            return base_proposal(st)
        except RuntimeError:
            raise
        except Exception as e:  # rare stranded-root cut -> clean rejection
            raise RuntimeError(f"proposal failed: {type(e).__name__}: {e}") from e

    coordinates = {
        str(n): [round(g.nodes[n]["LONG"], 6), round(g.nodes[n]["LAT"], 6)]
        for n in g.nodes
    }
    candidates = {
        str(n): bool(g.nodes[n].get("candidate"))
        and not g.nodes[n].get("candidate_artificial")
        for n in g.nodes
    }
    candidates_l2 = {str(n): bool(g.nodes[n].get("candidate_l2")) for n in g.nodes}
    candidates_repaired = {str(n): bool(g.nodes[n].get("candidate")) for n in g.nodes}

    params = {
        "w_unit": W, "epsilon": EPS_BASE, "epsilon_super": EPS_SUPER,
        "c_min": C_MIN_BASE, "capacity_level": C_MAX_BASE,
        "c_min_super": C_MIN_SUPER, "c_max_super": C_MAX_SUPER,
        "gamma": GAMMA_BASE, "demand_target": W, "seed": seed,
        "calibration": "capacity-block (paper Section 7.4, locked 2026-07-02)",
    }

    run_and_write(
        graph=g, initial_state=state, proposal=proposal, total_steps=steps,
        out_dir=out_dir, coordinates=coordinates, candidates=candidates,
        candidates_l2=candidates_l2, candidates_repaired=candidates_repaired,
        params=params, burn_in=max(1, steps // 10),
    )

    # blocks.json — catchment polygons with TOPOLOGY-PRESERVING
    # simplification, so adjacent features share vertices and the
    # viewer's super-boundary layer can match shared segments.
    print("  building blocks.json (coverage_simplify)…")
    lsoa = gpd.read_file(REPO / "data/raw/LSOA_2021_London.gpkg")[
        ["LSOA21CD", "LSOA21NM", "geometry"]]
    code_of = {str(n): g.nodes[n]["LSOA21CD"] for n in g.nodes}
    keep = set(code_of.values())
    lsoa = lsoa[lsoa["LSOA21CD"].isin(keep)].reset_index(drop=True)
    lsoa["geometry"] = shapely.coverage_simplify(lsoa.geometry.values, tolerance=25)
    lsoa = lsoa.to_crs(4326)
    id_of = {v: k for k, v in code_of.items()}
    features = []
    for _, row in lsoa.iterrows():
        geom = shapely.geometry.mapping(shapely.set_precision(row.geometry, 1e-6))
        features.append({
            "type": "Feature",
            "properties": {
                "id": id_of[row["LSOA21CD"]],
                "LSOA21CD": row["LSOA21CD"],
                "LSOA21NM": row["LSOA21NM"],
            },
            "geometry": geom,
        })
    (out_dir / "blocks.json").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features},
        separators=(",", ":")))
    print(f"  wrote blocks.json ({len(features)} features)")



# --------------------------------------------------------------------------
# Grid dataset (with Recorder substeps for the tree animation)
# --------------------------------------------------------------------------

def export_grid(steps: int, seed: int, out_dir: Path) -> None:
    from falcomchain.graph.grid import Grid
    from falcomchain.markovchain.energy import compute_energy
    from falcomchain.tree.snapshot import Recorder

    set_seed(seed)
    graph = Grid(dimensions=(20, 20), num_candidates=80, density="random").graph

    Assignment.travel_times = {
        (a, b): float(
            abs(graph.nodes[a]["C_X"] - graph.nodes[b]["C_X"])
            + abs(graph.nodes[a]["C_Y"] - graph.nodes[b]["C_Y"])
        )
        for a in graph.nodes for b in graph.nodes
    }

    demand_target = 2000
    total = sum(d["demand"] for _, d in graph.nodes(data=True))
    seed_target = total / max(1, math.ceil(total / demand_target))
    partition = Partition.from_random_assignment(
        graph=graph, epsilon=0.3, demand_target=seed_target,
        assignment_class=None, capacity_level=2,
    )
    state = ChainState.initial(
        partition=partition, energy=0.0, beta=1.0, energy_fn=compute_energy)

    proposal = partial(
        hierarchical_recom, epsilon_base=0.3, epsilon_super=0.3,
        demand_target=demand_target,
    )

    coordinates = {
        str(n): [graph.nodes[n]["C_X"], graph.nodes[n]["C_Y"]] for n in graph.nodes
    }
    candidates = {str(n): bool(d.get("candidate")) for n, d in graph.nodes(data=True)}
    params = {
        "epsilon": 0.3, "demand_target": demand_target, "capacity_level": 2,
        "seed": seed,
    }

    tmp = Path(tempfile.mkdtemp(prefix="fp_grid_rec_"))
    recorder = Recorder(str(tmp), record_substeps=True)
    recorder.write_header(graph, partition, params=params)

    run_and_write(
        graph=graph, initial_state=state, proposal=proposal, total_steps=steps,
        out_dir=out_dir, coordinates=coordinates, candidates=candidates,
        params=params, burn_in=max(1, steps // 10), recorder=recorder,
    )
    recorder.close()

    # Export the recorder's output and keep only the phase files (the
    # step JSON we already wrote ourselves, with supers/centers
    # included). The viewer fetches phases/phases_NNNN.json for the
    # spanning-tree animation.
    rec_out = tmp / "json"
    Recorder.export_to_json(str(tmp), str(rec_out))
    sub_src = rec_out / "phases"
    if sub_src.exists():
        sub_dst = out_dir / "phases"
        if sub_dst.exists():
            shutil.rmtree(sub_dst)
        shutil.copytree(sub_src, sub_dst)
        print(f"  copied {len(list(sub_dst.glob('*.json')))} phase files")
    else:
        print("  WARNING: recorder produced no phase files")
    shutil.rmtree(tmp, ignore_errors=True)

    # blocks.json — unit squares around each grid node.
    features = []
    for n, d in graph.nodes(data=True):
        x, y = d["C_X"], d["C_Y"]
        features.append({
            "type": "Feature",
            "properties": {"id": str(n), "demand": d.get("demand"),
                           "candidate": int(bool(d.get("candidate")))},
            "geometry": {"type": "Polygon", "coordinates": [[
                [x - 0.5, y - 0.5], [x + 0.5, y - 0.5],
                [x + 0.5, y + 0.5], [x - 0.5, y + 0.5], [x - 0.5, y - 0.5],
            ]]},
        })
    (out_dir / "blocks.json").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features},
        separators=(",", ":")))
    print(f"  wrote blocks.json ({len(features)} features)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["las", "grid"], required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset == "las":
        steps = args.steps or 1000
        out = Path(args.out) if args.out else SITE / "las_v2"
        export_las(steps, args.seed, out)
    else:
        steps = args.steps or 300
        out = Path(args.out) if args.out else SITE / "grid_20x20_v2"
        export_grid(steps, args.seed, out)
    print(f"done → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
