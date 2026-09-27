"""
Generate the three synthetic grids for the scalability experiment.

Reproducibility contract:
- Each grid is built deterministically from a fixed seed (42 + i for the
  i-th grid in PAPER_SIZES).
- Demands ~ Uniform(80, 120), ρ = 0.05 candidates, paper §6.4.
- The graph is then **repaired** to satisfy Assumption 6.1 at the
  experiment's chosen (demand_target, ε, c_min) using `fast_center`
  (deterministic when PYTHONHASHSEED is set, hence the runner script
  invokes Python with PYTHONHASHSEED=0).
- The repaired graph is serialised in NetworkX node_link_graph format
  to `data/grid_{N}.json`.
- A companion `data/grid_{N}.meta.json` records the hyperparameters
  used for repair, the candidate counts before/after, and the
  environment fingerprint (Python and library versions).

Usage::

    PYTHONHASHSEED=0 python3 generate_grids.py

This is run once; the resulting JSON files are committed (or
re-generated on demand) and consumed by `run_scalability.py`.
"""

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import networkx as nx

# Make the experiment package importable
_EXP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_ROOT))

from grids import make_grid  # noqa: E402  (sibling module)
from falcomchain.candidates.feasibility import (  # noqa: E402
    check_facility_density,
    repair_facility_density,
)
from las.cdba_scaffold import cdba_two_phase  # noqa: E402  (LAS-matched repair)


# ---------------------------------------------------------------------------
# Experiment hyperparameters (paper §6.4 / scalability §6.6)
# ---------------------------------------------------------------------------
SIZES = [10_000, 20_000, 50_000]

# Demand-target / capacity / tolerance (see notes/design.md)
# Two relaxations from paper §6.4 for the chain to run end-to-end at scale:
#  1. ε relaxed from 0.10 → 0.30. Spanning-tree retry exhausts at tighter ε
#     for |V| ≥ 2000 with U(80,120) demand; 0.30 is the smallest ε that
#     consistently builds the initial unit-capacity partition.
#  2. c¹_max = 1 (unit-capacity L1). The chain's super-edge admissibility
#     check enforces `2 ≤ teams ≤ c²_max` per super-district (hardcoded
#     `teams >= 2`, capped at c²_max). With paper's c²_max=3 and c¹_max=4,
#     any L1 district reaching c¹=4 teams cannot fit in any super-district.
#     Forcing c¹_max=1 keeps the chain in the unit-capacity refinement
#     regime where c²=2 or 3 always admits some merge.
DEMAND_TARGET = 10_000
EPSILON_L1 = 0.50
EPSILON_L2 = 0.50
C_MIN_L1 = 1
C_MAX_L1 = 3
C_MIN_L2 = 2
C_MAX_L2 = 6
GAMMA = 0.0

# Repair strategy — CDBA two-phase (capacitated diameter-bounded
# augmentation), matching the LAS pipeline. Phase 1 places candidates
# to cover the graph in demand-bounded balls (V_max = (1+ε)·c_max·w);
# Phase 2 places additional candidates at graph 1-centers of any
# facility-free component whose diameter exceeds CDBA_D_MAX, until
# every facility-free component has bounded diameter and volume.
# Produced 890 artificials for LAS (vs 1,519 with fast_center).
REPAIR_STRATEGY = "cdba_two_phase"
CDBA_D_MAX = 12        # matches the LAS-tuned value


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def _env_fingerprint() -> dict:
    """Return a dict describing the running environment."""
    import importlib.metadata as md

    def _ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "packages": {
            "networkx": _ver("networkx"),
            "numpy": _ver("numpy"),
            "scipy": _ver("scipy"),
            "matplotlib": _ver("matplotlib"),
            "falcomchain": _ver("falcomchain"),
        },
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# One-grid pipeline
# ---------------------------------------------------------------------------

def build_one_grid(n: int, seed: int, out_dir: Path) -> dict:
    """
    Build, repair, and serialise one grid. Return a dict of stats.
    """
    print(f"\n=== Generating grid |V|={n:,} (seed={seed}) ===")
    t0 = time.perf_counter()
    g_falcom = make_grid(n, seed=seed)
    g = nx.Graph()
    # Convert FalcomChain Graph subclass back to a plain nx.Graph so
    # node_link_graph round-trips cleanly. Copy node + edge attrs.
    for node, data in g_falcom.nodes(data=True):
        g.add_node(node, **data)
    for u, v, data in g_falcom.edges(data=True):
        g.add_edge(u, v, **data)
    g.graph.update(dict(g_falcom.graph))
    t_build = time.perf_counter() - t0

    n_real_candidates = sum(1 for _, d in g.nodes(data=True) if d.get("candidate"))
    total_demand = sum(d["demand"] for _, d in g.nodes(data=True))
    w_max = max(d["demand"] for _, d in g.nodes(data=True))
    print(f"  Built: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
          f"{n_real_candidates} real candidates, total demand={total_demand:,}, "
          f"w_max={w_max} (took {t_build:.2f}s)")

    # Granularity floor sanity check
    floor = w_max / (C_MIN_L1 * DEMAND_TARGET)
    print(f"  Granularity floor ε ≥ {floor:.4f}; chosen ε = {EPSILON_L1}")
    if floor > EPSILON_L1:
        raise RuntimeError(
            f"Granularity floor {floor:.4f} exceeds chosen ε={EPSILON_L1}. "
            f"Either raise DEMAND_TARGET or C_MIN_L1."
        )

    # Pre-repair feasibility check
    pre = check_facility_density(
        g, demand_target=DEMAND_TARGET, epsilon=EPSILON_L1, c_min=C_MIN_L1
    )
    print(f"  Pre-repair: passes={pre.passes}, "
          f"violating_components={len(pre.violating_components)}, "
          f"worst_demand={pre.worst_demand:,.0f}, threshold={pre.threshold:,.0f}")

    # CDBA two-phase repair — LAS-matched scaffolding.
    # V_max = max c-team band ceiling; cells below this volume are
    # absorbable by a single max-capacity district neighbor.
    V_MAX = (1 + EPSILON_L1) * C_MAX_L1 * DEMAND_TARGET
    t1 = time.perf_counter()
    if not pre.passes:
        added_list = cdba_two_phase(
            g, V_max=V_MAX, d_max=CDBA_D_MAX, log=False,
        )
        added = set(added_list)
    else:
        added = set()
    t_repair = time.perf_counter() - t1
    print(f"  Repair (cdba_two_phase, V_max={V_MAX:.0f}, d_max={CDBA_D_MAX}): "
          f"added {len(added):,} artificial candidates in {t_repair:.2f}s")

    # Post-repair feasibility check — soft, since CDBA scaffolds for the
    # relaxed Assumption 6.1' (bounded volume + bounded diameter per
    # facility-free component), not the strict 6.1.
    post = check_facility_density(
        g, demand_target=DEMAND_TARGET, epsilon=EPSILON_L1, c_min=C_MIN_L1
    )
    print(f"  Strict Assumption 6.1 post-check: passes={post.passes}, "
          f"violating={len(post.violating_components)}, "
          f"worst_demand={post.worst_demand:,.0f}")

    n_total_candidates = sum(1 for _, d in g.nodes(data=True) if d.get("candidate"))
    print(f"  Post-repair: {n_total_candidates} total candidates "
          f"({n_total_candidates - n_real_candidates} artificial)")

    # Serialise the graph
    grid_path = out_dir / f"grid_{n}.json"
    with open(grid_path, "w") as f:
        json.dump(nx.node_link_data(g, edges="adjacency"), f)
    grid_sha = _file_sha256(grid_path)
    print(f"  Wrote {grid_path.name} ({grid_path.stat().st_size:,} bytes, "
          f"sha256={grid_sha[:16]}…)")

    # Metadata
    stats = {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "rows": g.graph["rows"],
        "cols": g.graph["cols"],
        "seed": seed,
        "candidate_density_target": 0.05,
        "n_real_candidates": n_real_candidates,
        "n_artificial_candidates": n_total_candidates - n_real_candidates,
        "n_total_candidates": n_total_candidates,
        "total_demand": total_demand,
        "w_max": w_max,
        "demand_low": 80,
        "demand_high": 120,
        "demand_target": DEMAND_TARGET,
        "epsilon_l1": EPSILON_L1,
        "epsilon_l2": EPSILON_L2,
        "c_min_l1": C_MIN_L1,
        "c_max_l1": C_MAX_L1,
        "c_max_l2": C_MAX_L2,
        "gamma": GAMMA,
        "repair_strategy": REPAIR_STRATEGY,
        "granularity_floor": floor,
        "pre_repair_worst_demand": pre.worst_demand,
        "pre_repair_threshold": pre.threshold,
        "pre_repair_passes": pre.passes,
        "build_time_s": t_build,
        "repair_time_s": t_repair,
        "grid_sha256": grid_sha,
        "grid_path": str(grid_path.name),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    meta_path = out_dir / f"grid_{n}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote {meta_path.name}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if os.environ.get("PYTHONHASHSEED") not in ("0", "42"):
        print(
            "WARNING: PYTHONHASHSEED is not pinned. Set it for full "
            "determinism of the repair step. Re-running:\n"
            "    PYTHONHASHSEED=0 python3 generate_grids.py",
            file=sys.stderr,
        )

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _env_fingerprint()
    print(f"Environment: Python {env['python']} on {env['platform']}")
    print(f"  PYTHONHASHSEED={env['pythonhashseed']}")
    print(f"  Packages: {env['packages']}")

    all_stats = []
    for i, n in enumerate(SIZES):
        seed = 42 + i
        all_stats.append(build_one_grid(n, seed, out_dir))

    # Index file pointing at all generated artefacts
    index = {
        "experiment": "scalability",
        "paper_section": "6.6",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "environment": env,
        "grids": all_stats,
    }
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote index: {index_path}")

    # Headline summary
    print("\n=== Summary ===")
    print(f"{'V':>8} {'real':>5} {'artificial':>10} {'total':>7} {'repair_s':>10}")
    for s in all_stats:
        print(
            f"{s['n_nodes']:>8,} {s['n_real_candidates']:>5} "
            f"{s['n_artificial_candidates']:>10} "
            f"{s['n_total_candidates']:>7} {s['repair_time_s']:>10.2f}"
        )


if __name__ == "__main__":
    main()
