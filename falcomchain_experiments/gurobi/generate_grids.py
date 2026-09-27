"""
Small grid instances for the Gurobi MILP experiments (paper §3 model).

Two instances:
- ``grid_100.json``  — 10x10, paper-strict hyperparameters
- ``grid_400.json``  — 20x20, paper-strict hyperparameters

Same node schema as the scalability grids
(``id``, ``demand``, ``C_X``, ``C_Y``, ``candidate``, ``super_candidate``)
so the existing :mod:`plot_dual_plotly` works unchanged.

Paper hyperparameters (§6.4 / §7.1):
    - demand ~ Uniform(80, 120)
    - L1 candidate density rho = 0.05
    - L2 super-candidate density = 0.20 of L1
    - epsilon^1 = 0.10, epsilon^2 = 0.15
    - c^1_max = 4, c^2_max = 3

For these small sizes we *skip* CDBA repair: at n=100 the total
demand is approx 10,000 and the per-team workload defaults below keep the
MILP comfortably feasible without artificial candidates. Adjusted
per-size workload targets:

    grid_100:  w = 1,000  (~10 teams cover total demand)
    grid_400:  w = 2,000  (~20 teams cover total demand)

Usage::

    PYTHONHASHSEED=0 python3 generate_grids.py
"""

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import networkx as nx


_REPO_EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_EXP))

from grids import make_grid  # noqa: E402  (sibling module)


# ---------------------------------------------------------------------------
# Per-instance hyperparameters (paper §3 / §6.4)
# ---------------------------------------------------------------------------
INSTANCES = [
    dict(n_nodes=100, seed=42, demand_target=1_000,
         epsilon_l1=0.10, epsilon_l2=0.15,
         c_max_l1=4, c_max_l2=3, w=1_000),
    dict(n_nodes=400, seed=43, demand_target=2_000,
         epsilon_l1=0.10, epsilon_l2=0.15,
         c_max_l1=4, c_max_l2=3, w=2_000),
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_one(spec: dict, out_dir: Path) -> dict:
    n = spec["n_nodes"]
    seed = spec["seed"]
    print(f"\n=== grid |V|={n} (seed={seed}) ===")

    g_falcom = make_grid(n, seed=seed)
    # Convert to plain nx.Graph for node_link_data round-trip.
    g = nx.Graph()
    for node, data in g_falcom.nodes(data=True):
        g.add_node(node, **data)
    for u, v, data in g_falcom.edges(data=True):
        g.add_edge(u, v, **data)
    g.graph.update(dict(g_falcom.graph))

    n_l1 = sum(1 for _, d in g.nodes(data=True) if d.get("candidate"))
    n_l2 = sum(1 for _, d in g.nodes(data=True) if d.get("super_candidate"))
    total_demand = sum(d["demand"] for _, d in g.nodes(data=True))
    w_max = max(d["demand"] for _, d in g.nodes(data=True))
    print(
        f"  {g.number_of_nodes()} nodes, {g.number_of_edges()} edges; "
        f"L1: {n_l1}, L2: {n_l2}; total demand: {total_demand:,}; "
        f"w_max: {w_max}"
    )

    grid_path = out_dir / f"grid_{n}.json"
    with open(grid_path, "w") as f:
        json.dump(nx.node_link_data(g, edges="adjacency"), f)
    sha = _file_sha256(grid_path)
    print(f"  wrote {grid_path.name} ({grid_path.stat().st_size:,} bytes, "
          f"sha256={sha[:16]}...)")

    meta = {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "rows": g.graph["rows"],
        "cols": g.graph["cols"],
        "seed": seed,
        "n_l1_candidates": n_l1,
        "n_l2_candidates": n_l2,
        "total_demand": total_demand,
        "w_max": w_max,
        "demand_low": 80,
        "demand_high": 120,
        "demand_target_w": spec["w"],
        "epsilon_l1": spec["epsilon_l1"],
        "epsilon_l2": spec["epsilon_l2"],
        "c_max_l1": spec["c_max_l1"],
        "c_max_l2": spec["c_max_l2"],
        "grid_sha256": sha,
        "grid_path": grid_path.name,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    meta_path = out_dir / f"grid_{n}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path.name}")
    return meta


def main():
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = [build_one(s, out_dir) for s in INSTANCES]
    idx = {
        "experiment": "gurobi_milp",
        "paper_section": "3",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "instances": metas,
    }
    with open(out_dir / "index.json", "w") as f:
        json.dump(idx, f, indent=2)
    print(f"\nWrote {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
