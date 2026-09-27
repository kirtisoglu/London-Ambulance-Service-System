"""
Deterministic, single-entry regeneration of the MILP-benchmark grids.

Reproducibility contract
------------------------
Run as::

    PYTHONHASHSEED=0 python3 build_instances.py

Each instance is rebuilt from scratch by a fixed pipeline:

  1. ``make_grid(n, seed)`` — square grid, demand ~ Uniform(80,120),
     real L1 candidates at density rho = 0.05.  Deterministic in `seed`.
  2. CDBA two-phase repair at ``V_max = (1-eps)*w`` and
     ``d_max = floor(V_max / d_max_node) - 1`` to bound facility-free
     components, then a ``fast_center`` greedy patch so Assumption 6.1
     holds *strictly* at the MILP's (eps, w).
  3. L2 super-candidates: a fixed-seed random subset of the resulting
     L1 set, of size ``round(0.20 * |F^1|)``.

Determinism rests on (a) ``PYTHONHASHSEED=0`` (CDBA and the feasibility
checker iterate over Python sets), and (b) pinned RNG seeds. Each grid
is written with a complete, internally consistent meta file including
the SHA-256 of the JSON, so downstream consumers can verify the input.

All MILP / FalCom parameters used downstream are recorded in the meta
so the whole §7 benchmark is reconstructible from these files alone.
"""

import datetime as _dt
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import networkx as nx

_EXP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_ROOT))

from grids import make_grid  # noqa: E402
from las.cdba_scaffold import cdba_two_phase  # noqa: E402
from falcomchain.candidates.feasibility import (  # noqa: E402
    check_facility_density, repair_facility_density,
)


# ---------------------------------------------------------------------------
# Frozen instance specifications (the §7 benchmark configuration).
# ---------------------------------------------------------------------------
INSTANCES = {
    # w = 200*sqrt(n): grid_100->2000, grid_400->4000, grid_1000->6000,
    # grid_10000->20000, grid_50000->45000. eps = 0.15 throughout.
    100:   dict(n_nodes=100,   seed=42, rho=0.05, w=2000,  eps=0.15,
                c_min_l1=1, c_max_l1=2, c_min_l2=2, c_max_l2=5,
                mu2=2, l2_ratio=0.20),
    400:   dict(n_nodes=400,   seed=43, rho=0.05, w=4000,  eps=0.15,
                c_min_l1=1, c_max_l1=2, c_min_l2=2, c_max_l2=5,
                mu2=2, l2_ratio=0.20),
    1000:  dict(n_nodes=1000,  seed=44, rho=0.05, w=6000,  eps=0.15,
                c_min_l1=1, c_max_l1=2, c_min_l2=2, c_max_l2=5,
                mu2=2, l2_ratio=0.20),
    10000: dict(n_nodes=10000, seed=45, rho=0.05, w=20000, eps=0.15,
                c_min_l1=1, c_max_l1=2, c_min_l2=2, c_max_l2=5,
                mu2=2, l2_ratio=0.20),
    50000: dict(n_nodes=50000, seed=46, rho=0.05, w=45000, eps=0.15,
                c_min_l1=1, c_max_l1=2, c_min_l2=2, c_max_l2=5,
                mu2=2, l2_ratio=0.20),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_one(spec: dict, out_dir: Path) -> dict:
    n, seed, w, eps = spec["n_nodes"], spec["seed"], spec["w"], spec["eps"]
    print(f"\n=== building grid_{n} (seed={seed}, w={w}, eps={eps}) ===")

    # --- 1. base grid + real candidates --------------------------------
    gf = make_grid(n, seed=seed, candidate_density=spec["rho"])
    g = nx.Graph()
    for node, data in gf.nodes(data=True):
        g.add_node(node, **data)
    for u, v, data in gf.edges(data=True):
        g.add_edge(u, v, **data)
    g.graph.update(dict(gf.graph))
    n_real = sum(1 for _, d in g.nodes(data=True) if d.get("candidate"))
    total_d = sum(d["demand"] for _, d in g.nodes(data=True))
    d_max_node = max(d["demand"] for _, d in g.nodes(data=True))
    print(f"  base: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
          f"real L1={n_real}, total demand={total_d:,}, d_max={d_max_node}")

    # --- 2. CDBA repair + greedy patch to satisfy Assumption 6.1 -------
    V_max = (1 - eps) * w
    d_max = max(1, int(V_max // d_max_node) - 1)
    cdba_added = cdba_two_phase(g, V_max=V_max, d_max=d_max, log=False)
    chk = check_facility_density(g, demand_target=w, epsilon=eps,
                                 c_min=spec["c_min_l1"])
    greedy_added = []
    if not chk.passes:
        greedy_added = repair_facility_density(
            g, demand_target=w, epsilon=eps, c_min=spec["c_min_l1"],
            strategy="fast_center",
        ) or []
    chk = check_facility_density(g, demand_target=w, epsilon=eps,
                                 c_min=spec["c_min_l1"])
    n_l1 = sum(1 for _, d in g.nodes(data=True) if d.get("candidate"))
    assert chk.passes, "Assumption 6.1 not satisfied after repair"
    print(f"  repair: +{len(cdba_added)} CDBA +{len(greedy_added)} greedy "
          f"=> L1={n_l1}; Assumption 6.1 worst={chk.worst_demand:.0f} "
          f"< threshold={chk.threshold:.0f}")

    # --- 3. L2 super-candidates: fixed-seed subset of L1 ---------------
    n_l2 = max(1, round(spec["l2_ratio"] * n_l1))
    cand_ids = sorted(node for node, d in g.nodes(data=True)
                      if d.get("candidate"))
    rng = random.Random(seed)
    super_set = set(rng.sample(cand_ids, k=n_l2))
    for node, d in g.nodes(data=True):
        d["super_candidate"] = 1 if node in super_set else 0
    print(f"  L2: {n_l2} super-candidates (round({spec['l2_ratio']}*{n_l1}))")

    # --- 4. serialise + consistent meta --------------------------------
    grid_path = out_dir / f"grid_{n}.json"
    with open(grid_path, "w") as f:
        json.dump(nx.node_link_data(g, edges="adjacency"), f)
    sha = _sha256(grid_path)

    meta = {
        "n_nodes": n, "seed": seed,
        "rows": g.graph["rows"], "cols": g.graph["cols"],
        "n_edges": g.number_of_edges(),
        "total_demand": total_d, "d_max_node": d_max_node,
        "rho": spec["rho"],
        "n_l1_real": n_real,
        "n_l1_artificial": n_l1 - n_real,
        "n_l1_candidates": n_l1,
        "n_l2_candidates": n_l2,
        # MILP / FalCom parameters (single source of truth for §7):
        "demand_target_w": w,
        "epsilon_l1": eps, "epsilon_l2": eps,
        "c_min_l1": spec["c_min_l1"], "c_max_l1": spec["c_max_l1"],
        "c_min_l2": spec["c_min_l2"], "c_max_l2": spec["c_max_l2"],
        "min_l1_per_l2": spec["mu2"],
        # repair provenance:
        "repair": "cdba_two_phase + fast_center patch",
        "cdba_V_max": V_max, "cdba_d_max": d_max,
        "assumption_6_1_passes": True,
        "assumption_6_1_threshold": float(chk.threshold),
        "assumption_6_1_worst_demand": float(chk.worst_demand),
        "grid_sha256": sha,
        "grid_path": grid_path.name,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    meta_path = out_dir / f"grid_{n}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {grid_path.name} (sha256={sha[:16]}...) and {meta_path.name}")
    return meta


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("ERROR: set PYTHONHASHSEED=0 for deterministic CDBA/feasibility.\n"
              "  PYTHONHASHSEED=0 python3 build_instances.py", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Optional CLI: build only the requested sizes (default: all).
    sizes = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not sizes:
        sizes = sorted(INSTANCES)
    metas = [build_one(INSTANCES[n], out_dir) for n in sizes]
    # Index reflects ALL grids on disk (not only those built this run).
    all_meta = []
    for n in sorted(INSTANCES):
        mp = out_dir / f"grid_{n}.meta.json"
        if mp.exists():
            all_meta.append(json.load(open(mp)))
    index = {
        "experiment": "milp_benchmark",
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "pythonhashseed": "0",
        "instances": all_meta,
    }
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print("\n=== summary (all grids on disk) ===")
    for m in all_meta:
        print(f"  grid_{m['n_nodes']}: L1={m['n_l1_candidates']} "
              f"({m['n_l1_real']}+{m['n_l1_artificial']}), "
              f"L2={m['n_l2_candidates']}, sha={m['grid_sha256'][:12]}")


if __name__ == "__main__":
    main()
