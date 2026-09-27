"""
Resample the L2 super-candidate set in an existing scalability grid JSON.

The original generator ``make_grid`` draws L2 super-candidates as 20% of
the *pre-repair* real L1 set. For the 1,000-node instance this gives
only 10 L2 markers, which under-fills the visualisation. This script
loads ``data/grid_{N}.json``, redraws ``super_candidate`` flags as a
new random subset of the *post-repair* L1 candidate set at the
requested count (drawn with the same RNG semantics as ``make_grid`` so
the picks are reproducible), and overwrites the file in place. The
underlying L1 set, demand, and adjacency are untouched.

Usage::

    python3 resample_super_candidates.py 1000 100   # 100 L2 in the 1k grid

Side effects: rewrites ``data/grid_{N}.json``; updates
``data/grid_{N}.meta.json`` if present.
"""

import json
import random
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def resample(n_nodes: int, n_super: int, *, seed: int = 42):
    grid_path = DATA_DIR / f"grid_{n_nodes}.json"
    if not grid_path.exists():
        raise SystemExit(f"Grid file not found: {grid_path}")

    with open(grid_path) as f:
        d = json.load(f)
    nodes = d["nodes"]

    cand_ids = sorted(n["id"] for n in nodes if n.get("candidate"))
    if n_super > len(cand_ids):
        raise SystemExit(
            f"Requested {n_super} super-candidates but only "
            f"{len(cand_ids)} L1 candidates exist in grid_{n_nodes}.json."
        )

    rng = random.Random(seed)
    new_super = set(rng.sample(cand_ids, k=n_super))

    for n in nodes:
        n["super_candidate"] = 1 if n["id"] in new_super else 0

    with open(grid_path, "w") as f:
        json.dump(d, f)
    print(
        f"Rewrote {grid_path.name}: "
        f"super_candidates -> {n_super} (drawn from {len(cand_ids)} L1; "
        f"seed={seed})"
    )

    meta_path = DATA_DIR / f"grid_{n_nodes}.meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["n_super_candidates"] = n_super
        meta["super_candidate_seed"] = seed
        meta["super_candidate_resampled"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Updated {meta_path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    resample(int(sys.argv[1]), int(sys.argv[2]))
