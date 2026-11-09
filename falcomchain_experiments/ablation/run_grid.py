"""
Ablation run on a synthetic grid: one variant of ``variants.VARIANTS``.

    PYTHONHASHSEED=0 python -m falcomchain_experiments.ablation.run_grid \
        --size 1000 --variant noresid --steps 10000 --seed 42

Writes ``falcomchain_experiments/ablation/results/<variant>/grid_<N>_seed<S>.json``
in the format of ``scalability/run_scalability.py`` (per-step time, its split,
the level-1 access cost R1 and the demand spread, rejection causes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scalability"))

import run_scalability  # noqa: E402

from .variants import VARIANTS, patch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    a = ap.parse_args()
    overrides = patch(run_scalability, a.variant)
    out_dir = Path(a.out_dir) / a.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out = run_scalability.run_one(a.size, a.seed, a.steps, out_dir, track_structural=True)
    path = out_dir / f"grid_{a.size}_seed{a.seed}.json"
    out["ablation_variant"] = a.variant
    out["ablation_overrides"] = overrides
    path.write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
