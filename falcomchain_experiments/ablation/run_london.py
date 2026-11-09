"""
Ablation run on the London instance: one variant of ``variants.VARIANTS``.

    PYTHONHASHSEED=0 python -m falcomchain_experiments.ablation.run_london \
        --variant noresid --steps 5000 --seed 1

Uses ``las/run_real_stations.run`` unchanged (sector-wise start built with the
default components, 97 units) and writes its summary, trace and snapshots to
``data/derived/ablation_london/<variant>/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..las import run_real_stations
from .variants import VARIANTS, patch

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--steps", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default=str(REPO / "data/derived/ablation_london"))
    a = ap.parse_args()
    patch(run_real_stations, a.variant)
    run_real_stations.run(a.seed, a.steps, 40, 1000, Path(a.out_dir) / a.variant, tag="abl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
