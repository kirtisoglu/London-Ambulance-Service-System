"""
Compute the candidate-LSOA x destination-LSOA travel-time matrix for a
CDBA candidate set. Both origins and destinations are keyed by
LSOA21CD so the chain's Assignment travel_times dict can be indexed by
(candidate_lsoa, dest_lsoa) directly.

Input:
    data/derived/cdba_candidates_w{W}_eps{EPS}_cmin{CMIN}_dmax{DMAX}.csv
Output:
    data/derived/cdba_travel_times_w{W}_eps{EPS}_cmin{CMIN}_dmax{DMAX}.parquet
    data/derived/cdba_travel_times_w{W}_eps{EPS}_cmin{CMIN}_dmax{DMAX}_meta.json

Travel-time model: OSMnx driving network of Greater London with edge
free-flow speeds scaled by the ambulance lights-and-sirens multiplier
(default 1.6x). Matches compute_travel_times.py except for the origin
ID scheme (LSOA21CD here vs L1:<station_code> there).

Run:
    python -m falcomchain_experiments.las.compute_cdba_travel_times
    python -m falcomchain_experiments.las.compute_cdba_travel_times \\
        --cdba data/derived/cdba_candidates_w3650_eps15_cmin4_dmax12.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd
import falcomtravel as ft

REPO = Path(__file__).resolve().parents[2]
GRAPH_JSON = REPO / "data/raw/london_graph.json"
GROUP_PATH = REPO / "data/derived/lsoa_to_group.csv"
DEFAULT_CDBA = (
    REPO
    / "data/derived/cdba_candidates_w3650_eps15_cmin3_dmax12.csv"
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cdba", default=str(DEFAULT_CDBA),
                    help="CDBA candidates CSV path")
    ap.add_argument("--place", default="Greater London, United Kingdom")
    ap.add_argument("--speed-factor", type=float, default=1.6)
    ap.add_argument("--fallback-kmh", type=float, default=35.0)
    args = ap.parse_args()

    cdba_path = Path(args.cdba)
    if not cdba_path.exists():
        raise FileNotFoundError(cdba_path)
    stem = cdba_path.stem.replace("cdba_candidates_", "")
    out_parquet = REPO / f"data/derived/cdba_travel_times_{stem}.parquet"
    out_meta = REPO / f"data/derived/cdba_travel_times_{stem}_meta.json"

    print(f"[1/4] Reading inputs")
    print(f"      CDBA candidates: {cdba_path.relative_to(REPO)}")
    candidates = pd.read_csv(cdba_path)
    print(f"      |F^1| = {len(candidates)}  "
          f"({(candidates['is_real']==1).sum()} real + "
          f"{(candidates['is_real']==0).sum()} artificial)")

    with open(GRAPH_JSON) as f:
        raw = json.load(f)
    lsoa_coords = {
        n["LSOA21CD"]: (float(n["LONG"]), float(n["LAT"]))
        for n in raw["nodes"]
    }

    las_lsoas = set(pd.read_csv(GROUP_PATH)["LSOA21CD"])
    dest_ids = sorted(c for c in lsoa_coords if c in las_lsoas)
    print(f"      destinations: {len(dest_ids)} LAS-served LSOAs")

    origin_ids = sorted(set(candidates["LSOA21CD"]))
    coords = {code: lsoa_coords[code] for code in dest_ids}
    for code in origin_ids:
        if code not in coords:
            coords[code] = lsoa_coords[code]
    missing = [c for c in origin_ids if c not in lsoa_coords]
    if missing:
        raise ValueError(
            f"{len(missing)} candidate LSOAs have no centroid in "
            f"london_graph.json (first 5: {missing[:5]})"
        )
    print(f"      origins:      {len(origin_ids)} candidate LSOAs")

    print(f"[2/4] Fetching OSMnx driving network: {args.place!r}")
    t0 = time.time()
    backend = ft.OSMnxBackend.from_place(
        args.place,
        coords=coords,
        network_type="drive",
        default_speed_kmh=args.fallback_kmh,
    )
    n_nodes = backend.graph.number_of_nodes()
    n_edges = backend.graph.number_of_edges()
    print(f"      OSM graph: {n_nodes:,} nodes / {n_edges:,} edges "
          f"({time.time() - t0:.1f}s)")

    print(f"[3/4] Scaling speeds by ambulance factor {args.speed_factor}x")
    multi = backend.graph.is_multigraph()
    iterator = (
        backend.graph.edges(keys=True, data=True) if multi
        else backend.graph.edges(data=True)
    )
    n_scaled = 0
    for item in iterator:
        data = item[-1]
        if "speed_kph" in data:
            data["speed_kph"] = float(data["speed_kph"]) * args.speed_factor
        if "travel_time" in data:
            data["travel_time"] = float(data["travel_time"]) / args.speed_factor
            n_scaled += 1
    backend._csr = None
    backend._csr_weight = None
    print(f"      rescaled travel_time on {n_scaled:,} edges")

    print(f"[4/4] Computing matrix: {len(origin_ids)} origins x "
          f"{len(dest_ids)} destinations "
          f"= {len(origin_ids) * len(dest_ids):,} pairs")
    t0 = time.time()
    matrix = backend.compute(
        origins=origin_ids, destinations=dest_ids, mode=ft.Mode.DRIVE,
    )
    elapsed = time.time() - t0
    print(f"      done in {elapsed:.1f}s "
          f"({len(matrix):,} pairs, {len(matrix.unreachable):,} unreachable)")

    secs = list(matrix.data.values())
    summary = {}
    if secs:
        summary = {
            "mean": round(statistics.mean(secs) / 60.0, 2),
            "median": round(statistics.median(secs) / 60.0, 2),
            "p90": round(statistics.quantiles(secs, n=10)[-1] / 60.0, 2),
            "max": round(max(secs) / 60.0, 2),
        }
        print(f"      travel-time summary (minutes): {summary}")

    matrix.meta.update({
        "place": args.place,
        "speed_factor": args.speed_factor,
        "fallback_kmh": args.fallback_kmh,
        "ambulance_speed_model": (
            "OSM free-flow speeds (with fallback for untagged edges) "
            "multiplied by speed_factor; lights-and-sirens response."
        ),
        "id_scheme": {"origin": "LSOA21CD", "destination": "LSOA21CD"},
        "cdba_source": str(cdba_path.relative_to(REPO)),
    })

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out_parquet)
    out_meta.write_text(json.dumps({
        "n_origins": len(origin_ids),
        "n_destinations": len(dest_ids),
        "n_pairs": len(matrix),
        "n_unreachable": len(matrix.unreachable),
        "speed_factor": args.speed_factor,
        "fallback_kmh": args.fallback_kmh,
        "place": args.place,
        "time_unit": "seconds",
        "n_osm_nodes": n_nodes,
        "n_osm_edges": n_edges,
        "cdba_source": str(cdba_path.relative_to(REPO)),
        "elapsed_seconds": round(elapsed, 1),
        "summary_minutes": summary,
    }, indent=2))
    print(f"\n      wrote {out_parquet.relative_to(REPO)} "
          f"({out_parquet.stat().st_size / 1024:.0f} KB)")
    print(f"      wrote {out_meta.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
