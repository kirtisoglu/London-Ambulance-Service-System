"""
Travel-time lookup for the LAS chain, backed by the CDBA candidate
parquet produced by ``compute_cdba_travel_times.py``.

The parquet stores ``origin, dest, seconds`` rows keyed by LSOA21CD on
both sides. This adapter exposes a ``Mapping`` indexed by
``(graph_node_id, graph_node_id)`` returning **minutes** (matching the
FalCom Assignment.travel_times contract). For pairs absent from the
matrix it falls back to the haversine proxy at a fixed driving speed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import pandas as pd

from .proxy_travel import ProxyTravelTimes


class CdbaTravelTimes(Mapping):
    """``(graph_node_id, graph_node_id) -> minutes`` over the
    candidate-LSOA travel-time matrix."""

    def __init__(
        self,
        graph,
        *,
        matrix_path: str | Path,
        fallback_kmh: float = 25.0,
    ):
        self.graph = graph
        self.fallback = ProxyTravelTimes(graph, kmh=fallback_kmh, cache=True)

        df = pd.read_parquet(matrix_path)
        self._table: dict[tuple[str, str], float] = {
            (str(o), str(d)): float(s) / 60.0
            for o, d, s in zip(df["origin"], df["dest"], df["seconds"])
        }
        self._lsoa_of: dict = {
            n: graph.nodes[n]["LSOA21CD"] for n in graph.nodes
        }
        self.n_pairs = len(self._table)
        self.n_fallbacks = 0

    def __len__(self) -> int:
        return len(self._table)

    def __iter__(self):
        return iter(self._table)

    def __contains__(self, key) -> bool:
        try:
            self[key]
            return True
        except (KeyError, ValueError):
            return False

    def __getitem__(self, key: tuple) -> float:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        u, v = key
        u_lsoa = self._lsoa_of.get(u)
        v_lsoa = self._lsoa_of.get(v)
        if u_lsoa is not None and v_lsoa is not None:
            t = self._table.get((u_lsoa, v_lsoa))
            if t is not None:
                return t
            t = self._table.get((v_lsoa, u_lsoa))
            if t is not None:
                return t
        self.n_fallbacks += 1
        return float(self.fallback[(u, v)])

    def fallback_report(self) -> str:
        return (
            f"CdbaTravelTimes: {self.n_pairs:,} network pairs loaded; "
            f"haversine fallback used {self.n_fallbacks:,} times."
        )
