"""
Proxy travel-time matrix for the LAS case study.

Used as a stand-in for `data/raw/travel_times.parquet` (the future
FalcomTravel + r5r road-network output) so the FalCom plumbing can be
exercised today on real LAS geography.

Travel time between two LSOAs is computed lazily from their British
National Grid centroids:

    t_minutes(u, v) = euclid_metres(u, v) / 1000.0 / kmh * 60.0

with default `kmh = 25.0` (Implementation.md §9). The class implements
`__getitem__((u, v))` so it drops in wherever `Assignment.travel_times`
expects a dict; values are computed on access (with optional caching).

Replace this entire object with the real `(facility, node) -> minutes`
dict once `data/raw/travel_times.parquet` exists; the FalCom code path
is identical.
"""

from __future__ import annotations

import math
from typing import Mapping


class ProxyTravelTimes(Mapping):
    """Lazy `(u, v) -> minutes` travel-time table.

    `kmh` is the assumed average urban driving speed; tune as needed.
    `cache` toggles per-pair memoisation (default True). Symmetry is
    not exploited explicitly, but the class hashes both orderings to
    the same cache slot when caching is on.
    """

    __slots__ = ("_xy", "_kmh", "_kmh_to_min_per_m", "_cache")

    def __init__(self, graph, kmh: float = 25.0, cache: bool = True) -> None:
        # graph is networkx-like; nodes carry BNG_E, BNG_N (metres).
        self._xy: dict[int, tuple[float, float]] = {
            n: (graph.nodes[n]["BNG_E"], graph.nodes[n]["BNG_N"])
            for n in graph.nodes
        }
        self._kmh = kmh
        # 60 / (kmh * 1000) = minutes per metre.
        self._kmh_to_min_per_m = 60.0 / (kmh * 1000.0)
        self._cache: dict[tuple[int, int], float] | None = {} if cache else None

    def __getitem__(self, key: tuple[int, int]) -> float:
        a, b = key
        if a == b:
            return 0.0
        ckey = (a, b) if a <= b else (b, a)
        if self._cache is not None:
            cached = self._cache.get(ckey)
            if cached is not None:
                return cached
        ax, ay = self._xy[a]
        bx, by = self._xy[b]
        d_m = math.hypot(ax - bx, ay - by)
        t_min = d_m * self._kmh_to_min_per_m
        if self._cache is not None:
            self._cache[ckey] = t_min
        return t_min

    # Mapping protocol — implemented for completeness; FalCom only
    # ever calls __getitem__.

    def __iter__(self):
        # The full (a, b) Cartesian is huge; refuse to enumerate.
        raise NotImplementedError(
            "ProxyTravelTimes does not support iteration; "
            "use __getitem__((a, b)) instead."
        )

    def __len__(self) -> int:
        n = len(self._xy)
        return n * n

    def __contains__(self, key) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        return key[0] in self._xy and key[1] in self._xy
