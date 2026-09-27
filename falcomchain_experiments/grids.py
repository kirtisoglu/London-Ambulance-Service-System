"""
Synthetic grid instances for Section 7.1 experiments.

Spec (from paper §7.1):
  - Square grid graphs, |V| ∈ {20, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000}
  - Node demands i.i.d. Uniform(80, 120)
  - Facility candidates placed uniformly at random, density ρ = 0.05
    (one candidate per 20 nodes on average)
  - Two hierarchy levels: ε₁ = 0.10, ε₂ = 0.15, c_max¹ = 4, c_max² = 3
  - Integer node labels (row-major: node = row * cols + col)
  - Edge attribute: shared_perim = 1
  - Node attributes: demand, area, candidate, C_X, C_Y

Usage::

    from experiments.grids import make_grid, PAPER_SIZES

    g = make_grid(100)          # 10×10 grid, 100 nodes
    g = make_grid(500, seed=42) # reproducible
"""

import math
import random as _random
from typing import Optional

import networkx as nx

from falcomchain.graph import Graph


# Sizes used in the paper experiments
PAPER_SIZES = [20, 100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]

# Paper parameters
DEMAND_LOW = 80
DEMAND_HIGH = 120
CANDIDATE_DENSITY = 0.05   # ρ

# Hierarchy parameters (stored as metadata on the graph)
EPSILON_L1 = 0.10
EPSILON_L2 = 0.15
C_MAX_L1 = 4
C_MAX_L2 = 3


def _grid_dims(n: int):
    """
    Return (rows, cols) for the most square integer factorisation of n.
    For perfect squares this is (√n, √n). Otherwise finds the pair
    (r, c) with r ≤ c, r*c = n, and c/r minimised.
    Falls back to (1, n) if n is prime.
    """
    rows = int(math.isqrt(n))
    while rows > 1:
        if n % rows == 0:
            return rows, n // rows
        rows -= 1
    return 1, n


def make_grid(
    n_nodes: int,
    demand_low: int = DEMAND_LOW,
    demand_high: int = DEMAND_HIGH,
    candidate_density: float = CANDIDATE_DENSITY,
    seed: Optional[int] = None,
) -> Graph:
    """
    Build a synthetic square-grid Graph instance per the paper §7.1 spec.

    :param n_nodes: Number of nodes. Should ideally factorise into a
        near-square pair; see ``_grid_dims``.
    :param demand_low: Lower bound of uniform demand distribution.
    :param demand_high: Upper bound of uniform demand distribution.
    :param candidate_density: Fraction of nodes that are facility candidates.
    :param seed: Random seed for reproducibility.
    :returns: A ``falcomchain.graph.Graph`` with integer node labels and
        attributes ``demand``, ``area``, ``candidate``, ``C_X``, ``C_Y``.
    :rtype: Graph
    """
    rng = _random.Random(seed)

    rows, cols = _grid_dims(n_nodes)
    actual_n = rows * cols
    if actual_n != n_nodes:
        raise ValueError(
            f"n_nodes={n_nodes} cannot be factored into a rectangle. "
            f"Nearest factorisation gives {actual_n} nodes ({rows}×{cols}). "
            f"Try one of: {PAPER_SIZES}"
        )

    # Build grid with tuple nodes, then relabel to integers (row-major)
    raw = nx.grid_2d_graph(rows, cols)
    nx.set_edge_attributes(raw, 1, "shared_perim")

    label_map = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
    g = nx.relabel_nodes(raw, label_map)

    # Assign node attributes
    n_candidates = max(1, round(candidate_density * actual_n))
    candidate_nodes = set(rng.sample(list(g.nodes()), k=n_candidates))

    # Level-2 (super-)candidates: a random subset of the L1 candidates,
    # at 20% density of the L1 set (matches LAS-like ratio: ~7 sector HQs
    # among ~63 L1 stations ≈ 11%, rounded up for headroom). Drawn from
    # the L1 candidate set so super-candidates ⊂ candidates.
    n_super_candidates = max(1, round(0.20 * n_candidates))
    super_candidate_nodes = set(
        rng.sample(sorted(candidate_nodes), k=n_super_candidates)
    )

    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            g.nodes[node]["demand"] = rng.randint(demand_low, demand_high)
            g.nodes[node]["area"] = 1
            g.nodes[node]["C_X"] = c
            g.nodes[node]["C_Y"] = r
            g.nodes[node]["candidate"] = 1 if node in candidate_nodes else 0
            g.nodes[node]["super_candidate"] = (
                1 if node in super_candidate_nodes else 0
            )

    # Attach experiment metadata as graph-level attributes
    g.graph["rows"] = rows
    g.graph["cols"] = cols
    g.graph["n_nodes"] = actual_n
    g.graph["candidate_density"] = candidate_density
    g.graph["epsilon_l1"] = EPSILON_L1
    g.graph["epsilon_l2"] = EPSILON_L2
    g.graph["c_max_l1"] = C_MAX_L1
    g.graph["c_max_l2"] = C_MAX_L2

    return Graph.from_networkx(g)


def make_all_paper_grids(seed: Optional[int] = None):
    """
    Build all nine grid instances from the paper §7.1.

    :param seed: Base seed. Each grid uses ``seed + i`` so instances are
        independent but reproducible.
    :returns: Dict mapping ``n_nodes → Graph``.
    :rtype: dict
    """
    grids = {}
    for i, n in enumerate(PAPER_SIZES):
        s = (seed + i) if seed is not None else None
        grids[n] = make_grid(n, seed=s)
    return grids
