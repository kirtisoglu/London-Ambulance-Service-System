"""
Experiment 1: Detailed Balance Validation
==========================================

Validates that the FalCom Markov chain samples from the correct Boltzmann
distribution μ_β(s) ∝ exp(-β · E(s)).

Setup (paper §7.1):
  - 5×4 grid graph (|V| = 20)
  - 4 fixed candidate facilities (corners of the grid)
  - 2 districts, capacity_level = 1 (one team per district)
  - ε = 0.20 (demand balance tolerance)
  - γ = 0 (no candidate-awareness bias, so ψ = φ)
  - Demands drawn i.i.d. Uniform(80, 120), seed=0

Method:
  - Enumerate all feasible connected 2-partitions by brute force
  - Compute E(s) for each state using Manhattan travel times
  - Compute exact μ_β(s) = exp(-β·E(s)) / Z(β) for several β values
  - Run the FalCom chain (tree-cut proposal + MH acceptance) for T steps
  - Compare empirical ν̂(s) to μ_β(s) via chi-squared test,
    total variation distance, and scatter plots saved in figures/

Usage::

    python experiments/experiment1/run.py
"""

import math
import random
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).parent
FIGURES_DIR = EXPERIMENT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

ROWS, COLS = 5, 4          # |V| = 20
N_NODES = ROWS * COLS
SEED = 0
DEMAND_LOW, DEMAND_HIGH = 80, 120
EPSILON = 0.20
N_DISTRICTS = 2
CAPACITY_LEVEL = 1
GAMMA = 0.0                # candidate-awareness parameter (0 → ψ = φ)
BURN_IN = 10_000
T_STEPS = 1_000_000
# β is expressed in normalised units: β_actual = β_norm / E_std.
# Computed after enumeration so that β_norm = 1 concentrates the
# distribution by roughly one standard deviation of energy.
BETA_NORMS = [0.0, 0.5, 1.0, 2.0, 5.0]


def _node(r: int, c: int) -> int:
    """Row-major integer label."""
    return r * COLS + c


def build_grid(seed: int = SEED) -> Tuple[nx.Graph, Dict[int, int], List[int]]:
    """
    Build a 5×4 grid graph with:
      - integer node labels (row-major)
      - demand ~ Uniform(80, 120)
      - 4 candidate facilities at the four corners
      - Manhattan travel_times dict keyed (facility, node)

    Returns (g, demands, candidates).
    """
    rng = random.Random(seed)
    g = nx.grid_2d_graph(ROWS, COLS)
    label_map = {(r, c): _node(r, c) for r in range(ROWS) for c in range(COLS)}
    g = nx.relabel_nodes(g, label_map)

    demands = {}
    for r in range(ROWS):
        for c in range(COLS):
            node = _node(r, c)
            demands[node] = rng.randint(DEMAND_LOW, DEMAND_HIGH)
            g.nodes[node]["demand"] = demands[node]
            g.nodes[node]["C_X"] = c
            g.nodes[node]["C_Y"] = r

    # Fix 4 candidates at the corners
    candidates = [
        _node(0, 0),
        _node(0, COLS - 1),
        _node(ROWS - 1, 0),
        _node(ROWS - 1, COLS - 1),
    ]
    for n in g.nodes:
        g.nodes[n]["candidate"] = 1 if n in candidates else 0

    return g, demands, candidates


def manhattan(g: nx.Graph, u: int, v: int) -> float:
    """Manhattan distance between nodes u and v on the grid."""
    return abs(g.nodes[u]["C_X"] - g.nodes[v]["C_X"]) + \
           abs(g.nodes[u]["C_Y"] - g.nodes[v]["C_Y"])


def build_travel_times(g: nx.Graph, candidates: List[int]) -> Dict[Tuple, float]:
    """Build the full travel_times dict: (facility, node) → Manhattan distance."""
    tt = {}
    for fac in candidates:
        for node in g.nodes:
            tt[(fac, node)] = manhattan(g, fac, node)
    return tt


# ---------------------------------------------------------------------------
# Feasible-state enumeration
# ---------------------------------------------------------------------------

def _is_connected(g: nx.Graph, nodes: FrozenSet[int]) -> bool:
    if not nodes:
        return False
    return nx.is_connected(g.subgraph(nodes))


def _demand_ok(demands: Dict, district: FrozenSet, total: float) -> bool:
    target = total / N_DISTRICTS
    d = sum(demands[n] for n in district)
    return abs(d / target - 1.0) <= EPSILON


def enumerate_feasible_partitions(
    g: nx.Graph,
    demands: Dict[int, int],
    candidates: List[int],
) -> List[Dict]:
    """
    Enumerate all feasible connected 2-partitions of the 5×4 grid.

    A partition is feasible if:
      1. Both districts are connected.
      2. Both satisfy the ε-demand balance constraint.
      3. Each district contains at least one candidate facility.

    Returns a list of state dicts.
    """
    nodes = list(g.nodes)
    total_demand = sum(demands.values())
    states = []
    all_nodes = frozenset(nodes)

    for size in range(1, N_NODES // 2 + 1):
        for subset in combinations(nodes, size):
            d1 = frozenset(subset)
            d2 = all_nodes - d1
            # Deduplicate size-N/2 pairs
            if size * 2 == N_NODES and min(d1) > min(d2):
                continue

            if not _is_connected(g, d1) or not _is_connected(g, d2):
                continue
            if not _demand_ok(demands, d1, total_demand):
                continue
            if not _demand_ok(demands, d2, total_demand):
                continue

            cands_d1 = [c for c in candidates if c in d1]
            cands_d2 = [c for c in candidates if c in d2]
            if not cands_d1 or not cands_d2:
                continue

            states.append({
                "partition": (d1, d2),
                "cands_d1": cands_d1,
                "cands_d2": cands_d2,
            })

    return states


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

def minimax_center(
    candidates: List[int],
    nodes: FrozenSet[int],
    tt: Dict,
) -> Tuple[Optional[int], float]:
    """Minimax center: argmin_{c} max_{v} tt(c, v).  Returns (center, radius)."""
    best_center = None
    best_radius = math.inf
    for c in candidates:
        radius = max(tt[(c, v)] for v in nodes)
        if radius < best_radius:
            best_radius = radius
            best_center = c
    return best_center, best_radius


def compute_energy_raw(
    partition: Tuple[FrozenSet, FrozenSet],
    cands: List[List[int]],
    demands: Dict,
    tt: Dict,
) -> float:
    """E(s) = Σ_D Σ_v d_v · tt(f(D), v) where f(D) is the minimax center."""
    total = 0.0
    for district, district_cands in zip(partition, cands):
        center, _ = minimax_center(district_cands, district, tt)
        if center is None:
            continue
        for v in district:
            total += demands[v] * tt[(center, v)]
    return total


def annotate_states(states: List[Dict], demands: Dict, tt: Dict) -> List[Dict]:
    """Add energy, centers, radii to each state dict."""
    for s in states:
        d1, d2 = s["partition"]
        c1, r1 = minimax_center(s["cands_d1"], d1, tt)
        c2, r2 = minimax_center(s["cands_d2"], d2, tt)
        s["centers"] = {1: c1, 2: c2}
        s["radii"] = {1: r1, 2: r2}
        s["energy"] = compute_energy_raw(
            s["partition"], [s["cands_d1"], s["cands_d2"]], demands, tt,
        )
    return states


# ---------------------------------------------------------------------------
# Exact Boltzmann distribution
# ---------------------------------------------------------------------------

def boltzmann(states: List[Dict], beta: float) -> Dict[int, float]:
    """μ_β(s) = exp(-β·E(s)) / Z(β).  Returns {idx: probability}."""
    log_weights = [-beta * s["energy"] for s in states]
    max_lw = max(log_weights)
    weights = [math.exp(lw - max_lw) for lw in log_weights]
    Z = sum(weights)
    return {i: w / Z for i, w in enumerate(weights)}


# ---------------------------------------------------------------------------
# FalCom chain simulation (simplified for 2-partition, capacity_level=1)
#
# Mirrors the FalCom algorithm:
#   1. Sample a random spanning tree (random-weight Kruskal)
#   2. Root it and accumulate subtree demand + candidate count (bottom-up)
#   3. Find all feasible cuts; compute ψ = φ·exp(-γ·r) for each
#   4. Select a cut proportional to ψ
#   5. log_proposal_ratio = log(ψ_chosen / Σψ)  (forward cut probability)
#   6. MH acceptance: α = min(1, exp(-β·ΔE + log_proposal_ratio))
# ---------------------------------------------------------------------------

def _partition_to_key(partition: Tuple[FrozenSet, FrozenSet]) -> FrozenSet:
    """Canonical key for a 2-partition (order-independent)."""
    return frozenset([partition[0], partition[1]])


def _energy_of_partition(
    partition: Tuple[FrozenSet, FrozenSet],
    candidates: List[int],
    demands: Dict,
    tt: Dict,
) -> float:
    d1, d2 = partition
    cands_d1 = [c for c in candidates if c in d1]
    cands_d2 = [c for c in candidates if c in d2]
    return compute_energy_raw(partition, [cands_d1, cands_d2], demands, tt)


def _propose(
    g: nx.Graph,
    demands: Dict,
    candidates: List[int],
    epsilon: float,
    total_demand: float,
    gamma: float,
    rng: random.Random,
) -> Optional[Tuple[Tuple[FrozenSet, FrozenSet], float]]:
    """
    FalCom proposal for a 2-partition with capacity_level=1.

    1. Random spanning tree (random-weight Kruskal).
    2. Root, accumulate subtree demand and candidate count via post-order DFS.
    3. Find feasible cuts, compute ψ for each.
    4. Select cut ∝ ψ; compute log_proposal_ratio = log(ψ_chosen / Σψ).

    Returns ((d1, d2), log_proposal_ratio) or None.
    """
    # --- Step 1: random spanning tree ---
    for u, v in g.edges():
        g[u][v]["_rw"] = rng.random()
    tree = nx.minimum_spanning_tree(g, weight="_rw")
    all_nodes = frozenset(g.nodes())
    candidate_set = frozenset(candidates)

    # --- Step 2: root and accumulate ---
    root = rng.choice(list(tree.nodes()))

    # BFS to build parent map
    parent = {root: root}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nbr in tree[cur]:
            if nbr not in parent:
                parent[nbr] = cur
                queue.append(nbr)

    # Children map
    children = {n: [] for n in parent}
    for node, par in parent.items():
        if node != par:
            children[par].append(node)

    # Post-order traversal for accumulation
    order = []
    stack = [root]
    while stack:
        cur = stack.pop()
        order.append(cur)
        for ch in children[cur]:
            stack.append(ch)
    order.reverse()

    subtree_demand = {}
    subtree_cands = {}
    for node in order:
        sd = demands[node]
        sc = 1 if node in candidate_set else 0
        for ch in children[node]:
            sd += subtree_demand[ch]
            sc += subtree_cands[ch]
        subtree_demand[node] = sd
        subtree_cands[node] = sc

    total_cands = subtree_cands[root]
    target = total_demand / N_DISTRICTS

    # --- Step 3: find feasible cuts and compute ψ ---
    cuts = []  # list of (node, ψ)
    for node in parent:
        if node == root:
            continue
        sd = subtree_demand[node]
        sc = subtree_cands[node]
        comp_cands = total_cands - sc

        # Both sides must have a candidate (φ > 0)
        if sc == 0 or comp_cands == 0:
            continue
        # Both sides must be demand-balanced
        if abs(sd / target - 1.0) > epsilon:
            continue
        if abs((total_demand - sd) / target - 1.0) > epsilon:
            continue

        # ψ = φ · exp(-γ · r)  where φ = candidate count, r = 1/φ proxy
        phi = sc
        if gamma == 0.0:
            psi = float(phi)
        else:
            r = 1.0 / phi
            psi = phi * math.exp(-gamma * r)

        cuts.append((node, psi))

    if not cuts:
        return None

    # --- Step 4: select cut proportional to ψ ---
    total_psi = sum(psi for _, psi in cuts)
    weights = [psi / total_psi for _, psi in cuts]
    chosen_idx = rng.choices(range(len(cuts)), weights=weights, k=1)[0]
    chosen_node, chosen_psi = cuts[chosen_idx]

    # log_proposal_ratio = log(ψ_chosen / Σψ)
    log_proposal_ratio = math.log(chosen_psi) - math.log(total_psi)

    # Build the partition from the chosen cut
    # Collect subtree nodes
    result = []
    stack = [chosen_node]
    while stack:
        cur = stack.pop()
        result.append(cur)
        stack.extend(children[cur])
    d1 = frozenset(result)
    d2 = all_nodes - d1

    return (d1, d2), log_proposal_ratio


def run_chain(
    g: nx.Graph,
    demands: Dict,
    candidates: List[int],
    tt: Dict,
    beta: float,
    gamma: float,
    rng: random.Random,
    T: int = T_STEPS,
    burn_in: int = BURN_IN,
) -> Dict[FrozenSet, int]:
    """
    Run the FalCom chain for a 2-partition with capacity_level=1.

    MH acceptance mirrors accept.py:
        log_alpha = -beta * delta_energy + log_proposal_ratio

    At β=0, always accept (matching accept.py line 43-44).

    Returns visit counts {partition_key: count} after burn-in.
    """
    total_demand = sum(demands.values())

    # Find a feasible starting state
    current = None
    for _ in range(1000):
        result = _propose(g, demands, candidates, EPSILON, total_demand, gamma, rng)
        if result is not None:
            current, _ = result
            break
    if current is None:
        raise RuntimeError("Could not find a feasible starting partition.")

    current_energy = _energy_of_partition(current, candidates, demands, tt)
    visit_counts: Dict[FrozenSet, int] = defaultdict(int)

    for step in range(T + burn_in):
        result = _propose(g, demands, candidates, EPSILON, total_demand, gamma, rng)
        if result is not None:
            proposed, log_proposal_ratio = result
            proposed_energy = _energy_of_partition(proposed, candidates, demands, tt)

            # MH acceptance — mirrors falcomchain/markovchain/accept.py
            if beta == 0:
                accept = True
            else:
                delta_energy = proposed_energy - current_energy
                log_alpha = -beta * delta_energy + log_proposal_ratio
                accept = math.log(rng.random()) <= log_alpha

            if accept:
                current = proposed
                current_energy = proposed_energy

        if step >= burn_in:
            visit_counts[_partition_to_key(current)] += 1

    return dict(visit_counts)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def total_variation(p: Dict[int, float], q: Dict[int, float]) -> float:
    """TV(p, q) = 0.5 · Σ|p(s) - q(s)|."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def chi_squared(observed_counts: Dict[int, int], expected_probs: Dict[int, float],
                total: int) -> Tuple[float, float]:
    """Pearson chi-squared test.  Bins with expected < 5 are pooled."""
    from scipy.stats import chi2 as _chi2

    obs = []
    exp = []
    other_obs = 0
    other_exp = 0.0

    for i, mu in expected_probs.items():
        expected = total * mu
        observed = observed_counts.get(i, 0)
        if expected < 5:
            other_obs += observed
            other_exp += expected
        else:
            obs.append(observed)
            exp.append(expected)

    if other_exp > 0:
        obs.append(other_obs)
        exp.append(other_exp)

    chi2_stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e > 0)
    df = len(obs) - 1
    p_value = 1.0 - _chi2.cdf(chi2_stat, df) if df > 0 else float("nan")
    return chi2_stat, p_value


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scatter_plot(
    mu_exact: Dict[int, float],
    mu_empirical: Dict[int, float],
    beta_norm: float,
    n_states: int,
    output_path: Path,
) -> None:
    """Scatter plot of μ̂(s) vs μ_β(s) for each state."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not installed — scatter plot not generated.")
        return

    x = [mu_exact.get(i, 0.0) for i in range(n_states)]
    y = [mu_empirical.get(i, 0.0) for i in range(n_states)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=15, alpha=0.6, color="steelblue")

    pos = [v for v in x + y if v > 0]
    if pos:
        lo, hi = min(pos) * 0.5, max(pos) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(r"$\mu_\beta(s)$ (exact)")
    ax.set_ylabel(r"$\hat{\nu}(s)$ (empirical)")
    ax.set_title(rf"Detailed Balance Validation ($\tilde{{\beta}}={beta_norm}$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Scatter plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Experiment 1: Detailed Balance Validation")
    print("=" * 60)
    print(f"Grid: {ROWS}×{COLS}  |V|={N_NODES}  seed={SEED}")
    print(f"Districts: {N_DISTRICTS}  ε={EPSILON}  γ={GAMMA}  "
          f"capacity_level={CAPACITY_LEVEL}")
    print(f"Burn-in: {BURN_IN:,}  T: {T_STEPS:,}")
    print()

    # Build grid
    g, demands, candidates = build_grid(seed=SEED)
    tt = build_travel_times(g, candidates)
    total_demand = sum(demands.values())
    print(f"Total demand: {total_demand}  "
          f"Target per district: {total_demand / N_DISTRICTS:.1f}")
    print(f"Candidates (4 corners): {candidates}")
    print()

    # Enumerate feasible states
    print("Enumerating feasible states...")
    states = enumerate_feasible_partitions(g, demands, candidates)
    states = annotate_states(states, demands, tt)
    n_states = len(states)
    print(f"Found {n_states} feasible connected 2-partitions.")

    if n_states == 0:
        print("ERROR: No feasible states found.")
        return

    # Compute energy scale for β normalisation
    energies = [s["energy"] for s in states]
    E_std = (sum((e - sum(energies) / n_states)**2 for e in energies)
             / n_states) ** 0.5
    print(f"Energy: min={min(energies):.0f}, max={max(energies):.0f}, "
          f"std={E_std:.1f}")
    print(f"β_norm → β_actual via β = β̃ / σ_E  (σ_E = {E_std:.1f})")
    print()

    # Build index: partition_key → state index
    key_to_idx = {_partition_to_key(s["partition"]): i for i, s in enumerate(states)}

    results = {}

    for beta_norm in BETA_NORMS:
        beta = beta_norm / E_std if E_std > 0 else 0.0
        print(f"--- β̃ = {beta_norm}  (β = {beta:.6f}) ---")

        # Fresh RNG per β so runs are independent
        rng_chain = random.Random(SEED + 42 + int(beta_norm * 1000))

        # Exact distribution
        mu_exact = boltzmann(states, beta)

        # Run chain
        print(f"  Running chain (T={T_STEPS:,} + {BURN_IN:,} burn-in)...")
        t0 = time.time()
        visit_raw = run_chain(g, demands, candidates, tt, beta, GAMMA,
                              rng_chain, T=T_STEPS, burn_in=BURN_IN)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/(T_STEPS+BURN_IN)*1000:.2f} ms/step)")

        # Map visit counts back to state indices
        visit_counts: Dict[int, int] = {}
        unmatched = 0
        for key, count in visit_raw.items():
            idx = key_to_idx.get(key)
            if idx is not None:
                visit_counts[idx] = count
            else:
                unmatched += count

        total_visits = sum(visit_counts.values())
        if unmatched > 0:
            print(f"  WARNING: {unmatched} visits to states not in enumerated set.")

        # Empirical distribution
        mu_empirical: Dict[int, float] = {
            i: visit_counts.get(i, 0) / total_visits
            for i in range(n_states)
        } if total_visits > 0 else {i: 0.0 for i in range(n_states)}

        # TV distance
        tv = total_variation(mu_exact, mu_empirical)

        # Chi-squared test
        try:
            chi2, pval = chi_squared(visit_counts, mu_exact, total_visits)
            print(f"  χ² = {chi2:.2f},  p = {pval:.4f}")
        except ImportError:
            chi2, pval = float("nan"), float("nan")
            print("  [skip] scipy not installed — chi-squared not computed.")

        print(f"  TV distance = {tv:.6f}")
        print(f"  Distinct states visited: {len(visit_raw)}/{n_states}")

        # Scatter plot
        tag = str(beta_norm).replace('.', 'p')
        scatter_plot(
            mu_exact, mu_empirical, beta_norm, n_states,
            FIGURES_DIR / f"scatter_beta_{tag}.png",
        )

        results[beta_norm] = {
            "beta_actual": beta,
            "mu_exact": mu_exact,
            "mu_empirical": mu_empirical,
            "chi2": chi2,
            "pval": pval,
            "tv": tv,
            "total_visits": total_visits,
        }

    # Summary table
    print()
    print("Summary")
    print("-" * 65)
    print(f"{'β̃':>6}  {'β':>12}  {'χ²':>10}  {'p-value':>10}  {'TV dist':>10}")
    print("-" * 65)
    for bn in BETA_NORMS:
        r = results[bn]
        print(f"{bn:>6}  {r['beta_actual']:>12.6f}  "
              f"{r['chi2']:>10.2f}  {r['pval']:>10.4f}  {r['tv']:>10.6f}")
    print("-" * 65)
    print()
    print("Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
