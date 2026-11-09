"""
Component ablations of the FalCom proposal (paper, ablation section).

Each variant switches off or changes exactly one component of the proposal and
leaves everything else at the calibration of the instance. The initial plan is
always built with the default components, so a variant measures what the
component buys inside the chain, not when a plan is built from scratch.

The variants are applied without touching the library: ``patch`` wraps the
``hierarchical_recom`` name of an experiment module so that the keyword
arguments of the variant override the ones the experiment passes.
"""
from __future__ import annotations

import os
import time

from falcomchain.markovchain import proposals as _proposals

# Wall-clock budget of the chain (seconds, from the first proposal). A variant
# that removes a component can make a proposal hundreds of times slower; when
# the budget is spent the chain is stopped cleanly and the run reports the
# proposals it made.
BUDGET_S = float(os.environ.get("ABLATION_BUDGET_S", "900"))

VARIANTS = {
    # the sampler as used in the paper
    "default": {},
    # counting predicate of the level-1 cut off (only >= 1 candidate per side)
    "nocount": {"count_candidates_base": False},
    # residual predicate of the supergraph cut off
    "noresid": {"check_super_residual": False},
    # capacity-scaled debt window c * [L~, U~] instead of the absolute half-width
    "scaled": {"rule": "scaled"},
    # level-1 re-cut against the global per-unit demand instead of the local one
    "global": {"_global_target": True},
    # retry budget M (spanning trees per extraction), both levels
    "M200": {"max_attempts_base": 200, "max_attempts_super": 200},
    "M1000": {"max_attempts_base": 1000, "max_attempts_super": 1000},
    "M5000": {"max_attempts_base": 5000, "max_attempts_super": 5000},
}

_GLOBAL = {"target": None}
_ORIG_RECURSION = _proposals.capacitated_recursive_tree


def _global_target_recursion(*args, **kwargs):
    """Level-1 recursion with the demand target replaced by the global one."""
    if _GLOBAL["target"] is not None and "demand_target" in kwargs:
        kwargs["demand_target"] = _GLOBAL["target"]
    return _ORIG_RECURSION(*args, **kwargs)


def patch(module, variant: str) -> dict:
    """Make ``module.hierarchical_recom`` apply ``variant``. Returns the overrides."""
    overrides = dict(VARIANTS[variant])
    use_global = overrides.pop("_global_target", False)
    original = module.hierarchical_recom
    if use_global:
        # only the base-level call goes through proposals.capacitated_recursive_tree;
        # the supergraph recursion is called from super_partitioners and is untouched
        _proposals.capacitated_recursive_tree = _global_target_recursion

    clock = {"deadline": None}

    def wrapped(state, **kwargs):
        now = time.perf_counter()
        if clock["deadline"] is None:
            clock["deadline"] = now + BUDGET_S
        elif now > clock["deadline"]:
            raise StopIteration            # ends the chain iteration; not counted as a proposal
        if use_global and _GLOBAL["target"] is None:
            part = state.partition
            g = part.graph.graph
            total = sum(g.nodes[n]["demand"] for n in g.nodes)
            _GLOBAL["target"] = total / sum(part.teams.values())
        kwargs.update(overrides)
        return original(state, **kwargs)

    module.hierarchical_recom = wrapped
    return {**overrides, **({"global_target": True} if use_global else {})}
