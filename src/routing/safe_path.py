"""Safe-path search — Phase 8.

Single responsibility: find rescue routes on the hazard-weighted graph with edge weight
``w = length * (1 + lambda * risk)``. Sweep ``lambda`` to produce a Pareto front of
(distance, cumulative risk), with a naive shortest-path baseline on the same axes. A route
must never traverse an edge marked ``road_blocked`` (Phase 8 test).

Stub: implemented in Phase 8.
"""

from __future__ import annotations


def safe_path(*args: object, **kwargs: object) -> object:
    """Risk-aware route search; returns a Pareto front over lambda. Stub (Phase 8)."""
    raise NotImplementedError("Phase 8: safe-path search is not implemented yet.")
