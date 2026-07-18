"""Lagrangian particle advection for survivor drift — Phase 7.

Single responsibility: given a detection at ``(lat, lon, t0)`` and the flow field, project
the position distribution at ``t0 + dt`` via Monte Carlo particles:

    dx/dt = u(x, y) * leeway_factor + turbulent diffusion (random walk, K_h from config)

Emit a 2D probability density and 50 % / 90 % containment polygons. Adapted from USCG SAROPS
(leeway + Monte Carlo containment) — cite it; every constant comes from config.

Stub: implemented in Phase 7.
"""

from __future__ import annotations


def advect(*args: object, **kwargs: object) -> object:
    """Advect survivor particles and return containment polygons. Stub (Phase 7)."""
    raise NotImplementedError("Phase 7: drift advection is not implemented yet.")
