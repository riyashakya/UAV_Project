"""Lagrangian survivor-drift advection — Phase 7.

Projects where a survivor detected in flowing water will drift to, so the search targets a
**region** (a probability polygon), not a stale point. This is deliberately modelled on how
maritime search-and-rescue planning works — the USCG **SAROPS** system (Kratzke, Stone & Frost,
2010) advects Monte-Carlo particles under a current + **leeway** and reports **containment**
areas. Here the same method is adapted to UAV re-tasking in flood water.

Per particle, each step:  ``dx = u(x, y)·leeway·dt + sqrt(2·K_h·dt)·N(0, 1)``  (advection +
turbulent diffusion). The particle cloud → 50 % / 90 % containment polygons.

CPU-only (numpy + shapely); every stochastic call threads an explicit ``np.random.Generator``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

FlowField = Callable[[float, float], tuple[float, float]]


def advect_particles(
    start_xy,
    flow: FlowField,
    *,
    horizon_s: float,
    dt: float,
    n_particles: int,
    leeway_factor: float,
    k_h: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advect ``n_particles`` from ``start_xy`` for ``horizon_s`` seconds. Returns (N, 2) metres."""
    pos = np.tile(np.asarray(start_xy, dtype=float), (n_particles, 1))
    sigma = math.sqrt(2.0 * k_h * dt)
    for _ in range(int(round(horizon_s / dt))):
        vel = np.array([flow(float(x), float(y)) for x, y in pos])
        pos = pos + vel * leeway_factor * dt
        if sigma > 0.0:
            pos = pos + rng.normal(0.0, sigma, size=pos.shape)
    return pos


def containment_polygon(particles: np.ndarray, fraction: float) -> Polygon:
    """Smallest-ish region holding ``fraction`` of the cloud: convex hull of the nearest-to-centroid
    particles (robust to non-Gaussian spread; a practical SAROPS-style containment area)."""
    centroid = particles.mean(axis=0)
    d = np.linalg.norm(particles - centroid, axis=1)
    k = max(3, int(math.ceil(fraction * len(particles))))
    inner = particles[np.argsort(d)[:k]]
    return MultiPoint([tuple(p) for p in inner]).convex_hull


def drift_search_region(
    start_xy,
    flow: FlowField,
    *,
    rng: np.random.Generator,
    n_particles: int = 1000,
    horizon_s: float = 1800.0,
    dt: float = 30.0,
    leeway_factor: float = 1.0,
    k_h: float = 2.0,
    containment_levels=(0.5, 0.9),
) -> dict:
    """Full drift projection: particle cloud, centroid, and containment polygons + areas."""
    particles = advect_particles(
        start_xy,
        flow,
        horizon_s=horizon_s,
        dt=dt,
        n_particles=n_particles,
        leeway_factor=leeway_factor,
        k_h=k_h,
        rng=rng,
    )
    containment = {lvl: containment_polygon(particles, lvl) for lvl in containment_levels}
    return {
        "particles": particles,
        "centroid": particles.mean(axis=0),
        "containment": containment,
        "areas_m2": {lvl: poly.area for lvl, poly in containment.items()},
    }


def cells_in_region(polygon: Polygon, world) -> list[int]:
    """Grid cells whose centre lies inside ``polygon`` — the cells to re-task UAVs toward (RQ4)."""
    return [c.id for c in world.cells if polygon.covers(Point(c.center_xy))]
