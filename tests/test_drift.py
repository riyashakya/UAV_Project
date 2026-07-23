"""Survivor-drift advection + containment (Phase 7 acceptance)."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
from shapely.geometry import Point
from src.drift.advect import (
    advect_particles,
    cells_in_region,
    containment_polygon,
    drift_search_region,
)
from src.sim.world import World, make_flow_field


def _uniform(vx, vy):
    return make_flow_field(OmegaConf.create({"type": "uniform", "vx": vx, "vy": vy}))


def test_zero_diffusion_uniform_flow_displaces_exactly_v_times_dt():
    """Analytic case: no turbulence + uniform flow -> every particle at start + v·leeway·Δt."""
    p = advect_particles(
        (0.0, 0.0),
        _uniform(0.5, 0.0),
        horizon_s=1000.0,
        dt=10.0,
        n_particles=50,
        leeway_factor=1.0,
        k_h=0.0,
        rng=np.random.default_rng(0),
    )
    assert np.allclose(p, [500.0, 0.0])  # 0.5 m/s * 1000 s = 500 m east, no spread


def test_containment_area_grows_with_horizon():
    flow = _uniform(0.2, 0.0)
    areas = [
        drift_search_region(
            (0.0, 0.0),
            flow,
            rng=np.random.default_rng(1),
            n_particles=500,
            horizon_s=h,
            dt=30.0,
            k_h=2.0,
        )["areas_m2"][0.9]
        for h in (300.0, 900.0, 1800.0)
    ]
    assert areas[0] < areas[1] < areas[2]  # diffusion spreads the cloud over time


def test_90pct_containment_holds_across_particles():
    p = advect_particles(
        (0.0, 0.0),
        _uniform(0.3, 0.1),
        horizon_s=1200.0,
        dt=30.0,
        n_particles=1000,
        leeway_factor=1.0,
        k_h=3.0,
        rng=np.random.default_rng(2),
    )
    poly = containment_polygon(p, 0.9)
    inside = sum(poly.covers(Point(x, y)) for x, y in p)
    assert inside / len(p) >= 0.9  # the 90% polygon contains >= 90% of the cloud


def test_90pct_containment_generalises_to_a_fresh_cloud():
    flow = _uniform(0.3, 0.1)
    a = advect_particles(
        (0.0, 0.0),
        flow,
        horizon_s=1200.0,
        dt=30.0,
        n_particles=1000,
        leeway_factor=1.0,
        k_h=3.0,
        rng=np.random.default_rng(3),
    )
    b = advect_particles(
        (0.0, 0.0),
        flow,
        horizon_s=1200.0,
        dt=30.0,
        n_particles=1000,
        leeway_factor=1.0,
        k_h=3.0,
        rng=np.random.default_rng(4),
    )
    poly = containment_polygon(a, 0.9)
    inside = sum(poly.covers(Point(x, y)) for x, y in b)
    assert inside / len(b) >= 0.85  # generalises to an independent cloud (with sampling slack)


def test_search_region_maps_to_grid_cells():
    world = World(rows=6, cols=6, cell_size_m=200.0)
    # drift starting mid-grid; the 90% polygon should overlap a handful of cells
    region = drift_search_region(
        world.cell_center(14),
        _uniform(0.3, 0.0),
        rng=np.random.default_rng(5),
        n_particles=500,
        horizon_s=1200.0,
        k_h=3.0,
    )
    cells = cells_in_region(region["containment"][0.9], world)
    assert 1 <= len(cells) <= world.n_cells
