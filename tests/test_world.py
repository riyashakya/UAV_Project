"""Grid geometry and analytic flow fields (Phase 4)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from src.sim.world import World, make_flow_field


def test_grid_geometry_and_base():
    w = World(rows=3, cols=4, cell_size_m=100.0, base_cell=0)
    assert w.n_cells == 12
    assert w.cell_center(0) == (50.0, 50.0)  # NW cell centre
    assert w.cell_center(11) == (350.0, 250.0)  # SE cell (row 2, col 3)
    assert w.base_xy == (50.0, 50.0)


def test_uniform_flow_is_constant():
    f = make_flow_field(OmegaConf.create({"type": "uniform", "vx": 0.3, "vy": -0.1}))
    assert f(0.0, 0.0) == (0.3, -0.1)
    assert f(1000.0, 500.0) == (0.3, -0.1)


def test_channel_flow_peaks_on_axis():
    f = make_flow_field(
        OmegaConf.create({"type": "channel", "vmax": 2.0, "axis_y": 100.0, "half_width": 100.0})
    )
    assert f(0.0, 100.0) == (2.0, 0.0)  # on the axis -> vmax
    assert f(0.0, 200.0) == (0.0, 0.0)  # at the bank -> zero
    assert f(0.0, 150.0)[0] == pytest.approx(1.5)  # halfway -> vmax*(1-0.25)


def test_radial_flow_points_outward():
    f = make_flow_field(OmegaConf.create({"type": "radial", "center": [0.0, 0.0], "strength": 1.0}))
    assert f(10.0, 0.0) == pytest.approx((1.0, 0.0))  # unit vector east
    assert f(0.0, -5.0) == pytest.approx((0.0, -1.0))  # unit vector south


def test_unknown_flow_type_raises():
    with pytest.raises(ValueError, match="unknown flow field"):
        make_flow_field(OmegaConf.create({"type": "whirlpool"}))
