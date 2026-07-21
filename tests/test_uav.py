"""UAV energy model and return-to-home (Phase 4 acceptance)."""

from __future__ import annotations

import pytest
from src.sim.uav import UAV, Status, UAVParams

PARAMS = UAVParams(
    cruise_speed_ms=15.0,
    p_hover_w=200.0,
    k_drag=0.35,
    battery_capacity_j=300_000.0,
    rth_margin=1.3,
)


def test_energy_model_matches_formula():
    u = UAV(0, PARAMS, (0.0, 0.0))
    assert u.power_cruise() == pytest.approx(200.0 + 0.35 * 15.0**2)  # P_hover + k v^2
    # energy = power * distance / speed
    assert u.energy_to_reach(1500.0) == pytest.approx(u.power_cruise() * 1500.0 / 15.0)


def test_returns_home_from_5km_with_energy_to_spare():
    """The Phase 4 headline test: a UAV starting at its RTH threshold, 5 km out, lands safely."""
    base = (0.0, 0.0)
    u = UAV(0, PARAMS, (5000.0, 0.0))
    u.energy = PARAMS.rth_margin * u.energy_to_reach(5000.0)  # exactly at the RTH threshold

    for _ in range(100_000):
        u.step(5.0, base)
        if u.status in (Status.LANDED, Status.DEAD):
            break

    assert u.status == Status.LANDED  # made it, did not die
    assert u.energy > 0.0  # with energy to spare
    assert u._dist(base) == pytest.approx(0.0, abs=1.0)  # actually at base


def test_rth_triggers_when_energy_low():
    base = (0.0, 0.0)
    u = UAV(0, PARAMS, (3000.0, 0.0))
    u.energy = 1.1 * u.energy_to_reach(3000.0)  # below the 1.3 margin -> should turn back
    events = u.step(5.0, base)
    assert ("rth_triggered", None) in events
    assert u.status == Status.RETURNING


def test_energy_is_monotonically_non_increasing():
    u = UAV(0, PARAMS, (2000.0, 0.0))
    u.target = (0.0, 0.0)
    prev = u.energy
    for _ in range(20):
        u.step(5.0, (0.0, 0.0))
        assert u.energy <= prev
        prev = u.energy
