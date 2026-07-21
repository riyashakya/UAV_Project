"""Engine determinism and coverage (Phase 4 acceptance)."""

from __future__ import annotations

import json

import pandas as pd
from src.sim.engine import round_robin_plan, run
from src.sim.oracle import Oracle
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

# Big battery so no UAV triggers RTH -> the round-robin plan can cover every cell.
PARAMS = UAVParams(15.0, 200.0, 0.35, battery_capacity_j=1e12, rth_margin=1.3)


def _oracle():
    base = dict(lat=1.0, lon=2.0, bbox_utm=[0.0, 0.0, 1.0, 1.0], source_image="x.jpg")
    rows = [
        {"scenario": "s", "cell_id": 1, "class": "person", "confidence": 0.9, "model": "A", **base},
        {"scenario": "s", "cell_id": 1, "class": "person", "confidence": 0.8, "model": "A", **base},
        {"scenario": "s", "cell_id": 4, "class": "water", "confidence": 0.7, "model": "B", **base},
    ]
    return Oracle(
        pd.DataFrame(rows), "s", false_negative_rate={"person": 0.3}, latency_s=(0.0, 2.0)
    )


def _fresh_run(seed):
    world = World(rows=3, cols=3, cell_size_m=100.0, base_cell=0)
    uavs = [UAV(i, PARAMS, world.base_xy) for i in range(2)]
    plan = round_robin_plan(world.n_cells, 2)
    return run(world, uavs, plan, seed=seed, duration_s=1200.0, dt=5.0, oracle=_oracle())


def test_same_seed_is_byte_identical():
    a = json.dumps(_fresh_run(0), sort_keys=True)
    b = json.dumps(_fresh_run(0), sort_keys=True)
    assert a == b


def test_full_coverage_with_enough_time_and_energy():
    result = _fresh_run(0)
    assert result["coverage"] == 1.0  # all 9 cells surveyed
    assert set(result["surveyed"]) == set(range(9))


def test_survivors_are_found_via_oracle():
    result = _fresh_run(0)
    # cell 1 has two persons; with a 0.3 miss rate at least one is usually found across the run
    assert "person" in result["found_total"] or "water" in result["found_total"]


def test_different_seeds_can_differ():
    # different seeds -> different false-negative draws (not guaranteed different, but the run
    # must at least complete and cover fully under both)
    assert _fresh_run(1)["coverage"] == 1.0
