"""Auction reallocation vs baselines (Phase 6 — the core contribution)."""

from __future__ import annotations

from src.coordination.allocation import Coordinator
from src.sim.engine import run
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

# Big battery so energy never limits the run: the test isolates the *reallocation* logic.
PARAMS = UAVParams(15.0, 200.0, 0.35, battery_capacity_j=5e7, rth_margin=1.3)


def _run(strategy, fail_at=None, seed=0):
    world = World(rows=6, cols=6, cell_size_m=200.0, base_cell=0)
    uavs = [UAV(i, PARAMS, world.base_xy) for i in range(4)]
    coord = Coordinator(strategy, world, 4)
    return run(
        world, uavs, coordinator=coord, seed=seed, duration_s=3600.0, dt=5.0, fail_at=fail_at
    )


def test_auction_recovers_abandoned_cells_static_does_not():
    """The Phase 6 headline: UAV-2 dies with unfinished cells. Auction recovers to >95%;
    static_partition_no_realloc must NOT (it loses those cells)."""
    fail = {2: 30.0}  # UAV 2 dies 30 s in, mid-sector, leaving several cells unsurveyed
    auction = _run("auction", fail_at=fail)
    static = _run("static_partition_no_realloc", fail_at=fail)

    assert auction["coverage"] > 0.95  # abandoned cells picked up by other UAVs
    assert static["coverage"] < 0.95  # abandoned cells lost -> baseline fails the same test
    assert auction["coverage"] > static["coverage"]
    assert static["lost_cells"]  # static leaves cells permanently unsurveyed


def test_reallocation_event_is_logged():
    auction = _run("auction", fail_at={2: 30.0})
    reassigned = [e for e in auction["events"] if e["event"] == "reassigned"]
    assert reassigned  # a reallocation actually happened
    moved_cells = {c for e in reassigned for c in e["cells"]}
    assert moved_cells <= set(auction["surveyed"])  # and those cells did get surveyed


def test_no_failure_all_strategies_cover_fully():
    for strategy in ("auction", "static_partition_no_realloc", "single_uav", "random_walk"):
        assert _run(strategy)["coverage"] == 1.0  # with no failures + time, everyone finishes


def test_single_uav_leaves_three_uavs_idle():
    result = _run("single_uav")
    # only UAV 0 is assigned work; 1-3 have nothing and land at (near) full battery
    assert result["uav_end"][1]["energy"] > result["uav_end"][0]["energy"]


def test_bid_prefers_near_and_high_priority():
    world = World(rows=4, cols=4, cell_size_m=100.0)
    coord = Coordinator("auction", world, 2)
    near = UAV(0, PARAMS, world.cell_center(0))  # sitting on cell 0
    far = UAV(1, PARAMS, world.cell_center(15))  # far corner
    assert coord._bid(near, 0) < coord._bid(far, 0)  # nearer UAV bids lower
    bid_normal = coord._bid(near, 0)
    coord.priority[0] = 10.0
    assert coord._bid(near, 0) < bid_normal  # higher priority -> lower (more eager) bid


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown strategy"):
        Coordinator("greedy", World(2, 2, 100.0), 2)
