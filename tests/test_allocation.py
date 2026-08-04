"""Auction reallocation vs baselines (Phase 6 — the core contribution)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from src.coordination.allocation import Coordinator
from src.sim.engine import run
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

# Big battery so energy never limits the run: the test isolates the *reallocation* logic.
PARAMS = UAVParams(15.0, 200.0, 0.35, battery_capacity_j=5e7, rth_margin=1.3)


@dataclass
class _Det:  # a minimal detection exposing the `.cls` the coordinator reads
    cls: str


# A strong, purely eastward flood current: a survivor drifts east and nowhere else.
_EAST_FLOW = {"n_particles": 300, "horizon_s": 600.0, "dt": 30.0, "leeway_factor": 1.0, "k_h": 0.5}


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


def _drift_coord(seed=0):
    # vx=1 m/s over 600 s ≈ 600 m ≈ 3 cells east of the source; tight spread keeps it on-grid.
    world = World(rows=6, cols=6, cell_size_m=200.0, flow=lambda x, y: (1.0, 0.0))
    return world, Coordinator(
        "auction", world, 2, drift_retask=True, drift_params=_EAST_FLOW, drift_seed=seed
    )


def test_drift_retask_boosts_only_cells_downstream_of_the_survivor():
    """RQ4: on a `person` detection, drift re-tasking boosts the cells the survivor drifts INTO
    (east, with the current) — never cells upstream (west) of the detection."""
    world, coord = _drift_coord()
    src = 13  # row 2, col 1 — interior, with room to drift east
    src_col = src % world.cols
    before = coord.priority.copy()
    coord.on_survey(uav=None, cell=src, detections=[_Det("person")])

    boosted = [int(c) for c in np.flatnonzero(coord.priority > before)]
    assert boosted, "a survivor detection must re-task at least one cell"
    assert all(c % world.cols >= src_col for c in boosted)  # never upstream (west) of detection
    assert any(c % world.cols > src_col for c in boosted)  # genuinely carried downstream (east)


def test_drift_retask_differs_from_neighbour_boost():
    """Drift mode targets downstream cells; the default neighbour boost includes the upstream
    (west) neighbour — so the two modes re-task provably different cells."""
    world, drift = _drift_coord()
    plain = Coordinator("auction", world, 2)  # default neighbour-boost mode
    src = 13
    west_neighbour = src - 1

    for coord in (drift, plain):
        coord.on_survey(uav=None, cell=src, detections=[_Det("person")])

    assert plain.priority[west_neighbour] > 1.0  # neighbour mode boosts the upstream cell
    assert drift.priority[west_neighbour] == 1.0  # drift mode does not — the survivor went east


def test_drift_retask_is_reproducible_under_seed():
    """Same drift seed → identical re-tasking (its own RNG keeps runs deterministic)."""
    (_, a), (_, b) = _drift_coord(seed=7), _drift_coord(seed=7)
    a.on_survey(uav=None, cell=13, detections=[_Det("person")])
    b.on_survey(uav=None, cell=13, detections=[_Det("person")])
    assert np.array_equal(a.priority, b.priority)


def test_greedy_priority_searches_high_priority_cell_first():
    """Probability-guided ordering: a UAV goes to the high-priority (high-survivor-likelihood) cell
    first, whereas the default FIFO order starts at cell 0."""
    world = World(rows=4, cols=4, cell_size_m=100.0)
    world.priority[10] = 50.0  # a strongly-preferred cell (a survivor-likelihood prior)
    guided = Coordinator("single_uav", world, 1, greedy_priority=True)
    assert guided.next_cell(UAV(0, PARAMS, world.base_xy), np.random.default_rng(0)) == 10
    plain = Coordinator("single_uav", world, 1)  # FIFO drains from cell 0
    assert plain.next_cell(UAV(0, PARAMS, world.base_xy), np.random.default_rng(0)) == 0


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown strategy"):
        Coordinator("greedy", World(2, 2, 100.0), 2)
