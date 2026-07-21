"""Partitioning: workload balance for weighted Voronoi (Phase 5 acceptance)."""

from __future__ import annotations

import numpy as np
from src.coordination.partition import partition, sector_workloads
from src.sim.world import World


def _world(rows=6, cols=6):
    return World(rows=rows, cols=cols, cell_size_m=200.0)


def test_partition_covers_every_cell_exactly_once():
    world = _world()
    for strategy in ("grid", "weighted_voronoi"):
        plan = partition(world, 4, strategy=strategy)
        assigned = [c for cells in plan.values() for c in cells]
        assert sorted(assigned) == list(range(world.n_cells))  # a true partition, no overlaps


def test_weighted_voronoi_balances_workload_within_5pct():
    """The Phase 5 headline: uniform-priority sectors are within 5% of each other."""
    world = _world()  # uniform priority (all ones)
    plan = partition(world, 4, strategy="weighted_voronoi", tol=0.05)
    loads = np.array(list(sector_workloads(plan, world.priority).values()))
    assert (loads.max() - loads.min()) <= 0.05 * loads.mean()
    assert loads.sum() == world.n_cells  # nothing lost


def test_grid_baseline_is_less_balanced_than_voronoi():
    world = _world()
    grid = np.array(
        list(sector_workloads(partition(world, 4, strategy="grid"), world.priority).values())
    )
    vor = np.array(
        list(
            sector_workloads(
                partition(world, 4, strategy="weighted_voronoi"), world.priority
            ).values()
        )
    )
    grid_spread = grid.max() - grid.min()
    vor_spread = vor.max() - vor.min()
    assert vor_spread <= grid_spread  # balancing should not be worse than the naive baseline


def test_weighted_voronoi_respects_priority():
    """With a clustered high-priority region, weighted Voronoi balances *workload* far better
    than the naive equal-area grid (exact balance is limited by coarse priority granularity)."""
    world = _world()
    world.priority[:6] = 5.0  # first row is high-priority
    vor = np.array(
        list(
            sector_workloads(
                partition(world, 4, strategy="weighted_voronoi"), world.priority
            ).values()
        )
    )
    grid = np.array(
        list(sector_workloads(partition(world, 4, strategy="grid"), world.priority).values())
    )
    assert (vor.max() - vor.min()) < (grid.max() - grid.min())  # much tighter than equal-area
    assert (vor.max() - vor.min()) <= 0.2 * vor.mean()


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown partition strategy"):
        partition(_world(), 4, strategy="kmeans")
