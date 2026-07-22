"""Monte-Carlo sweep runner (Phase 9). Small grid, no oracle -> fast, GPU-free."""

from __future__ import annotations

from omegaconf import OmegaConf
from src.eval.runner import run_sweep
from src.sim.uav import UAVParams
from src.sim.world import World

CFG = OmegaConf.create(
    {
        "strategies": ["auction", "static_partition_no_realloc"],
        "n_uavs": [4],
        "conditions": ["two_fail"],
        "n_seeds": 8,
        "fail_window_s": [20.0, 60.0],
        "duration_min": 60,
        "timestep_s": 5.0,
        "coverage_target": 0.9,
        "complete_threshold": 0.95,
    }
)


def _sweep():
    world = World(rows=6, cols=6, cell_size_m=200.0, base_cell=0)
    params = UAVParams(15.0, 200.0, 0.35, battery_capacity_j=5e7, rth_margin=1.3)
    return run_sweep(CFG, world, params, oracle=None)


def test_sweep_produces_tidy_rows_and_ci_summary():
    tidy, summary = _sweep()
    assert len(tidy) == 2 * 1 * 1 * 8  # strategies × n_uavs × conditions × seeds
    assert {"strategy", "n_uavs", "condition", "seed", "coverage"} <= set(tidy.columns)
    assert "coverage_ci95" in summary.columns and "coverage_mean" in summary.columns


def test_auction_beats_static_under_failures():
    """The headline: under mid-mission failures, auction covers more than static partitioning."""
    _, summary = _sweep()
    auction = summary[summary.strategy == "auction"].coverage_mean.iloc[0]
    static = summary[summary.strategy == "static_partition_no_realloc"].coverage_mean.iloc[0]
    assert auction > static


def test_reproducible():
    a, _ = _sweep()
    b, _ = _sweep()
    assert a.equals(b)  # same seeds -> identical rows
