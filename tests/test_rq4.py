"""RQ4: re-tasking search to the drift region vs the stale sighting (quantitative result)."""

from __future__ import annotations

from omegaconf import OmegaConf
from src.eval.rq4 import run_rq4

_BASE = dict(
    survivor_row=2,
    survivor_col=1,
    cell_size_m=200.0,
    horizon_s=1800.0,
    dt=30.0,
    leeway_factor=1.0,
    n_particles=400,
    containment=0.9,
    n_seeds=120,
)


def _cfg(**over):
    d = dict(
        _BASE, flow={"type": "channel", "vmax": 0.5, "axis_y": 500.0, "half_width": 700.0}, k_h=2.0
    )
    d.update(over)
    return OmegaConf.create(d)


def test_zero_diffusion_is_exact():
    """With no turbulence the drift is deterministic: the prediction centroid equals the true
    position (drift error ~0), and the stale sighting is off by exactly the drift distance."""
    r = run_rq4(_cfg(k_h=0.0, n_seeds=5))
    assert r["drift"]["mean_err"] < 1e-6
    # vmax*leeway*horizon = 0.5 * 1 * 1800 = 900 m of eastward drift along the channel axis
    assert abs(r["stale"]["mean_err"] - 900.0) < 1.0


def test_drift_aware_locates_better_than_stale():
    """The headline: searching the 90% drift zone locates the survivor far more often, and much
    closer, than searching the stale detection cell they have already drifted out of."""
    r = run_rq4(_cfg())
    assert r["drift"]["located_rate"] >= 0.85  # 90% containment holds (calibration)
    assert r["stale"]["located_rate"] < r["drift"]["located_rate"]  # stale point misses the drifter
    assert r["drift"]["mean_err"] < r["stale"]["mean_err"]  # drift-aware target is closer
