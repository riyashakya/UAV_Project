"""Vision-estimated flood current (PIV) — tests-first contract.

The drift model already advects under a ``FlowField = (x, y) -> (vx, vy)``. Here we *estimate* that
field from a (synthetic, known-truth) drone clip via particle image velocimetry (FFT
cross-correlation), instead of assuming it. These analytic cases pin the contract:

* PIV recovers a KNOWN uniform pixel shift;
* pixel displacement converts to the correct metric velocity;
* the estimated ``FlowField`` drives the tested drift advection to the analytic displacement;
* synthesis is deterministic under an explicit ``np.random.Generator``.

CPU-only, pure numpy — no OpenCV, no datasets (fast; not marked slow).
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
from src.drift.advect import advect_particles
from src.perception.flow import (
    estimate_flow_piv,
    piv_to_flowfield,
    synthesize_flow_video,
)
from src.sim.world import make_flow_field


def _uniform(vx, vy):
    return make_flow_field(OmegaConf.create({"type": "uniform", "vx": vx, "vy": vy}))


def test_synth_video_is_deterministic():
    """Same seed -> identical frames (never touch the global RNG)."""
    kw = dict(
        truth_flow=_uniform(3.0, 0.0),
        n_frames=4,
        frame_px=(96, 96),
        metres_per_pixel=1.0,
        fps=1.0,
        density=0.04,
    )
    a = synthesize_flow_video(rng=np.random.default_rng(0), **kw)
    b = synthesize_flow_video(rng=np.random.default_rng(0), **kw)
    assert a.shape == (4, 96, 96)
    assert np.array_equal(a, b)


def test_piv_recovers_known_uniform_shift():
    """A clip that moves 3 px/frame east must be recovered as dX≈+3, dY≈0."""
    frames = synthesize_flow_video(
        truth_flow=_uniform(3.0, 0.0),  # 3 m/s east, mpp=1, fps=1 -> 3 px/frame east
        n_frames=6,
        frame_px=(128, 128),
        metres_per_pixel=1.0,
        fps=1.0,
        density=0.05,
        rng=np.random.default_rng(1),
    )
    piv = estimate_flow_piv(frames, window_px=32, step_px=16)
    mean_dy, mean_dx = piv["disp_px"].mean(axis=0)
    assert abs(mean_dx - 3.0) < 0.6  # east displacement recovered
    assert abs(mean_dy - 0.0) < 0.6  # no north/south drift


def test_static_clip_gives_near_zero_flow():
    frames = synthesize_flow_video(
        truth_flow=_uniform(0.0, 0.0),
        n_frames=5,
        frame_px=(96, 96),
        metres_per_pixel=1.0,
        fps=1.0,
        density=0.05,
        rng=np.random.default_rng(2),
    )
    piv = estimate_flow_piv(frames, window_px=32, step_px=16)
    assert np.linalg.norm(piv["disp_px"].mean(axis=0)) < 0.6


def test_pixel_to_metric_conversion():
    """disp_px * metres_per_pixel * fps = velocity (m/s). One vector -> constant field."""
    piv = {
        "centers_px": np.array([[50.0, 50.0]]),
        "disp_px": np.array([[0.0, 2.0]]),  # 2 px/frame east, 0 south
    }
    flow, _ = piv_to_flowfield(piv, metres_per_pixel=0.5, fps=4.0)
    vx, vy = flow(123.0, 456.0)  # single sample -> same vector everywhere (IDW)
    assert abs(vx - (2.0 * 0.5 * 4.0)) < 1e-9  # 4.0 m/s east
    assert abs(vy - 0.0) < 1e-9


def test_estimated_flowfield_drives_drift_to_analytic_displacement():
    """End-to-end: PIV-estimated field, fed to the tested advection with no diffusion, must
    displace a particle by ~v·t — same analytic contract as the drift model's own test."""
    frames = synthesize_flow_video(
        truth_flow=_uniform(2.0, 0.0),  # 2 m/s east
        n_frames=6,
        frame_px=(128, 128),
        metres_per_pixel=1.0,
        fps=1.0,
        density=0.05,
        rng=np.random.default_rng(3),
    )
    piv = estimate_flow_piv(frames, window_px=32, step_px=16)
    flow, _ = piv_to_flowfield(piv, metres_per_pixel=1.0, fps=1.0)
    pos = advect_particles(
        (0.0, 0.0),
        flow,
        horizon_s=100.0,
        dt=10.0,
        n_particles=1,
        leeway_factor=1.0,
        k_h=0.0,
        rng=np.random.default_rng(0),
    )[0]
    assert abs(pos[0] - 200.0) < 40.0  # ~2 m/s * 100 s = 200 m east (within PIV tolerance)
    assert abs(pos[1] - 0.0) < 40.0
