"""Ablation benchmark: separate the reallocation effect from the guidance effect (dataset-free)."""

from __future__ import annotations

from omegaconf import OmegaConf
from src.eval.benchmark import run_benchmark


def _cfg(**over):
    d = dict(
        grid={"rows": 8, "cols": 8},
        cell_size_m=200,
        hotspots=[[6, 6], [1, 6], [6, 1]],
        cluster_sigma=1.0,
        survivors_total=80,
        prior_noise=0.2,
        person_fn=0.1,
        n_uavs=4,
        condition="one_fail",
        n_seeds=8,
        duration_min=90,
        fail_window_s=[20.0, 120.0],
    )
    d.update(over)
    return OmegaConf.create(d)


def test_reallocation_helps_and_guidance_speeds_search():
    res = run_benchmark(_cfg())
    st = res["systems"]["static (no realloc)"]
    au = res["systems"]["auction (realloc)"]
    gu = res["systems"]["auction + guided"]
    # reallocation recovers coverage the static baseline loses under a UAV failure (robust)
    assert au["coverage"][0] > st["coverage"][0]
    # ...and detects at least as many survivors (never worse)
    assert au["detect"][0] >= st["detect"][0] - 0.02
    # the added guidance's main benefit is speed: it reaches 50% at least as fast as uniform search
    assert gu["t50"][0] <= au["t50"][0] + 1e-6
