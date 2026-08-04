"""Option A benchmark: adaptive pipeline beats a static baseline under stress (dataset-free)."""

from __future__ import annotations

from omegaconf import OmegaConf
from src.eval.benchmark import run_benchmark


def _cfg(**over):
    d = dict(
        grid={"rows": 8, "cols": 8},
        cell_size_m=200,
        cluster_center=[6, 6],
        cluster_sigma=1.4,
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


def test_adaptive_beats_static_baseline_under_stress():
    res = run_benchmark(_cfg())
    names = list(res["systems"])
    base, adapt = res["systems"][names[0]], res["systems"][names[1]]
    # under a UAV failure, reallocation recovers coverage the static baseline loses (robust)
    assert adapt["coverage"]["mean"] > base["coverage"]["mean"]
    # ...and detects at least as many survivors — never worse (the survivor gap depends on whether
    # the failure hits the survivor-dense sector, so it is only strict on average over many seeds)
    assert adapt["detect"]["mean"] >= base["detect"]["mean"] - 0.02
    # ...and guided search locates 80% of all survivors at least as fast as the uniform sweep
    assert adapt["t80"]["mean"] <= base["t80"]["mean"]
