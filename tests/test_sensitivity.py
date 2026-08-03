"""Contribution B: perception × coordination sensitivity (dataset-free via synthetic detections)."""

from __future__ import annotations

import pandas as pd
from omegaconf import OmegaConf
from src.eval.sensitivity import run_sensitivity


def _synthetic_detections(scenario="flood_a", n_cells=36, per_cell=3) -> pd.DataFrame:
    """A person in every cell — so a UAV failure that abandons cells really costs survivors."""
    rows = []
    for cid in range(n_cells):
        for k in range(per_cell):
            rows.append(
                {
                    "scenario": scenario,
                    "cell_id": cid,
                    "class": "person",
                    "confidence": 0.5 + 0.01 * k,
                    "lat": 29.75,
                    "lon": -95.36,
                    "bbox_utm": [0.0, 0.0, 1.0, 1.0],
                    "source_image": f"{cid}_{k}.jpg",
                    "model": "A",
                    "synthetic_geo": True,
                }
            )
    return pd.DataFrame(rows)


def _cfg(**over):
    d = dict(
        scenario="flood_a",
        n_uavs=6,
        condition="two_fail",
        strategies=["auction", "static_partition_no_realloc"],
        fn_rates=[0.0, 0.3],
        n_seeds=8,
        fail_window_s=[20.0, 120.0],
    )
    d.update(over)
    return OmegaConf.create(d)


def test_perception_error_lowers_detection_and_auction_recovers_coverage():
    _, agg = run_sensitivity(_cfg(), detections=_synthetic_detections())

    def rate(strat, fn):
        r = agg[(agg.strategy == strat) & (agg.fn == fn)]
        return float(r.detect_rate_mean.iloc[0])

    # more detector error -> fewer survivors detected, for both strategies
    assert rate("auction", 0.3) < rate("auction", 0.0)
    assert rate("static_partition_no_realloc", 0.3) < rate("static_partition_no_realloc", 0.0)
    # under a failure, adaptive re-tasking recovers coverage -> detects at least as many survivors
    assert rate("auction", 0.0) >= rate("static_partition_no_realloc", 0.0)
    assert rate("auction", 0.3) >= rate("static_partition_no_realloc", 0.3)
    # with no perception error and full recovery, the auction finds ~all survivors
    assert rate("auction", 0.0) >= 0.95
