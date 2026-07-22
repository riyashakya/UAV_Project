"""Coordination metrics from an event log (Phase 9)."""

from __future__ import annotations

import math

from src.eval.metrics import compute_metrics


def test_metrics_on_a_synthetic_run():
    result = {
        "events": [
            {
                "t": 10.0,
                "uav": 0,
                "event": "arrived",
                "x": 0.0,
                "y": 0.0,
                "cell": 0,
                "found": ["person", "person"],
            },
            {"t": 20.0, "uav": 0, "event": "arrived", "x": 100.0, "y": 0.0, "cell": 1, "found": []},
            {
                "t": 30.0,
                "uav": 0,
                "event": "arrived",
                "x": 100.0,
                "y": 0.0,
                "cell": 0,
                "found": ["person"],
            },  # revisit of cell 0
        ],
        "lost_cells": [5, 6],
    }
    m = compute_metrics(result, n_cells=4, coverage_target=0.5, complete_threshold=0.95)
    assert m["coverage"] == 0.5  # 2 unique cells of 4
    assert m["redundant_ratio"] == 1.5  # 3 surveys / 2 unique
    assert m["survivors_found"] == 3
    assert m["lost_cells"] == 2
    assert m["completed"] == 0.0  # below 0.95
    assert m["time_to_target_s"] == 20.0  # reached 50% at the second arrival
    assert m["total_distance_m"] == 100.0  # (0,0)->(100,0)->(100,0)


def test_time_to_target_is_nan_when_unreached():
    result = {
        "events": [{"t": 5.0, "uav": 0, "event": "arrived", "x": 0.0, "y": 0.0, "cell": 0}],
        "lost_cells": [],
    }
    m = compute_metrics(result, n_cells=10, coverage_target=0.9)
    assert math.isnan(m["time_to_target_s"])  # only 10% covered
    assert m["coverage"] == 0.1
