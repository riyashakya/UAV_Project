"""Oracle: cell/scenario filtering, deterministic false-negatives, and latency (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.sim.oracle import Oracle


def _df():
    base = dict(lat=1.0, lon=2.0, bbox_utm=[10.0, 20.0, 5.0, 5.0], source_image="x.jpg")
    return pd.DataFrame(
        [
            {
                "scenario": "s",
                "cell_id": 0,
                "class": "person",
                "confidence": 0.9,
                "model": "A",
                **base,
            },
            {
                "scenario": "s",
                "cell_id": 0,
                "class": "water",
                "confidence": 0.8,
                "model": "B",
                **base,
            },
            {
                "scenario": "s",
                "cell_id": 1,
                "class": "vehicle",
                "confidence": 0.7,
                "model": "A",
                **base,
            },
            {
                "scenario": "other",
                "cell_id": 0,
                "class": "person",
                "confidence": 0.5,
                "model": "A",
                **base,
            },
        ]
    )


def test_filters_by_scenario_and_cell():
    o = Oracle(_df(), "s")
    assert o.cell_ids == [0, 1]  # 'other' scenario excluded
    d0 = o.get_detections(0, np.random.default_rng(0))
    assert {d.cls for d in d0} == {"person", "water"}
    assert o.get_detections(1, np.random.default_rng(0))[0].cls == "vehicle"
    assert o.get_detections(99, np.random.default_rng(0)) == []


def test_deterministic_under_seed():
    o = Oracle(_df(), "s", false_negative_rate={"person": 0.5, "water": 0.5}, latency_s=(0.0, 3.0))
    a = o.get_detections(0, np.random.default_rng(42))
    b = o.get_detections(0, np.random.default_rng(42))
    assert [(d.cls, d.latency_s) for d in a] == [(d.cls, d.latency_s) for d in b]


def test_false_negative_rate_extremes():
    always = Oracle(_df(), "s", false_negative_rate={"person": 1.0, "water": 1.0})
    assert always.get_detections(0, np.random.default_rng(1)) == []  # everything missed
    never = Oracle(_df(), "s", false_negative_rate={"person": 0.0, "water": 0.0})
    assert len(never.get_detections(0, np.random.default_rng(1))) == 2  # nothing missed


def test_latency_within_range_and_fields():
    o = Oracle(_df(), "s", latency_s=(1.0, 1.0))
    det = o.get_detections(1, np.random.default_rng(0))[0]
    assert det.latency_s == pytest.approx(1.0)
    assert det.bbox_utm == (10.0, 20.0, 5.0, 5.0)
    assert det.model == "A" and det.cls == "vehicle"
