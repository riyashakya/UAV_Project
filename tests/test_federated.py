"""Federated partitioning + FedAvg aggregation — dataset-free, no flwr/torch/GPU needed."""

from __future__ import annotations

import numpy as np
from src.perception.federated.fedavg import weighted_average
from src.perception.federated.partition import group_by_source


def test_group_by_source_splits_by_prefix():
    files = [
        "visdrone_0001.jpg",
        "visdrone_0002.jpg",
        "sard_0001.jpg",
        "other_0001.jpg",  # not a requested source -> dropped
    ]
    groups = group_by_source(files, ["visdrone", "sard"])
    assert set(groups) == {"visdrone", "sard"}
    assert len(groups["visdrone"]) == 2
    assert groups["sard"] == ["sard_0001.jpg"]
    assert "other" not in groups  # unknown source excluded


def test_group_by_source_is_non_iid_imbalanced():
    files = [f"visdrone_{i}.jpg" for i in range(5)] + [f"sard_{i}.jpg" for i in range(2)]
    groups = group_by_source(files, ["visdrone", "sard"])
    assert len(groups["visdrone"]) == 5 and len(groups["sard"]) == 2  # imbalanced, as in real data


def test_fedavg_sample_weighted_average():
    """FedAvg = sample-count-weighted mean of client weights (checked on numpy tensors)."""
    a = {"w": np.array([2.0, 4.0])}
    b = {"w": np.array([4.0, 8.0])}
    out = weighted_average([a, b], [1, 3])  # (1*a + 3*b) / 4
    assert np.allclose(out["w"], [3.5, 7.0])


def test_fedavg_equal_weights_is_plain_mean():
    a = {"x": np.array([0.0]), "y": np.array([10.0])}
    b = {"x": np.array([2.0]), "y": np.array([20.0])}
    out = weighted_average([a, b], [1, 1])
    assert np.allclose(out["x"], [1.0]) and np.allclose(out["y"], [15.0])


def test_fedavg_rejects_bad_input():
    import pytest

    with pytest.raises(ValueError):
        weighted_average([], [])
    with pytest.raises(ValueError):
        weighted_average([{"w": np.array([1.0])}], [0])  # zero total weight
