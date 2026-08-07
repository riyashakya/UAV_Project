"""IoU + recall/precision matching for the SAHI operating-point eval — dataset-free tests."""

from __future__ import annotations

import numpy as np
from src.perception.sahi_eval import iou, recall_precision


def test_iou_identical_box_is_one():
    b = np.array([0.0, 0.0, 10.0, 10.0])
    assert np.isclose(iou(b, b[None, :])[0], 1.0)


def test_iou_disjoint_is_zero_and_half_overlap():
    b = np.array([0.0, 0.0, 10.0, 10.0])
    boxes = np.array([[20.0, 20.0, 30.0, 30.0], [5.0, 0.0, 15.0, 10.0]])  # disjoint, half-overlap
    ious = iou(b, boxes)
    assert np.isclose(ious[0], 0.0)
    assert np.isclose(ious[1], 50 / 150)  # inter 50, union 150 -> 1/3


def test_iou_empty():
    assert len(iou(np.array([0.0, 0, 1, 1]), np.zeros((0, 4)))) == 0


def test_recall_precision_perfect():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    rec, prec = recall_precision([box], [box], 0.5)
    assert rec == 1.0 and prec == 1.0


def test_recall_precision_miss_and_false_positive():
    gt = [np.array([[0.0, 0.0, 10.0, 10.0]])]
    # one detection far away: misses the GT (recall 0) and is a false positive (precision 0)
    det = [np.array([[50.0, 50.0, 60.0, 60.0]])]
    rec, prec = recall_precision(det, gt, 0.5)
    assert rec == 0.0 and prec == 0.0


def test_recall_precision_partial():
    # 2 GT, 1 found + 1 spurious detection -> recall 1/2, precision 1/2
    gt = [np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])]
    det = [np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])]
    rec, prec = recall_precision(det, gt, 0.5)
    assert np.isclose(rec, 0.5) and np.isclose(prec, 0.5)
