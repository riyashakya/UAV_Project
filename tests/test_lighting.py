"""Brightness transform for the lighting-robustness eval — dataset-free unit tests."""

from __future__ import annotations

import numpy as np
from src.perception.lighting_eval import adjust_brightness


def test_brighten_and_darken_scale_pixels():
    img = np.full((2, 2, 3), 100, dtype=np.uint8)
    assert np.all(adjust_brightness(img, 1.5) == 150)  # brighter
    assert np.all(adjust_brightness(img, 0.5) == 50)  # darker


def test_identity_at_factor_one():
    img = np.array([[[10, 128, 250]]], dtype=np.uint8)
    assert np.array_equal(adjust_brightness(img, 1.0), img)


def test_clips_to_255_and_keeps_uint8():
    img = np.full((3, 3), 200, dtype=np.uint8)
    out = adjust_brightness(img, 1.8)  # 200*1.8 = 360 -> clipped
    assert out.dtype == np.uint8
    assert out.max() == 255
    assert np.all(out == 255)


def test_dark_never_negative():
    img = np.array([[0, 5, 10]], dtype=np.uint8)
    out = adjust_brightness(img, 0.1)
    assert out.min() >= 0 and out.dtype == np.uint8
