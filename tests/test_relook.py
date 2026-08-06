"""Re-look detection model — dataset-free analytic tests for miss_after_looks."""

from __future__ import annotations

import numpy as np
from src.eval.relook import miss_after_looks


def test_single_look_equals_fn():
    for fn in (0.1, 0.4, 0.7):
        for ps in (0.0, 0.3, 1.0):
            assert abs(miss_after_looks(fn, ps, 1) - fn) < 1e-12  # k=1 must reproduce FN exactly


def test_independent_looks_are_fn_power_k():
    """persistent_share = 0 -> misses are independent -> miss = FN**k."""
    assert abs(miss_after_looks(0.5, 0.0, 2) - 0.25) < 1e-12
    assert abs(miss_after_looks(0.5, 0.0, 3) - 0.125) < 1e-12


def test_fully_persistent_looks_do_nothing():
    """persistent_share = 1 -> every miss is structural -> re-looks never help."""
    for k in (1, 2, 5, 10):
        assert abs(miss_after_looks(0.5, 1.0, k) - 0.5) < 1e-12


def test_monotone_non_increasing_in_k_and_bounded_by_floor():
    fn, ps = 0.5, 0.4
    vals = [miss_after_looks(fn, ps, k) for k in range(1, 8)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))  # never increases with more looks
    floor = ps * fn  # permanent-miss floor
    assert vals[-1] >= floor - 1e-9
    assert np.isclose(miss_after_looks(fn, ps, 1000), floor, atol=1e-6)  # converges to the floor


def test_zero_looks_always_miss():
    assert miss_after_looks(0.4, 0.3, 0) == 1.0
