"""Mask -> YOLO-seg polygon conversion (Model B input; ADR-002: never box a mask)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon
from src.perception.datasets.masks import mask_to_polygons


def _polys(mask, index_to_unified, **kw):
    kw.setdefault("simplify_px", 1.0)
    kw.setdefault("min_polygon_points", 4)
    kw.setdefault("min_region_area_px", 32)
    return mask_to_polygons(mask, index_to_unified, **kw)


def test_single_square_yields_one_normalised_polygon():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1  # a 20x20 block, area 400
    polys = _polys(mask, {1: "water"})

    assert len(polys) == 1
    cls, ring = polys[0]
    assert cls == "water"
    # normalised to [0, 1]
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in ring)
    # area of the normalised ring ~ (20/100)*(20/100) = 0.04
    assert Polygon(ring).area == pytest.approx(0.04, abs=0.01)


def test_two_disconnected_regions_are_two_instances():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 2
    mask[10:30, 60:80] = 2
    polys = _polys(mask, {2: "building_damaged"})
    assert len(polys) == 2
    assert {c for c, _ in polys} == {"building_damaged"}


def test_small_regions_are_dropped():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:3, 0:3] = 1  # area 9 < min_region_area_px (32)
    assert _polys(mask, {1: "water"}) == []


def test_unmapped_indices_are_ignored():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 7  # index 7 not in the map
    assert _polys(mask, {1: "water"}) == []
