"""Detection-cache georeferencing (pure functions; no GPU/models) — Phase 3."""

from __future__ import annotations

import pytest
from src.perception.detect_cache import enu_to_wgs84, georeference_rows


def test_enu_origin_and_directions():
    lat0, lon0 = 29.75, -95.36
    assert enu_to_wgs84(0.0, 0.0, lat0, lon0) == pytest.approx((lat0, lon0))
    # north (+northing) raises latitude; east (+easting) raises longitude
    lat_n, _ = enu_to_wgs84(0.0, 1000.0, lat0, lon0)
    _, lon_e = enu_to_wgs84(1000.0, 0.0, lat0, lon0)
    assert lat_n > lat0
    assert lon_e > lon0


def test_georeference_rows_places_detection_in_its_cell():
    # cell_id 7 in a 6-col grid -> row 1, col 1; a detection at the cell centre.
    raw = [
        {"cls": "person", "confidence": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "model": "A"}
    ]
    rows = georeference_rows(
        raw,
        scenario="flood_a",
        cell_id=7,
        n_cols=6,
        cell_size_m=200.0,
        origin_lat=29.75,
        origin_lon=-95.36,
        source_image="img.jpg",
    )
    assert len(rows) == 1
    r = rows[0]
    # col 1 + cx 0.5 -> easting 1.5*200 = 300; row 1 + cy 0.5 -> northing -1.5*200 = -300
    assert r["bbox_utm"][0] == pytest.approx(300.0)
    assert r["bbox_utm"][1] == pytest.approx(-300.0)
    assert r["bbox_utm"][2] == pytest.approx(40.0)  # w 0.2 * 200
    assert r["scenario"] == "flood_a" and r["cell_id"] == 7
    assert r["class"] == "person" and r["model"] == "A"
    assert r["synthetic_geo"] is True
    # south of the origin row -> latitude below origin
    assert r["lat"] < 29.75


def test_empty_detections_yield_no_rows():
    assert (
        georeference_rows(
            [],
            scenario="s",
            cell_id=0,
            n_cols=6,
            cell_size_m=200.0,
            origin_lat=0.0,
            origin_lon=0.0,
            source_image="x.jpg",
        )
        == []
    )
