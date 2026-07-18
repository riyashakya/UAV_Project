"""Source loaders map native annotations to unified instances with the expected counts."""

from __future__ import annotations

from collections import Counter

import pytest
from src.perception.datasets.loaders import build_loader

from tests.fixtures.synthetic import EXPECTED


def _count(loader) -> Counter:
    counts: Counter = Counter()
    for image in loader.iter_images():
        for det in image.detections:
            counts[det.cls] += 1
        for seg in image.segments:
            counts[seg.cls] += 1
    return counts


@pytest.mark.parametrize("source", ["visdrone", "sard", "rescuenet", "floodnet"])
def test_loader_counts_match_expected(source, taxonomy, datasets_cfg):
    root = datasets_cfg.roots[source]
    build_cfg = dict(datasets_cfg.build)
    loader = build_loader(source, root, taxonomy.sources[source], build_cfg)

    assert loader.is_available()
    assert dict(_count(loader)) == EXPECTED[source]


def test_loader_reports_unavailable_when_absent(tmp_path, taxonomy):
    loader = build_loader("visdrone", tmp_path / "nope", taxonomy.sources["visdrone"], {})
    assert not loader.is_available()
    assert list(loader.iter_images()) == []


def test_sard_keeps_pose_metadata(taxonomy, datasets_cfg):
    loader = build_loader("sard", datasets_cfg.roots["sard"], taxonomy.sources["sard"], {})
    poses = {d.meta.get("pose") for img in loader.iter_images() for d in img.detections}
    assert {"standing", "walking", "running", "sitting", "lying", "not-defined"} == poses
    assert all(d.cls == "person" for img in loader.iter_images() for d in img.detections)


def test_unknown_source_class_raises(tmp_path, taxonomy):
    """A VisDrone label with an out-of-range class id must fail loudly (no silent drop)."""
    base = tmp_path / "visdrone" / "VisDrone2019-DET-train" / "labels"
    base.mkdir(parents=True)
    (base / "x.txt").write_text("99 0.5 0.5 0.1 0.1\n")  # id 99 is undeclared
    loader = build_loader("visdrone", tmp_path / "visdrone", taxonomy.sources["visdrone"], {})
    with pytest.raises(IndexError):
        list(loader.iter_images())
