"""Source loaders map native annotations to unified instances with the expected counts."""

from __future__ import annotations

from collections import Counter

import pytest
from PIL import Image
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


def test_sard_maps_human_to_person_with_metadata(taxonomy, datasets_cfg):
    loader = build_loader("sard", datasets_cfg.roots["sard"], taxonomy.sources["sard"], {})
    dets = [d for img in loader.iter_images() for d in img.detections]
    assert dets and all(d.cls == "person" for d in dets)
    assert {d.meta.get("pose") for d in dets} == {"human"}  # Roboflow export flattens poses


def test_loader_infers_splits(taxonomy, datasets_cfg):
    loader = build_loader(
        "visdrone", datasets_cfg.roots["visdrone"], taxonomy.sources["visdrone"], {}
    )
    splits = {img.image_path.name: img.split for img in loader.iter_images()}
    assert splits["000001.jpg"] == "train"
    assert splits["000003.jpg"] == "val"


def test_unknown_source_class_raises(tmp_path, taxonomy):
    """A VisDrone annotation with an out-of-range category id must fail loudly (no silent drop)."""
    base = tmp_path / "visdrone" / "VisDrone2019-DET-train"
    (base / "annotations").mkdir(parents=True)
    (base / "images").mkdir(parents=True)
    (base / "annotations" / "x.txt").write_text("10,10,20,20,1,99,0,0\n")  # category 99 undeclared
    Image.new("RGB", (100, 100)).save(base / "images" / "x.jpg")
    loader = build_loader("visdrone", tmp_path / "visdrone", taxonomy.sources["visdrone"], {})
    with pytest.raises(IndexError):
        list(loader.iter_images())
