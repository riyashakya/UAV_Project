"""End-to-end Phase 1 build: dry-run counts and on-disk unified datasets."""

from __future__ import annotations

from pathlib import Path

from src.perception.datasets.build import DatasetWriter, collect, render_table


def test_dry_run_totals(taxonomy, datasets_cfg):
    report = collect(taxonomy, datasets_cfg)

    # detect set
    assert report.total("detect", "person") == 9  # VisDrone 3 + SARD 6
    assert report.total("detect", "vehicle") == 5  # VisDrone
    # seg set
    assert report.total("segment", "building_damaged") == 4  # RescueNet 3 + FloodNet 1
    assert report.total("segment", "road_blocked") == 3  # RescueNet 2 + FloodNet 1
    assert report.total("segment", "water") == 2  # RescueNet 1 + FloodNet 1

    assert all(report.available.values())
    assert report.splits["train"] > 0 and report.splits["val"] > 0


def test_render_table_lists_both_sets(taxonomy, datasets_cfg):
    table = render_table(collect(taxonomy, datasets_cfg), taxonomy)
    assert "DETECT (Model A)" in table
    assert "SEGMENT (Model B)" in table
    for cls in ("person", "vehicle", "building_damaged", "road_blocked", "water"):
        assert cls in table


def test_writer_emits_yolo_and_yoloseg_labels(taxonomy, datasets_cfg):
    detect_dir = Path(datasets_cfg.output.detect_dir)
    seg_dir = Path(datasets_cfg.output.seg_dir)
    writer = DatasetWriter(taxonomy, detect_dir, seg_dir)

    collect(taxonomy, datasets_cfg, writer=writer)
    writer.write_data_yaml()

    detect_labels = list((detect_dir / "labels").rglob("*.txt"))
    seg_labels = list((seg_dir / "labels").rglob("*.txt"))
    assert detect_labels and seg_labels
    assert (detect_dir / "data.yaml").exists()
    assert (seg_dir / "data.yaml").exists()
    # real train/val split folders exist and an image was placed next to its label
    assert (detect_dir / "images" / "train").is_dir()
    assert (detect_dir / "labels" / "val").is_dir()
    assert list((detect_dir / "images" / "train").glob("*.jpg"))

    # a detect label line is "class xc yc w h" (5 fields), class id in {0, 1}
    det_lines = [ln for p in detect_labels for ln in p.read_text().splitlines() if ln]
    assert det_lines
    for line in det_lines:
        fields = line.split()
        assert len(fields) == 5
        assert fields[0] in {"0", "1"}

    # a seg label line is "class x1 y1 ... xn yn" -> odd field count, >= 1 + 3*2
    seg_lines = [ln for p in seg_labels for ln in p.read_text().splitlines() if ln]
    assert seg_lines
    for line in seg_lines:
        fields = line.split()
        assert fields[0] in {"0", "1", "2"}
        assert len(fields) % 2 == 1 and len(fields) >= 7


def test_restrict_to_single_source(taxonomy, datasets_cfg):
    report = collect(taxonomy, datasets_cfg, sources=["sard"])
    assert report.total("detect", "person") == 6
    assert report.total("detect", "vehicle") == 0
    assert report.available == {"sard": True}
