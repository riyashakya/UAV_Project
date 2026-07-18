"""Builders that materialise tiny synthetic datasets on disk.

Expected unified instance counts (asserted by tests):

    detect  : person = 8 (VisDrone 2 + SARD 6), vehicle = 4 (VisDrone)
    segment : building_damaged = 3 (RescueNet 2 + FloodNet 1),
              road_blocked     = 2 (RescueNet 1 + FloodNet 1),
              water            = 2 (RescueNet 1 + FloodNet 1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Expected per-source unified instance counts, for tests to import.
EXPECTED = {
    "visdrone": {"person": 2, "vehicle": 4},
    "sard": {"person": 6},
    "rescuenet": {"building_damaged": 2, "road_blocked": 1, "water": 1},
    "floodnet": {"building_damaged": 1, "road_blocked": 1, "water": 1},
}


def _tiny_jpg(path: Path, size: tuple[int, int] = (48, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 120, 120)).save(path)


def make_visdrone(root: Path) -> None:
    """VisDrone YOLO export: 2 images. Classes chosen to exercise map + drop."""
    base = root / "VisDrone2019-DET-train"
    labels, images = base / "labels", base / "images"
    labels.mkdir(parents=True, exist_ok=True)
    # img1: pedestrian, people (-> person x2); car (-> vehicle); bicycle, motor (dropped)
    (labels / "000001.txt").write_text(
        "0 0.5 0.5 0.10 0.10\n"
        "1 0.2 0.2 0.05 0.05\n"
        "3 0.8 0.8 0.20 0.20\n"
        "2 0.1 0.1 0.05 0.05\n"
        "9 0.9 0.9 0.05 0.05\n"
    )
    # img2: van, truck, bus (-> vehicle x3); tricycle, awning-tricycle (dropped)
    (labels / "000002.txt").write_text(
        "4 0.5 0.5 0.10 0.10\n"
        "5 0.3 0.3 0.10 0.10\n"
        "8 0.6 0.6 0.10 0.10\n"
        "6 0.2 0.2 0.10 0.10\n"
        "7 0.1 0.1 0.10 0.10\n"
    )
    _tiny_jpg(images / "000001.jpg")
    _tiny_jpg(images / "000002.jpg")


def _voc(width: int, height: int, objects: list[str]) -> str:
    obj_xml = "".join(
        f"<object><name>{name}</name>"
        f"<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>90</ymax></bndbox>"
        f"</object>"
        for name in objects
    )
    return (
        f"<annotation><size><width>{width}</width><height>{height}</height>"
        f"<depth>3</depth></size>{obj_xml}</annotation>"
    )


def make_sard(root: Path) -> None:
    """SARD Pascal VOC: 6 person boxes across 2 images, spanning all 6 pose labels."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "000001.xml").write_text(
        _voc(1920, 1080, ["Standing", "Walking", "Lying", "Not Defined"])
    )
    (root / "000002.xml").write_text(_voc(1920, 1080, ["Sitting", "Running"]))
    _tiny_jpg(root / "000001.jpg")
    _tiny_jpg(root / "000002.jpg")


def _mask(size: int, blocks: dict[int, tuple[slice, slice]]) -> np.ndarray:
    arr = np.zeros((size, size), dtype=np.uint8)
    for value, (rows, cols) in blocks.items():
        arr[rows, cols] = value
    return arr


def make_rescuenet(root: Path) -> None:
    """RescueNet index mask: water(1), building-major(4), building-total(5), road-blocked(8),
    plus a dropped vehicle(6) region."""
    d = root / "label-masks"
    d.mkdir(parents=True, exist_ok=True)
    mask = _mask(
        100,
        {
            1: (slice(10, 30), slice(10, 30)),  # water
            4: (slice(10, 30), slice(40, 60)),  # building-major-damage -> building_damaged
            5: (slice(10, 30), slice(70, 90)),  # building-total-destruction -> building_damaged
            8: (slice(60, 80), slice(40, 60)),  # road-blocked
            6: (slice(60, 80), slice(10, 30)),  # vehicle -> dropped
        },
    )
    Image.fromarray(mask, mode="L").save(d / "0001_lab.png")
    _tiny_jpg(d / "0001.jpg")


def make_floodnet(root: Path) -> None:
    """FloodNet index mask: building-flooded(1), road-flooded(3), water(5); grass(9) dropped."""
    d = root / "label-masks"
    d.mkdir(parents=True, exist_ok=True)
    mask = _mask(
        100,
        {
            1: (slice(10, 30), slice(10, 30)),  # building-flooded -> building_damaged
            3: (slice(10, 30), slice(40, 60)),  # road-flooded -> road_blocked
            5: (slice(60, 80), slice(10, 30)),  # water
            9: (slice(60, 80), slice(40, 60)),  # grass -> dropped
        },
    )
    Image.fromarray(mask, mode="L").save(d / "0001_lab.png")
    _tiny_jpg(d / "0001.jpg")


def make_all(base: Path) -> dict[str, Path]:
    """Build all four datasets under ``base`` and return their root paths."""
    roots = {
        "visdrone": base / "visdrone",
        "sard": base / "sard",
        "rescuenet": base / "rescuenet",
        "floodnet": base / "floodnet",
    }
    make_visdrone(roots["visdrone"])
    make_sard(roots["sard"])
    make_rescuenet(roots["rescuenet"])
    make_floodnet(roots["floodnet"])
    return roots
