"""Builders that materialise tiny synthetic datasets on disk.

These mirror the *real* download formats (raw VisDrone annotations, Roboflow-VOC SARD with a
single ``human`` class, RescueNet/FloodNet index masks with originals in a sibling folder,
and train/val/test splits), so loader and build tests exercise the real code paths.

Expected unified instance counts (asserted by tests):

    detect  : person = 9 (VisDrone 3 + SARD 6), vehicle = 5 (VisDrone)
    segment : building_damaged = 4 (RescueNet 3 + FloodNet 1),
              road_blocked     = 3 (RescueNet 2 + FloodNet 1),
              water            = 2 (RescueNet 1 + FloodNet 1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Expected per-source unified instance counts (across all splits), for tests to import.
EXPECTED = {
    "visdrone": {"person": 3, "vehicle": 5},
    "sard": {"person": 6},
    "rescuenet": {"building_damaged": 3, "road_blocked": 2, "water": 1},
    "floodnet": {"building_damaged": 1, "road_blocked": 1, "water": 1},
}


def _jpg(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 120, 120)).save(path)


def make_visdrone(root: Path) -> None:
    """RAW VisDrone-DET: `left,top,w,h,score,category,trunc,occ`, split train/val folders."""

    def subset(name: str, files: dict[str, str]) -> None:
        base = root / name
        for stem, ann in files.items():
            (base / "annotations").mkdir(parents=True, exist_ok=True)
            (base / "annotations" / f"{stem}.txt").write_text(ann)
            _jpg(base / "images" / f"{stem}.jpg")

    subset(
        "VisDrone2019-DET-train",
        {
            # cat 1 ped, 2 people -> person; 4 car -> vehicle; 3 bicycle, 0 ignored -> drop
            "000001": "10,10,20,20,1,1,0,0\n30,30,10,10,1,2,0,0\n50,50,20,20,1,4,0,0\n"
            "70,70,10,10,1,3,0,0\n80,80,5,5,1,0,0,0\n",
            # cat 5 van, 6 truck, 9 bus -> vehicle; 10 motor, 11 others -> drop
            "000002": "10,10,20,20,1,5,0,0\n40,40,10,10,1,6,0,0\n60,60,10,10,1,9,0,0\n"
            "80,80,5,5,1,10,0,0\n85,85,5,5,1,11,0,0\n",
        },
    )
    subset(
        "VisDrone2019-DET-val",
        {"000003": "10,10,20,20,1,1,0,0\n40,40,10,10,1,4,0,0\n"},  # person1, vehicle1
    )


def _voc(objects: list[str], size: int = 100) -> str:
    obj_xml = "".join(
        f"<object><name>{n}</name>"
        f"<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>90</ymax></bndbox>"
        f"</object>"
        for n in objects
    )
    return (
        f"<annotation><size><width>{size}</width><height>{size}</height>"
        f"<depth>3</depth></size>{obj_xml}</annotation>"
    )


def make_sard(root: Path) -> None:
    """SARD Roboflow-VOC: single `human` class across train/valid/test."""
    for split, stem, n in (("train", "g1", 3), ("valid", "g2", 2), ("test", "g3", 1)):
        d = root / "Sard" / split
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}.xml").write_text(_voc(["human"] * n))
        _jpg(d / f"{stem}.jpg")


def _mask(size: int, blocks: dict[int, tuple[slice, slice]]) -> np.ndarray:
    arr = np.zeros((size, size), dtype=np.uint8)
    for value, (rows, cols) in blocks.items():
        arr[rows, cols] = value
    return arr


def _write_mask_pair(label_dir: Path, org_dir: Path, stem: str, mask: np.ndarray) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(label_dir / f"{stem}_lab.png")
    _jpg(org_dir / f"{stem}.jpg")


def make_rescuenet(root: Path) -> None:
    """RescueNet index masks; originals in the sibling `*-org-img` folder; train + val."""
    train = root / "segmentation-trainset"
    _write_mask_pair(
        train / "train-label-img",
        train / "train-org-img",
        "0001",
        _mask(
            100,
            {
                1: (slice(10, 30), slice(10, 30)),  # water
                4: (slice(10, 30), slice(40, 60)),  # building-major -> building_damaged
                5: (slice(10, 30), slice(70, 90)),  # building-total -> building_damaged
                8: (slice(60, 80), slice(40, 60)),  # road-blocked
                6: (slice(60, 80), slice(10, 30)),  # vehicle -> dropped
            },
        ),
    )
    val = root / "segmentation-validationset"
    _write_mask_pair(
        val / "val-label-img",
        val / "val-org-img",
        "0002",
        _mask(
            100,
            {
                4: (slice(10, 30), slice(10, 30)),  # building-major -> building_damaged
                8: (slice(60, 80), slice(60, 80)),  # road-blocked
            },
        ),
    )


def make_floodnet(root: Path) -> None:
    """FloodNet index masks (the CORRECT format) + originals in a sibling folder; train."""
    train = root / "train"
    _write_mask_pair(
        train / "train-label-img",
        train / "train-org-img",
        "0001",
        _mask(
            100,
            {
                1: (slice(10, 30), slice(10, 30)),  # building-flooded -> building_damaged
                3: (slice(10, 30), slice(40, 60)),  # road-flooded -> road_blocked
                5: (slice(60, 80), slice(10, 30)),  # water
                9: (slice(60, 80), slice(40, 60)),  # grass -> dropped
            },
        ),
    )


def make_all(base: Path) -> dict[str, Path]:
    """Build all four datasets under ``base`` and return their root paths."""
    roots = {name: base / name for name in ("visdrone", "sard", "rescuenet", "floodnet")}
    make_visdrone(roots["visdrone"])
    make_sard(roots["sard"])
    make_rescuenet(roots["rescuenet"])
    make_floodnet(roots["floodnet"])
    return roots
