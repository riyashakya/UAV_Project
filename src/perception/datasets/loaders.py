"""Source-dataset loaders: read native annotations, emit ``UnifiedImage`` records.

Four loaders behind one interface. Detect sources (VisDrone, SARD) yield boxes; segment
sources (RescueNet, FloodNet) yield polygons traced from masks. Every loader routes native
classes through its :class:`SourceMapping`, so an undeclared class raises instead of being
silently dropped.

Expected on-disk layouts (documented in docs/datasets.md):
  * VisDrone : ``**/labels/*.txt`` YOLO boxes (via Ultralytics export), images in ``**/images/``.
  * SARD     : ``**/*.xml`` Pascal VOC, images alongside.
  * RescueNet: index-mask PNGs (``*lab*.png``), original ``.jpg`` alongside.
  * FloodNet : index-mask PNGs (``*lab*.png``), original ``.jpg`` alongside.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from .masks import mask_to_polygons
from .model import BBox, DetectInstance, SegInstance, UnifiedImage
from .taxonomy import SourceMapping

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _normalise_class(name: str) -> str:
    """Canonicalise a native class string: lowercase, spaces/underscores -> hyphen."""
    return name.strip().lower().replace(" ", "-").replace("_", "-")


class SourceLoader(ABC):
    """Base loader. Concrete subclasses set ``task`` and implement iteration."""

    task: str

    def __init__(self, root: Path, mapping: SourceMapping, build_cfg: dict | None = None):
        self.root = Path(root)
        self.mapping = mapping
        self.build_cfg = build_cfg or {}

    @property
    def name(self) -> str:
        return self.mapping.name

    @abstractmethod
    def is_available(self) -> bool:
        """True if the dataset appears to be present on disk (used by dry-run)."""

    @abstractmethod
    def iter_images(self) -> Iterator[UnifiedImage]:
        """Yield one :class:`UnifiedImage` per source image with unified annotations."""

    def _find_image(self, stem_dir: Path, stem: str) -> Path | None:
        for ext in _IMAGE_EXTS:
            candidate = stem_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None


class VisDroneLoader(SourceLoader):
    """VisDrone-DET in YOLO format (Ultralytics export)."""

    task = "detect"

    def _label_files(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*.txt") if p.parent.name == "labels")

    def is_available(self) -> bool:
        return bool(self._label_files())

    def iter_images(self) -> Iterator[UnifiedImage]:
        for label_path in self._label_files():
            image_dir = label_path.parent.parent / "images"
            image_path = self._find_image(image_dir, label_path.stem) or label_path
            detections: list[DetectInstance] = []
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                native = self.mapping.native[class_id]
                unified = self.mapping.unified_for(native)
                if unified is None:  # explicitly dropped
                    continue
                xc, yc, w, h = (float(v) for v in parts[1:5])
                detections.append(DetectInstance(cls=unified, bbox=BBox(xc, yc, w, h)))
            yield UnifiedImage(
                source=self.name,
                image_path=image_path,
                width=0,  # boxes already normalised; native size not needed for detect
                height=0,
                detections=detections,
            )


class SardLoader(SourceLoader):
    """SARD in Pascal VOC XML. Every box is a person; pose kept as metadata."""

    task = "detect"

    def _xml_files(self) -> list[Path]:
        return sorted(self.root.rglob("*.xml"))

    def is_available(self) -> bool:
        return bool(self._xml_files())

    def iter_images(self) -> Iterator[UnifiedImage]:
        keep_meta = self.mapping.keep_metadata
        for xml_path in self._xml_files():
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            width = int(size.findtext("width")) if size is not None else 0
            height = int(size.findtext("height")) if size is not None else 0
            image_path = self._find_image(xml_path.parent, xml_path.stem) or xml_path

            detections: list[DetectInstance] = []
            for obj in root.findall("object"):
                native = _normalise_class(obj.findtext("name", ""))
                unified = self.mapping.unified_for(native)
                if unified is None:
                    continue
                box = obj.find("bndbox")
                xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
                xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
                bbox = BBox(
                    xc=((xmin + xmax) / 2) / width,
                    yc=((ymin + ymax) / 2) / height,
                    w=(xmax - xmin) / width,
                    h=(ymax - ymin) / height,
                )
                meta = {keep_meta: native} if keep_meta else {}
                detections.append(DetectInstance(cls=unified, bbox=bbox, meta=meta))
            yield UnifiedImage(
                source=self.name,
                image_path=image_path,
                width=width,
                height=height,
                detections=detections,
            )


class _MaskLoader(SourceLoader):
    """Shared logic for index-mask segment datasets (RescueNet, FloodNet)."""

    task = "segment"

    def _mask_files(self) -> list[Path]:
        pngs = sorted(self.root.rglob("*.png"))
        labelled = [p for p in pngs if "lab" in p.stem.lower() or "lab" in p.parent.name.lower()]
        return labelled or pngs

    def is_available(self) -> bool:
        return bool(self._mask_files())

    def iter_images(self) -> Iterator[UnifiedImage]:
        # Authoritative values live in configs/datasets/default.yaml; these fallbacks only
        # apply if a caller omits the build config entirely, and must mirror that file.
        index_to_unified = self.mapping.index_to_unified()
        simplify_px = float(self.build_cfg.get("polygon_simplify_px", 1.0))
        min_points = int(self.build_cfg.get("min_polygon_points", 4))
        min_area = int(self.build_cfg.get("min_region_area_px", 32))

        for mask_path in self._mask_files():
            image = Image.open(mask_path)
            if image.mode not in ("L", "P", "I"):
                image = image.convert("L")
            mask = np.asarray(image)
            height, width = mask.shape[:2]

            polys = mask_to_polygons(
                mask,
                index_to_unified,
                simplify_px=simplify_px,
                min_polygon_points=min_points,
                min_region_area_px=min_area,
            )
            segments = [SegInstance(cls=cls, polygon=poly) for cls, poly in polys]
            original = self._find_image(mask_path.parent, mask_path.stem.replace("_lab", ""))
            yield UnifiedImage(
                source=self.name,
                image_path=original or mask_path,
                width=width,
                height=height,
                segments=segments,
            )


class RescueNetLoader(_MaskLoader):
    """RescueNet index masks (11 classes)."""


class FloodNetLoader(_MaskLoader):
    """FloodNet index masks (10 classes)."""


LOADERS: dict[str, type[SourceLoader]] = {
    "visdrone": VisDroneLoader,
    "sard": SardLoader,
    "rescuenet": RescueNetLoader,
    "floodnet": FloodNetLoader,
}


def build_loader(
    name: str, root: Path, mapping: SourceMapping, build_cfg: dict | None = None
) -> SourceLoader:
    """Instantiate the loader registered for ``name``."""
    if name not in LOADERS:
        raise KeyError(f"No loader registered for {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name](root, mapping, build_cfg)
