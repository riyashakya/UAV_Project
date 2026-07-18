"""Source-dataset loaders: read native annotations, emit ``UnifiedImage`` records.

Four loaders behind one interface. Detect sources (VisDrone, SARD) yield boxes; segment
sources (RescueNet, FloodNet) yield polygons traced from index masks. Every loader routes
native classes through its :class:`SourceMapping`, so an undeclared class raises instead of
being silently dropped. Each image is tagged with its ``split`` (train/val/test), inferred
from the source folder name.

Expected on-disk layouts (documented in docs/datasets.md), matching the real downloads:
  * VisDrone : ``VisDrone2019-DET-{train,val}/annotations/*.txt`` (RAW format), ``images/*.jpg``.
  * SARD     : ``Sard/{train,valid,test}/*.xml`` Pascal VOC (Roboflow export), images alongside.
  * RescueNet: ``segmentation-{trainset,validationset}/*-label-img/*_lab.png`` index masks,
               originals in the sibling ``*-org-img/`` folder.
  * FloodNet : index masks + originals (same shape as RescueNet). NOTE: the RGB "ColorMasks"
               distribution is not usable here — see docs/datasets.md.
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

    def _split_for(self, path: Path) -> str:
        """Infer train/val/test from the path *below the dataset root* (avoids false hits)."""
        try:
            parts = [p.lower() for p in path.relative_to(self.root).parts]
        except ValueError:
            parts = [p.lower() for p in path.parts]
        joined = "/".join(parts)
        if "test" in joined:
            return "test"
        if "val" in joined:
            return "val"
        return "train"


class VisDroneLoader(SourceLoader):
    """VisDrone-DET in its RAW annotation format.

    Each line is ``left,top,width,height,score,category,truncation,occlusion`` in absolute
    pixels; category id is field index 5 (0..11). Boxes are normalised by the image size.
    """

    task = "detect"

    def _label_files(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*.txt") if p.parent.name == "annotations")

    def is_available(self) -> bool:
        return bool(self._label_files())

    def iter_images(self) -> Iterator[UnifiedImage]:
        for label_path in self._label_files():
            image_dir = label_path.parent.parent / "images"
            image_path = self._find_image(image_dir, label_path.stem)
            if image_path is None:
                continue  # cannot normalise boxes without the image size
            width, height = Image.open(image_path).size

            detections: list[DetectInstance] = []
            for line in label_path.read_text().splitlines():
                fields = [f for f in line.split(",") if f != ""]
                if len(fields) < 6:
                    continue
                left, top, w, h = (float(v) for v in fields[:4])
                category = int(fields[5])
                native = self.mapping.native[category]
                unified = self.mapping.unified_for(native)
                if unified is None:  # explicitly dropped (ignored-regions, bicycle, ...)
                    continue
                if w <= 0 or h <= 0:
                    continue
                bbox = BBox(
                    xc=(left + w / 2) / width,
                    yc=(top + h / 2) / height,
                    w=w / width,
                    h=h / height,
                )
                detections.append(DetectInstance(cls=unified, bbox=bbox))
            yield UnifiedImage(
                source=self.name,
                image_path=image_path,
                width=width,
                height=height,
                split=self._split_for(label_path),
                detections=detections,
            )


class SardLoader(SourceLoader):
    """SARD in Pascal VOC XML. Every box is a person; native label kept as metadata."""

    task = "detect"

    def _xml_files(self) -> list[Path]:
        return sorted(self.root.rglob("*.xml"))

    def is_available(self) -> bool:
        return bool(self._xml_files())

    def iter_images(self) -> Iterator[UnifiedImage]:
        keep_meta = self.mapping.keep_metadata
        for xml_path in self._xml_files():
            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            width = int(size.findtext("width")) if size is not None else 0
            height = int(size.findtext("height")) if size is not None else 0
            if width == 0 or height == 0:
                continue
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
                split=self._split_for(xml_path),
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

    def _find_original(self, mask_path: Path) -> Path | None:
        """Locate the RGB source image for a mask, incl. the sibling ``*-org-img`` folder."""
        stem = mask_path.stem.replace("_lab", "").replace("-lab", "")
        parent = mask_path.parent
        sibling = parent.parent / parent.name.lower().replace("label", "org")
        for directory in (sibling, parent):
            found = self._find_image(directory, stem)
            if found is not None:
                return found
        return None

    def iter_images(self) -> Iterator[UnifiedImage]:
        index_to_unified = self.mapping.index_to_unified()
        # Authoritative values live in configs/datasets/default.yaml; these fallbacks only
        # apply if a caller omits the build config entirely, and must mirror that file.
        simplify_px = float(self.build_cfg.get("polygon_simplify_px", 1.0))
        min_points = int(self.build_cfg.get("min_polygon_points", 4))
        min_area = int(self.build_cfg.get("min_region_area_px", 32))

        for mask_path in self._mask_files():
            image = Image.open(mask_path)
            if image.mode == "P":
                image = image.convert("L")
            mask = np.asarray(image)
            if mask.ndim != 2:
                raise ValueError(
                    f"{self.name}: {mask_path.name} is an RGB colour mask, not an index mask. "
                    "Index masks (one class number per pixel) are required — see docs/datasets.md."
                )
            height, width = mask.shape

            polys = mask_to_polygons(
                mask,
                index_to_unified,
                simplify_px=simplify_px,
                min_polygon_points=min_points,
                min_region_area_px=min_area,
            )
            segments = [SegInstance(cls=cls, polygon=poly) for cls, poly in polys]
            original = self._find_original(mask_path)
            yield UnifiedImage(
                source=self.name,
                image_path=original or mask_path,
                width=width,
                height=height,
                split=self._split_for(mask_path),
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
