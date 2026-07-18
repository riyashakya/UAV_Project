"""In-memory representation of a unified sample.

One ``UnifiedImage`` per source image, holding either detect boxes (Model A) or seg polygons
(Model B) already expressed in the unified taxonomy and in YOLO-normalised coordinates
(``[0, 1]``). Loaders produce these; the writer and the dry-run counter consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BBox:
    """YOLO-normalised bounding box; all fields in ``[0, 1]`` relative to image size."""

    xc: float
    yc: float
    w: float
    h: float

    def as_yolo(self) -> tuple[float, float, float, float]:
        return (self.xc, self.yc, self.w, self.h)


@dataclass(frozen=True)
class DetectInstance:
    """One detect-set instance: a unified class + normalised box + optional metadata."""

    cls: str
    bbox: BBox
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SegInstance:
    """One seg-set instance: a unified class + a YOLO-seg polygon.

    ``polygon`` is a flat list of normalised ``(x, y)`` vertices in ``[0, 1]``.
    """

    cls: str
    polygon: tuple[tuple[float, float], ...]


@dataclass
class UnifiedImage:
    """A single source image mapped into the unified taxonomy."""

    source: str
    image_path: Path
    width: int
    height: int
    split: str = "train"  # "train" | "val" | "test", inferred from the source folder
    detections: list[DetectInstance] = field(default_factory=list)
    segments: list[SegInstance] = field(default_factory=list)
