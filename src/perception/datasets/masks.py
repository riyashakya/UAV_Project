"""Convert integer segmentation masks to YOLO-seg polygons.

RescueNet and FloodNet ship pixel masks whose values are class-palette indices. For Model B
we need YOLO-seg polygons in normalised coordinates. Per unified class we take the union of
its source indices, split into connected components, trace each component's outer boundary,
simplify it (Douglas-Peucker), and normalise by image size. Holes are ignored (YOLO-seg has
no hole concept); tiny regions and degenerate polygons are dropped per the build config.

Never reduce a mask to a bounding box (ADR-002): a box around a flood region is meaningless.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from skimage import measure


def mask_to_polygons(
    mask: np.ndarray,
    index_to_unified: dict[int, str],
    *,
    simplify_px: float,
    min_polygon_points: int,
    min_region_area_px: int,
) -> list[tuple[str, tuple[tuple[float, float], ...]]]:
    """Trace ``mask`` into ``(unified_class, normalised_polygon)`` pairs.

    Args:
        mask: 2D array of palette indices, shape ``(H, W)``.
        index_to_unified: palette index -> unified class, mapped classes only.
        simplify_px: Douglas-Peucker tolerance in pixels.
        min_polygon_points: drop polygons with fewer vertices after simplification.
        min_region_area_px: drop connected components smaller than this many pixels.

    Returns:
        One entry per polygon; vertices are ``(x, y)`` in ``[0, 1]``.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W); got shape {mask.shape}")
    height, width = mask.shape
    out: list[tuple[str, tuple[tuple[float, float], ...]]] = []

    for index, unified in index_to_unified.items():
        binary = mask == index
        if not binary.any():
            continue

        labelled = measure.label(binary, connectivity=2)
        for region in measure.regionprops(labelled):
            if region.area < min_region_area_px:
                continue

            component = labelled == region.label
            # Pad so components touching the image edge still yield a closed contour.
            padded = np.pad(component.astype(float), pad_width=1)
            contours = measure.find_contours(padded, level=0.5)
            if not contours:
                continue

            # Longest contour is the outer boundary; inner contours are holes we ignore.
            contour = max(contours, key=len)
            # find_contours yields (row, col); undo the 1px pad and flip to (x, y).
            xy = [(col - 1.0, row - 1.0) for row, col in contour]

            polygon = Polygon(xy)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # repair self-intersections
            polygon = polygon.simplify(simplify_px, preserve_topology=True)
            if polygon.is_empty:
                continue

            parts = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
            for part in parts:
                coords = list(part.exterior.coords)
                if len(coords) < min_polygon_points:
                    continue
                norm = tuple((x / width, y / height) for x, y in coords)
                out.append((unified, norm))

    return out
