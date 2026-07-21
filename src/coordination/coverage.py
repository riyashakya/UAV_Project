"""Coverage-path planning — Phase 5.

Boustrophedon ("lawnmower") sweep within a sector, parameterised by camera footprint width and
desired sidelap. Sweep lines step by ``footprint × (1 - sidelap)`` so adjacent passes overlap.

The classic bug this guards against: when the sector height is not an integer multiple of the
step, a naive loop stops one strip short and never covers the last row. ``sweep_line_ys`` adds an
explicit final line flush with the far edge so the whole sector is always covered (Phase 5 test).

CPU-only; no perception imports (ADR-001).
"""

from __future__ import annotations

from src.sim.world import World


def sweep_line_ys(
    ymin: float, ymax: float, footprint_width_m: float, sidelap: float
) -> list[float]:
    """Y-centres of the sweep lines covering ``[ymin, ymax]``, incl. the last-row fix."""
    fw = footprint_width_m
    stride = fw * (1.0 - sidelap)
    if stride <= 0:
        raise ValueError("sidelap must be < 1 so the sweep advances")
    top = ymax - fw / 2
    if top <= ymin + fw / 2:  # sector thinner than one footprint -> a single centred line
        return [(ymin + ymax) / 2]

    ys: list[float] = []
    y = ymin + fw / 2
    while y <= top + 1e-9:
        ys.append(y)
        y += stride
    # LAST-ROW FIX: ensure a sweep line sits flush with the far edge.
    if ys[-1] < top - 1e-9:
        ys.append(top)
    return ys


def _sector_bbox(sector_cells: list[int], world: World) -> tuple[float, float, float, float]:
    xs = [world.cells[c].center_xy[0] for c in sector_cells]
    ys = [world.cells[c].center_xy[1] for c in sector_cells]
    h = world.cell_size_m / 2
    return min(xs) - h, max(xs) + h, min(ys) - h, max(ys) + h


def coverage_path(
    sector_cells: list[int], world: World, footprint_width_m: float, sidelap: float
) -> list[tuple[float, float]]:
    """Ordered boustrophedon waypoints covering the sector's bounding box."""
    if not sector_cells:
        return []
    xmin, xmax, ymin, ymax = _sector_bbox(sector_cells, world)
    path: list[tuple[float, float]] = []
    for i, y in enumerate(sweep_line_ys(ymin, ymax, footprint_width_m, sidelap)):
        leg = [(xmin, y), (xmax, y)] if i % 2 == 0 else [(xmax, y), (xmin, y)]
        path.extend(leg)
    return path


def coverage_fraction(
    sector_cells: list[int], world: World, footprint_width_m: float, sidelap: float
) -> float:
    """Fraction of the sector's cells whose centre falls within a sweep line's footprint."""
    if not sector_cells:
        return 0.0
    _, _, ymin, ymax = _sector_bbox(sector_cells, world)
    lines = sweep_line_ys(ymin, ymax, footprint_width_m, sidelap)
    half = footprint_width_m / 2
    covered = sum(
        1
        for c in sector_cells
        if any(abs(world.cells[c].center_xy[1] - ly) <= half + 1e-9 for ly in lines)
    )
    return covered / len(sector_cells)
