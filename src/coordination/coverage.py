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


def spiral_path(
    sector_cells: list[int], world: World, footprint_width_m: float, sidelap: float
) -> list[tuple[float, float]]:
    """Ordered inward rectangular-spiral waypoints covering the sector's bounding box, with rings
    spaced by the same ``footprint × (1 - sidelap)`` stride as the lawnmower sweep (so both cover
    to the same footprint). An alternative to boustrophedon, for the coverage-pattern comparison."""
    if not sector_cells:
        return []
    xmin, xmax, ymin, ymax = _sector_bbox(sector_cells, world)
    fw = footprint_width_m
    stride = fw * (1.0 - sidelap)
    if stride <= 0:
        raise ValueError("sidelap must be < 1 so the spiral advances")
    left, right = xmin + fw / 2, xmax - fw / 2
    top, bottom = ymin + fw / 2, ymax - fw / 2
    path: list[tuple[float, float]] = []
    while left <= right + 1e-9 and top <= bottom + 1e-9:
        path.append((left, top))  # traverse this ring: right, down, left, up-into-next
        path.append((right, top))
        path.append((right, bottom))
        path.append((left, bottom))
        left += stride
        top += stride
        right -= stride
        bottom -= stride
        if left <= right + 1e-9 and top <= bottom + 1e-9:
            path.append((left - stride, top))  # step inward to the next ring's start
    return path or [((xmin + xmax) / 2, (ymin + ymax) / 2)]


def path_length(path: list[tuple[float, float]]) -> float:
    """Total Euclidean length of a waypoint path (metres)."""
    import math

    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))


def path_coverage_fraction(
    path: list[tuple[float, float]], sector_cells: list[int], world: World, footprint_width_m: float
) -> float:
    """Fraction of sector cells whose centre lies within ``footprint/2`` of the path (2-D: densely
    samples each leg, so it scores any pattern — lawnmower or spiral — on the same basis)."""
    import math

    if not sector_cells or len(path) < 1:
        return 0.0
    half = footprint_width_m / 2
    step = max(1.0, footprint_width_m / 4)
    pts: list[tuple[float, float]] = [path[0]]
    for a, b in zip(path[:-1], path[1:]):  # sample points along each leg
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        for k in range(1, int(d / step) + 1):
            t = k * step / d
            pts.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        pts.append(b)
    covered = 0
    for c in sector_cells:
        cx, cy = world.cells[c].center_xy
        if any(math.hypot(px - cx, py - cy) <= half + 1e-9 for px, py in pts):
            covered += 1
    return covered / len(sector_cells)


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
