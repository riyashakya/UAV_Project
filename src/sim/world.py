"""Grid world and flow field — Phase 4.

Single responsibility: hold the area-of-interest as a grid with per-cell hazard state, plus
a 2D flow field ``u(x, y) -> (vx, vy)`` loaded from config (analytic fields for now: uniform,
channel, radial). Coordinates: EPSG:4326 for storage, local UTM for anything metric
(``_wgs84`` / ``_utm`` naming).

Stub: implemented in Phase 4.
"""

from __future__ import annotations


class World:
    """Grid AOI with per-cell hazard state and a flow field. Stub (Phase 4)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Phase 4: simulator world is not implemented yet.")
