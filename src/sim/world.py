"""Grid world and analytic flow field — Phase 4.

Holds the area-of-interest as a metric grid of equal cells (reusing the scenario geometry), a
base location, per-cell priority, and a 2D flow field ``u(x, y) -> (vx, vy)`` selected from
config (analytic fields: uniform, channel, radial). Coordinates are local metres
(x = east, y = south — row/col increase east/south); the drift model (Phase 7) uses the flow.

CPU-only; imports nothing heavier than numpy (ADR-001).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

FlowField = Callable[[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Cell:
    """One grid cell: id, (row, col), and metric centre in metres."""

    id: int
    row: int
    col: int
    center_xy: tuple[float, float]


def make_flow_field(cfg) -> FlowField:
    """Build an analytic flow field ``(x, y) -> (vx, vy)`` from config."""
    kind = cfg.get("type", "uniform")
    if kind == "uniform":
        vx, vy = float(cfg.get("vx", 0.0)), float(cfg.get("vy", 0.0))
        return lambda x, y: (vx, vy)
    if kind == "channel":  # unidirectional flow, parabolic across the channel axis
        vmax, axis_y, half = float(cfg.vmax), float(cfg.axis_y), float(cfg.half_width)

        def channel(x: float, y: float) -> tuple[float, float]:
            t = (y - axis_y) / half
            return (vmax * max(0.0, 1.0 - t * t), 0.0)

        return channel
    if kind == "radial":  # source/sink from a centre (e.g. dam break)
        cx, cy, strength = float(cfg.center[0]), float(cfg.center[1]), float(cfg.strength)

        def radial(x: float, y: float) -> tuple[float, float]:
            dx, dy = x - cx, y - cy
            r = math.hypot(dx, dy) or 1e-9
            return (strength * dx / r, strength * dy / r)

        return radial
    raise ValueError(f"unknown flow field type {kind!r}")


class World:
    """Metric grid AOI with a base, per-cell priority, and a flow field."""

    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size_m: float,
        *,
        base_cell: int = 0,
        flow: FlowField | None = None,
        priority: np.ndarray | None = None,
    ):
        self.rows, self.cols = int(rows), int(cols)
        self.cell_size_m = float(cell_size_m)
        self.cells = [
            Cell(
                r * self.cols + c,
                r,
                c,
                ((c + 0.5) * self.cell_size_m, (r + 0.5) * self.cell_size_m),
            )
            for r in range(self.rows)
            for c in range(self.cols)
        ]
        self.flow: FlowField = flow or (lambda x, y: (0.0, 0.0))
        self.base_xy = self.cell_center(int(base_cell))
        self.priority = (
            np.ones(self.n_cells) if priority is None else np.asarray(priority, dtype=float)
        )

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        return self.cells[cell_id].center_xy

    def flow_at(self, x: float, y: float) -> tuple[float, float]:
        return self.flow(x, y)

    @classmethod
    def from_configs(cls, scenario_cfg, world_cfg) -> World:
        return cls(
            rows=int(scenario_cfg.grid.rows),
            cols=int(scenario_cfg.grid.cols),
            cell_size_m=float(scenario_cfg.cell_size_m),
            base_cell=int(world_cfg.get("base_cell", 0)),
            flow=make_flow_field(world_cfg.get("flow", {"type": "uniform"})),
        )
