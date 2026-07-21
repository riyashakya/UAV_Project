"""UAV kinematics and energy model — Phase 4.

A point-mass UAV: constant cruise speed with instantaneous turns (documented simplification;
turn dynamics are negligible at survey scale), and a standard rotorcraft power model
``P = P_hover + k·v²``. It triggers **return-to-home** when remaining energy drops below
``rth_margin × energy_needed_to_reach_base`` — the margin (and every other constant) comes from
config. Deterministic: no randomness lives here (the engine owns the RNG).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    IDLE = "idle"
    CRUISING = "cruising"
    RETURNING = "returning"
    LANDED = "landed"
    DEAD = "dead"  # battery exhausted before reaching base


@dataclass(frozen=True)
class UAVParams:
    cruise_speed_ms: float
    p_hover_w: float
    k_drag: float
    battery_capacity_j: float
    rth_margin: float

    @classmethod
    def from_cfg(cls, cfg) -> UAVParams:
        return cls(
            cruise_speed_ms=float(cfg.cruise_speed_ms),
            p_hover_w=float(cfg.p_hover_w),
            k_drag=float(cfg.k_drag),
            battery_capacity_j=float(cfg.battery_capacity_j),
            rth_margin=float(cfg.rth_margin),
        )


class UAV:
    """A single simulated UAV. The engine sets ``target``; the UAV handles motion + energy + RTH."""

    def __init__(self, uav_id, params: UAVParams, pos_xy, energy_j: float | None = None):
        self.id = uav_id
        self.p = params
        self.pos: tuple[float, float] = (float(pos_xy[0]), float(pos_xy[1]))
        self.energy = params.battery_capacity_j if energy_j is None else float(energy_j)
        self.target: tuple[float, float] | None = None
        self.status = Status.IDLE

    # --- energy model -----------------------------------------------------------------
    def power_cruise(self) -> float:
        return self.p.p_hover_w + self.p.k_drag * self.p.cruise_speed_ms**2

    def energy_to_reach(self, distance_m: float) -> float:
        """Energy (J) to cruise ``distance_m`` at cruise speed."""
        return self.power_cruise() * distance_m / self.p.cruise_speed_ms

    def _dist(self, xy) -> float:
        return math.hypot(xy[0] - self.pos[0], xy[1] - self.pos[1])

    # --- one timestep -----------------------------------------------------------------
    def step(self, dt: float, base_xy) -> list[tuple[str, object]]:
        """Advance ``dt`` seconds. Returns ``(event, payload)`` tuples for the engine to log."""
        if self.status in (Status.LANDED, Status.DEAD):
            return []
        events: list[tuple[str, object]] = []

        # 1. Return-to-home decision overrides the survey target.
        if self.status != Status.RETURNING:
            if self.energy < self.p.rth_margin * self.energy_to_reach(self._dist(base_xy)):
                self.status = Status.RETURNING
                self.target = (float(base_xy[0]), float(base_xy[1]))
                events.append(("rth_triggered", None))

        # 2. No target -> hover in place (still burns hover power).
        if self.target is None:
            self.energy -= self.p.p_hover_w * dt
            if self.energy <= 0:
                self.status = Status.DEAD
                events.append(("dead", None))
            return events

        # 3. Move toward the target at cruise speed.
        v = self.p.cruise_speed_ms
        d = self._dist(self.target)
        step_d = min(v * dt, d)
        t_move = step_d / v if v > 0 else 0.0
        if d > 1e-12:
            self.pos = (
                self.pos[0] + (self.target[0] - self.pos[0]) * step_d / d,
                self.pos[1] + (self.target[1] - self.pos[1]) * step_d / d,
            )
        # cruise for the moving fraction, hover for any remainder of the step
        self.energy -= self.power_cruise() * t_move + self.p.p_hover_w * max(0.0, dt - t_move)
        if self.energy <= 0:
            self.status = Status.DEAD
            events.append(("dead", None))
            return events

        # 4. Arrival.
        if step_d >= d - 1e-9:
            if self.status == Status.RETURNING:
                self.status = Status.LANDED
                events.append(("landed", None))
            else:
                self.status = Status.IDLE
                events.append(("arrived", None))
            self.target = None
        return events
