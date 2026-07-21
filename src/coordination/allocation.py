"""Dynamic task reallocation — Phase 6 (the core contribution).

Auction-based reallocation in the style of the **Contract Net Protocol** (Smith, 1980). In the
**Gerkey & Matarić (2004) MRTA taxonomy** this is single-task, single-robot, instantaneous
assignment (ST-SR-IA) applied repeatedly online, i.e. a time-extended assignment policy.

**Reallocation triggers**
1. A UAV hits its return-to-home threshold or dies and **abandons** its unfinished cells → those
   cells are auctioned to the still-flying UAVs (lowest bid wins).
2. A **high-priority detection** (a `person`) upweights the surrounding cells, so they attract
   stronger bids and are visited sooner.

**Bid (a cost; lowest wins):** ``a·travel_cost + b·energy_penalty − c·cell_priority`` — a nearby
UAV with plenty of energy bidding on a high-priority cell bids lowest. Weights come from config.

**Baselines behind the same interface** (the comparison *is* the result):
``single_uav`` · ``static_partition_no_realloc`` · ``random_walk``.

CPU-only; imports `sim.world`/`sim.uav` (numpy), never the perception detector (ADR-001).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from src.sim.uav import Status
from src.sim.world import World

from .partition import partition

STRATEGIES = ("auction", "static_partition_no_realloc", "single_uav", "random_walk")
REALLOCATING = ("auction",)  # strategies that reassign abandoned work


class Coordinator:
    """Assigns cells to UAVs and (for the auction strategy) reallocates abandoned work online."""

    def __init__(
        self,
        strategy: str,
        world: World,
        n_uavs: int,
        *,
        plan: dict[int, list[int]] | None = None,
        bid_weights=(1.0, 0.0005, 50.0),
        priority_boost: float = 3.0,
        partition_tol: float = 0.05,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected {STRATEGIES}")
        self.strategy = strategy
        self.world = world
        self.n_uavs = n_uavs
        self.a, self.b, self.c = bid_weights
        self.priority_boost = priority_boost
        self.priority = np.asarray(world.priority, dtype=float).copy()
        self.visited: set[int] = set()
        self.queues: dict[int, deque[int]] = {i: deque() for i in range(n_uavs)}
        self.pool: list[int] = []  # abandoned cells nobody could take (lost coverage)
        self._init_assignment(plan, partition_tol)

    def _init_assignment(self, plan, tol) -> None:
        if plan is not None:
            for s, cells in plan.items():
                self.queues[s] = deque(cells)
        elif self.strategy == "single_uav":
            self.queues[0] = deque(range(self.world.n_cells))  # one UAV does everything
        elif self.strategy == "random_walk":
            pass  # targets chosen dynamically
        else:  # auction, static_partition_no_realloc -> a balanced static partition to start
            for s, cells in partition(self.world, self.n_uavs, "weighted_voronoi", tol=tol).items():
                self.queues[s] = deque(cells)

    # --- queries used by the engine ---------------------------------------------------
    def has_pending(self) -> bool:
        return len(self.visited) < self.world.n_cells

    def reallocates(self) -> bool:
        return self.strategy in REALLOCATING

    def next_cell(self, uav, rng: np.random.Generator) -> int | None:
        """The next cell for ``uav`` to survey, or None if it currently has nothing to do."""
        if self.strategy == "random_walk":
            unvisited = [c for c in range(self.world.n_cells) if c not in self.visited]
            return int(rng.choice(unvisited)) if unvisited else None
        q = self.queues[uav.id]
        while q:
            cell = q.popleft()
            if cell not in self.visited:
                return cell
        return None

    # --- events -----------------------------------------------------------------------
    def on_survey(self, uav, cell: int, detections) -> None:
        self.visited.add(cell)
        if self.strategy == "auction" and any(
            getattr(d, "cls", None) == "person" for d in detections
        ):
            for nb in self._neighbors(cell):  # re-task toward survivors
                self.priority[nb] *= self.priority_boost

    def on_abandon(self, dead_uav, uavs, current_target: int | None) -> list[int]:
        """A UAV abandoned its work. Return the cells reassigned (auction) or lost (baselines)."""
        abandoned = [c for c in self.queues[dead_uav.id] if c not in self.visited]
        if current_target is not None and current_target not in self.visited:
            abandoned.append(current_target)
        self.queues[dead_uav.id].clear()
        abandoned = sorted(set(abandoned), key=lambda c: -self.priority[c])
        if not abandoned:
            return []
        if not self.reallocates():
            self.pool.extend(abandoned)  # baseline: never reassigned
            return []
        eligible = [
            u for u in uavs if u.id != dead_uav.id and u.status in (Status.IDLE, Status.CRUISING)
        ]
        reassigned = []
        for cell in abandoned:
            if not eligible:
                self.pool.append(cell)
                continue
            winner = min(eligible, key=lambda u: self._bid(u, cell))
            self.queues[winner.id].append(cell)
            reassigned.append(cell)
        return reassigned

    # --- helpers ----------------------------------------------------------------------
    def _bid(self, uav, cell: int) -> float:
        cx, cy = self.world.cell_center(cell)
        travel = math.hypot(cx - uav.pos[0], cy - uav.pos[1])
        energy_penalty = uav.p.battery_capacity_j - uav.energy
        return self.a * travel + self.b * energy_penalty - self.c * float(self.priority[cell])

    def _neighbors(self, cell: int) -> list[int]:
        w, (r, c) = self.world, divmod(cell, self.world.cols)
        out = []
        if r > 0:
            out.append((r - 1) * w.cols + c)
        if r < w.rows - 1:
            out.append((r + 1) * w.cols + c)
        if c > 0:
            out.append(r * w.cols + (c - 1))
        if c < w.cols - 1:
            out.append(r * w.cols + (c + 1))
        return out
