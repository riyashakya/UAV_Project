"""Area partitioning — Phase 5.

Divide the AOI into N sectors behind one interface:

* ``grid`` — naive contiguous column-strips (the baseline).
* ``weighted_voronoi`` — sectors seeded spatially (Lloyd relaxation) then rebalanced by greedy
  boundary swaps so each sector's **workload** (sum of per-cell priority) is equal to within a
  tolerance — i.e. equal-workload, not equal-area (CLAUDE.md / Gerkey & Matarić framing).

Returns ``{sector_id: [cell_id, ...]}``, which plugs straight into the engine as a per-UAV plan.
CPU-only (numpy); imported by coordination, never by the perception detector.
"""

from __future__ import annotations

import numpy as np

from src.sim.world import World


def grid_partition(world: World, n_sectors: int) -> dict[int, list[int]]:
    """Contiguous vertical column-strips (naive baseline; equal-*area*, not workload)."""
    plan: dict[int, list[int]] = {s: [] for s in range(n_sectors)}
    for c in range(world.cols):
        s = min(c * n_sectors // world.cols, n_sectors - 1)
        for r in range(world.rows):
            plan[s].append(r * world.cols + c)
    return {s: sorted(cells) for s, cells in plan.items()}


def _grid_neighbors(world: World) -> dict[int, list[int]]:
    nbr: dict[int, list[int]] = {}
    for cell in world.cells:
        r, c, out = cell.row, cell.col, []
        if r > 0:
            out.append((r - 1) * world.cols + c)
        if r < world.rows - 1:
            out.append((r + 1) * world.cols + c)
        if c > 0:
            out.append(r * world.cols + (c - 1))
        if c < world.cols - 1:
            out.append(r * world.cols + (c + 1))
        nbr[cell.id] = out
    return nbr


def _lloyd_labels(centers: np.ndarray, n_sectors: int, iters: int = 15) -> np.ndarray:
    """Spatial Voronoi via Lloyd relaxation from evenly-spaced (deterministic) seeds."""
    seed_idx = np.linspace(0, len(centers) - 1, n_sectors).astype(int)
    seeds = centers[seed_idx].astype(float).copy()
    labels = np.zeros(len(centers), dtype=int)
    for _ in range(iters):
        d = ((centers[:, None, :] - seeds[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        for s in range(n_sectors):
            m = labels == s
            if m.any():
                seeds[s] = centers[m].mean(0)
    return labels


def _greedy_balance(
    labels: np.ndarray,
    priority: np.ndarray,
    neighbors: dict[int, list[int]],
    n_sectors: int,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """Move boundary cells from the heaviest sector to a lighter adjacent one until balanced."""
    labels = labels.copy()
    for _ in range(max_iter):
        load = np.array([priority[labels == s].sum() for s in range(n_sectors)])
        if load.max() - load.min() <= tol * load.mean():
            break
        heavy = int(load.argmax())
        moved = False
        # Prefer moving a heavy boundary cell to its lightest adjacent sector.
        best = None
        for cid in np.where(labels == heavy)[0]:
            for nb in neighbors[cid]:
                s = int(labels[nb])
                if s != heavy and load[s] < load[heavy]:
                    if best is None or load[s] < load[best[1]]:
                        best = (int(cid), s)
        if best is not None:
            labels[best[0]] = best[1]
            moved = True
        if not moved:
            break
    return labels


def weighted_voronoi_partition(
    world: World, n_sectors: int, *, tol: float = 0.05, max_iter: int = 2000
) -> dict[int, list[int]]:
    """Spatially-seeded sectors rebalanced to equal workload (sum of priority) within ``tol``."""
    centers = np.array([cell.center_xy for cell in world.cells], dtype=float)
    labels = _lloyd_labels(centers, n_sectors)
    labels = _greedy_balance(
        labels, np.asarray(world.priority, float), _grid_neighbors(world), n_sectors, tol, max_iter
    )
    return {s: sorted(int(i) for i in np.where(labels == s)[0]) for s in range(n_sectors)}


def partition(
    world: World,
    n_sectors: int,
    strategy: str = "weighted_voronoi",
    *,
    tol: float = 0.05,
    max_iter: int = 2000,
) -> dict[int, list[int]]:
    """Partition the AOI into ``n_sectors`` sectors using the named strategy."""
    if strategy == "grid":
        return grid_partition(world, n_sectors)
    if strategy == "weighted_voronoi":
        return weighted_voronoi_partition(world, n_sectors, tol=tol, max_iter=max_iter)
    raise ValueError(f"unknown partition strategy {strategy!r}")


def sector_workloads(plan: dict[int, list[int]], priority: np.ndarray) -> dict[int, float]:
    """Total priority per sector — used to check workload balance."""
    return {s: float(np.asarray([priority[c] for c in cells]).sum()) for s, cells in plan.items()}
