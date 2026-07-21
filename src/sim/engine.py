"""Fixed-timestep simulation engine — Phase 4 (coordinator-driven from Phase 6).

Drives the world + UAVs through a deterministic, headless loop and emits an event log. Given the
same seed, two runs produce byte-identical logs. A ``Coordinator`` decides each UAV's next cell
and (for the auction strategy) reallocates work abandoned when a UAV dies or returns home. The
single ``np.random.Generator`` is threaded through the oracle + coordinator so all noise is
reproducible. No plotting, no I/O in the loop.

    make sim SCEN=flood_a SEED=0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .uav import UAV, Status, UAVParams
from .world import World

REPO_ROOT = Path(__file__).resolve().parents[2]


def round_robin_plan(n_cells: int, n_uavs: int) -> dict[int, list[int]]:
    """Assign cells to UAVs round-robin (a naive plan; real partitioning is Phase 5)."""
    plan: dict[int, list[int]] = {i: [] for i in range(n_uavs)}
    for cell_id in range(n_cells):
        plan[cell_id % n_uavs].append(cell_id)
    return plan


def run(
    world: World,
    uavs: list[UAV],
    plan: dict | None = None,
    *,
    coordinator=None,
    seed: int,
    duration_s: float,
    dt: float,
    oracle=None,
    fail_at: dict[int, float] | None = None,
) -> dict:
    """Run the simulation. ``fail_at`` maps uav_id -> time (s) to force a mid-mission failure."""
    if coordinator is None:  # backward-compatible: wrap a static plan
        from src.coordination.allocation import Coordinator

        coordinator = Coordinator("static_partition_no_realloc", world, len(uavs), plan=plan)
    rng = np.random.default_rng(seed)
    fail_at = fail_at or {}
    current_cell = {u.id: None for u in uavs}
    found_total: dict[str, int] = {}
    events: list[dict] = []
    n_steps = int(round(duration_s / dt))

    def rec(u, kind, t, **extra):
        events.append(
            {
                "t": t,
                "uav": u.id,
                "event": kind,
                "x": round(u.pos[0], 3),
                "y": round(u.pos[1], 3),
                "energy": round(u.energy, 1),
                **extra,
            }
        )

    for step_i in range(n_steps):
        t = round((step_i + 1) * dt, 3)
        for u in uavs:
            # Scripted failure (battery dies mid-mission) -> abandon and reallocate.
            if (
                u.id in fail_at
                and t >= float(fail_at[u.id])
                and u.status not in (Status.LANDED, Status.DEAD)
            ):
                u.status, u.target = Status.DEAD, None
                rec(u, "failed", t)
                reassigned = coordinator.on_abandon(u, uavs, current_cell[u.id])
                current_cell[u.id] = None
                if reassigned:
                    rec(u, "reassigned", t, cells=reassigned)
                continue

            # Assign the next target when idle and able to fly.
            if u.target is None and u.status not in (Status.RETURNING, Status.LANDED, Status.DEAD):
                cell = coordinator.next_cell(u, rng)
                if cell is not None:
                    u.target = world.cell_center(cell)
                    current_cell[u.id] = cell
                    u.status = Status.CRUISING
                elif coordinator.reallocates() and coordinator.has_pending():
                    u.status = Status.IDLE  # loiter to stay available for reassignment
                else:
                    u.status, u.target = Status.RETURNING, world.base_xy  # nothing to do -> home

            for kind, _ in u.step(dt, world.base_xy):
                if kind == "rth_triggered":
                    reassigned = coordinator.on_abandon(u, uavs, current_cell[u.id])
                    current_cell[u.id] = None
                    rec(u, kind, t, **({"reassigned": reassigned} if reassigned else {}))
                elif kind == "arrived":
                    cid = current_cell[u.id]
                    current_cell[u.id] = None
                    dets = oracle.get_detections(cid, rng) if oracle is not None else []
                    coordinator.on_survey(u, cid, dets)
                    found = sorted(d.cls for d in dets)
                    for c in found:
                        found_total[c] = found_total.get(c, 0) + 1
                    rec(u, kind, t, cell=cid, **({"found": found} if oracle is not None else {}))
                else:
                    rec(u, kind, t)

        if all(u.status in (Status.LANDED, Status.DEAD) for u in uavs):
            break

    surveyed = sorted(coordinator.visited)
    return {
        "events": events,
        "surveyed": surveyed,
        "coverage": round(len(surveyed) / world.n_cells, 4),
        "uav_end": {u.id: {"status": u.status.value, "energy": round(u.energy, 1)} for u in uavs},
        "found_total": dict(sorted(found_total.items())),
        "lost_cells": sorted(coordinator.pool),
    }


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one simulation.")
    parser.add_argument("--scenario", default=str(REPO_ROOT / "configs/scenario/flood_a.yaml"))
    parser.add_argument("--world", default=str(REPO_ROOT / "configs/sim/world.yaml"))
    parser.add_argument("--uav", default=str(REPO_ROOT / "configs/sim/uav.yaml"))
    parser.add_argument("--oracle", default=str(REPO_ROOT / "configs/sim/oracle.yaml"))
    parser.add_argument("--coord", default=str(REPO_ROOT / "configs/coordination/default.yaml"))
    parser.add_argument("--strategy", default=None, help="override allocation strategy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scenario = OmegaConf.load(args.scenario)
    world_cfg = OmegaConf.load(args.world)
    uav_cfg = OmegaConf.load(args.uav)
    coord_cfg = OmegaConf.load(args.coord)

    world = World.from_configs(scenario, world_cfg)
    params = UAVParams.from_cfg(uav_cfg)
    n_uavs = int(world_cfg.get("n_uavs", 4))
    uavs = [UAV(i, params, world.base_xy) for i in range(n_uavs)]

    from src.coordination.allocation import Coordinator

    alloc = coord_cfg.allocation
    strategy = args.strategy or alloc.strategy
    bw = alloc.bid_weights
    coordinator = Coordinator(
        strategy,
        world,
        n_uavs,
        bid_weights=(float(bw.travel), float(bw.energy), float(bw.priority)),
        priority_boost=float(alloc.priority_boost),
    )

    oracle = None
    cache = REPO_ROOT / "data" / "cache" / "detections.parquet"
    if cache.exists():
        from .oracle import Oracle

        ocfg = OmegaConf.load(args.oracle)
        oracle = Oracle(
            cache,
            scenario.name,
            false_negative_rate=dict(ocfg.false_negative_rate),
            latency_s=tuple(ocfg.latency_s),
        )

    duration_s = float(world_cfg.get("duration_min", 60)) * 60.0
    dt = float(world_cfg.get("timestep_s", 5.0))
    result = run(
        world,
        uavs,
        coordinator=coordinator,
        seed=args.seed,
        duration_s=duration_s,
        dt=dt,
        oracle=oracle,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"{scenario.name}_{strategy}_seed{args.seed}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(json.dumps(result, indent=2))
    OmegaConf.save(
        OmegaConf.merge(scenario, world_cfg, uav_cfg, coord_cfg), out_dir / "resolved_config.yaml"
    )

    print(f"[sim] scenario={scenario.name} strategy={strategy} seed={args.seed} uavs={n_uavs}")
    print(
        f"[sim] coverage={result['coverage'] * 100:.0f}%  "
        f"surveyed={len(result['surveyed'])}/{world.n_cells}  found={result['found_total']}  "
        f"lost={len(result['lost_cells'])}"
    )
    print(f"[sim] uav end: {result['uav_end']}")
    print(f"[sim] wrote {out_dir}/events.json ({len(result['events'])} events)")


if __name__ == "__main__":
    main()
