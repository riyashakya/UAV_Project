"""Fixed-timestep simulation engine — Phase 4.

Drives the world + UAVs through a deterministic, headless loop and emits an event log. Given
the same seed, two runs produce byte-identical logs. The single ``np.random.Generator`` is
threaded through the oracle so survey noise is reproducible. No plotting, no I/O in the loop.

    make sim SCEN=flood_a SEED=0
"""

from __future__ import annotations

import argparse
import json
from collections import deque
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
    plan: dict,
    *,
    seed: int,
    duration_s: float,
    dt: float,
    oracle=None,
) -> dict:
    """Run the simulation. Returns ``{events, surveyed, coverage, uav_end, found_total}``."""
    rng = np.random.default_rng(seed)
    queues = {u.id: deque(plan.get(u.id, [])) for u in uavs}
    current_cell = {u.id: None for u in uavs}
    surveyed: set[int] = set()
    found_total: dict[str, int] = {}
    events: list[dict] = []
    n_steps = int(round(duration_s / dt))

    for step_i in range(n_steps):
        t = round((step_i + 1) * dt, 3)
        for u in uavs:
            # Assign the next planned cell when idle and able to fly.
            if (
                u.target is None
                and u.status not in (Status.RETURNING, Status.LANDED, Status.DEAD)
                and queues[u.id]
            ):
                cid = queues[u.id].popleft()
                u.target = world.cell_center(cid)
                current_cell[u.id] = cid
                u.status = Status.CRUISING

            for kind, _ in u.step(dt, world.base_xy):
                ev = {
                    "t": t,
                    "uav": u.id,
                    "event": kind,
                    "x": round(u.pos[0], 3),
                    "y": round(u.pos[1], 3),
                    "energy": round(u.energy, 1),
                }
                if kind == "arrived":
                    cid = current_cell[u.id]
                    ev["cell"] = cid
                    surveyed.add(cid)
                    if oracle is not None:
                        found = sorted(d.cls for d in oracle.get_detections(cid, rng))
                        ev["found"] = found
                        for c in found:
                            found_total[c] = found_total.get(c, 0) + 1
                events.append(ev)

        if all(u.status in (Status.LANDED, Status.DEAD) for u in uavs) and not any(
            q for q in queues.values()
        ):
            break

    return {
        "events": events,
        "surveyed": sorted(surveyed),
        "coverage": round(len(surveyed) / world.n_cells, 4),
        "uav_end": {u.id: {"status": u.status.value, "energy": round(u.energy, 1)} for u in uavs},
        "found_total": dict(sorted(found_total.items())),
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scenario = OmegaConf.load(args.scenario)
    world_cfg = OmegaConf.load(args.world)
    uav_cfg = OmegaConf.load(args.uav)

    world = World.from_configs(scenario, world_cfg)
    params = UAVParams.from_cfg(uav_cfg)
    n_uavs = int(world_cfg.get("n_uavs", 4))
    uavs = [UAV(i, params, world.base_xy) for i in range(n_uavs)]
    plan = round_robin_plan(world.n_cells, n_uavs)

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
    result = run(world, uavs, plan, seed=args.seed, duration_s=duration_s, dt=dt, oracle=oracle)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"{scenario.name}_seed{args.seed}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "events.json").write_text(json.dumps(result, indent=2))
    OmegaConf.save(OmegaConf.merge(scenario, world_cfg, uav_cfg), out_dir / "resolved_config.yaml")

    print(
        f"[sim] scenario={scenario.name} seed={args.seed} uavs={n_uavs} dt={dt}s "
        f"duration={duration_s / 60:.0f}min"
    )
    print(
        f"[sim] coverage={result['coverage'] * 100:.0f}%  "
        f"surveyed={len(result['surveyed'])}/{world.n_cells} cells  "
        f"detections found={result['found_total']}"
    )
    print(f"[sim] uav end states: {result['uav_end']}")
    print(f"[sim] wrote {out_dir}/events.json ({len(result['events'])} events)")


if __name__ == "__main__":
    main()
