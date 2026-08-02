"""Mission animation — a GIF of the fleet flying the survey (`make animate`).

Reruns one scripted-failure mission with per-timestep recording and renders it: the grid, cells
filling in as they are surveyed, each UAV on its own coloured trail, survivor detections revealed
as their cell is reached, and a caption when a UAV fails and its work is reauctioned. This is the
one moving picture of the core contribution — static output only otherwise (CLAUDE.md: no UI).

    make animate

CPU-only; matplotlib imported lazily (Agg backend) and written with Pillow (a core dep).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from src.coordination.allocation import Coordinator
from src.sim.engine import run
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]

# One fixed colour per UAV (its trail + marker); status is shown by marker shape.
_UAV_COLORS = ["#1565C0", "#2E9E6B", "#E08A1E", "#8E44AD", "#0C7B93", "#C0392B"]
_DEAD, _LANDED = "dead", "landed"


def _survivor_reveal(events: list[dict]) -> dict[int, float]:
    """cell id -> time a survivor (person) was first detected there, from the event log."""
    reveal: dict[int, float] = {}
    for e in events:
        if e.get("event") == "arrived" and "person" in e.get("found", []):
            reveal.setdefault(int(e["cell"]), float(e["t"]))
    return reveal


def render(world: World, result: dict, cfg, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    traj = result["trajectory"]
    size = world.cell_size_m
    extent_x, extent_y = world.cols * size, world.rows * size
    reveal = _survivor_reveal(result["events"])
    fail_uav, fail_at = int(cfg.fail_uav), float(cfg.fail_at_s)
    n_uavs = len(traj[0]["uavs"])
    # trim the return-to-base tail: end a few frames after the survey is complete, so the GIF
    # spends its time on the action (sweeping + the failure) rather than UAVs flying home
    full = next(
        (i for i, s in enumerate(traj) if len(s["visited"]) == world.n_cells), len(traj) - 1
    )
    traj = traj[: min(len(traj), full + 4)]
    frames = list(range(0, len(traj), max(1, int(cfg.stride))))
    if frames[-1] != len(traj) - 1:
        frames.append(len(traj) - 1)

    fig, ax = plt.subplots(figsize=(6.6, 6.9))

    def cell_xy(cid: int) -> tuple[float, float]:
        r, c = divmod(cid, world.cols)
        return c * size, r * size

    def update(idx: int):
        ax.clear()
        snap = traj[idx]
        t = snap["t"]
        # grid
        for k in range(world.cols + 1):
            ax.axvline(k * size, color="#EAEFF4", lw=0.8, zorder=0)
        for k in range(world.rows + 1):
            ax.axhline(k * size, color="#EAEFF4", lw=0.8, zorder=0)
        # surveyed cells
        for cid in snap["visited"]:
            x, y = cell_xy(cid)
            ax.add_patch(
                Rectangle((x, y), size, size, facecolor="#D6ECDD", edgecolor="none", zorder=1)
            )
        # survivor detections revealed so far (kept subtle — the saturated cache marks most cells)
        for cid, rt in reveal.items():
            if rt <= t:
                cx, cy = cell_xy(cid)
                ax.scatter(
                    cx + size / 2,
                    cy + size / 2,
                    marker="*",
                    s=70,
                    color="#B85042",
                    alpha=0.55,
                    edgecolor="none",
                    zorder=2,
                )
        # base
        bx, by = world.base_xy
        ax.scatter(bx, by, marker="s", s=90, color="#111111", zorder=3)
        # UAV trails + markers
        for i in range(n_uavs):
            colour = _UAV_COLORS[i % len(_UAV_COLORS)]
            xs = [traj[j]["uavs"][i][0] for j in range(idx + 1)]
            ys = [traj[j]["uavs"][i][1] for j in range(idx + 1)]
            ax.plot(xs, ys, color=colour, lw=1.4, alpha=0.55, zorder=2)
            x, y, status = snap["uavs"][i]
            marker = "X" if status == _DEAD else ("s" if status == _LANDED else "o")
            ax.scatter(
                x, y, marker=marker, s=130, color=colour, edgecolor="white", lw=1.2, zorder=4
            )

        n_vis = len(snap["visited"])
        ax.set_title(f"t = {t / 60:4.1f} min     coverage {n_vis}/{world.n_cells}", fontsize=12)
        if t >= fail_at:  # a banner inside the top of the plot (won't collide with the title)
            ax.text(
                0.5,
                0.97,
                f"UAV-{fail_uav} failed → work reauctioned",
                transform=ax.transAxes,
                ha="center",
                va="top",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#C0392B", "edgecolor": "none"},
                zorder=6,
            )
        ax.set_xlim(0, extent_x)
        ax.set_ylim(0, extent_y)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        handles = [
            Line2D(
                [], [], marker="o", color="w", markerfacecolor="#555", markersize=9, label="UAV"
            ),
            Line2D(
                [],
                [],
                marker="X",
                color="w",
                markerfacecolor="#C0392B",
                markersize=9,
                label="failed",
            ),
            Line2D(
                [],
                [],
                marker="*",
                color="w",
                markerfacecolor="#B85042",
                markersize=12,
                label="survivor",
            ),
            Line2D(
                [], [], marker="s", color="w", markerfacecolor="#111", markersize=9, label="base"
            ),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.92, ncol=2)
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=int(cfg.fps)))
    plt.close(fig)


def main() -> None:
    scenario = OmegaConf.load(REPO_ROOT / "configs/scenario/flood_a.yaml")
    world_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/world.yaml")
    uav_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/uav.yaml")
    coord_cfg = OmegaConf.load(REPO_ROOT / "configs/coordination/default.yaml")
    cfg = OmegaConf.load(REPO_ROOT / "configs/viz/mission.yaml")

    world = World.from_configs(scenario, world_cfg)
    params = UAVParams.from_cfg(uav_cfg)
    n_uavs = int(world_cfg.get("n_uavs", 4))
    uavs = [UAV(i, params, world.base_xy) for i in range(n_uavs)]
    bw = coord_cfg.allocation.bid_weights
    coord = Coordinator(
        str(cfg.strategy),
        world,
        n_uavs,
        bid_weights=(float(bw.travel), float(bw.energy), float(bw.priority)),
        priority_boost=float(coord_cfg.allocation.priority_boost),
    )

    oracle = None
    cache = REPO_ROOT / "data" / "cache" / "detections.parquet"
    if cache.exists():
        from src.sim.oracle import Oracle

        ocfg = OmegaConf.load(REPO_ROOT / "configs/sim/oracle.yaml")
        oracle = Oracle(
            cache,
            scenario.name,
            false_negative_rate=dict(ocfg.false_negative_rate),
            latency_s=tuple(ocfg.latency_s),
        )

    result = run(
        world,
        uavs,
        coordinator=coord,
        seed=int(cfg.seed),
        duration_s=float(world_cfg.get("duration_min", 60)) * 60.0,
        dt=float(world_cfg.get("timestep_s", 5.0)),
        oracle=oracle,
        fail_at={int(cfg.fail_uav): float(cfg.fail_at_s)},
        record_trajectory=True,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"mission_{scenario.name}_{ts}"
    render(world, result, cfg, out_dir / "mission.gif")

    print(
        f"[animate] {scenario.name}: {n_uavs} UAVs, UAV-{int(cfg.fail_uav)} fails at "
        f"{float(cfg.fail_at_s):.0f}s -> coverage {result['coverage'] * 100:.0f}%"
    )
    print(f"[animate] {len(result['trajectory'])} timesteps -> wrote {out_dir}/mission.gif")


if __name__ == "__main__":
    main()
