"""Draw the survivor-drift projection — Phase 7 figure (`make drift`).

Renders one static map (CLAUDE.md: matplotlib only, no UI): the Monte-Carlo particle cloud, the
50 % / 90 % containment polygons, the detection point → drifted centroid, and the grid cells the
auction would re-task UAVs toward (`cells_in_region` — the RQ4 link). All the maths lives in the
already-tested ``drift.advect``; this module is I/O only. Writes to ``outputs/drift/<ts>/``.

    make drift

CPU-only; matplotlib imported lazily (Agg backend).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.drift.advect import cells_in_region, drift_search_region
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]


def plot_drift(world: World, region: dict, start_xy, retask_cells, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.patches import Rectangle

    size = world.cell_size_m
    fig, ax = plt.subplots(figsize=(6.8, 6.6))

    # grid cell boundaries
    for k in range(world.cols + 1):
        ax.axvline(k * size, color="#E3E9F0", lw=0.8, zorder=0)
    for k in range(world.rows + 1):
        ax.axhline(k * size, color="#E3E9F0", lw=0.8, zorder=0)

    # cells the auction would re-task toward (drift-region cells)
    for cid in retask_cells:
        cell = world.cells[cid]
        ax.add_patch(
            Rectangle(
                (cell.col * size, cell.row * size),
                size,
                size,
                facecolor="#1C7293",
                alpha=0.14,
                edgecolor="#1C7293",
                lw=1.0,
                zorder=1,
                label="_nolegend_",
            )
        )

    # particle cloud
    pts = region["particles"]
    ax.scatter(
        pts[:, 0], pts[:, 1], s=3, color="#4C6A82", alpha=0.25, zorder=2, label="drift cloud"
    )

    # containment polygons (draw larger first so the tighter one sits on top)
    shades = {0.9: "#7FB3D5", 0.5: "#1C7293"}
    for lvl in sorted(region["containment"], reverse=True):
        poly = region["containment"][lvl]
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        xy = np.asarray(poly.exterior.coords)
        ax.add_patch(
            MplPolygon(
                xy,
                closed=True,
                facecolor=shades.get(lvl, "#7FB3D5"),
                alpha=0.22,
                edgecolor=shades.get(lvl, "#1C7293"),
                lw=1.8,
                zorder=3,
                label=f"{int(lvl * 100)}% containment",
            )
        )

    # detection point and drifted centroid
    cen = region["centroid"]
    ax.scatter(
        *start_xy,
        marker="*",
        s=320,
        color="#B85042",
        edgecolor="white",
        lw=1.0,
        zorder=5,
        label="detection (t=0)",
    )
    ax.scatter(
        *cen,
        marker="X",
        s=150,
        color="#0C3B6E",
        edgecolor="white",
        lw=1.0,
        zorder=5,
        label="drifted centroid",
    )

    # flow arrow at the detection point
    vx, vy = world.flow(float(start_xy[0]), float(start_xy[1]))
    if vx or vy:
        ax.annotate(
            "",
            xy=(start_xy[0] + vx * 220, start_xy[1] + vy * 220),
            xytext=start_xy,
            arrowprops=dict(arrowstyle="-|>", color="#B85042", lw=1.6),
            zorder=4,
        )

    ax.set_xlim(0, world.cols * size)
    ax.set_ylim(0, world.rows * size)
    ax.invert_yaxis()  # row 0 (north) at the top -> reads like a map
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("south (m)")
    ax.set_title("Survivor-drift projection: search where they are now")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    scenario = OmegaConf.load(REPO_ROOT / "configs/scenario/flood_a.yaml")
    world_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/world.yaml")
    cfg = OmegaConf.load(REPO_ROOT / "configs/drift/default.yaml")
    world = World.from_configs(scenario, world_cfg)

    survivor_cell = int(cfg.demo.survivor_cell)
    start_xy = world.cell_center(survivor_cell)
    rng = np.random.default_rng(int(cfg.demo.seed))
    region = drift_search_region(
        start_xy,
        world.flow,
        rng=rng,
        n_particles=int(cfg.n_particles),
        horizon_s=float(cfg.horizon_s),
        dt=float(cfg.timestep_s),
        leeway_factor=float(cfg.leeway_factor),
        k_h=float(cfg.k_h),
        containment_levels=tuple(cfg.containment_levels),
    )
    retask_cells = cells_in_region(region["containment"][max(cfg.containment_levels)], world)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "drift" / ts
    plot_drift(world, region, start_xy, retask_cells, out_dir / "drift.png")

    drift_m = float(np.linalg.norm(np.asarray(region["centroid"]) - np.asarray(start_xy)))
    areas = region["areas_m2"]
    at = tuple(round(v) for v in start_xy)
    print(f"[drift] survivor detected in cell {survivor_cell} at {at} m")
    print(
        f"[drift] drifted {drift_m:.0f} m over {float(cfg.horizon_s) / 60:.0f} min "
        f"-> centroid {tuple(round(float(v)) for v in region['centroid'])} m"
    )
    print(
        "[drift] containment areas: "
        + ", ".join(f"{int(k * 100)}%={v / 1e4:.1f} ha" for k, v in sorted(areas.items()))
    )
    print(f"[drift] re-task cells (90% region): {sorted(retask_cells)}")
    print(f"[drift] wrote {out_dir}/drift.png")


if __name__ == "__main__":
    main()
