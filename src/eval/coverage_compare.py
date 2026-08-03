"""Coverage-pattern comparison — lawnmower (boustrophedon) sweep vs spiral (Phase 5 extension).

Answers "is only one flight pattern used?": both patterns cover the same sector to the same
footprint; the question is efficiency. Reports **path length at equal coverage** (shorter = less
flight time / energy) and draws both paths. Standalone geometry — no engine change.

    make coverage-compare

CPU-only; matplotlib imported lazily for the figure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from src.coordination.coverage import (
    coverage_path,
    path_coverage_fraction,
    path_length,
    spiral_path,
)
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]


def compare(world: World, footprint_width_m: float, sidelap: float) -> dict:
    cells = [c.id for c in world.cells]
    law = coverage_path(cells, world, footprint_width_m, sidelap)
    spi = spiral_path(cells, world, footprint_width_m, sidelap)
    return {
        "lawnmower": {
            "path": law,
            "length_m": round(path_length(law), 1),
            "coverage": round(path_coverage_fraction(law, cells, world, footprint_width_m), 3),
        },
        "spiral": {
            "path": spi,
            "length_m": round(path_length(spi), 1),
            "coverage": round(path_coverage_fraction(spi, cells, world, footprint_width_m), 3),
        },
    }


def plot_compare(world: World, res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = world.cell_size_m
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, (name, colour) in zip(axes, (("lawnmower", "#1565C0"), ("spiral", "#B85042"))):
        r = res[name]
        xs = [p[0] for p in r["path"]]
        ys = [p[1] for p in r["path"]]
        for k in range(world.cols + 1):
            ax.axvline(k * size, color="#EAEFF4", lw=0.8, zorder=0)
        for k in range(world.rows + 1):
            ax.axhline(k * size, color="#EAEFF4", lw=0.8, zorder=0)
        ax.plot(xs, ys, color=colour, lw=1.8, zorder=2)
        ax.scatter([xs[0]], [ys[0]], color=colour, s=45, zorder=3)
        ax.set_title(f"{name}: {r['length_m']:.0f} m · coverage {r['coverage'] * 100:.0f}%")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Coverage patterns: path length at equal coverage", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/coverage.yaml")
    world = World(int(cfg.grid.rows), int(cfg.grid.cols), float(cfg.cell_size_m))
    res = compare(world, float(cfg.footprint_width_m), float(cfg.sidelap))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"coverage_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_compare(world, res, out_dir / "coverage_compare.png")

    law, spi = res["lawnmower"], res["spiral"]
    longer, shorter = (spi, law) if spi["length_m"] >= law["length_m"] else (law, spi)
    pct = 100 * (longer["length_m"] - shorter["length_m"]) / longer["length_m"]
    print(f"[coverage] lawnmower: {law['length_m']:.0f} m, coverage {law['coverage'] * 100:.0f}%")
    print(f"[coverage] spiral   : {spi['length_m']:.0f} m, coverage {spi['coverage'] * 100:.0f}%")
    winner = "lawnmower" if law["length_m"] <= spi["length_m"] else "spiral"
    print(f"[coverage] at equal coverage, {winner} is {pct:.0f}% shorter → less flight time/energy")
    print(f"[coverage] wrote {out_dir}/coverage_compare.png")


if __name__ == "__main__":
    main()
