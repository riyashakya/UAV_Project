"""Probability-guided vs uniform search ordering — addresses the SARCPPF gap.

Probability-map planners (e.g. Wu et al., 2024) search the most-likely cells first; a plain sweep
does not. This experiment gives one searcher a grid with **clustered** survivors and an
**imperfect** prior probability map (informative + noise, so the comparison is not circular), and
compares survivors-found-over-time for a **guided** search (nearest-high-prior cell first, via the
Coordinator's `greedy_priority`) against a travel-efficient **uniform** boustrophedon sweep. Both
use the real engine, so travel and energy are accounted for.

    make search-order

CPU-only; reuses engine + oracle + Coordinator — no core change beyond the guarded greedy rule.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.coordination.allocation import Coordinator
from src.sim.engine import run
from src.sim.oracle import Oracle
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clustered_survivors(cfg, rng) -> tuple[np.ndarray, pd.DataFrame]:
    """A Gaussian cluster of survivors → integer count per cell + a synthetic detection table."""
    rows, cols = int(cfg.grid.rows), int(cfg.grid.cols)
    cr, cc = float(cfg.cluster_center[0]), float(cfg.cluster_center[1])
    sig = float(cfg.cluster_sigma)
    dens = np.zeros(rows * cols)
    for r in range(rows):
        for c in range(cols):
            dens[r * cols + c] = np.exp(-((r - cr) ** 2 + (c - cc) ** 2) / (2 * sig**2))
    counts = np.floor(dens / dens.sum() * int(cfg.survivors_total)).astype(int)
    recs = []
    for cid, n in enumerate(counts):
        for k in range(int(n)):
            recs.append(
                {
                    "scenario": "search",
                    "cell_id": cid,
                    "class": "person",
                    "confidence": float(rng.uniform(0.4, 0.9)),
                    "lat": 29.75,
                    "lon": -95.36,
                    "bbox_utm": [0.0, 0.0, 1.0, 1.0],
                    "source_image": f"{cid}_{k}.jpg",
                    "model": "A",
                    "synthetic_geo": True,
                }
            )
    return counts, pd.DataFrame(recs)


def _boustrophedon(rows: int, cols: int) -> list[int]:
    order = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        order += [r * cols + c for c in cs]
    return order


def _detection_curve(events) -> tuple[np.ndarray, np.ndarray, int]:
    """Cumulative survivors found vs time, from the event log."""
    ts, cum, running = [], [], 0
    for e in sorted((e for e in events if e.get("event") == "arrived"), key=lambda e: e["t"]):
        n = e.get("found", []).count("person")
        if n:
            running += n
            ts.append(float(e["t"]))
            cum.append(running)
    return np.array(ts), np.array(cum), running


def run_search_order(cfg, detections=None) -> dict:
    rows, cols = int(cfg.grid.rows), int(cfg.grid.cols)
    seed_rng = np.random.default_rng(0)
    counts, df = (
        (detections["counts"], detections["df"])
        if detections is not None
        else _clustered_survivors(cfg, seed_rng)
    )
    total = int(counts.sum())

    # imperfect prior: informative cluster blended with a uniform prior by `prior_noise`
    noise = float(cfg.prior_noise)
    true_p = counts / counts.sum() if counts.sum() else np.ones(rows * cols) / (rows * cols)
    uni = np.ones(rows * cols) / (rows * cols)
    prior = (1 - noise) * true_p + noise * uni
    prior = prior / prior.max() * 10.0  # scale so it moves the bid meaningfully

    world = World(rows, cols, float(cfg.cell_size_m), priority=prior)
    uav_cfg = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs/sim/uav.yaml"), {"battery_capacity_j": 5e7}
    )
    params = UAVParams.from_cfg(uav_cfg)
    plan = {0: _boustrophedon(rows, cols)}
    oracle = Oracle(df, "search", false_negative_rate={"person": float(cfg.person_fn)})
    duration_s = float(cfg.duration_min) * 60.0

    grid_t = np.linspace(0, duration_s, 200)
    out = {}
    for policy, guided in (("uniform", False), ("guided", True)):
        curves, t80 = [], []
        for seed in range(int(cfg.n_seeds)):
            coord = Coordinator(
                "single_uav", world, 1, plan={0: list(plan[0])}, greedy_priority=guided
            )
            res = run(
                world,
                [UAV(0, params, world.base_xy)],
                coordinator=coord,
                seed=seed,
                duration_s=duration_s,
                dt=5.0,
                oracle=oracle,
            )
            ts, cum, found = _detection_curve(res["events"])
            frac = (
                np.zeros_like(grid_t) if found == 0 else np.interp(grid_t, ts, cum / found, left=0)
            )
            curves.append(frac)
            # time to locate 80% of the survivors this run detects
            target = 0.8 * found
            t80.append(float(ts[np.argmax(cum >= target)]) if found else duration_s)
        out[policy] = {
            "curve": np.mean(curves, axis=0),
            "t80_mean": float(np.mean(t80)),
            "t80_ci": float(1.96 * np.std(t80, ddof=1) / np.sqrt(len(t80))),
        }
    return {"grid_t": grid_t, "total": total, "policies": out}


def plot_search_order(res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = res["grid_t"] / 60.0
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for policy, colour in (("guided", "#1565C0"), ("uniform", "#B85042")):
        p = res["policies"][policy]
        label = (
            f"{'probability-guided' if policy == 'guided' else 'uniform sweep'} "
            f"(80% at {p['t80_mean'] / 60:.1f} min)"
        )
        ax.plot(t, p["curve"] * 100, lw=2.2, color=colour, label=label)
    ax.set_xlabel("mission time (min)")
    ax.set_ylabel("survivors located (% of those detected)")
    ax.set_ylim(0, 102)
    ax.set_title("Probability-guided vs uniform search: survivors located over time")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/search.yaml")
    res = run_search_order(cfg)
    g, u = res["policies"]["guided"], res["policies"]["uniform"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"search_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_search_order(res, out_dir / "search_order.png")

    print(
        f"[search-order] {res['total']} clustered survivors · {cfg.n_seeds} seeds · "
        f"prior_noise={cfg.prior_noise}"
    )
    print(
        f"[search-order] time to locate 80%: guided "
        f"{g['t80_mean'] / 60:.1f}±{g['t80_ci'] / 60:.1f} min "
        f"vs uniform {u['t80_mean'] / 60:.1f}±{u['t80_ci'] / 60:.1f} min"
    )
    speedup = u["t80_mean"] / g["t80_mean"] if g["t80_mean"] else float("nan")
    print(
        f"[search-order] probability-guided search locates 80% of survivors {speedup:.1f}× faster"
    )
    print(f"[search-order] wrote {out_dir}/search_order.png")


if __name__ == "__main__":
    main()
