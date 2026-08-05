"""Ablation benchmark (Option A, corrected) — decompose the adaptive advantage.

The earlier single "adaptive vs static" comparison conflated two things and used an unfair capped
metric. This version fixes both: it runs THREE systems under a UAV failure with **sparse** survivor
hotspots (not one per cell), and reports the honest **detection curves** (survivors located vs time)
plus a *reachable* threshold, so the reallocation effect and the guidance effect are separated:

* **static** — fixed partition, uniform sweep, no reallocation (the static-SOTA baseline);
* **auction** — reallocation on failure, uniform search (isolates reallocation);
* **auction + guided** — reallocation + probability-guided search (isolates the added guidance).

    make benchmark

CPU-only; reuses the engine + Coordinator flags — no new mechanism.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.coordination.allocation import Coordinator
from src.eval.runner import _failures
from src.eval.search_order import _detection_curve
from src.sim.engine import run
from src.sim.oracle import Oracle
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]

# name -> (strategy, greedy_priority)
SYSTEMS = {
    "static (no realloc)": ("static_partition_no_realloc", False),
    "auction (realloc)": ("auction", False),
    "auction + guided": ("auction", True),
}


def _hotspot_survivors(cfg) -> tuple[np.ndarray, pd.DataFrame]:
    """Sparse survivors: a few Gaussian hotspots in different sectors, most cells empty."""
    rows, cols = int(cfg.grid.rows), int(cfg.grid.cols)
    sig = float(cfg.cluster_sigma)
    dens = np.zeros(rows * cols)
    for hr, hc in cfg.hotspots:
        for r in range(rows):
            for c in range(cols):
                dens[r * cols + c] += np.exp(-((r - hr) ** 2 + (c - hc) ** 2) / (2 * sig**2))
    counts = np.floor(dens / dens.sum() * int(cfg.survivors_total)).astype(int)
    recs = []
    for cid, n in enumerate(counts):
        for k in range(int(n)):
            recs.append(
                {
                    "scenario": "bench",
                    "cell_id": cid,
                    "class": "person",
                    "confidence": 0.6,
                    "lat": 29.75,
                    "lon": -95.36,
                    "bbox_utm": [0.0, 0.0, 1.0, 1.0],
                    "source_image": f"{cid}_{k}.jpg",
                    "model": "A",
                    "synthetic_geo": True,
                }
            )
    return counts, pd.DataFrame(recs)


def run_benchmark(cfg, detections=None, prior=None) -> dict:
    rows, cols = int(cfg.grid.rows), int(cfg.grid.cols)
    counts, df = (
        (detections["counts"], detections["df"])
        if detections is not None
        else _hotspot_survivors(cfg)
    )
    total = int(counts.sum())
    ncell = rows * cols
    if prior is None:  # default: an imperfect prior derived from the (noised) survivor density
        noise = float(cfg.prior_noise)
        true_p = counts / counts.sum() if counts.sum() else np.ones(ncell) / ncell
        prior = (1 - noise) * true_p + noise * (np.ones(ncell) / ncell)
    prior = np.asarray(prior, dtype=float)
    prior = prior / prior.max() * 10.0 if prior.max() > 0 else np.ones(ncell)

    world = World(rows, cols, float(cfg.cell_size_m), priority=prior)
    uav_cfg = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs/sim/uav.yaml"), {"battery_capacity_j": 5e7}
    )
    params = UAVParams.from_cfg(uav_cfg)
    ccfg = OmegaConf.load(REPO_ROOT / "configs/coordination/default.yaml")
    bw = ccfg.allocation.bid_weights
    bid_weights = (float(bw.travel), float(bw.energy), float(bw.priority))
    boost = float(ccfg.allocation.priority_boost)
    oracle = Oracle(df, "bench", false_negative_rate={"person": float(cfg.person_fn)})
    duration_s = float(cfg.duration_min) * 60.0
    n_uavs = int(cfg.n_uavs)
    window = tuple(cfg.fail_window_s)
    grid_t = np.linspace(0, duration_s, 200)

    out = {}
    for name, (strategy, greedy) in SYSTEMS.items():
        cov, det, t50, curves = [], [], [], []
        for seed in range(int(cfg.n_seeds)):
            coord = Coordinator(
                strategy,
                world,
                n_uavs,
                bid_weights=bid_weights,
                priority_boost=boost,
                greedy_priority=greedy,
            )
            fail_at = _failures(
                str(cfg.condition), n_uavs, np.random.default_rng(seed + 100_000), window
            )
            res = run(
                world,
                [UAV(i, params, world.base_xy) for i in range(n_uavs)],
                coordinator=coord,
                seed=seed,
                duration_s=duration_s,
                dt=5.0,
                oracle=oracle,
                fail_at=fail_at,
            )
            ts, cum, found = _detection_curve(res["events"])
            cov.append(res["coverage"])
            det.append(found / total if total else 0.0)
            frac = np.interp(grid_t, ts, cum / total, left=0) if len(ts) else np.zeros_like(grid_t)
            curves.append(frac)
            reached = ts[cum >= 0.5 * total] if (total and len(cum)) else np.array([])
            t50.append(float(reached[0]) if len(reached) else duration_s)

        def mean_ci(v):
            v = np.asarray(v, dtype=float)
            n = len(v)
            return (float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)

        out[name] = {
            "coverage": mean_ci(cov),
            "detect": mean_ci(det),
            "t50": mean_ci(t50),
            "curve": np.mean(curves, axis=0),
        }
    return {"total": total, "grid_t": grid_t, "systems": out}


def plot_benchmark(res: dict, out_path: Path, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = res["grid_t"] / 60.0
    colours = {
        "static (no realloc)": "#B85042",
        "auction (realloc)": "#E0A500",
        "auction + guided": "#1565C0",
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, s in res["systems"].items():
        ax.plot(
            t,
            s["curve"] * 100,
            lw=2.2,
            color=colours.get(name, "#888"),
            label=f"{name} (50% at {s['t50'][0] / 60:.1f} min, final {s['detect'][0] * 100:.0f}%)",
        )
    ax.axhline(50, color="#999", ls=":", lw=1)
    ax.set_xlabel("mission time (min)")
    ax.set_ylabel("survivors located (% of all survivors)")
    ax.set_ylim(0, 100)
    ax.set_title(
        title or "Ablation: reallocation vs added guidance (sparse survivors, 1 UAV failure)"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/benchmark.yaml")
    res = run_benchmark(cfg)
    s = res["systems"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"benchmark_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_benchmark(res, out_dir / "benchmark.png")

    st, au, gu = s["static (no realloc)"], s["auction (realloc)"], s["auction + guided"]
    print(f"[benchmark] {res['total']} sparse survivors · {cfg.n_uavs} UAVs · {cfg.condition}")
    for name, v in s.items():
        print(
            f"[benchmark] {name:22s} coverage {v['coverage'][0] * 100:4.0f}% · "
            f"detected {v['detect'][0] * 100:4.0f}% · 50% at {v['t50'][0] / 60:4.1f} min"
        )
    print(
        f"[benchmark] reallocation gain (static→auction): "
        f"+{(au['detect'][0] - st['detect'][0]) * 100:.0f} detected pts"
    )
    print(
        f"[benchmark] guidance gain (auction→+guided):    "
        f"+{(gu['detect'][0] - au['detect'][0]) * 100:.0f} detected pts, "
        f"{au['t50'][0] / gu['t50'][0]:.1f}× faster to 50%"
    )
    print(f"[benchmark] wrote {out_dir}/benchmark.png")


if __name__ == "__main__":
    main()
