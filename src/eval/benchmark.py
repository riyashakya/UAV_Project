"""Option A — controlled head-to-head benchmark: adaptive pipeline vs a static-SOTA-class baseline.

This makes no novelty claim. It reproduces the *static* class of integrated UAV-SAR systems (fixed
partition + uniform sweep, no reallocation, no probability guidance — cf. AI-Enhanced UAV Clusters,
2026) as a baseline in our own simulator, and compares it against this project's **adaptive**
pipeline (auction reallocation + probability-guided search) on ONE controlled scenario under stress
(clustered survivors + an imperfect prior + a UAV failure + detector false-negatives). Reports, mean
± 95 % CI over seeds:

* **coverage** (area surveyed),
* **survivors detected** (fraction of ground truth),
* **time to locate 80 %** of the survivors.

    make benchmark

CPU-only; reuses the engine + Coordinator flags + the clustered-survivor generator (no new method).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.coordination.allocation import Coordinator
from src.eval.runner import _failures
from src.eval.search_order import _clustered_survivors, _detection_curve
from src.sim.engine import run
from src.sim.oracle import Oracle
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]

# The two systems compared: name -> (strategy, greedy_priority, uses the survivor-likelihood prior)
SYSTEMS = {
    "baseline (static + uniform sweep)": ("static_partition_no_realloc", False, False),
    "adaptive (auction + guided search)": ("auction", True, True),
}


def run_benchmark(cfg, detections=None) -> dict:
    rows, cols = int(cfg.grid.rows), int(cfg.grid.cols)
    counts, df = (
        (detections["counts"], detections["df"])
        if detections is not None
        else _clustered_survivors(cfg, np.random.default_rng(0))
    )
    total = int(counts.sum())
    ncell = rows * cols
    noise = float(cfg.prior_noise)
    true_p = counts / counts.sum() if counts.sum() else np.ones(ncell) / ncell
    prior = (1 - noise) * true_p + noise * (np.ones(ncell) / ncell)
    prior = prior / prior.max() * 10.0
    uniform = np.ones(ncell)

    world = World(rows, cols, float(cfg.cell_size_m))
    uav_cfg = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs/sim/uav.yaml"), {"battery_capacity_j": 5e7}
    )
    params = UAVParams.from_cfg(uav_cfg)
    coord_cfg = OmegaConf.load(REPO_ROOT / "configs/coordination/default.yaml")
    bw = coord_cfg.allocation.bid_weights
    bid_weights = (float(bw.travel), float(bw.energy), float(bw.priority))
    boost = float(coord_cfg.allocation.priority_boost)
    oracle = Oracle(df, "search", false_negative_rate={"person": float(cfg.person_fn)})
    duration_s = float(cfg.duration_min) * 60.0
    n_uavs = int(cfg.n_uavs)
    window = tuple(cfg.fail_window_s)

    results = {}
    for name, (strategy, greedy, use_prior) in SYSTEMS.items():
        world.priority = prior.copy() if use_prior else uniform.copy()
        cov, det, t80 = [], [], []
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
            # time to locate 80% of ALL ground-truth survivors (same denominator for both systems;
            # a system that never reaches it — e.g. lost coverage — is capped at the mission end)
            reached = ts[cum >= 0.8 * total] if (total and len(cum)) else np.array([])
            t80.append(float(reached[0]) if len(reached) else duration_s)

        def agg(v):
            v = np.asarray(v, dtype=float)
            n = len(v)
            return {
                "mean": float(v.mean()),
                "ci": float(1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            }

        results[name] = {"coverage": agg(cov), "detect": agg(det), "t80": agg(t80)}
    return {"total": total, "systems": results}


def plot_benchmark(res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res["systems"])
    colours = {names[0]: "#B85042", names[1]: "#1565C0"}
    metrics = [("coverage", "area covered"), ("detect", "survivors detected")]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for i, name in enumerate(names):
        vals = [res["systems"][name][m]["mean"] * 100 for m, _ in metrics]
        errs = [res["systems"][name][m]["ci"] * 100 for m, _ in metrics]
        t80 = res["systems"][name]["t80"]["mean"] / 60
        ax.bar(
            x + (i - 0.5) * w,
            vals,
            w,
            yerr=errs,
            capsize=3,
            color=colours[name],
            label=f"{name} — 80% located at {t80:.1f} min",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("percent")
    ax.set_ylim(0, 108)
    ax.set_title("Adaptive pipeline vs static baseline (clustered survivors, 1 UAV failure)")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/benchmark.yaml")
    res = run_benchmark(cfg)
    names = list(res["systems"])
    base, adapt = res["systems"][names[0]], res["systems"][names[1]]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"benchmark_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_benchmark(res, out_dir / "benchmark.png")

    print(f"[benchmark] {res['total']} clustered survivors · {cfg.n_uavs} UAVs · {cfg.condition}")
    for name in names:
        s = res["systems"][name]
        print(
            f"[benchmark] {name:38s} coverage {s['coverage']['mean'] * 100:4.0f}% · "
            f"survivors {s['detect']['mean'] * 100:4.0f}% · 80% at {s['t80']['mean'] / 60:.1f} min"
        )
    d_cov = (adapt["coverage"]["mean"] - base["coverage"]["mean"]) * 100
    d_det = (adapt["detect"]["mean"] - base["detect"]["mean"]) * 100
    speed = base["t80"]["mean"] / adapt["t80"]["mean"] if adapt["t80"]["mean"] else float("nan")
    print(
        f"[benchmark] adaptive vs baseline: +{d_cov:.0f} coverage pts, "
        f"+{d_det:.0f} survivors pts, {speed:.1f}× faster to locate 80%"
    )
    print(f"[benchmark] wrote {out_dir}/benchmark.png")


if __name__ == "__main__":
    main()
