"""Contribution B — perception × coordination sensitivity study.

Perception papers report detector AP; coordination papers usually assume perfect perception. The
decoupled oracle (ADR-001) lets the *measured* detector error be a controlled knob over the
coordination layer. This sweeps the **survivor false-negative rate** and, under a UAV failure,
measures the end-to-end **survivor-detection rate** (found / ground-truth survivors) for adaptive
auction vs static partitioning — exposing how perception error and coordination robustness combine.

Reference line ``1 - FN`` = the ceiling with perfect coverage (only perception loss); the gap below
it is the coordination (coverage) loss that re-tasking recovers.

    make sensitivity

CPU-only; reuses ``engine.run`` + ``Oracle`` — no engine change. matplotlib imported lazily.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.coordination.allocation import Coordinator
from src.eval.runner import _failures
from src.sim.engine import run
from src.sim.oracle import Oracle
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]


def _total_survivors(detections, scenario: str) -> int:
    df = detections if isinstance(detections, pd.DataFrame) else pd.read_parquet(detections)
    return int(((df["scenario"] == scenario) & (df["class"] == "person")).sum())


def run_sensitivity(cfg, detections=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep FN × strategy × seed; return (tidy per-run rows, aggregated mean ± 95% CI)."""
    scenario = OmegaConf.load(REPO_ROOT / f"configs/scenario/{cfg.scenario}.yaml")
    world_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/world.yaml")
    uav_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/uav.yaml")
    coord_cfg = OmegaConf.load(REPO_ROOT / "configs/coordination/default.yaml")
    ocfg = OmegaConf.load(REPO_ROOT / "configs/sim/oracle.yaml")

    world = World.from_configs(scenario, world_cfg)
    params = UAVParams.from_cfg(uav_cfg)
    bw = coord_cfg.allocation.bid_weights
    bid_weights = (float(bw.travel), float(bw.energy), float(bw.priority))
    boost = float(coord_cfg.allocation.priority_boost)
    det = detections if detections is not None else REPO_ROOT / "data/cache/detections.parquet"
    total = _total_survivors(det, scenario.name)
    base_fn = dict(ocfg.false_negative_rate)
    window = tuple(cfg.fail_window_s)
    duration_s = float(world_cfg.get("duration_min", 60)) * 60.0
    dt = float(world_cfg.get("timestep_s", 5.0))
    n_uavs = int(cfg.n_uavs)

    rows: list[dict] = []
    for strategy in cfg.strategies:
        for fn in cfg.fn_rates:
            fn = float(fn)
            oracle = Oracle(
                det,
                scenario.name,
                false_negative_rate={**base_fn, "person": fn},
                latency_s=tuple(ocfg.latency_s),
            )
            for seed in range(int(cfg.n_seeds)):
                uavs = [UAV(i, params, world.base_xy) for i in range(n_uavs)]
                coord = Coordinator(
                    strategy, world, n_uavs, bid_weights=bid_weights, priority_boost=boost
                )
                fail_at = _failures(
                    str(cfg.condition), n_uavs, np.random.default_rng(seed + 100_000), window
                )
                result = run(
                    world,
                    uavs,
                    coordinator=coord,
                    seed=seed,
                    duration_s=duration_s,
                    dt=dt,
                    oracle=oracle,
                    fail_at=fail_at,
                )
                found = result["found_total"].get("person", 0)
                rows.append(
                    {
                        "strategy": strategy,
                        "fn": fn,
                        "seed": seed,
                        "detect_rate": found / total if total else 0.0,
                        "coverage": result["coverage"],
                    }
                )

    tidy = pd.DataFrame(rows)
    agg = []
    for (strategy, fn), sub in tidy.groupby(["strategy", "fn"]):
        v = sub["detect_rate"].to_numpy(dtype=float)
        n = len(v)
        agg.append(
            {
                "strategy": strategy,
                "fn": round(fn, 3),
                "n": n,
                "detect_rate_mean": round(float(v.mean()), 4),
                "detect_rate_ci95": round(
                    float(1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0, 4
                ),
                "coverage_mean": round(float(sub["coverage"].mean()), 4),
            }
        )
    return tidy, pd.DataFrame(agg)


def plot_sensitivity(agg: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "auction": ("Adaptive auction", "#1565C0"),
        "static_partition_no_realloc": ("Static partition", "#B85042"),
    }
    fns = sorted(agg["fn"].unique())
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        fns,
        [1 - f for f in fns],
        "--",
        color="#8C8C8C",
        lw=1.4,
        label="perfect-coverage ceiling (1 − FN)",
    )
    for strat, (name, colour) in style.items():
        sub = agg[agg.strategy == strat].sort_values("fn")
        if sub.empty:
            continue
        ax.errorbar(
            sub["fn"],
            sub["detect_rate_mean"],
            yerr=sub["detect_rate_ci95"],
            marker="o",
            capsize=3,
            color=colour,
            label=name,
            lw=2,
        )
    ax.set_xlabel("survivor detector false-negative rate (FN)")
    ax.set_ylabel("survivors detected (fraction of ground truth)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Perception × coordination: survivor detection under a 2-UAV failure")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/sensitivity.yaml")
    tidy, agg = run_sensitivity(cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"sensitivity_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "sensitivity.csv", index=False)
    plot_sensitivity(agg, out_dir / "sensitivity.png")

    def rate(strat, fn):
        r = agg[(agg.strategy == strat) & (np.isclose(agg.fn, fn))]
        return float(r.detect_rate_mean.iloc[0]) if not r.empty else float("nan")

    fmax = max(float(f) for f in cfg.fn_rates)
    print(
        f"[sensitivity] {len(tidy)} runs · scenario={cfg.scenario} · "
        f"{cfg.n_uavs} UAVs · {cfg.condition}"
    )
    print(
        f"[sensitivity] at FN=0.0: auction {rate('auction', 0.0) * 100:.0f}% vs static "
        f"{rate('static_partition_no_realloc', 0.0) * 100:.0f}% survivors detected"
    )
    print(
        f"[sensitivity] at FN={fmax}: auction {rate('auction', fmax) * 100:.0f}% vs static "
        f"{rate('static_partition_no_realloc', fmax) * 100:.0f}%"
    )
    print(
        "[sensitivity] perception error (FN) sets the ceiling (1−FN); adaptive re-tasking recovers "
        "the coverage gap below it, static does not"
    )
    print(f"[sensitivity] wrote {out_dir}/sensitivity.{{png,csv}}")


if __name__ == "__main__":
    main()
