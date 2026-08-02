"""Monte-Carlo sweep runner — Phase 9.

Runs the grid {strategies} × {n_uavs} × {failure conditions} × {seeds}, computes coordination
metrics per run, and aggregates to mean ± 95% CI — the evidence for whether adaptive reallocation
beats static partitioning. Writes tidy Parquet + a summary to ``outputs/runs/<timestamp>/``.

    make sweep
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.coordination.allocation import Coordinator
from src.eval.metrics import compute_metrics
from src.sim.engine import run
from src.sim.uav import UAV, UAVParams
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = [
    "coverage",
    "time_to_target_s",
    "redundant_ratio",
    "survivors_found",
    "total_distance_m",
    "completed",
    "lost_cells",
]
_N_FAIL = {"none": 0, "one_fail": 1, "two_fail": 2}


def _resolve_strategy(label: str, default_boost: float) -> tuple[str, float]:
    if label == "auction_no_priority":
        return "auction", 1.0  # priority-upweight ablation
    return ("auction" if label == "auction" else label), default_boost


def _failures(condition: str, n_uavs: int, rng: np.random.Generator, window) -> dict[int, float]:
    n = min(_N_FAIL[condition], n_uavs)
    if n == 0:
        return {}
    ids = rng.choice(n_uavs, size=n, replace=False)
    return {int(i): float(rng.uniform(*window)) for i in ids}


def run_sweep(
    cfg,
    world: World,
    params: UAVParams,
    *,
    oracle=None,
    bid_weights=(1.0, 0.0005, 50.0),
    default_boost=3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the sweep; return (tidy per-run rows, aggregated mean ± 95% CI)."""
    duration_s = float(cfg.duration_min) * 60.0
    dt = float(cfg.timestep_s)
    window = tuple(cfg.fail_window_s)
    rows: list[dict] = []

    for label in cfg.strategies:
        base_strategy, boost = _resolve_strategy(label, default_boost)
        for n_uavs in cfg.n_uavs:
            for condition in cfg.conditions:
                for seed in range(int(cfg.n_seeds)):
                    uavs = [UAV(i, params, world.base_xy) for i in range(int(n_uavs))]
                    coord = Coordinator(
                        base_strategy,
                        world,
                        int(n_uavs),
                        bid_weights=bid_weights,
                        priority_boost=boost,
                    )
                    fail_at = _failures(
                        condition, int(n_uavs), np.random.default_rng(seed + 100_000), window
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
                    m = compute_metrics(
                        result,
                        world.n_cells,
                        coverage_target=float(cfg.coverage_target),
                        complete_threshold=float(cfg.complete_threshold),
                    )
                    rows.append(
                        {
                            "strategy": label,
                            "n_uavs": int(n_uavs),
                            "condition": condition,
                            "seed": seed,
                            **m,
                        }
                    )

    tidy = pd.DataFrame(rows)
    return tidy, _aggregate(tidy)


def _aggregate(tidy: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (strategy, n_uavs, condition), sub in tidy.groupby(["strategy", "n_uavs", "condition"]):
        n = len(sub)
        row = {"strategy": strategy, "n_uavs": n_uavs, "condition": condition, "n": n}
        for metric in METRICS:
            vals = sub[metric].to_numpy(dtype=float)
            with warnings.catch_warnings():  # some groups never reach the coverage target (all-NaN)
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = float(np.nanmean(vals))
                sd = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = 1.96 * sd / np.sqrt(n) if n > 0 else 0.0
        out.append(row)
    return pd.DataFrame(out)


_STRATEGY_STYLE = {  # label -> (display name, bar colour)
    "auction": ("Adaptive auction", "#1565C0"),
    "auction_no_priority": ("Auction (no priority)", "#5B9BD5"),
    "static_partition_no_realloc": ("Static partition", "#B85042"),
    "single_uav": ("Single UAV", "#8C8C8C"),
    "random_walk": ("Random walk", "#C8A31A"),
}
_CONDITION_LABEL = {"none": "no failure", "one_fail": "1 failure", "two_fail": "2 failures"}


def plot_sweep(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of coverage (mean ± 95% CI) by failure condition, one bar per strategy,
    at the largest UAV count — the visual form of the headline table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_uavs = int(summary["n_uavs"].max())
    conditions = [c for c in ("none", "one_fail", "two_fail") if c in set(summary.condition)]
    strategies = [s for s in _STRATEGY_STYLE if s in set(summary.strategy)]
    x = np.arange(len(conditions))
    width = 0.8 / max(1, len(strategies))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, strat in enumerate(strategies):
        means, cis = [], []
        for cond in conditions:
            row = summary[
                (summary.n_uavs == n_uavs)
                & (summary.condition == cond)
                & (summary.strategy == strat)
            ]
            means.append(float(row.coverage_mean.iloc[0]) * 100 if not row.empty else 0.0)
            cis.append(float(row.coverage_ci95.iloc[0]) * 100 if not row.empty else 0.0)
        name, colour = _STRATEGY_STYLE[strat]
        ax.bar(
            x + (i - (len(strategies) - 1) / 2) * width,
            means,
            width,
            yerr=cis,
            capsize=3,
            color=colour,
            label=name,
            error_kw={"elinewidth": 1, "ecolor": "#333333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_CONDITION_LABEL[c] for c in conditions])
    ax.set_ylabel("Area covered (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Coverage under UAV failure ({n_uavs} UAVs, mean ± 95% CI)")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _headline(summary: pd.DataFrame) -> str:
    """Auction vs static coverage under each failure condition (at the largest UAV count)."""
    lines = ["\nHEADLINE — coverage (mean ± 95% CI), auction vs static partitioning:"]
    n_uavs = int(summary["n_uavs"].max())
    for condition in ("none", "one_fail", "two_fail"):
        sel = summary[(summary.n_uavs == n_uavs) & (summary.condition == condition)]
        a = sel[sel.strategy == "auction"]
        s = sel[sel.strategy == "static_partition_no_realloc"]
        if a.empty or s.empty:
            continue
        av, ac = a.coverage_mean.iloc[0], a.coverage_ci95.iloc[0]
        sv, sc = s.coverage_mean.iloc[0], s.coverage_ci95.iloc[0]
        lines.append(
            f"  [{n_uavs} UAVs, {condition:9s}] auction {av * 100:5.1f}±{ac * 100:.1f}%  "
            f"vs static {sv * 100:5.1f}±{sc * 100:.1f}%   Δ={(av - sv) * 100:+.1f} pts"
        )
    return "\n".join(lines)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/sweep.yaml")
    scenario = OmegaConf.load(REPO_ROOT / f"configs/scenario/{cfg.scenario}.yaml")
    world_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/world.yaml")
    uav_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/uav.yaml")
    coord_cfg = OmegaConf.load(REPO_ROOT / "configs/coordination/default.yaml")

    world = World.from_configs(scenario, world_cfg)
    params = UAVParams.from_cfg(uav_cfg)
    bw = coord_cfg.allocation.bid_weights
    bid_weights = (float(bw.travel), float(bw.energy), float(bw.priority))

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

    n_runs = len(cfg.strategies) * len(cfg.n_uavs) * len(cfg.conditions) * int(cfg.n_seeds)
    print(
        f"[sweep] {n_runs} runs: {list(cfg.strategies)} × {list(cfg.n_uavs)} UAVs × "
        f"{list(cfg.conditions)} × {cfg.n_seeds} seeds ..."
    )
    tidy, summary = run_sweep(
        cfg,
        world,
        params,
        oracle=oracle,
        bid_weights=bid_weights,
        default_boost=float(coord_cfg.allocation.priority_boost),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"sweep_{cfg.scenario}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(out_dir / "sweep.parquet", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")
    headline = _headline(summary)
    (out_dir / "headline.txt").write_text(headline + "\n")
    plot_sweep(summary, out_dir / "results.png")

    print(headline)
    print(f"\n[sweep] {len(tidy)} runs -> {out_dir}/sweep.parquet + summary.csv + results.png")


if __name__ == "__main__":
    main()
