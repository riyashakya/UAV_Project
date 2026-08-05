"""Real-data grounding of the ablation — attacks the "synthetic scenario" weakness.

Instead of invented Gaussian survivor hotspots, this runs the same static / auction / auction+guided
ablation on the **real detection cache**: the real per-cell survivor distribution and confidences
(real YOLO on real disaster imagery), with a **real, independent** search prior derived from the
flood-water segmentation. Survivors were measured to sit *away* from the water (corr ≈ -0.5), so the
prior is the **inverted water map** — a genuine, non-circular predictive signal (not the survivor
locations). Honest limitations (documented): the georeferencing is still synthetic, and it is still
simulation-only.

    make benchmark-real

CPU-only; reuses run_benchmark — no new mechanism.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.eval.benchmark import plot_benchmark, run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[2]


def real_data(scenario: str, rows: int, cols: int):
    """Return (real per-cell survivor counts, the real person detections, a flood-derived prior)."""
    cache = REPO_ROOT / "data" / "cache" / "detections.parquet"
    df = pd.read_parquet(cache)
    fa = df[df["scenario"] == scenario].copy()
    person = fa[fa["class"] == "person"].copy()
    person["scenario"] = "bench"  # relabel so the oracle serves these under the benchmark scenario
    n = rows * cols
    counts = np.zeros(n, dtype=int)
    for cid, k in person.groupby("cell_id").size().items():
        counts[int(cid)] = int(k)
    water = np.zeros(n)
    for cid, k in fa[fa["class"] == "water"].groupby("cell_id").size().items():
        water[int(cid)] = float(k)
    # real, independent prior: survivors fled AWAY from the flood -> invert the water density
    prior = 1.0 - (water / water.max() if water.max() > 0 else water)
    return counts, person, prior


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/benchmark_real.yaml")
    counts, person, prior = real_data(cfg.scenario, int(cfg.grid.rows), int(cfg.grid.cols))
    res = run_benchmark(cfg, detections={"counts": counts, "df": person}, prior=prior)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"benchmark_real_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_benchmark(
        res,
        out_dir / "benchmark_real.png",
        title="REAL detection distribution + flood-derived prior (1 UAV failure)",
    )

    s = res["systems"]
    st, au, gu = s["static (no realloc)"], s["auction (realloc)"], s["auction + guided"]
    print(
        f"[benchmark-real] {res['total']} REAL survivor detections · "
        f"{cfg.n_uavs} UAVs · {cfg.condition}"
    )
    for name, v in s.items():
        print(
            f"[benchmark-real] {name:22s} coverage {v['coverage'][0] * 100:4.0f}% · "
            f"detected {v['detect'][0] * 100:4.0f}% · 50% at {v['t50'][0] / 60:4.1f} min"
        )
    print(
        f"[benchmark-real] reallocation gain (static→auction): "
        f"+{(au['detect'][0] - st['detect'][0]) * 100:.0f} detected pts"
    )
    print(
        f"[benchmark-real] guidance gain (auction→+guided):    "
        f"+{(gu['detect'][0] - au['detect'][0]) * 100:.0f} detected pts, "
        f"{au['t50'][0] / gu['t50'][0]:.1f}× faster to 50%"
    )
    print(f"[benchmark-real] wrote {out_dir}/benchmark_real.png")


if __name__ == "__main__":
    main()
