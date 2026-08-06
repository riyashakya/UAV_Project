"""Re-look experiment — can coordination partly beat the perception bottleneck by looking twice?

Section 4.5 found the detector is the bottleneck: once coverage is recovered, a survivor the camera
misses on a single pass is simply lost. But that assumed **one look per cell**. Independent re-looks
lower the effective miss rate (two looks: miss goes from FN to ~FN^2), so a coordinator *can* claw
some of it back — at the cost of covering less new ground, and only down to the fraction of misses
that are *structural* rather than bad luck.

This is deliberately honest about that last point via a ``persistent_share`` knob: a share of misses
are permanent (occluded / too small / underwater) and never recovered no matter how many looks. With
it at 0 the model is the over-optimistic FN^k; at 1 re-looks do nothing.

    make relook

Focused Monte-Carlo (like rq4 / sensitivity), CPU-only; reuses the benchmark's sparse survivor
generator. The detection model ``miss_after_looks`` is pure and unit-tested. Not a new method — a
concrete, bounded refinement of the bottleneck finding (multiple-looks-improve-detection is classic
search theory; the value is measuring the trade-off against *this* project's result).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.eval.benchmark import _hotspot_survivors

REPO_ROOT = Path(__file__).resolve().parents[2]


def miss_after_looks(fn: float, persistent_share: float, k: int) -> float:
    """Probability a present survivor is still missed after ``k`` independent looks.

    A share ``persistent_share`` of the single-look miss rate is structural (never recovered); the
    rest is transient and re-rolled each look. Calibrated so ``k == 1`` gives exactly ``fn``.
    """
    if k <= 0:
        return 1.0
    rho = float(persistent_share) * float(fn)  # permanent-miss floor (<= fn by construction)
    denom = 1.0 - rho
    q = (fn - rho) / denom if denom > 0 else 0.0  # transient per-look miss
    return rho + denom * (q**k)


def _mean_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    return (float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)


def run_relook(cfg) -> dict:
    rows, cols = int(cfg.grid.rows), int(cfg.grid.cols)
    ncell = rows * cols
    counts, _ = _hotspot_survivors(cfg)
    total = int(counts.sum())
    true_p = counts / counts.sum() if counts.sum() else np.ones(ncell) / ncell

    fn = float(cfg.fn)
    ps = float(cfg.persistent_share)
    budget = int(round(float(cfg.coverage_factor) * ncell))  # total looks available
    p_detect = {int(k): 1.0 - miss_after_looks(fn, ps, int(k)) for k in cfg.looks}

    by_noise: dict[float, dict] = {}
    for noise in cfg.prior_noise_levels:
        noise = float(noise)
        per_k = {}
        for k in cfg.looks:
            k = int(k)
            n_targeted = min(ncell, budget // k)  # distinct cells affordable at k looks each
            found_frac, cover_frac = [], []
            for seed in range(int(cfg.n_seeds)):
                rng = np.random.default_rng(seed)
                prior = (1 - noise) * true_p + noise * rng.random(ncell)  # imperfect per-seed prior
                targeted = np.argsort(prior)[::-1][:n_targeted]
                found = sum(int(rng.binomial(int(counts[c]), p_detect[k])) for c in targeted)
                found_frac.append(found / total if total else 0.0)
                cover_frac.append(n_targeted / ncell)
            per_k[k] = {
                "looks": k,
                "p_detect": p_detect[k],
                "found": _mean_ci(found_frac),
                "coverage": _mean_ci(cover_frac),
            }
        by_noise[noise] = per_k

    return {
        "total": total,
        "fn": fn,
        "persistent_share": ps,
        "single_look_ceiling": 1.0 - fn,
        "persistent_floor_detect": 1.0 - ps * fn,
        "by_noise": by_noise,
    }


def plot_relook(res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colours = ["#0C3B6E", "#1E88C7", "#8CC0E4"]  # good -> poor prior
    for (noise, per_k), colour in zip(sorted(res["by_noise"].items()), colours):
        ks = sorted(per_k)
        found = [per_k[k]["found"][0] * 100 for k in ks]
        ci = [per_k[k]["found"][1] * 100 for k in ks]
        quality = "good" if noise <= 0.15 else ("medium" if noise <= 0.45 else "poor")
        ax.errorbar(
            ks,
            found,
            yerr=ci,
            marker="o",
            lw=2,
            color=colour,
            label=f"{quality} prior (noise {noise:g})",
        )
    ax.axhline(
        res["single_look_ceiling"] * 100,
        ls=":",
        color="#B85042",
        label=f"single-look ceiling (1-FN) = {res['single_look_ceiling'] * 100:.0f}%",
    )
    ax.set_xlabel("looks per targeted cell (k) — higher k = more re-looks, less coverage")
    ax.set_ylabel("survivors found (% of all)")
    any_k = sorted(next(iter(res["by_noise"].values())))
    ax.set_xticks(any_k)
    ax.set_ylim(0, 100)
    ax.set_title("Re-look helps only with a good prior; otherwise coverage loss wins")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/relook.yaml")
    res = run_relook(cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"relook_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "relook.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["prior_noise", "looks", "p_detect", "found_pct", "found_ci", "coverage_pct"])
        for noise, per_k in sorted(res["by_noise"].items()):
            for k in sorted(per_k):
                s = per_k[k]
                wr.writerow(
                    [
                        noise,
                        k,
                        round(s["p_detect"], 3),
                        round(s["found"][0] * 100, 1),
                        round(s["found"][1] * 100, 1),
                        round(s["coverage"][0] * 100, 1),
                    ]
                )
    plot_relook(res, out_dir / "relook.png")

    print(
        f"[relook] FN={res['fn']} persistent_share={res['persistent_share']} · "
        f"{res['total']} survivors · single-look ceiling {res['single_look_ceiling'] * 100:.0f}% · "
        f"floor if re-look all = {res['persistent_floor_detect'] * 100:.0f}%"
    )
    for noise, per_k in sorted(res["by_noise"].items()):
        base = per_k[1]["found"][0] * 100
        best_k = max(per_k.values(), key=lambda s: s["found"][0])
        gain = best_k["found"][0] * 100 - base
        verdict = "re-look WINS" if best_k["looks"] > 1 and gain > 0.5 else "cover-more wins"
        print(
            f"[relook] prior noise {noise:g}: k=1 finds {base:4.1f}% · "
            f"best k={best_k['looks']} finds {best_k['found'][0] * 100:4.1f}% "
            f"({gain:+.1f} pts) -> {verdict}"
        )
    print(f"[relook] wrote {out_dir}/relook.{{png,csv}}")


if __name__ == "__main__":
    main()
