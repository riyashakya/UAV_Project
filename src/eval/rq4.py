"""RQ4 quantitative result — drift-aware search re-tasking vs the stale sighting (Phase 7).

A survivor detected in flowing water drifts downstream; searching the *stale detection point* sends
rescue to where they *were*, while searching the *predicted drift region* targets where they *are*.
This Monte-Carlo experiment measures both over many seeds:

* **localisation rate** — fraction of runs where the survivor's true position falls in the searched
  region (the 90% drift zone, or the detection cell for the stale policy);
* **localisation error** — metres from the search target to the survivor's true position.

Each seed advects the survivor's *true* position with the flow, and — with an **independent** RNG,
so the predictor never sees the true draw — predicts the drift region. Reuses only the tested
Phase-7 advection; no engine/coordinator change (ADR-001 untouched).

    make rq4

CPU-only (numpy + shapely); matplotlib imported lazily for the figure.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from shapely.geometry import Point

from src.drift.advect import advect_particles, drift_search_region
from src.sim.world import make_flow_field

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_rq4(cfg) -> dict:
    """Run the experiment; return localisation rate + error (mean ± 95% CI) for both policies."""
    flow = make_flow_field(cfg.flow)
    size = float(cfg.cell_size_m)
    s0 = np.array([(float(cfg.survivor_col) + 0.5) * size, (float(cfg.survivor_row) + 0.5) * size])
    lvl = float(cfg.containment)
    kw = dict(
        horizon_s=float(cfg.horizon_s),
        dt=float(cfg.dt),
        leeway_factor=float(cfg.leeway_factor),
        k_h=float(cfg.k_h),
    )

    drift_hit, stale_hit, drift_err, stale_err, trues = [], [], [], [], []
    for seed in range(int(cfg.n_seeds)):
        # the survivor's TRUE final position (one advected particle)
        p_true = advect_particles(s0, flow, n_particles=1, rng=np.random.default_rng(seed), **kw)[0]
        # an INDEPENDENT drift prediction (its own RNG -> a genuine forecast, not circular)
        region = drift_search_region(
            tuple(s0),
            flow,
            rng=np.random.default_rng(seed + 1_000_000),
            n_particles=int(cfg.n_particles),
            containment_levels=(lvl,),
            **kw,
        )
        centroid = np.asarray(region["centroid"])
        drift_hit.append(bool(region["containment"][lvl].covers(Point(p_true))))
        stale_hit.append(
            bool(abs(p_true[0] - s0[0]) <= size / 2 and abs(p_true[1] - s0[1]) <= size / 2)
        )
        drift_err.append(float(np.hypot(*(p_true - centroid))))
        stale_err.append(float(np.hypot(*(p_true - s0))))
        trues.append(p_true)

    def agg(hits, errs) -> dict:
        errs = np.asarray(errs)
        n = len(errs)
        ci = float(1.96 * errs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return {
            "located_rate": round(float(np.mean(hits)), 3),
            "mean_err": round(float(errs.mean()), 1),
            "ci_err": round(ci, 1),
        }

    return {
        "n_seeds": int(cfg.n_seeds),
        "drift_distance_m": round(float(np.hypot(*(np.mean(trues, axis=0) - s0))), 1),
        "drift": agg(drift_hit, drift_err),
        "stale": agg(stale_hit, stale_err),
        "_trues": np.asarray(trues),
        "_s0": s0,
        "_flow": flow,
        "_kw": kw,
        "_lvl": lvl,
        "_n_particles": int(cfg.n_particles),
    }


def plot_rq4(res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    s0, trues = res["_s0"], res["_trues"]
    region = drift_search_region(
        tuple(s0),
        res["_flow"],
        rng=np.random.default_rng(7),
        n_particles=res["_n_particles"],
        containment_levels=(0.5, res["_lvl"]),
        **res["_kw"],
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(
        trues[:, 0],
        trues[:, 1],
        s=10,
        color="#4C6A82",
        alpha=0.35,
        label="true survivor position (per seed)",
    )
    for level, shade in ((res["_lvl"], "#9CC6E8"), (0.5, "#5E9FD0")):
        poly = region["containment"].get(level)
        if poly is not None and poly.geom_type == "Polygon":
            ax.add_patch(
                MplPolygon(
                    np.asarray(poly.exterior.coords),
                    closed=True,
                    facecolor=shade,
                    alpha=0.25,
                    edgecolor=shade,
                    lw=1.6,
                    label=f"{int(level * 100)}% drift zone (searched)",
                )
            )
    ax.scatter(
        *s0,
        marker="X",
        s=160,
        color="#B85042",
        edgecolor="white",
        lw=1.2,
        zorder=5,
        label="stale sighting (searched by baseline)",
    )
    ax.scatter(
        *region["centroid"],
        marker="P",
        s=160,
        color="#0C3B6E",
        edgecolor="white",
        lw=1.2,
        zorder=5,
        label="drift centroid (search target)",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("south (m)")
    ax.set_title("RQ4: survivors drift away from the sighting — search the drift zone")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/eval/rq4.yaml")
    res = run_rq4(cfg)
    d, s = res["drift"], res["stale"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"rq4_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rq4.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["policy", "located_rate", "mean_err_m", "ci95_m"])
        wr.writerow(["drift_aware", d["located_rate"], d["mean_err"], d["ci_err"]])
        wr.writerow(["stale_sighting", s["located_rate"], s["mean_err"], s["ci_err"]])
    plot_rq4(res, out_dir / "rq4.png")

    print(
        f"[rq4] {res['n_seeds']} seeds · survivor drifts "
        f"~{res['drift_distance_m']:.0f} m from the sighting"
    )
    print(
        f"[rq4] DRIFT-AWARE (90% zone): located {d['located_rate'] * 100:.0f}% of the time, "
        f"error {d['mean_err']:.0f}±{d['ci_err']:.0f} m"
    )
    print(
        f"[rq4] STALE SIGHTING       : located {s['located_rate'] * 100:.0f}% of the time, "
        f"error {s['mean_err']:.0f}±{s['ci_err']:.0f} m"
    )
    print(
        f"[rq4] re-tasking search to the drift zone cuts localisation error by "
        f"{s['mean_err'] - d['mean_err']:.0f} m and raises the hit-rate by "
        f"{(d['located_rate'] - s['located_rate']) * 100:.0f} pts"
    )
    print(f"[rq4] wrote {out_dir}/rq4.{{png,csv}}")


if __name__ == "__main__":
    main()
