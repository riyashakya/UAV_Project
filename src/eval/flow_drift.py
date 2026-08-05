"""Measured vs assumed current for survivor-drift forecasting (the "new" integration).

Upgrades the drift model's weakest input — the flow field — from *assumed* to *measured*. A
synthetic drone clip is advected by a KNOWN flood-channel current; PIV (`src/perception/flow.py`)
estimates that current straight off the video; then the survivor-drift forecast is run twice — once
from an ASSUMED (hand-set, wrong) current, once from the PIV-MEASURED current — and both are scored
against where the survivor TRULY drifts. Reuses only the tested drift advection (ADR-001 untouched:
the field is estimated offline and cached, exactly like detections).

    make flow-drift

Honest scope: no real flood video with a known current was available, so ground truth is synthetic;
this is a proof-of-concept of the integration, not a field-validated result. CPU-only.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from shapely.geometry import Point

from src.drift.advect import advect_particles, drift_search_region
from src.perception.flow import (
    estimate_flow_piv,
    piv_to_flowfield,
    save_flow_field,
    synthesize_flow_video,
)
from src.sim.world import make_flow_field

REPO_ROOT = Path(__file__).resolve().parents[2]


def _flow_recovery(true_flow, sampled: dict) -> dict:
    """Score the PIV field against the truth at the sampled points: speed RMSE + angular error."""
    pts, vest = sampled["points_xy"], sampled["vel"]
    vtrue = np.array([true_flow(float(x), float(y)) for x, y in pts])
    rmse = float(np.sqrt(np.mean(np.sum((vest - vtrue) ** 2, axis=1))))
    # angular error where both have appreciable speed
    st, se = np.linalg.norm(vtrue, axis=1), np.linalg.norm(vest, axis=1)
    m = (st > 0.3) & (se > 0.3)
    if m.any():
        cos = np.sum(vest[m] * vtrue[m], axis=1) / (se[m] * st[m])
        ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
    else:
        ang = float("nan")
    return {
        "rmse_m_s": round(rmse, 3),
        "mean_speed_true": round(float(st.mean()), 3),
        "ang_deg": round(ang, 1),
    }


def run_flow_drift(cfg) -> dict:
    true_flow = make_flow_field(cfg.true_flow)
    assumed_flow = make_flow_field(cfg.assumed_flow)

    # 1) synthesise a clip under the TRUE current, then MEASURE the current back off it with PIV
    frames = synthesize_flow_video(
        true_flow,
        n_frames=int(cfg.synth.n_frames),
        frame_px=tuple(cfg.synth.frame_px),
        metres_per_pixel=float(cfg.synth.metres_per_pixel),
        fps=float(cfg.synth.fps),
        density=float(cfg.synth.density),
        noise=float(cfg.synth.noise),
        rng=np.random.default_rng(0),
    )
    piv = estimate_flow_piv(frames, window_px=int(cfg.piv.window_px), step_px=int(cfg.piv.step_px))
    measured_flow, sampled = piv_to_flowfield(
        piv, metres_per_pixel=float(cfg.synth.metres_per_pixel), fps=float(cfg.synth.fps)
    )
    recovery = _flow_recovery(true_flow, sampled)

    # 2) forecast the survivor's drift from ASSUMED vs MEASURED current; score against TRUE drift
    s0 = np.array([float(cfg.survivor_xy[0]), float(cfg.survivor_xy[1])])
    lvl = float(cfg.containment)
    kw = dict(
        horizon_s=float(cfg.horizon_s),
        dt=float(cfg.dt),
        leeway_factor=float(cfg.leeway_factor),
        k_h=float(cfg.k_h),
    )
    fields = {"assumed": assumed_flow, "measured": measured_flow}
    hits = {k: [] for k in fields}
    errs = {k: [] for k in fields}
    trues = []
    for seed in range(int(cfg.n_seeds)):
        p_true = advect_particles(
            s0, true_flow, n_particles=1, rng=np.random.default_rng(seed), **kw
        )[0]
        trues.append(p_true)
        for name, flow in fields.items():
            region = drift_search_region(
                tuple(s0),
                flow,
                rng=np.random.default_rng(seed + (1 if name == "assumed" else 2) * 1_000_000),
                n_particles=int(cfg.n_particles),
                containment_levels=(lvl,),
                **kw,
            )
            hits[name].append(bool(region["containment"][lvl].covers(Point(p_true))))
            errs[name].append(float(np.hypot(*(p_true - np.asarray(region["centroid"])))))

    def agg(name) -> dict:
        e = np.asarray(errs[name])
        n = len(e)
        ci = float(1.96 * e.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return {
            "located_rate": round(float(np.mean(hits[name])), 3),
            "mean_err": round(float(e.mean()), 1),
            "ci_err": round(ci, 1),
        }

    trues = np.asarray(trues)
    return {
        "n_seeds": int(cfg.n_seeds),
        "recovery": recovery,
        "assumed": agg("assumed"),
        "measured": agg("measured"),
        "true_drift_m": round(float(np.hypot(*(trues.mean(axis=0) - s0))), 1),
        "_s0": s0,
        "_trues": trues,
        "_true_flow": true_flow,
        "_measured_flow": measured_flow,
        "_assumed_flow": assumed_flow,
        "_sampled": sampled,
        "_kw": kw,
        "_lvl": lvl,
        "_np": int(cfg.n_particles),
        "_frame_px": tuple(cfg.synth.frame_px),
        "_mpp": float(cfg.synth.metres_per_pixel),
    }


def _region_poly(flow, s0, kw, lvl, npart, seed):
    r = drift_search_region(
        tuple(s0),
        flow,
        rng=np.random.default_rng(seed),
        n_particles=npart,
        containment_levels=(lvl,),
        **kw,
    )
    return r["containment"][lvl], np.asarray(r["centroid"])


def plot_flow_drift(res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    s0, trues, kw, lvl = res["_s0"], res["_trues"], res["_kw"], res["_lvl"]
    extent = res["_frame_px"][1] * res["_mpp"], res["_frame_px"][0] * res["_mpp"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4))

    # -- left: true vs PIV-measured current (quiver on a coarse grid)
    gx = np.linspace(20, extent[0] - 20, 9)
    gy = np.linspace(20, extent[1] - 20, 9)
    GX, GY = np.meshgrid(gx, gy)
    for flow, colour, lab in (
        (res["_true_flow"], "#9AA7B4", "true current"),
        (res["_measured_flow"], "#0C3B6E", "PIV-measured current"),
    ):
        UV = np.array([flow(float(x), float(y)) for x, y in zip(GX.ravel(), GY.ravel())])
        ax1.quiver(
            GX.ravel(),
            GY.ravel(),
            UV[:, 0],
            UV[:, 1],
            color=colour,
            alpha=0.85,
            angles="xy",
            scale_units="xy",
            scale=0.12,
            width=0.004,
            label=lab,
        )
    ax1.set_title(f"PIV recovers the current (RMSE {res['recovery']['rmse_m_s']} m/s)")
    ax1.set_xlabel("east (m)")
    ax1.set_ylabel("south (m)")
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    ax1.legend(fontsize=8, loc="upper right")

    # -- right: true drift endpoints vs assumed / measured forecast regions
    ax2.scatter(
        trues[:, 0],
        trues[:, 1],
        s=10,
        color="#4C6A82",
        alpha=0.4,
        label="true survivor position (per seed)",
    )
    for flow, shade, name, seed in (
        (res["_assumed_flow"], "#B85042", "assumed", 11),
        (res["_measured_flow"], "#2E86AB", "measured", 12),
    ):
        poly, cen = _region_poly(flow, s0, kw, lvl, res["_np"], seed)
        if poly.geom_type == "Polygon":
            ax2.add_patch(
                MplPolygon(
                    np.asarray(poly.exterior.coords),
                    closed=True,
                    facecolor=shade,
                    alpha=0.22,
                    edgecolor=shade,
                    lw=1.8,
                    label=(
                        f"{name} flow -> {int(lvl * 100)}% zone "
                        f"({res[name]['located_rate'] * 100:.0f}% hit)"
                    ),
                )
            )
        ax2.scatter(*cen, marker="P", s=120, color=shade, edgecolor="white", lw=1.1, zorder=5)
    ax2.scatter(
        *s0, marker="X", s=150, color="#333", edgecolor="white", lw=1.1, zorder=6, label="sighting"
    )
    ax2.set_title("Forecast drift: measured current tracks the truth, assumed misses")
    ax2.set_xlabel("east (m)")
    ax2.set_ylabel("south (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.invert_yaxis()
    ax2.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/perception/flow.yaml")
    res = run_flow_drift(cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "runs" / f"flow_drift_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_flow_field(
        REPO_ROOT / "data" / "cache" / "flow_field.npz",
        res["_sampled"],
        meta={"source": "synthetic_piv", "true_flow": OmegaConf.to_container(cfg.true_flow)},
    )
    with open(out_dir / "flow_drift.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["policy", "located_rate", "mean_err_m", "ci95_m"])
        for name in ("assumed", "measured"):
            wr.writerow(
                [name, res[name]["located_rate"], res[name]["mean_err"], res[name]["ci_err"]]
            )
    plot_flow_drift(res, out_dir / "flow_drift.png")

    a, m, rec = res["assumed"], res["measured"], res["recovery"]
    print(
        f"[flow-drift] {res['n_seeds']} seeds · survivor drifts ~{res['true_drift_m']:.0f} m east"
    )
    print(
        f"[flow-drift] PIV current recovery: RMSE {rec['rmse_m_s']} m/s "
        f"(mean true speed {rec['mean_speed_true']} m/s), angular error {rec['ang_deg']}°"
    )
    print(
        f"[flow-drift] ASSUMED current  : located {a['located_rate'] * 100:.0f}% · "
        f"error {a['mean_err']:.0f}±{a['ci_err']:.0f} m"
    )
    print(
        f"[flow-drift] MEASURED current : located {m['located_rate'] * 100:.0f}% · "
        f"error {m['mean_err']:.0f}±{m['ci_err']:.0f} m"
    )
    print(
        f"[flow-drift] measuring the current cuts localisation error by "
        f"{a['mean_err'] - m['mean_err']:.0f} m and raises the hit-rate by "
        f"{(m['located_rate'] - a['located_rate']) * 100:.0f} pts"
    )
    print(f"[flow-drift] wrote {out_dir}/flow_drift.{{png,csv}}")


if __name__ == "__main__":
    main()
