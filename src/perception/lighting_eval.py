"""Lighting-robustness evaluation for the detector (bright / normal / dark).

The detector was trained with brightness augmentation (``hsv_v: 0.4``, pinned in
``configs/perception/model_a.yaml``). This measures whether that pays off: it re-scores the *same*
validation set at several brightness levels and reports how mAP holds up. No retraining — it only
runs the already-trained model on brightened / dimmed copies of the val images.

    make lighting-robustness

Perception-only (may import ultralytics; ADR-001 keeps it out of src/sim). Needs the detect dataset
and a trained checkpoint; not part of `make test`. The brightness transform itself is unit-tested.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "runs"


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel intensities by ``factor`` and clip to [0, 255] — the value/brightness change.

    ``factor < 1`` darkens, ``factor > 1`` brightens, ``factor == 1`` is identity.
    """
    return np.clip(img.astype(np.float32) * float(factor), 0.0, 255.0).astype(np.uint8)


def _pick_device() -> str | int:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _find_weights(cfg_weights) -> Path:
    if cfg_weights:
        return Path(cfg_weights)
    cands = sorted(
        (REPO_ROOT / "outputs" / "perception").glob("*detect*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(
            "No detect checkpoint found under outputs/perception/*detect*/weights/best.pt; "
            "set `weights:` in configs/perception/lighting.yaml."
        )
    return cands[-1]


def _build_variant(
    src_img_dir: Path, src_lbl_dir: Path, out_root: Path, factor: float, names, limit: int
) -> Path:
    """Write brightness-adjusted images + copied labels to ``out_root``; return its data.yaml."""
    img_out, lbl_out = out_root / "images", out_root / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in src_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    # deterministic shuffle so the subset spans both sources (files are grouped by source name)
    np.random.default_rng(0).shuffle(imgs)
    for p in imgs[: int(limit)]:
        arr = np.asarray(Image.open(p).convert("RGB"))
        Image.fromarray(adjust_brightness(arr, factor)).save(img_out / p.name)
        lbl = src_lbl_dir / (p.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, lbl_out / lbl.name)
    yaml_path = out_root / "data.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "path": str(out_root),
                "train": "images",
                "val": "images",
                "names": list(names),
                "nc": len(names),
            }
        ),
        yaml_path,
    )
    return yaml_path


def run_lighting(cfg) -> dict:
    from ultralytics import YOLO  # lazy (ADR-001)

    src = OmegaConf.load(REPO_ROOT / cfg.data)
    base = Path(src.path)
    val_img = base / str(src.val)
    val_lbl = base / str(src.val).replace("images", "labels")
    names = list(src.names)
    weights = _find_weights(cfg.get("weights"))
    device = _pick_device()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"lighting_{ts}"
    print(f"[lighting] weights={weights.name} device={device} limit={int(cfg.limit)} imgs/level")

    rows = []
    for label, factor in cfg.factors:
        variant = _build_variant(
            val_img, val_lbl, out_dir / label, float(factor), names, int(cfg.limit)
        )
        metrics = YOLO(str(weights)).val(
            data=str(variant), imgsz=int(cfg.imgsz), device=device, plots=False, verbose=False
        )
        rows.append(
            {
                "lighting": label,
                "factor": float(factor),
                "map50": float(metrics.box.map50),
                "map": float(metrics.box.map),
            }
        )
        print(
            f"[lighting] {label:6s} (x{factor}): mAP@50={rows[-1]['map50']:.3f} "
            f"mAP@50-95={rows[-1]['map']:.3f}"
        )
    return {"rows": rows, "out_dir": out_dir, "weights": str(weights)}


def plot_lighting(res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    labels = [f"{r['lighting']}\n(x{r['factor']})" for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(labels, [r["map50"] * 100 for r in rows], color="#1565C0", width=0.6)
    for i, r in enumerate(rows):
        ax.text(i, r["map50"] * 100 + 1, f"{r['map50'] * 100:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("mAP@50 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Detector robustness to lighting (same val set, re-brightened)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/perception/lighting.yaml")
    res = run_lighting(cfg)
    out_dir = res["out_dir"]
    with open(out_dir / "lighting.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["lighting", "factor", "map50", "map"])
        wr.writeheader()
        wr.writerows(res["rows"])
    plot_lighting(res, out_dir / "lighting.png")

    normal = next((r for r in res["rows"] if r["lighting"] == "normal"), res["rows"][0])
    for r in res["rows"]:
        drop = (normal["map50"] - r["map50"]) * 100
        print(
            f"[lighting] {r['lighting']:6s}: mAP@50 {r['map50'] * 100:4.1f}% "
            f"({'-' if drop >= 0 else '+'}{abs(drop):.1f} pts vs normal)"
        )
    print(f"[lighting] wrote {out_dir}/lighting.{{png,csv}}")


if __name__ == "__main__":
    main()
