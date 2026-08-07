"""SAHI at a realistic operating point — recall/precision, full-frame vs tiled, base vs fine-tuned.

Section 4.1's AP result (at conf 0.001, as AP requires) made SAHI look worse. That AP is real but is
dominated by the low-confidence regime where SAHI's cross-tile merge drowns in weak boxes. The one
that matters for search-and-rescue is *recall at a realistic operating threshold* — how many real
people the system finds. This measures that, for base vs fine-tuned detectors, full-frame vs SAHI.

    make sahi-recall

Perception-only (may import ultralytics/sahi; ADR-001). Needs the detect dataset + both checkpoints;
not in `make test`. The IoU and recall/precision maths are pure and unit-tested.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "runs"


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy ``box`` against an (N,4) array; empty -> empty."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + areas - inter + 1e-9)


def recall_precision(dets_per_img, gts_per_img, iou_thr: float) -> tuple[float, float]:
    """Match detections to GT (IoU >= ``iou_thr``); return (recall, precision) over all images."""
    gt_total = gt_hit = det_total = det_hit = 0
    for dets, gts in zip(dets_per_img, gts_per_img):
        for g in gts:
            gt_total += 1
            if len(dets) and iou(g, dets).max() >= iou_thr:
                gt_hit += 1
        for d in dets:
            det_total += 1
            if len(gts) and iou(d, gts).max() >= iou_thr:
                det_hit += 1
    recall = gt_hit / gt_total if gt_total else 0.0
    precision = det_hit / det_total if det_total else 0.0
    return recall, precision


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_gt(img_path: Path, labels_dir: Path) -> np.ndarray:
    w, h = Image.open(img_path).size
    out = []
    lp = labels_dir / f"{img_path.stem}.txt"
    if lp.exists():
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                _, xc, yc, bw, bh = (float(v) for v in parts)
                out.append(
                    [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
                )
    return np.array(out) if out else np.zeros((0, 4))


def _sample(images_dir: Path, counts: dict, full: bool = False) -> list[Path]:
    imgs = sorted(images_dir.glob("*.jpg"))
    if full:
        return imgs
    picked = []
    for src, n in counts.items():
        picked += [p for p in imgs if p.name.startswith(src)][: int(n)]
    return picked


def filter_boxes(boxes: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Keep boxes with score >= ``thr`` (so a sweep runs off one low-conf inference pass)."""
    if len(boxes) == 0:
        return boxes
    return boxes[np.asarray(scores) >= thr]


def _fullframe(weights, paths, conf, imgsz, device):
    """Return per-image (boxes Nx4, scores N)."""
    from ultralytics import YOLO

    m = YOLO(str(weights))
    out = []
    for p in paths:
        r = m.predict(str(p), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        if r.boxes is None:
            out.append((np.zeros((0, 4)), np.zeros(0)))
        else:
            out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
    return out


def _sahi(weights, paths, conf, imgsz, slice_px, overlap, device):
    """Return per-image (boxes Nx4, scores N) from SAHI tiled inference."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    m = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=conf,
        image_size=imgsz,
        device=device,
    )
    out = []
    for p in paths:
        res = get_sliced_prediction(
            str(p),
            m,
            slice_height=slice_px,
            slice_width=slice_px,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )
        boxes = [o.bbox.to_xyxy() for o in res.object_prediction_list]
        scores = [o.score.value for o in res.object_prediction_list]
        out.append((np.array(boxes) if boxes else np.zeros((0, 4)), np.array(scores)))
    return out


def run_sahi_eval(cfg) -> dict:
    images_dir = REPO_ROOT / cfg.val_images
    labels_dir = REPO_ROOT / cfg.val_labels
    device = _pick_device()
    paths = _sample(images_dir, dict(cfg.sample))
    gts = [_load_gt(p, labels_dir) for p in paths]
    n_gt = sum(len(g) for g in gts)
    conf, imgsz = float(cfg.conf), int(cfg.imgsz)
    iou_thr = float(cfg.iou_match)
    weights = {
        "base": REPO_ROOT / cfg.base_weights,
        "fine-tuned": REPO_ROOT / cfg.finetuned_weights,
    }

    rows = []
    for model_name, w in weights.items():
        ff = _fullframe(w, paths, conf, imgsz, device)
        sh = _sahi(w, paths, conf, imgsz, int(cfg.slice), float(cfg.overlap), device)
        for method, dets in (("full-frame", ff), ("SAHI", sh)):
            boxes = [b for b, _ in dets]  # already filtered at cfg.conf by the detector
            rec, prec = recall_precision(boxes, gts, iou_thr)
            rows.append({"model": model_name, "method": method, "recall": rec, "precision": prec})
            print(
                f"[sahi] {model_name:11s} {method:11s}  "
                f"recall {rec * 100:5.1f}%  precision {prec * 100:5.1f}%"
            )
    return {"rows": rows, "n_gt": n_gt, "n_img": len(paths), "conf": conf, "iou": iou_thr}


def run_sweep(cfg) -> dict:
    """Full-val recall/precision over a confidence sweep (one inference pass per config)."""
    images_dir = REPO_ROOT / cfg.val_images
    labels_dir = REPO_ROOT / cfg.val_labels
    device = _pick_device()
    paths = _sample(images_dir, dict(cfg.sample), full=bool(cfg.get("full_val", False)))
    gts = [_load_gt(p, labels_dir) for p in paths]
    n_gt = sum(len(g) for g in gts)
    base_conf, imgsz = float(cfg.base_conf), int(cfg.imgsz)
    iou_thr = float(cfg.iou_match)
    thresholds = [float(t) for t in cfg.sweep_thresholds]
    weights = {
        "base": REPO_ROOT / cfg.base_weights,
        "fine-tuned": REPO_ROOT / cfg.finetuned_weights,
    }

    curves: dict[str, list] = {}
    for model_name, w in weights.items():
        for method, fn in (
            ("full-frame", lambda w=w: _fullframe(w, paths, base_conf, imgsz, device)),
            (
                "SAHI",
                lambda w=w: _sahi(
                    w, paths, base_conf, imgsz, int(cfg.slice), float(cfg.overlap), device
                ),
            ),
        ):
            dets = fn()  # one inference pass at base_conf, keep all boxes+scores
            key = f"{model_name} · {method}"
            pts = []
            for thr in thresholds:
                boxes = [filter_boxes(b, s, thr) for b, s in dets]
                rec, prec = recall_precision(boxes, gts, iou_thr)
                pts.append({"thr": thr, "recall": rec, "precision": prec})
                print(
                    f"[sweep] {key:24s} thr {thr:.2f}: recall {rec * 100:5.1f}%  "
                    f"precision {prec * 100:5.1f}%"
                )
            curves[key] = pts
    return {"curves": curves, "n_gt": n_gt, "n_img": len(paths), "iou": iou_thr}


def plot_sweep(res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "base · full-frame": ("#9AA7B4", "o", "-"),
        "base · SAHI": ("#9AA7B4", "s", "--"),
        "fine-tuned · full-frame": ("#0C3B6E", "o", "-"),
        "fine-tuned · SAHI": ("#B85042", "s", "--"),
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for key, pts in res["curves"].items():
        c, mk, ls = styles.get(key, ("#888", "o", "-"))
        rec = [p["recall"] * 100 for p in pts]
        prec = [p["precision"] * 100 for p in pts]
        ax.plot(rec, prec, marker=mk, ls=ls, color=c, lw=2, label=key)
    ax.set_xlabel("recall (% of real survivors found)")
    ax.set_ylabel("precision (%)")
    ax.set_title(f"Recall–precision across a confidence sweep (full val, IoU {res['iou']})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sahi(res: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    labels = [f"{r['model']}\n{r['method']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(x - 0.2, [r["recall"] * 100 for r in rows], 0.4, label="recall", color="#1565C0")
    ax.bar(x + 0.2, [r["precision"] * 100 for r in rows], 0.4, label="precision", color="#B85042")
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r["recall"] * 100 + 1, f"{r['recall'] * 100:.0f}", ha="center", fontsize=8)
        ax.text(
            i + 0.2,
            r["precision"] * 100 + 1,
            f"{r['precision'] * 100:.0f}",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title(f"SAHI trades precision for recall (conf {res['conf']}, IoU {res['iou']})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs/perception/sahi.yaml")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"sahi_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("sweep", False)):
        print(f"[sweep] full-val sweep, IoU={cfg.iou_match}, thr={list(cfg.sweep_thresholds)}")
        res = run_sweep(cfg)
        with open(out_dir / "sahi_sweep.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["config", "threshold", "recall", "precision"])
            for key, pts in res["curves"].items():
                for p in pts:
                    wr.writerow([key, p["thr"], round(p["recall"], 4), round(p["precision"], 4)])
        plot_sweep(res, out_dir / "sahi_sweep.png")
        print(f"[sweep] {res['n_img']} imgs, {res['n_gt']} GT -> {out_dir.name}/sahi_sweep.png")
        return

    print(f"[sahi] operating point conf={cfg.conf} IoU={cfg.iou_match}")
    res = run_sahi_eval(cfg)
    with open(out_dir / "sahi.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["model", "method", "recall", "precision"])
        wr.writeheader()
        wr.writerows(res["rows"])
    plot_sahi(res, out_dir / "sahi.png")
    print(f"[sahi] {res['n_img']} imgs, {res['n_gt']} GT boxes -> wrote {out_dir}/sahi.{{png,csv}}")


if __name__ == "__main__":
    main()
