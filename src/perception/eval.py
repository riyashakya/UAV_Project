"""Evaluate Model A (detect) — Phase 2b.

Compares **full-frame** vs **SAHI** tiled inference on the detect val set, and reports COCO
metrics including **AP stratified by object size** (small/medium/large) plus a **per-source**
breakdown (VisDrone-val vs SARD-val). Inference runs once per method; the per-source views are
just COCO-eval on image-id subsets, so no retraining and no extra inference.

    python -m src.perception.eval --config configs/perception/eval.yaml
    make eval-perception

Perception-only module (may import ultralytics/sahi; ADR-001 keeps these out of src/sim).
Imports are lazy so ``--help`` / test collection don't need torch.
"""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "perception"

# COCO summary stats order from COCOeval.stats
STAT_KEYS = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def find_latest_weights() -> Path:
    runs = sorted(glob.glob(str(OUTPUT_ROOT / "detect_*" / "weights" / "best.pt")), key=lambda p: p)
    if not runs:
        raise FileNotFoundError("No detect_*/weights/best.pt under outputs/perception/.")
    return Path(runs[-1])


def build_coco_gt(images_dir: Path, labels_dir: Path, classes: list[str]) -> tuple[dict, dict]:
    """Build a COCO ground-truth dict from YOLO val labels. Returns (coco_gt, filename->id)."""
    images, annotations = [], []
    name_to_id: dict[str, int] = {}
    ann_id = 1
    for img_path in sorted(images_dir.glob("*.jpg")):
        img_id = len(images) + 1
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        name_to_id[img_path.name] = img_id

        label = labels_dir / f"{img_path.stem}.txt"
        if not label.exists():
            continue
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = (float(v) for v in parts)
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            bw_abs, bh_abs = bw * w, bh * h
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x, y, bw_abs, bh_abs],
                    "area": bw_abs * bh_abs,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    return coco_gt, name_to_id


def run_fullframe(weights: Path, image_paths, name_to_id, *, imgsz, conf, max_det, device):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    preds = []
    for path in image_paths:
        r = model.predict(
            str(path), imgsz=imgsz, conf=conf, max_det=max_det, device=device, verbose=False
        )[0]
        img_id = name_to_id[path.name]
        b = r.boxes
        if b is None:
            continue
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()
        clss = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), sc, cl in zip(xyxy, confs, clss):
            preds.append(
                {
                    "image_id": img_id,
                    "category_id": int(cl) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(sc),
                }
            )
    return preds


def run_sahi(weights: Path, image_paths, name_to_id, *, imgsz, conf, slice_px, overlap, device):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=conf,
        image_size=imgsz,
        device=device,
    )
    preds = []
    for path in image_paths:
        result = get_sliced_prediction(
            str(path),
            det_model,
            slice_height=slice_px,
            slice_width=slice_px,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )
        img_id = name_to_id[path.name]
        for obj in result.object_prediction_list:
            x, y, w, h = obj.bbox.to_xywh()
            preds.append(
                {
                    "image_id": img_id,
                    "category_id": int(obj.category.id) + 1,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(obj.score.value),
                }
            )
    return preds


def coco_eval(coco_gt: dict, preds: list, img_ids: list[int] | None = None) -> dict:
    """Run COCOeval and return the 12 summary stats. Empty preds -> zeros."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with tempfile.TemporaryDirectory() as td:
        gt_path = Path(td) / "gt.json"
        dt_path = Path(td) / "dt.json"
        gt_path.write_text(json.dumps(coco_gt))
        dt_path.write_text(json.dumps(preds))
        gt = COCO(str(gt_path))
        if not preds:
            return dict.fromkeys(STAT_KEYS, 0.0)
        dt = gt.loadRes(str(dt_path))
        e = COCOeval(gt, dt, "bbox")
        if img_ids is not None:
            e.params.imgIds = img_ids
        e.evaluate()
        e.accumulate()
        e.summarize()
        return dict(zip(STAT_KEYS, [float(x) for x in e.stats]))


def _subset_ids(coco_gt: dict, source: str) -> list[int]:
    return [im["id"] for im in coco_gt["images"] if im["file_name"].startswith(source + "_")]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 2b perception evaluation.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/perception/eval.yaml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="eval only N images (smoke test)")
    parser.add_argument("--no-sahi", action="store_true")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    device = pick_device(args.device)
    weights = (
        Path(args.weights)
        if args.weights
        else (_resolve(cfg.detect_weights) if cfg.get("detect_weights") else find_latest_weights())
    )
    classes = list(cfg.classes)

    images_dir = _resolve(cfg.val_images)
    labels_dir = _resolve(cfg.val_labels)
    print(f"[eval] weights={weights}\n[eval] device={device}  val={images_dir}")
    coco_gt, name_to_id = build_coco_gt(images_dir, labels_dir, classes)

    image_paths = [images_dir / im["file_name"] for im in coco_gt["images"]]
    if args.limit:
        image_paths = image_paths[: args.limit]
        keep = {p.name for p in image_paths}
        coco_gt["images"] = [im for im in coco_gt["images"] if im["file_name"] in keep]
        ids = {im["id"] for im in coco_gt["images"]}
        coco_gt["annotations"] = [a for a in coco_gt["annotations"] if a["image_id"] in ids]

    print(f"[eval] {len(image_paths)} val images, {len(coco_gt['annotations'])} GT boxes")

    methods = {}
    print("\n[eval] running full-frame inference ...")
    methods["full_frame"] = run_fullframe(
        weights,
        image_paths,
        name_to_id,
        imgsz=int(cfg.imgsz),
        conf=float(cfg.conf),
        max_det=int(cfg.max_det),
        device=device,
    )
    if not args.no_sahi:
        print("[eval] running SAHI tiled inference (slower) ...")
        methods["sahi"] = run_sahi(
            weights,
            image_paths,
            name_to_id,
            imgsz=int(cfg.imgsz),
            conf=float(cfg.conf),
            slice_px=int(cfg.sahi.slice),
            overlap=float(cfg.sahi.overlap),
            device=device,
        )

    # Evaluate: overall + per source, per method.
    results: dict[str, dict] = {}
    subsets = {"all": None}
    for src in cfg.get("sources", []):
        subsets[src] = _subset_ids(coco_gt, src)
    for method, preds in methods.items():
        for subset_name, ids in subsets.items():
            key = f"{method}/{subset_name}"
            print(f"\n===== {key} =====")
            results[key] = coco_eval(coco_gt, preds, ids)

    # Emit table + files.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["AP", "AP50", "AP_small", "AP_medium", "AP_large", "AR100"]
    lines = ["", f"{'method/subset':<22}" + "".join(f"{c:>11}" for c in cols)]
    lines.append("-" * len(lines[-1]))
    for key, stats in results.items():
        lines.append(f"{key:<22}" + "".join(f"{stats[c]:>11.4f}" for c in cols))
    table = "\n".join(lines)
    print("\n" + table)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "results.txt").write_text(table + "\n")
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")
    print(f"\n[eval] wrote {out_dir}/results.{{json,txt}}")

    if "sahi" in methods:
        d = results["sahi/all"]["AP_small"] - results["full_frame/all"]["AP_small"]
        print(f"[eval] SAHI vs full-frame  AP_small delta = {d:+.4f}")


if __name__ == "__main__":
    main()
