"""Slice the unified detect set into 640x640 tiles — slicing-aided fine-tuning (Phase 2b fix).

Phase 2b found that naive SAHI *reduced* AP: the detector trained on whole (downscaled) frames, but
SAHI feeds 640x640 **slices** at inference, so objects appear at a scale the model never trained on.
This builds a sliced training set — tiles at the SAHI slice size, plus (by default) the original
images — so a fine-tune sees both scales and SAHI inference matches. YOLO format in and out.

Pillow only (no torch), so it can run before the perception extra is installed.

    python -m src.perception.slice_dataset --data data/unified/detect/data.yaml \
        --out data/unified/detect_sliced

Defaults follow the SAHI settings in CLAUDE.md (640 slices, 0.2 overlap).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_boxes(label_path: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    """YOLO normalized (cls cx cy bw bh) -> absolute pixel (cls x1 y1 x2 y2)."""
    out = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append(
                (cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            )
    return out


def _tile_origins(extent: int, size: int, step: int) -> list[int]:
    if extent <= size:
        return [0]
    xs = list(range(0, extent - size + 1, step))
    if xs[-1] != extent - size:
        xs.append(extent - size)  # a final tile flush with the far edge (no gap)
    return xs


def _slice_one(img: Image.Image, boxes, size: int, overlap: float, min_vis: float):
    """Yield (crop, [ (cls, cx, cy, bw, bh) normalized to the crop ])."""
    w, h = img.size
    step = max(1, int(size * (1 - overlap)))
    for y0 in _tile_origins(h, size, step):
        for x0 in _tile_origins(w, size, step):
            tw, th = min(size, w - x0), min(size, h - y0)
            labels = []
            for cls, bx1, by1, bx2, by2 in boxes:
                ix1, iy1 = max(bx1, x0), max(by1, y0)
                ix2, iy2 = min(bx2, x0 + tw), min(by2, y0 + th)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                orig = (bx2 - bx1) * (by2 - by1)
                if orig <= 0 or (ix2 - ix1) * (iy2 - iy1) / orig < min_vis:
                    continue  # too much of the box was cut off by the tile edge
                cx = ((ix1 + ix2) / 2 - x0) / tw
                cy = ((iy1 + iy2) / 2 - y0) / th
                labels.append((cls, cx, cy, (ix2 - ix1) / tw, (iy2 - iy1) / th))
            yield img.crop((x0, y0, x0 + tw, y0 + th)), labels


def slice_split(src, split, out, size, overlap, min_vis, keep_empty, include_orig, limit, rng):
    img_dir = src / "images" / split
    lbl_dir = src / "labels" / split
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit:
        imgs = imgs[:limit]
    n_tiles = n_kept = 0
    for ip in imgs:
        try:
            img = Image.open(ip).convert("RGB")
        except Exception:
            continue
        boxes = _read_boxes(lbl_dir / f"{ip.stem}.txt", *img.size)
        if include_orig:  # keep the full frame too, so the fine-tune sees both scales
            shutil.copy(ip, out / "images" / split / ip.name)
            lp = lbl_dir / f"{ip.stem}.txt"
            if lp.exists():
                shutil.copy(lp, out / "labels" / split / f"{ip.stem}.txt")
        for i, (crop, labels) in enumerate(_slice_one(img, boxes, size, overlap, min_vis)):
            n_tiles += 1
            if not labels and rng.random() > keep_empty:
                continue  # drop most empty tiles to keep the set from exploding
            n_kept += 1
            stem = f"{ip.stem}_t{i}"
            crop.save(out / "images" / split / f"{stem}.jpg", quality=90)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "".join(
                    f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for c, cx, cy, bw, bh in labels
                )
            )
    return len(imgs), n_tiles, n_kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Slice a YOLO detect dataset for slicing-aided fine-tuning."
    )
    ap.add_argument("--data", default="data/unified/detect/data.yaml")
    ap.add_argument("--out", default="data/unified/detect_sliced")
    ap.add_argument("--size", type=int, default=640, help="slice size (SAHI 640; CLAUDE.md)")
    ap.add_argument("--overlap", type=float, default=0.2, help="tile overlap ratio (CLAUDE.md)")
    ap.add_argument(
        "--min-vis",
        type=float,
        default=0.3,
        help="keep a box if >= this fraction stays in the tile",
    )
    ap.add_argument(
        "--keep-empty", type=float, default=0.05, help="probability of keeping a tile with no boxes"
    )
    ap.add_argument(
        "--no-originals", action="store_true", help="tiles only (default also keeps full frames)"
    )
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_cfg = OmegaConf.load(REPO_ROOT / args.data)
    src = (REPO_ROOT / args.data).parent
    out = REPO_ROOT / args.out
    rng = np.random.default_rng(args.seed)
    for split in ("train", "val"):
        if not (src / "images" / split).exists():
            continue
        n_img, n_tiles, n_kept = slice_split(
            src,
            split,
            out,
            args.size,
            args.overlap,
            args.min_vis,
            args.keep_empty,
            not args.no_originals,
            args.limit,
            rng,
        )
        print(
            f"[slice] {split}: {n_img} images -> {n_tiles} tiles, "
            f"kept {n_kept} (+ originals={not args.no_originals})"
        )

    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "nc": int(src_cfg.nc),
        "names": list(src_cfg.names),
    }
    OmegaConf.save(OmegaConf.create(data_yaml), out / "data.yaml")
    print(f"[slice] wrote {out}/data.yaml")


if __name__ == "__main__":
    main()
