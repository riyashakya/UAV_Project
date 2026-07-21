"""Build the detection cache — Phase 3.

Runs Model A (detect) and Model B (segment) over a scenario's imagery **once** and writes
``data/cache/detections.parquet`` with columns
``scenario, cell_id, class, confidence, lat, lon, bbox_utm, source_image, model, synthetic_geo``.

Our datasets have no geotags, so georeferencing is **synthetic**: images are laid out on a grid
of equal cells anchored at a config lat/lon, and each detection's position is placed within its
cell's footprint. ``synthetic_geo=True`` records this honestly (never silently faked).

This is the last GPU stage (ADR-001); everything downstream reads the parquet via the oracle.

    python -m src.perception.detect_cache --scenario configs/scenario/flood_a.yaml
    make cache-dets
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "perception"
CACHE_PATH = REPO_ROOT / "data" / "cache" / "detections.parquet"

_METERS_PER_DEG_LAT = 111_320.0  # flat-earth approximation (synthetic geo only)


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def enu_to_wgs84(east_m: float, north_m: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Flat-earth local-ENU metres → lat/lon. Valid only for the small synthetic AOIs here."""
    lat = lat0 + north_m / _METERS_PER_DEG_LAT
    lon = lon0 + east_m / (_METERS_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat, lon


def georeference_rows(
    raw_dets: list[dict],
    *,
    scenario: str,
    cell_id: int,
    n_cols: int,
    cell_size_m: float,
    origin_lat: float,
    origin_lon: float,
    source_image: str,
) -> list[dict]:
    """Map a cell's normalised detections into georeferenced parquet rows (pure function).

    ``raw_dets`` items: ``{cls, confidence, cx, cy, w, h, model}`` with box in normalised
    image coords ([0, 1]). Row 0 / col 0 is the north-west cell; north decreases with row.
    """
    row, col = divmod(cell_id, n_cols)
    rows: list[dict] = []
    for d in raw_dets:
        east_m = (col + d["cx"]) * cell_size_m
        north_m = -(row + d["cy"]) * cell_size_m
        lat, lon = enu_to_wgs84(east_m, north_m, origin_lat, origin_lon)
        rows.append(
            {
                "scenario": scenario,
                "cell_id": cell_id,
                "class": d["cls"],
                "confidence": float(d["confidence"]),
                "lat": lat,
                "lon": lon,
                "bbox_utm": [east_m, north_m, d["w"] * cell_size_m, d["h"] * cell_size_m],
                "source_image": source_image,
                "model": d["model"],
                "synthetic_geo": True,
            }
        )
    return rows


def _find_weights(prefix: str) -> Path:
    runs = sorted(glob.glob(str(OUTPUT_ROOT / f"{prefix}_*" / "weights" / "best.pt")))
    if not runs:
        raise FileNotFoundError(f"No {prefix}_*/weights/best.pt under outputs/perception/.")
    return Path(runs[-1])


def _collect_images(cfg) -> list[Path]:
    paths: list[Path] = []
    for src in cfg.sources:
        d = _resolve(src["dir"])
        paths += sorted(d.glob(src.get("glob", "*.jpg")))[: int(src.get("n", 10**9))]
    max_cells = int(cfg.grid.rows) * int(cfg.grid.cols)
    return paths[:max_cells]


def _run_models(image_paths, weights_a, weights_b, device):
    """Run both models on each image → {image_path: [raw det dicts]}. Uses ultralytics (ADR-001)."""
    from ultralytics import YOLO  # lazy, perception-only

    model_a, model_b = YOLO(str(weights_a)), YOLO(str(weights_b))
    out: dict[Path, list[dict]] = {}
    for path in image_paths:
        dets: list[dict] = []
        for model, tag in ((model_a, "A"), (model_b, "B")):
            r = model.predict(str(path), device=device, verbose=False)[0]
            if r.boxes is None:
                continue
            xywhn = r.boxes.xywhn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (cx, cy, w, h), sc, cl in zip(xywhn, confs, clss):
                dets.append(
                    {
                        "cls": r.names[int(cl)],
                        "confidence": float(sc),
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": float(w),
                        "h": float(h),
                        "model": tag,
                    }
                )
        out[path] = dets
    return out


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


def build_cache(scenario_path: Path, *, weights_a=None, weights_b=None, device=None) -> Path:
    cfg = OmegaConf.load(scenario_path)
    weights_a = Path(weights_a) if weights_a else _find_weights("detect")
    weights_b = Path(weights_b) if weights_b else _find_weights("segment")
    device = pick_device(device)

    image_paths = _collect_images(cfg)
    print(f"[cache] scenario={cfg.name}  cells={len(image_paths)}  device={device}")
    print(f"[cache] model A={weights_a.name}  model B={weights_b.name}")
    per_image = _run_models(image_paths, weights_a, weights_b, device)

    all_rows: list[dict] = []
    for cell_id, path in enumerate(image_paths):
        all_rows += georeference_rows(
            per_image[path],
            scenario=cfg.name,
            cell_id=cell_id,
            n_cols=int(cfg.grid.cols),
            cell_size_m=float(cfg.cell_size_m),
            origin_lat=float(cfg.origin_wgs84.lat),
            origin_lon=float(cfg.origin_wgs84.lon),
            source_image=path.name,
        )

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(CACHE_PATH, index=False)
    by_class = df["class"].value_counts().to_dict() if not df.empty else {}
    print(f"[cache] wrote {len(df)} detections -> {CACHE_PATH}\n[cache] by class: {by_class}")
    return CACHE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the detection cache for a scenario.")
    parser.add_argument("--scenario", default=str(REPO_ROOT / "configs/scenario/flood_a.yaml"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    build_cache(Path(args.scenario), device=args.device)


if __name__ == "__main__":
    main()
