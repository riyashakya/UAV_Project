"""Train Model A (detect) or Model B (segment) locally with Ultralytics YOLO11 — Phase 2.

Reads a perception config (``configs/perception/model_a.yaml`` or ``model_b.yaml``), trains the
single "most suitable" model (``yolo11s`` / ``yolo11s-seg`` by default), and writes weights +
curves to ``outputs/perception/<timestamp>/``. Auto-selects the device: Apple-Silicon GPU
(``mps``) → CUDA → CPU. Every hyperparameter comes from the config; CLI flags override for
quick experiments (e.g. a smoke test).

    python -m src.perception.train --config configs/perception/model_a.yaml
    python -m src.perception.train --config configs/perception/model_b.yaml

This module (perception) may import ``ultralytics``; ``src/sim`` may not (ADR-001). The import
is lazy so ``--help`` and unit-test collection do not require torch.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "perception"


def pick_device(explicit: str | None) -> str | int:
    """Return the training device: explicit override, else mps → cuda → cpu."""
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch not installed / probe failed → CPU
        pass
    return "cpu"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a YOLO11 perception model locally.")
    p.add_argument("--config", required=True, help="perception config YAML (model_a / model_b)")
    p.add_argument("--model", default=None, help="override the model weights (e.g. yolo11m.pt)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="batch size; auto-batch is CUDA-only")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--device", default=None, help="mps | cpu | 0 (cuda); default: auto")
    p.add_argument(
        "--fraction", type=float, default=1.0, help="fraction of train data (smoke tests)"
    )
    p.add_argument("--name", default=None, help="run name; default <task>_<model>_<timestamp>")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = OmegaConf.load(args.config)

    model_weights = args.model or cfg.model
    task = cfg.task
    data = str((REPO_ROOT / cfg.data).resolve())
    epochs = args.epochs if args.epochs is not None else int(cfg.get("final_epochs", 100))
    imgsz = args.imgsz or int(cfg.imgsz)
    patience = args.patience if args.patience is not None else int(cfg.get("patience", 20))
    seed = int(cfg.get("seed", 0))
    device = pick_device(args.device)

    # Auto-batch (-1) only works on CUDA; fall back to a safe fixed batch on mps/cpu.
    batch = args.batch if args.batch is not None else int(cfg.get("batch", -1))
    if batch < 0 and device != 0:
        batch = 16

    stem = Path(model_weights).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"{task}_{stem}_{ts}"

    if not Path(data).exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data}\n"
            "Build it first, e.g. `make build-datasets "
            'ARGS="--sources visdrone sard --copy-images"`.'
        )

    print(
        f"[train] model={model_weights} task={task} device={device} imgsz={imgsz} "
        f"batch={batch} epochs={epochs} fraction={args.fraction}\n[train] data={data}"
    )

    from ultralytics import YOLO  # lazy: perception-only import (ADR-001)

    model = YOLO(model_weights)
    model.train(
        data=data,
        task=task,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        seed=seed,
        device=device,
        fraction=args.fraction,
        project=str(OUTPUT_ROOT),
        name=run_name,
        exist_ok=False,  # never overwrite a previous run (CLAUDE.md)
        plots=True,
    )

    metrics = model.val(device=device)  # keep val on the GPU (MPS), not CPU
    box = metrics.seg if task == "segment" else metrics.box
    run_dir = OUTPUT_ROOT / run_name
    OmegaConf.save(OmegaConf.create(dict(cfg)), run_dir / "resolved_config.yaml")
    print(
        f"\n[train] done → {run_dir}\n"
        f"[train] weights: {run_dir / 'weights' / 'best.pt'}\n"
        f"[train] mAP@50={box.map50:.4f}  mAP@50-95={box.map:.4f}"
    )


if __name__ == "__main__":
    main()
