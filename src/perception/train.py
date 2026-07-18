"""Train Model A (detect) and Model B (segment) — Phase 2.

Single responsibility: fit the two YOLO11 models on the unified datasets produced by
``src.perception.datasets`` and write weights + training curves to ``outputs/perception/``.
Configs live under ``configs/perception/``.

Stub: implemented in Phase 2.
"""

from __future__ import annotations


def main() -> None:
    """CLI entry point for training (``make ...`` / ``python -m src.perception.train``)."""
    raise NotImplementedError("Phase 2: YOLO11 training is not implemented yet.")


if __name__ == "__main__":
    main()
