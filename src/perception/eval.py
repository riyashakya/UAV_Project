"""Evaluate perception — Phase 2.

Single responsibility: score Model A / Model B on the val split and emit the results tables
to ``outputs/perception/<timestamp>/``. Reports mAP@50 and mAP@50-95 (overall + per class),
**AP stratified by COCO object size** (the headline number), a full-frame vs SAHI ablation,
and a per-source breakdown (VisDrone-only vs SARD-only vs combined) to expose the domain gap.

Never tune to make numbers look better; report them as measured (CLAUDE.md).

Stub: implemented in Phase 2.
"""

from __future__ import annotations


def main() -> None:
    """CLI entry point (``make eval-perception``)."""
    raise NotImplementedError("Phase 2: perception evaluation is not implemented yet.")


if __name__ == "__main__":
    main()
