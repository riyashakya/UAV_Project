"""Offline perception pipeline (GPU).

Single responsibility: turn real labelled UAV imagery into evaluated detections cached to
disk. Covers dataset unification (Phase 1), YOLO11 training/eval (Phase 2), and detection
caching (Phase 3).

This subpackage — and only this subpackage — may import ``ultralytics``/``torch``/``sahi``.
Nothing under ``src.sim`` may reach it at import time (ADR-001).
"""
