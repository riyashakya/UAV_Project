"""Detection oracle — the only bridge from perception to the simulator (Phase 3).

Single responsibility: given a ``cell_id`` and an ``np.random.Generator``, return the cached
detections for that cell, applying a configurable per-class false-negative rate and detection
latency. This is the *only* place the simulator learns about the world (ADR-001).

Reads ``data/cache/detections.parquet``; must never import ``ultralytics``/``torch``/``sahi``.
Deterministic given a seed.

Stub: implemented in Phase 3.
"""

from __future__ import annotations


class Oracle:
    """Serves cached detections to the simulator with configurable noise. Stub (Phase 3)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("Phase 3: detection oracle is not implemented yet.")

    def get_detections(self, cell_id: int, rng: object) -> list:
        """Return cached detections for ``cell_id`` with FN/latency noise. Stub (Phase 3)."""
        raise NotImplementedError("Phase 3: detection oracle is not implemented yet.")
