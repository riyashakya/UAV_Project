"""Detection oracle — the only bridge from perception to the simulator (Phase 3).

When a UAV enters a cell, the oracle returns that cell's cached detections, applying a
configurable per-class **false-negative rate** and a **detection latency** drawn from config.
This is the *only* place the simulator learns about the world (ADR-001).

Reads ``data/cache/detections.parquet`` with pandas — it must never import
``ultralytics``/``torch``/``sahi`` (enforced by ``tests/test_sim_isolation.py``). Given the same
seeded ``np.random.Generator``, results are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Detection:
    """A single cached detection served to the simulator."""

    cell_id: int
    cls: str
    confidence: float
    lat_wgs84: float
    lon_wgs84: float
    bbox_utm: tuple[float, float, float, float]  # (easting_m, northing_m, width_m, height_m)
    model: str
    source_image: str
    latency_s: float  # simulated delay before this detection is reported


class Oracle:
    """Serves cached detections for a scenario with configurable false-negative + latency noise.

    Args:
        detections: parquet path or a preloaded DataFrame with the cache schema.
        scenario: scenario name to filter on.
        false_negative_rate: per-class probability a real detection is missed.
        latency_s: ``(min, max)`` uniform range for per-detection reporting latency.
    """

    def __init__(
        self,
        detections: Path | str | pd.DataFrame,
        scenario: str,
        *,
        false_negative_rate: dict[str, float] | None = None,
        latency_s: tuple[float, float] = (0.0, 0.0),
    ):
        df = detections if isinstance(detections, pd.DataFrame) else pd.read_parquet(detections)
        df = df[df["scenario"] == scenario]
        # Group once, stable order, so a seeded generator reproduces the same draws.
        self._by_cell: dict[int, list[dict]] = {}
        for cell_id, group in df.sort_values(["cell_id", "source_image", "class"]).groupby(
            "cell_id", sort=True
        ):
            self._by_cell[int(cell_id)] = group.to_dict("records")
        self.scenario = scenario
        self.false_negative_rate = false_negative_rate or {}
        self.latency_s = latency_s

    @property
    def cell_ids(self) -> list[int]:
        return sorted(self._by_cell)

    def get_detections(self, cell_id: int, rng: np.random.Generator) -> list[Detection]:
        """Return the cell's detections, dropping some per the false-negative rate and drawing a
        latency for each survivor. Deterministic for a given ``rng`` state."""
        out: list[Detection] = []
        for rec in self._by_cell.get(int(cell_id), []):
            cls = rec["class"]
            fn = self.false_negative_rate.get(cls, 0.0)
            # Draw for every detection (even at fn=0) so the rng stream is order-stable.
            missed = rng.random() < fn
            latency = float(rng.uniform(*self.latency_s))
            if missed:
                continue
            out.append(
                Detection(
                    cell_id=int(rec["cell_id"]),
                    cls=cls,
                    confidence=float(rec["confidence"]),
                    lat_wgs84=float(rec["lat"]),
                    lon_wgs84=float(rec["lon"]),
                    bbox_utm=tuple(float(v) for v in rec["bbox_utm"]),
                    model=rec["model"],
                    source_image=rec["source_image"],
                    latency_s=latency,
                )
            )
        return out
