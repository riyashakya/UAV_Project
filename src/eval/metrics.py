"""Coordination metrics from an engine event log — Phase 9.

Pure functions over one run's result dict (from ``src.sim.engine.run``). No randomness, no I/O —
the runner calls these and aggregates them to mean ± 95% CI across seeds.
"""

from __future__ import annotations

import math


def compute_metrics(
    result: dict, n_cells: int, *, coverage_target: float = 0.9, complete_threshold: float = 0.95
) -> dict:
    """Mission metrics for one run.

    Returns coverage, time-to-``coverage_target`` (s, NaN if never reached), redundant-coverage
    ratio (surveys / unique cells; 1.0 = no waste), survivors found, total flight distance (m,
    a proxy from event positions), mission completion (coverage ≥ ``complete_threshold``), and
    the number of permanently-lost cells.
    """
    seen: set[int] = set()
    surveys = 0
    survivors = 0
    time_to_target = math.nan
    last_pos: dict[int, tuple[float, float]] = {}
    distance = 0.0

    for e in result["events"]:
        uav, pos = e["uav"], (e["x"], e["y"])
        if uav in last_pos:
            distance += math.dist(pos, last_pos[uav])
        last_pos[uav] = pos
        if e["event"] == "arrived":
            surveys += 1
            seen.add(e["cell"])
            if math.isnan(time_to_target) and len(seen) / n_cells >= coverage_target:
                time_to_target = e["t"]
            survivors += e.get("found", []).count("person")

    coverage = len(seen) / n_cells
    return {
        "coverage": coverage,
        "time_to_target_s": time_to_target,
        "redundant_ratio": surveys / len(seen) if seen else 0.0,
        "survivors_found": survivors,
        "total_distance_m": distance,
        "completed": float(coverage >= complete_threshold),
        "lost_cells": len(result.get("lost_cells", [])),
    }
