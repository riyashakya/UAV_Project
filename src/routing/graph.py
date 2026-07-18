"""Road graph construction — Phase 8.

Single responsibility: build the AOI road graph with OSMnx and **cache the extract to disk**
(never hit the Overpass API on every run — CLAUDE.md non-goal: no live network at run time).
Apply detections: ``road_blocked`` removes edges; ``water``/``building_damaged`` proximity
raises edge risk.

Stub: implemented in Phase 8.
"""

from __future__ import annotations


def build_graph(*args: object, **kwargs: object) -> object:
    """Build (and disk-cache) the hazard-weighted road graph. Stub (Phase 8)."""
    raise NotImplementedError("Phase 8: road graph is not implemented yet.")
