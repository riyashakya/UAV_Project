"""Dataset unification (Phase 1).

Convert each source dataset (VisDrone, SARD, RescueNet, FloodNet) into two unified datasets
under one taxonomy (ADR-002): a detect set (``person``, ``vehicle``) and a seg set
(``building_damaged``, ``road_blocked``, ``water``). The mapping is total — every native
class is mapped or explicitly dropped, enforced by tests.

Entry point: ``python -m src.perception.datasets.build [--dry-run]``.
"""

from .taxonomy import SourceMapping, UnifiedTaxonomy, load_taxonomy

__all__ = ["SourceMapping", "UnifiedTaxonomy", "load_taxonomy"]
