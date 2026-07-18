"""Hazard-weighted rescue routing (CPU).

Single responsibility: build a road graph for the AOI, fold detections into edge weights
(``road_blocked`` removes edges; ``water``/``building_damaged`` proximity raises risk), and
search for safe rescue paths — reported as a Pareto front of (distance, cumulative risk)
rather than one arbitrary compromise. Phase 8.
"""
