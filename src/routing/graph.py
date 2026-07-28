"""Road graph construction + hazard application — Phase 8.

Two ways to get the road graph:

* ``road_graph_from_world`` — a synthetic 4-connected lattice over the (synthetic-geo) grid.
  Offline and testable; matches the synthetic scenario.
* ``build_osm_road_graph`` — a real OSMnx extract for a bounding box, **cached to disk** so the
  Overpass API is hit at most once (CLAUDE.md non-goal: no live network at run time). Lazy import.

``apply_detections`` folds segmentation hazards into the graph: ``road_blocked`` **removes** a
cell's edges; ``water`` / ``building_damaged`` **raise** the risk of edges touching that cell.

CPU-only (networkx); never imports the perception detector (ADR-001).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx

from src.sim.world import World


def road_graph_from_world(world: World) -> nx.Graph:
    """A 4-connected road lattice: one node per cell, edges between adjacent cells."""
    g = nx.Graph()
    for cell in world.cells:
        g.add_node(cell.id, x=cell.center_xy[0], y=cell.center_xy[1], risk=0.0)
    for cell in world.cells:
        if cell.col < world.cols - 1:
            g.add_edge(cell.id, cell.id + 1, length=world.cell_size_m, risk=0.0)
        if cell.row < world.rows - 1:
            g.add_edge(cell.id, cell.id + world.cols, length=world.cell_size_m, risk=0.0)
    return g


def apply_detections(
    graph: nx.Graph,
    detections,
    *,
    risk_weights: dict[str, float] | None = None,
) -> nx.Graph:
    """Fold hazards into the graph (in place, also returned).

    ``detections``: iterable of items each exposing a class and a cell id — objects with
    ``.cls``/``.cell_id`` (oracle ``Detection``), ``(cls, cell_id)`` tuples, or ``{class,
    cell_id}`` dicts. ``road_blocked`` removes the cell's edges; ``water``/``building_damaged``
    add node risk.
    """
    risk_weights = risk_weights or {"water": 1.0, "building_damaged": 0.7}
    node_risk: dict[int, float] = defaultdict(float)
    blocked: set[int] = set()

    for det in detections:
        cls, cell = _cls_cell(det)
        if cls == "road_blocked":
            blocked.add(cell)
        elif cls in risk_weights and cell in graph:
            node_risk[cell] += risk_weights[cls]

    for node, risk in node_risk.items():
        if node in graph.nodes:
            graph.nodes[node]["risk"] = graph.nodes[node].get("risk", 0.0) + risk
    # edge risk = mean of its endpoints' node risk (computed before removing blocked edges)
    for u, v in graph.edges:
        graph[u][v]["risk"] = 0.5 * (
            graph.nodes[u].get("risk", 0.0) + graph.nodes[v].get("risk", 0.0)
        )
    graph.remove_edges_from([(u, v) for u, v in list(graph.edges) if u in blocked or v in blocked])
    return graph


def apply_flood_corridor(
    graph: nx.Graph,
    world: World,
    *,
    columns,
    risk_north: float,
    risk_south: float,
    decay: float = 0.55,
) -> nx.Graph:
    """Overlay a contiguous flooded corridor across ``columns``, with crossing risk decaying
    north→south so hazard-avoidance costs distance.

    Scattered point-detections on a dense grid don't constrain routing (a free equal-length detour
    always exists); a real flood is a *barrier*. Risk decays **geometrically** with row (``decay``)
    so the distance-vs-risk trade-off is *convex* — every crossing row is a distinct Pareto vertex a
    weighted-sum λ-sweep can recover (a linear gradient makes the points collinear and the front
    collapses to its two endpoints). A documented, illustrative flood structure.
    """
    for cell in world.cells:
        if cell.col in columns:
            graph.nodes[cell.id]["risk"] = risk_south + (risk_north - risk_south) * (
                decay**cell.row
            )
    for u, v in graph.edges:
        graph[u][v]["risk"] = 0.5 * (
            graph.nodes[u].get("risk", 0.0) + graph.nodes[v].get("risk", 0.0)
        )
    return graph


def _cls_cell(det) -> tuple[str, int]:
    if hasattr(det, "cls") and hasattr(det, "cell_id"):
        return det.cls, int(det.cell_id)
    if isinstance(det, dict):
        return det["class"], int(det["cell_id"])
    return det[0], int(det[1])


def detections_from_cache(parquet_path: Path, scenario: str) -> list[tuple[str, int]]:
    """Read ``(class, cell_id)`` hazard/detection pairs from the detection cache for a scenario."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    df = df[df["scenario"] == scenario]
    return [(row["class"], int(row["cell_id"])) for _, row in df.iterrows()]


def build_osm_road_graph(
    bbox_wgs84: tuple[float, float, float, float], cache_path: Path
) -> nx.Graph:
    """Real road graph from OSMnx for ``(north, south, east, west)``, cached to ``cache_path``.

    Lazy-imports osmnx (the ``geo`` extra). Loads the cache if present so Overpass is hit once.
    """
    cache_path = Path(cache_path)
    import osmnx as ox

    if cache_path.exists():
        return ox.load_graphml(cache_path)
    graph = ox.graph_from_bbox(bbox=bbox_wgs84, network_type="drive")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, cache_path)
    return graph
