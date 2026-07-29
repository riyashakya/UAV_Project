"""Hazard-weighted routing: risk application, blocked-edge avoidance, Pareto front (Phase 8)."""

from __future__ import annotations

import networkx as nx
import numpy as np
from src.routing.graph import (
    apply_detections,
    apply_flood_corridor,
    apply_flood_zone,
    nearest_node,
    road_graph_from_world,
    simple_graph_from_osmnx,
)
from src.routing.safe_path import pareto_front, route_metrics, safe_path, shortest_path
from src.sim.world import World


def test_graph_from_world_is_a_4connected_lattice():
    g = road_graph_from_world(World(rows=3, cols=3, cell_size_m=100.0))
    assert g.number_of_nodes() == 9
    assert g.number_of_edges() == 12  # 6 horizontal + 6 vertical
    assert g[0][1]["length"] == 100.0


def test_water_raises_risk_and_road_blocked_removes_edges():
    g = road_graph_from_world(World(rows=3, cols=3, cell_size_m=100.0))
    apply_detections(g, [("water", 4), ("road_blocked", 0)], risk_weights={"water": 1.0})
    # cell 4 (centre) is risky -> its edges carry risk
    assert g[4][1]["risk"] > 0
    # cell 0's edges are gone
    assert not g.has_edge(0, 1) and not g.has_edge(0, 3)


def test_route_never_traverses_a_road_blocked_edge():
    """Phase 8 acceptance: a blocked cell's edges are removed, so no route can use them."""
    world = World(rows=4, cols=4, cell_size_m=100.0)
    g = road_graph_from_world(world)
    apply_detections(g, [("road_blocked", 5)])  # block an interior cell
    route = safe_path(g, 0, 15, lam=0.0)["path"]
    assert 5 not in route  # detoured around the blocked cell
    assert route[0] == 0 and route[-1] == 15  # still a valid start->end route
    # every edge of the route exists and none touches the blocked cell
    for u, v in zip(route[:-1], route[1:]):
        assert g.has_edge(u, v)
        assert u != 5 and v != 5


def _tradeoff_graph():
    g = nx.Graph()
    g.add_edge("s", "a", length=1.0, risk=10.0)  # short but risky
    g.add_edge("a", "t", length=1.0, risk=10.0)
    g.add_edge("s", "b", length=3.0, risk=0.0)  # long but safe
    g.add_edge("b", "t", length=3.0, risk=0.0)
    return g


def test_baseline_is_shortest_ignoring_risk():
    base = shortest_path(_tradeoff_graph(), "s", "t")
    assert base["path"] == ("s", "a", "t")  # the length-2 route despite its risk
    assert base["length"] == 2.0
    assert base["risk"] == 20.0  # 1*10 + 1*10


def test_pareto_front_trades_distance_for_risk():
    g = _tradeoff_graph()
    front = pareto_front(g, "s", "t", [0.0, 0.1, 1.0, 5.0, 50.0])
    assert len(front) == 2  # the short-risky and the long-safe route, both non-dominated
    assert front[0]["length"] < front[-1]["length"]  # sorted by length
    assert front[0]["risk"] > front[-1]["risk"]  # ...and risk falls as distance rises
    # the safest route has zero risk
    assert front[-1]["risk"] == 0.0


def test_route_metrics_sum_along_path():
    g = _tradeoff_graph()
    length, risk = route_metrics(g, ["s", "b", "t"])
    assert length == 6.0 and risk == 0.0


def _osm_like():
    """A tiny OSMnx-shaped MultiDiGraph: directed + parallel edges, GraphML-style string lengths,
    WGS84 lon/lat on each node. Lets the real-map graph logic be tested without hitting Overpass."""
    g = nx.MultiDiGraph()
    g.add_node(1, x=-0.10, y=51.52)  # NW
    g.add_node(2, x=-0.09, y=51.52)  # NE
    g.add_node(3, x=-0.10, y=51.51)  # SW
    g.add_node(4, x=-0.09, y=51.51)  # SE
    g.add_edge(1, 2, length="100")  # string length, as GraphML round-trips it
    g.add_edge(2, 1, length="100")  # reverse direction -> same undirected pair
    g.add_edge(1, 2, length="80")  # a shorter parallel edge: this one should win
    g.add_edge(1, 3, length="120")
    g.add_edge(2, 4, length="120")
    g.add_edge(3, 4, length="100")
    return g


def test_simple_graph_from_osmnx_collapses_parallel_edges():
    g = simple_graph_from_osmnx(_osm_like())
    assert not g.is_directed()
    assert g.number_of_nodes() == 4
    assert g[1][2]["length"] == 80.0  # shortest of the parallel 1<->2 spans, coerced to float
    assert g.nodes[1]["x"] == -0.10 and g.nodes[1]["risk"] == 0.0


def test_apply_flood_zone_raises_node_and_edge_risk():
    g = simple_graph_from_osmnx(_osm_like())
    # flood the northern row (lat 51.52) only -> nodes 1 and 2
    apply_flood_zone(g, lon_min=-0.11, lon_max=-0.08, lat_min=51.515, lat_max=51.525, risk_peak=5.0)
    assert g.nodes[1]["risk"] == 5.0 and g.nodes[3]["risk"] == 0.0
    assert g[1][2]["risk"] == 5.0  # both endpoints flooded
    assert g[1][3]["risk"] == 2.5  # one endpoint flooded -> mean of the two


def test_nearest_node_snaps_to_closest():
    g = simple_graph_from_osmnx(_osm_like())
    assert nearest_node(g, -0.089, 51.509) == 4  # closest to the SE node


def test_flood_corridor_gives_a_multipoint_pareto_front():
    """Phase 8 acceptance: a flood barrier yields >= 5 non-dominated distance-vs-risk routes."""
    world = World(rows=6, cols=6, cell_size_m=200.0)
    g = road_graph_from_world(world)
    apply_flood_corridor(g, world, columns=[2, 3], risk_north=6.0, risk_south=0.5)
    lambdas = [0.0] + list(np.logspace(-2, np.log10(50.0), 40))
    front = pareto_front(g, 0, 5, lambdas)  # west base -> east target, across the corridor
    assert len(front) >= 5
    assert front[0]["length"] <= front[-1]["length"]  # sorted by length
    assert front[0]["risk"] >= front[-1]["risk"]  # ...safest route is the longest, lowest-risk
