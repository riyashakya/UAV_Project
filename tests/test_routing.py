"""Hazard-weighted routing: risk application, blocked-edge avoidance, Pareto front (Phase 8)."""

from __future__ import annotations

import networkx as nx
import numpy as np
from src.routing.graph import apply_detections, apply_flood_corridor, road_graph_from_world
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
