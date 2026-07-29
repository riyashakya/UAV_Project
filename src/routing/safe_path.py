"""Safe-path search + Pareto front — Phase 8.

Edge weight ``w = length · (1 + λ·risk)``. Sweeping λ from 0 (shortest, hazard-blind) upward
traces the **Pareto front** of (total distance, cumulative risk exposure) — a menu of routes, not
one arbitrary compromise. A route can never traverse a ``road_blocked`` edge because those edges
are removed from the graph (Phase 8 test).

    make routes            # Pareto front for the flood_a scenario -> outputs/routing/

CPU-only (networkx); matplotlib is imported lazily only for the figure.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import networkx as nx

from src.routing.graph import (
    apply_detections,
    apply_flood_corridor,
    apply_flood_zone,
    build_osm_road_graph,
    detections_from_cache,
    nearest_node,
    road_graph_from_world,
    simple_graph_from_osmnx,
)
from src.sim.world import World

REPO_ROOT = Path(__file__).resolve().parents[2]


def _weight(lam: float):
    def w(u, v, data):
        return data["length"] * (1.0 + lam * data.get("risk", 0.0))

    return w


def route_metrics(graph: nx.Graph, path: list[int]) -> tuple[float, float]:
    """(total length m, cumulative risk exposure = Σ length·risk) along ``path``."""
    length = risk = 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = graph[u][v]
        length += d["length"]
        risk += d["length"] * d.get("risk", 0.0)
    return length, risk


def safe_path(graph: nx.Graph, source: int, target: int, lam: float) -> dict:
    """Min-``(length·(1+λ·risk))`` path. Raises ``nx.NetworkXNoPath`` if disconnected."""
    path = nx.shortest_path(graph, source, target, weight=_weight(lam))
    length, risk = route_metrics(graph, path)
    return {"lambda": lam, "path": tuple(path), "length": length, "risk": risk}


def shortest_path(graph: nx.Graph, source: int, target: int) -> dict:
    """Hazard-blind shortest path (the naive baseline) = ``safe_path`` at λ = 0."""
    return safe_path(graph, source, target, 0.0)


def pareto_front(graph: nx.Graph, source: int, target: int, lambdas) -> list[dict]:
    """Sweep λ, keep the non-dominated routes on (length, risk), sorted by length."""
    routes: dict[tuple, dict] = {}
    for lam in lambdas:
        try:
            r = safe_path(graph, source, target, lam)
        except nx.NetworkXNoPath:
            continue
        # key by the (length, risk) trade-off point: two distinct paths that reach the
        # same point are one Pareto vertex, so keep the first (smallest-λ) representative.
        key = (round(r["length"], 6), round(r["risk"], 6))
        routes.setdefault(key, r)
    pts = list(routes.values())
    front = [
        r
        for r in pts
        if not any(
            o["path"] != r["path"]
            and o["length"] <= r["length"]
            and o["risk"] <= r["risk"]
            and (o["length"] < r["length"] or o["risk"] < r["risk"])
            for o in pts
        )
    ]
    return sorted(front, key=lambda r: r["length"])


def plot_pareto(front: list[dict], baseline: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs = [r["risk"] for r in front]
    ys = [r["length"] for r in front]
    ax.plot(xs, ys, "-o", color="#1C7293", label="Pareto front (safe routes)")
    ax.scatter(
        [baseline["risk"]],
        [baseline["length"]],
        color="#B85042",
        zorder=5,
        s=70,
        label="Naive shortest path",
    )
    ax.set_xlabel("Cumulative risk exposure  (Σ length·risk)")
    ax.set_ylabel("Route distance (m)")
    ax.set_title("Rescue routing: distance vs risk trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_osm_routes(graph: nx.Graph, front: list[dict], baseline: dict, out_path: Path) -> None:
    """Draw the real street network faintly, shade the flooded nodes, and overlay the naive
    shortest path (red) and the safest Pareto route (blue)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for u, v in graph.edges:  # base network
        ax.plot(
            [graph.nodes[u]["x"], graph.nodes[v]["x"]],
            [graph.nodes[u]["y"], graph.nodes[v]["y"]],
            color="#C9D6E3",
            lw=0.6,
            zorder=1,
        )
    flooded = [n for n in graph.nodes if graph.nodes[n].get("risk", 0.0) > 0]
    if flooded:
        ax.scatter(
            [graph.nodes[n]["x"] for n in flooded],
            [graph.nodes[n]["y"] for n in flooded],
            c=[graph.nodes[n]["risk"] for n in flooded],
            cmap="Blues",
            s=14,
            zorder=2,
            label="flooded (node risk)",
        )

    def _draw(path, color, label):
        ax.plot(
            [graph.nodes[n]["x"] for n in path],
            [graph.nodes[n]["y"] for n in path],
            color=color,
            lw=2.6,
            zorder=4,
            label=label,
        )

    _draw(baseline["path"], "#B85042", "naive shortest path")
    _draw(front[-1]["path"], "#1C7293", "safest route")
    src, tgt = baseline["path"][0], baseline["path"][-1]
    ax.scatter(
        [graph.nodes[src]["x"], graph.nodes[tgt]["x"]],
        [graph.nodes[src]["y"], graph.nodes[tgt]["y"]],
        color="#0C3B6E",
        s=60,
        zorder=5,
    )
    ax.set_title("Rescue routing on a real street network")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _lambdas(lambda_max: float, n: int) -> list[float]:
    import numpy as np

    return [0.0] + list(np.logspace(-2, np.log10(lambda_max), max(1, n - 1)))


def main() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(REPO_ROOT / "configs/routing/default.yaml")
    scenario = OmegaConf.load(REPO_ROOT / "configs/scenario/flood_a.yaml")
    world_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/world.yaml")
    world = World.from_configs(scenario, world_cfg)

    graph = road_graph_from_world(world)
    cache = REPO_ROOT / "data" / "cache" / "detections.parquet"
    dets = detections_from_cache(cache, scenario.name) if cache.exists() else []
    apply_detections(graph, dets, risk_weights=dict(cfg.risk))  # real hazards (road_blocked, etc.)
    fc = cfg.demo.flood_corridor
    apply_flood_corridor(
        graph,
        world,
        columns=list(fc.columns),
        risk_north=float(fc.risk_north),
        risk_south=float(fc.risk_south),
    )

    src, tgt = int(cfg.demo.source_cell), int(cfg.demo.target_cell)
    lambdas = _lambdas(float(cfg.pareto.lambda_max), int(cfg.pareto.n_lambda))
    front = pareto_front(graph, src, tgt, lambdas)
    baseline = shortest_path(graph, src, tgt)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "routing" / f"{scenario.name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pareto.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["lambda", "length_m", "risk_exposure", "n_hops"])
        for r in front:
            wr.writerow(
                [f"{r['lambda']:.3f}", f"{r['length']:.1f}", f"{r['risk']:.1f}", len(r["path"]) - 1]
            )
    plot_pareto(front, baseline, out_dir / "pareto.png")

    print(f"[routes] scenario={scenario.name}  {src}->{tgt}  hazards={len(dets)}")
    print(f"[routes] {len(front)} non-dominated routes on the Pareto front")
    print(
        f"[routes] shortest: {baseline['length']:.0f} m, risk {baseline['risk']:.1f}  |  "
        f"safest: {front[-1]['length']:.0f} m, risk {front[-1]['risk']:.1f}"
    )
    print(f"[routes] wrote {out_dir}/pareto.{{png,csv}}")


def main_osm() -> None:
    """Same Pareto routing, but on a REAL cached OSMnx street network (configs/routing/osm.yaml)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(REPO_ROOT / "configs/routing/osm.yaml")
    bbox = tuple(float(v) for v in cfg.area.bbox)
    cache = REPO_ROOT / "data" / "cache" / "osm" / f"{cfg.area.name}.graphml"
    graph = simple_graph_from_osmnx(build_osm_road_graph(bbox, cache))
    for z in cfg.flood_zones:
        apply_flood_zone(
            graph,
            lon_min=float(z.lon_min),
            lon_max=float(z.lon_max),
            lat_min=float(z.lat_min),
            lat_max=float(z.lat_max),
            risk_peak=float(z.risk_peak),
        )
    src = nearest_node(graph, float(cfg.demo.source[0]), float(cfg.demo.source[1]))
    tgt = nearest_node(graph, float(cfg.demo.target[0]), float(cfg.demo.target[1]))
    lambdas = _lambdas(float(cfg.pareto.lambda_max), int(cfg.pareto.n_lambda))
    front = pareto_front(graph, src, tgt, lambdas)
    baseline = shortest_path(graph, src, tgt)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "routing" / f"osm_{cfg.area.name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pareto.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["lambda", "length_m", "risk_exposure", "n_hops"])
        for r in front:
            wr.writerow(
                [f"{r['lambda']:.3f}", f"{r['length']:.1f}", f"{r['risk']:.1f}", len(r["path"]) - 1]
            )
    plot_pareto(front, baseline, out_dir / "pareto.png")
    plot_osm_routes(graph, front, baseline, out_dir / "map.png")

    print(
        f"[routes-osm] area={cfg.area.name}  "
        f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}"
    )
    print(f"[routes-osm] {len(front)} non-dominated routes on the Pareto front")
    print(
        f"[routes-osm] shortest: {baseline['length']:.0f} m, risk {baseline['risk']:.1f}  |  "
        f"safest: {front[-1]['length']:.0f} m, risk {front[-1]['risk']:.1f}"
    )
    print(f"[routes-osm] wrote {out_dir}/pareto.{{png,csv}} + map.png")


if __name__ == "__main__":
    import sys

    (main_osm if len(sys.argv) > 1 and sys.argv[1] == "osm" else main)()
