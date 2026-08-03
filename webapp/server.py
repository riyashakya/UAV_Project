"""Local web demo server — an OPT-IN visualisation tool, not part of the evaluated pipeline.

Runs the existing simulation engine on demand and returns the mission as JSON; the browser
(``index.html``) animates it live. Standard-library HTTP only (no Flask, no new dependency), bound
to localhost. See ``docs/adr/ADR-003-web-demo-tool.md`` for why this deviates from the CLAUDE.md
"no web dashboards" non-goal and how it is scoped so the dissertation's offline evaluation is
unaffected. Coordination-side only — never imports the perception detector (ADR-001).

    make web        # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # run as `python webapp/server.py` from anywhere

from omegaconf import OmegaConf  # noqa: E402
from src.coordination.allocation import STRATEGIES, Coordinator  # noqa: E402
from src.sim.engine import run  # noqa: E402
from src.sim.uav import UAV, UAVParams  # noqa: E402
from src.sim.world import World  # noqa: E402

HERE = Path(__file__).resolve().parent


def _survivor_reveal(events) -> dict[str, float]:
    reveal: dict[str, float] = {}
    for e in events:
        if e.get("event") == "arrived" and "person" in e.get("found", []):
            reveal.setdefault(str(int(e["cell"])), float(e["t"]))
    return reveal


def _survivors(detections) -> dict:
    """Summarise the detected people: total, mean confidence, and a per-cell breakdown — including
    the (georeferenced) lat/lon of the highest-confidence detection in each cell (the pinpoint)."""
    persons = [d for d in detections if d["cls"] == "person"]
    by_cell: dict[int, dict] = {}
    for d in persons:
        e = by_cell.setdefault(
            int(d["cell"]),
            {"count": 0, "best": -1.0, "sum": 0.0, "t": d["t"], "lat": None, "lon": None},
        )
        e["count"] += 1
        e["sum"] += d["confidence"]
        if d["confidence"] > e["best"]:  # keep the pinpoint of the strongest detection
            e["best"] = d["confidence"]
            e["lat"], e["lon"] = round(float(d["lat"]), 6), round(float(d["lon"]), 6)
    cells = [
        {
            "cell": c,
            "count": e["count"],
            "best_conf": round(e["best"], 3),
            "mean_conf": round(e["sum"] / e["count"], 3),
            "first_t": e["t"],
            "lat": e["lat"],
            "lon": e["lon"],
        }
        for c, e in by_cell.items()
    ]
    cells.sort(key=lambda x: -x["best_conf"])
    total = len(persons)
    mean_conf = round(sum(d["confidence"] for d in persons) / total, 3) if total else 0.0
    return {"total": total, "mean_conf": mean_conf, "cells": cells}


def _to_latlon(origin, x_east_m, y_south_m):
    """Grid metres → WGS84, using the scenario's synthetic-geo anchor (origin = the grid's NW
    corner). Equirectangular approximation — fine over a ~km-scale area."""
    import math

    lat = float(origin[0]) - y_south_m / 111320.0
    lon = float(origin[1]) + x_east_m / (111320.0 * math.cos(math.radians(float(origin[0]))))
    return [round(lat, 6), round(lon, 6)]


def _gmaps_dir_url(latlon_waypoints) -> str:
    """A Google Maps directions URL through the given (lat, lon) waypoints (Maps snaps to real
    roads between them). Capped to keep the URL short."""
    pts = latlon_waypoints
    if len(pts) > 10:  # keep endpoints, evenly sample the middle
        idx = [round(i * (len(pts) - 1) / 9) for i in range(10)]
        pts = [pts[i] for i in sorted(set(idx))]
    return "https://www.google.com/maps/dir/" + "/".join(f"{a},{b}" for a, b in pts)


def _response_plan(world, focus_cell, det_scenario, seed, origin, flood_overlay=None) -> dict:
    """Perception → drift → routing for the top survivor: project their drift probability zones,
    then route from base to where they are predicted to drift (not the stale detection cell).
    ``flood_overlay`` overlays a contiguous flood barrier so fastest and safest diverge."""
    import numpy as np
    from src.drift.advect import drift_search_region
    from src.routing.graph import (
        apply_detections,
        apply_flood_corridor,
        detections_from_cache,
        road_graph_from_world,
    )
    from src.routing.safe_path import pareto_front

    d = OmegaConf.load(REPO_ROOT / "configs/drift/default.yaml")
    region = drift_search_region(
        world.cell_center(focus_cell),
        world.flow,
        rng=np.random.default_rng(int(seed)),
        n_particles=int(d.n_particles),
        horizon_s=float(d.horizon_s),
        dt=float(d.timestep_s),
        leeway_factor=float(d.leeway_factor),
        k_h=float(d.k_h),
        containment_levels=(0.5, 0.9),
    )

    def poly_xy(p):
        if p.is_empty or p.geom_type != "Polygon":
            return []
        return [[round(x, 1), round(y, 1)] for x, y in p.exterior.coords]

    drift = {
        "centroid": [round(float(v), 1) for v in region["centroid"]],
        "contain50": poly_xy(region["containment"][0.5]),
        "contain90": poly_xy(region["containment"][0.9]),
        "area90_ha": round(region["areas_m2"][0.9] / 1e4, 2),
        "horizon_min": round(float(d.horizon_s) / 60),
    }

    # rescue target = the cell nearest the drift centroid (search where they will be, not were)
    cx, cy = region["centroid"]
    size = world.cell_size_m
    col = min(world.cols - 1, max(0, round(cx / size - 0.5)))
    row = min(world.rows - 1, max(0, round(cy / size - 0.5)))
    target_cell = int(row * world.cols + col)

    g = road_graph_from_world(world)
    cache = REPO_ROOT / "data" / "cache" / "detections.parquet"
    if cache.exists():
        apply_detections(g, detections_from_cache(cache, det_scenario))
    if flood_overlay is not None:  # heavy-flood variant: a graded barrier so routes diverge
        apply_flood_corridor(
            g,
            world,
            columns=list(flood_overlay["columns"]),
            risk_north=float(flood_overlay["risk_north"]),
            risk_south=float(flood_overlay["risk_south"]),
        )
    routes = None
    gmaps_url = None
    base = 0
    if target_cell != base and g.has_node(target_cell):
        lambdas = [0.0] + list(np.logspace(-2, np.log10(50.0), 30))
        front = pareto_front(g, base, target_cell, lambdas)
        if front:

            def route(r):
                pts = [world.cell_center(c) for c in r["path"]]
                return {
                    "path_xy": [[round(x, 1), round(y, 1)] for x, y in pts],
                    "length_m": round(r["length"]),
                    "risk": round(r["risk"], 1),
                }

            routes = {"fastest": route(front[0]), "safest": route(front[-1])}
            # Google Maps directions through the SAFEST (recommended) route, in real coordinates
            safest_pts = [world.cell_center(c) for c in front[-1]["path"]]
            gmaps_url = _gmaps_dir_url([_to_latlon(origin, x, y) for x, y in safest_pts])
    return {
        "focus_cell": int(focus_cell),
        "target_cell": target_cell,
        "target_latlon": _to_latlon(origin, *world.cell_center(target_cell)),
        "drift": drift,
        "routes": routes,
        "gmaps_url": gmaps_url,
    }


def _scenario_names() -> list[str]:
    return sorted(p.stem for p in (REPO_ROOT / "configs/scenario").glob("*.yaml"))


def simulate(
    *, seed, strategy, fail_uav, fail_at, drift_retask, n_uavs=None, scenario="flood_a"
) -> dict:
    """Run one mission with the given controls; return everything the browser needs to draw it."""
    if scenario not in _scenario_names():
        scenario = "flood_a"
    scen = OmegaConf.load(REPO_ROOT / f"configs/scenario/{scenario}.yaml")
    world_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/world.yaml")
    uav_cfg = OmegaConf.load(REPO_ROOT / "configs/sim/uav.yaml")
    coord_cfg = OmegaConf.load(REPO_ROOT / "configs/coordination/default.yaml")

    # a scenario may override the flow field (World reads flow from the world config)
    if scen.get("flow") is not None:
        world_cfg = OmegaConf.merge(world_cfg, {"flow": scen.flow})
    world = World.from_configs(scen, world_cfg)
    params = UAVParams.from_cfg(uav_cfg)
    if n_uavs is None:
        n_uavs = int(world_cfg.get("n_uavs", 4))
    n_uavs = max(1, min(8, int(n_uavs)))  # the engine supports any count; clamp for the demo
    uavs = [UAV(i, params, world.base_xy) for i in range(n_uavs)]
    bw = coord_cfg.allocation.bid_weights

    drift_params = {}
    if drift_retask:
        d = OmegaConf.load(REPO_ROOT / "configs/drift/default.yaml")
        drift_params = {
            "n_particles": int(d.n_particles),
            "horizon_s": float(d.horizon_s),
            "dt": float(d.timestep_s),
            "leeway_factor": float(d.leeway_factor),
            "k_h": float(d.k_h),
        }
    coord = Coordinator(
        strategy,
        world,
        n_uavs,
        bid_weights=(float(bw.travel), float(bw.energy), float(bw.priority)),
        priority_boost=float(coord_cfg.allocation.priority_boost),
        drift_retask=bool(drift_retask),
        drift_params=drift_params,
        drift_level=float(coord_cfg.allocation.get("drift_level", 0.9)),
        drift_seed=int(seed),
    )

    # a variant scenario may reuse another scenario's real cached detections
    det_scenario = str(scen.get("detections_scenario", scen.name))
    oracle = None
    cache = REPO_ROOT / "data" / "cache" / "detections.parquet"
    if cache.exists():
        from src.sim.oracle import Oracle

        ocfg = OmegaConf.load(REPO_ROOT / "configs/sim/oracle.yaml")
        oracle = Oracle(
            cache,
            det_scenario,
            false_negative_rate=dict(ocfg.false_negative_rate),
            latency_s=tuple(ocfg.latency_s),
        )

    fail_map = {int(fail_uav): float(fail_at)} if fail_at is not None else None
    result = run(
        world,
        uavs,
        coordinator=coord,
        seed=int(seed),
        duration_s=float(world_cfg.get("duration_min", 60)) * 60.0,
        dt=float(world_cfg.get("timestep_s", 5.0)),
        oracle=oracle,
        fail_at=fail_map,
        record_trajectory=True,
        record_detections=True,
    )
    flood_overlay = scen.get("flood_overlay")
    if flood_overlay is not None:
        flood_overlay = OmegaConf.to_container(flood_overlay, resolve=True)
    origin = (float(scen.origin_wgs84.lat), float(scen.origin_wgs84.lon))
    survivors = _survivors(result.get("detections", []))
    plan = (
        _response_plan(
            world, survivors["cells"][0]["cell"], det_scenario, seed, origin, flood_overlay
        )
        if survivors["cells"]
        else None
    )
    return {
        "params": {
            "seed": int(seed),
            "strategy": strategy,
            "scenario": scen.name,
            "drift_retask": bool(drift_retask),
            "fail": ({"uav": int(fail_uav), "t": float(fail_at)} if fail_map else None),
        },
        "world": {
            "rows": world.rows,
            "cols": world.cols,
            "cell_size": world.cell_size_m,
            "n_cells": world.n_cells,
            "base_xy": list(world.base_xy),
            "n_uavs": n_uavs,
        },
        "trajectory": result["trajectory"],
        "survivor_reveal": _survivor_reveal(result["events"]),
        "coverage": result["coverage"],
        "found": result["found_total"],
        "uav_end": result["uav_end"],
        "lost_cells": result["lost_cells"],
        "survivors": survivors,
        "plan": plan,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away / cancelled — don't crash the server

    def do_GET(self):  # noqa: N802 (stdlib naming)
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif route.path == "/api/meta":
            body = json.dumps(
                {"strategies": list(STRATEGIES), "scenarios": _scenario_names()}
            ).encode()
            self._send(200, body, "application/json")
        elif route.path == "/api/run":
            self._run(parse_qs(route.query))
        else:
            self._send(404, b"not found", "text/plain")

    def _run(self, q):
        try:
            fail_on = q.get("fail", ["1"])[0] == "1"
            out = simulate(
                seed=int(q.get("seed", ["0"])[0]),
                strategy=q.get("strategy", ["auction"])[0],
                scenario=q.get("scenario", ["flood_a"])[0],
                n_uavs=int(q.get("n_uavs", ["4"])[0]),
                fail_uav=int(q.get("fail_uav", ["2"])[0]),
                fail_at=(float(q.get("fail_at", ["100"])[0]) if fail_on else None),
                drift_retask=q.get("drift", ["0"])[0] == "1",
            )
            self._send(200, json.dumps(out).encode(), "application/json")
        except Exception as exc:  # surface the error to the browser rather than a blank 500
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")

    def log_message(self, *_):  # keep the console quiet
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    # Single-threaded on purpose: numpy/pandas in a spawned request thread can segfault on macOS
    # (Error 139). A local single-user demo has no need for concurrency; each run takes ~0.2 s.
    httpd = HTTPServer(("127.0.0.1", port), Handler)
    print(f"[web] Multi-UAV mission visualiser running -> http://127.0.0.1:{port}")
    print("[web] open that URL in a browser; Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped.")


if __name__ == "__main__":
    main()
