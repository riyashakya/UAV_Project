# Pipeline map — where each step lives (for presenting)

Point to this when you present: every stage of *area → deploy drones → cover → detect → drift →
route → evaluate*, mapped to the code and the command that runs it. Everything is **simulation
only** (no real drones), and perception is **decoupled** (offline, cached — see
[`adr/ADR-001`](adr/ADR-001-decouple-perception-from-coordination.md)).

| # | Pipeline step | Lives in | Run it | Output |
|---|---|---|---|---|
| 0 | **Area + coordinates** (grid AOI, synthetic-geo anchor) | [`configs/scenario/flood_a.yaml`](../configs/scenario/flood_a.yaml), [`src/sim/world.py`](../src/sim/world.py) | — (config) | grid geometry + `origin_wgs84` |
| 1 | **Perception — train** (Model A detect, Model B segment) | [`src/perception/train.py`](../src/perception/train.py) | `make train-a` / `make train-b` | weights in `outputs/perception/` |
| 2 | **Perception — evaluate** (AP by object size, SAHI) | [`src/perception/eval.py`](../src/perception/eval.py) | `make eval-perception` | tables in `outputs/perception/` |
| 3 | **Detection cache** (offline YOLO pass → ADR-001 bridge) | [`src/perception/detect_cache.py`](../src/perception/detect_cache.py) | `make cache-dets` | `data/cache/detections.parquet` (class, **confidence**, lat/lon) |
| 4 | **Oracle** (serves cached detections to the sim, with noise) | [`src/sim/oracle.py`](../src/sim/oracle.py) | (used by the sim) | `Detection` objects |
| 5 | **Deploy drones + energy** (world, UAV, timed loop) | [`src/sim/{world,uav,engine}.py`](../src/sim/engine.py) | `make sim SCEN=flood_a SEED=0` | `outputs/runs/…/events.json` |
| 6 | **Partition the area** (grid vs weighted-Voronoi) | [`src/coordination/partition.py`](../src/coordination/partition.py) | (used by the sim) | one sector per drone |
| 7 | **Cover a sector** (lawnmower **vs spiral** — comparison) | [`src/coordination/coverage.py`](../src/coordination/coverage.py) | `make coverage-compare` | `outputs/runs/coverage_*/coverage_compare.png` |
| 8 | **Detect survivors** (who/what, with confidence) | oracle + [`src/perception/`](../src/perception/) | seen in `make sim` `found=…` | per-cell detections + confidence |
| 9 | **Coordinate + re-plan on failure** (auction + 3 baselines) | [`src/coordination/allocation.py`](../src/coordination/allocation.py) | `make sim` / `make sweep` | reallocation events |
| 10 | **Predict survivor drift** (SAROPS particle advection) | [`src/drift/advect.py`](../src/drift/advect.py) | `make drift` | `outputs/drift/…/drift.png` |
| 11 | **Drift → re-task, quantified (RQ4)** | [`src/eval/rq4.py`](../src/eval/rq4.py) | `make rq4` | `outputs/runs/rq4_*/rq4.png` (88% vs 0% located) |
| 12 | **Safest/fastest route** (hazard-weighted Pareto) | [`src/routing/{graph,safe_path}.py`](../src/routing/safe_path.py) | `make routes` | `outputs/routing/…/pareto.png` |
| 13 | **Route on a REAL street map** (OpenStreetMap) | [`src/routing/graph.py`](../src/routing/graph.py) (`build_osm_road_graph`) | `make routes-osm` | `outputs/routing/osm_*/map.png` |
| 14 | **Evaluation — the evidence** (1,800-run Monte Carlo) | [`src/eval/{runner,metrics}.py`](../src/eval/runner.py) | `make sweep` | `outputs/runs/sweep_*/{results.png,summary.csv,headline.txt}` |
| 15 | **Mission animation** (drones + failure, live) | [`src/sim/animate.py`](../src/sim/animate.py) | `make animate` | `outputs/runs/mission_*/mission.gif` |
| — | **Interactive browser demo** (opt-in tool, ADR-003) | [`webapp/`](../webapp/) | `make web` → http://127.0.0.1:8000 | live mission + survivors + routes + Google Maps link |

## The one thing to say clearly in the viva

**Perception is offline and cached, not a live video feed** (ADR-001). The AI ran once on real
labelled imagery → `detections.parquet`; the simulator reads that through the oracle when a drone
enters a cell. This is a *strength* — it lets coordination run hundreds of seeded Monte-Carlo runs
in minutes while the AI is scored honestly against real ground truth — and it makes perception and
coordination two clean, separately-evaluated contributions.

## One-command tour

```bash
make demo            # mission → sweep → drift → routing, end to end (~7 s)
```
