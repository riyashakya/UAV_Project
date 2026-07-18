# Claude Code task prompts

## Why your original prompt needs splitting, not just rewriting

Your prompt asked for four different things in one message: a full plan, an
implementation, the datasets, and a checklist. Claude Code will attempt all four and do
each one shallowly — you'll get a plausible-looking repo skeleton with stub functions
and a generic dataset list, and you'll spend longer fixing it than writing it.

The other problems:

| Issue in the original | Consequence |
|---|---|
| No acceptance criteria | Claude Code can't tell when a task is done, so it stops at "looks reasonable" |
| No tech stack pinned | It will pick for you. There is a real chance it reaches for AirSim and burns two weeks. |
| No non-goals | Scope creeps into flight controllers, dashboards, real-time |
| "simulate with drones used" is ambiguous | Reads as *real* drones on a bad day |
| No evaluation metrics | For an MSc this is the whole marked contribution, and it's the thing that gets skipped |
| One 200-word run-on | Constraints get lost in the middle; the model anchors on the last clause |

**The fix:** persistent context goes in `CLAUDE.md` (loaded automatically every session,
survives `/compact`). Tasks get pasted one at a time, each scoped to one module with a
testable definition of done.

Start each session with `claude` from the repo root, and run `/context` once to confirm
`CLAUDE.md` actually loaded.

---

## Phase 0 — Scaffold

> Set up the repo skeleton for this project. Python 3.10, `uv` for dependency
> management, Hydra for config, pytest for tests, ruff for lint.
>
> Create: the directory layout implied by CLAUDE.md, a `Makefile` with the targets
> listed there, `pyproject.toml` with pinned deps, `.gitignore` covering `data/`,
> `outputs/`, `*.pt`, and a `docs/adr/` folder with ADR-001 and ADR-002 written up from
> CLAUDE.md.
>
> Every module gets a `__init__.py` and a docstring stating its single responsibility.
> No implementation yet — stubs raising `NotImplementedError` are correct here.
>
> **Done when:** `make setup && make lint && make test` all pass on a clean clone, with
> zero tests collected but no collection errors.

---

## Phase 1 — Perception: dataset unification

> Build `src/perception/datasets/`.
>
> Write loaders that convert each source dataset to a unified taxonomy, plus the label
> mapping table. Sources and their real formats:
>
> - **VisDrone-DET** — YOLO format via Ultralytics auto-download (`data="VisDrone.yaml"`).
>   Map `pedestrian`+`people` → `person`; `car`+`van`+`truck`+`bus` → `vehicle`. Drop the rest.
> - **SARD** — VOC XML, 1,981 images, 6 pose subclasses. Collapse all poses → `person`.
>   Keep the pose in a metadata column; we will use it for stratified evaluation later.
> - **RescueNet** — pixel masks, 4,494 images at 3000×4000, 11 classes. Map
>   `Building-Major-Damage`+`Building-Total-Destruction` → `building_damaged`;
>   `Road-Blocked` → `road_blocked`; `Water` → `water`. Emit YOLO-seg polygons.
> - **FloodNet** — pixel masks. `building-flooded` → `building_damaged`;
>   `road-flooded` → `road_blocked`; `water` → `water`.
>
> Two output datasets, not one (see ADR-002): a detect set (person, vehicle) and a seg
> set (building_damaged, road_blocked, water).
>
> Write `docs/datasets.md` recording, per source: licence, image count, native
> resolution, capture platform, and the exact mapping applied. This is dissertation
> material — be precise.
>
> **Done when:** `python -m src.perception.datasets.build --dry-run` prints a per-class
> instance count table for both output sets, and tests verify the mapping is total
> (every source class is either mapped or explicitly listed as dropped, with no silent
> drops).

---

## Phase 2 — Perception: train and evaluate

> Implement `src/perception/train.py` and `src/perception/eval.py`.
>
> Train Model A (`yolo11s.pt`, detect) on the unified detect set and Model B
> (`yolo11s-seg.pt`) on the seg set. Configs under `configs/perception/`.
>
> Evaluation must report, for Model A:
> - mAP@50 and mAP@50-95, overall and per class
> - **AP stratified by COCO object size** (small/medium/large) — this is the headline number
> - Full-frame inference vs SAHI tiled inference (640 slices, 0.2 overlap) as an ablation
> - A per-source breakdown (VisDrone-only vs SARD-only vs combined) to expose the domain gap
>
> Use the val split for tuning; do not touch test until Phase 9.
>
> **Done when:** `make eval-perception` writes a results table to
> `outputs/perception/<timestamp>/` and I can see the SAHI-vs-full-frame delta on small
> objects. Report the numbers to me; do not tune to make them look better.

---

## Phase 3 — Detection cache

> Implement `src/perception/detect_cache.py` and `src/sim/oracle.py`.
>
> `detect_cache.py` runs both models over a scenario's imagery once and writes
> `data/cache/detections.parquet` with columns: `scenario, cell_id, class, confidence,
> lat, lon, bbox_utm, source_image, model`. Georeference using the image footprint
> metadata; if a dataset lacks geotags, synthesise a plausible grid layout and record
> that fact in a `synthetic_geo: bool` column — do not silently fake it.
>
> `oracle.py` exposes `get_detections(cell_id, rng) -> list[Detection]`, applying a
> configurable per-class false-negative rate and detection latency drawn from the config.
> This is the *only* place the simulator learns about the world.
>
> **Done when:** the cache builds for one scenario, `oracle.get_detections` is
> deterministic given a seed, and a test asserts `src/sim/` has no import path reaching
> `ultralytics` (walk the AST — I want this enforced, not documented).

---

## Phase 4 — Simulator core

> Implement `src/sim/world.py`, `src/sim/uav.py`, `src/sim/engine.py`.
>
> - `world.py` — grid over the scenario AOI, per-cell hazard state, and a 2D flow field
>   `u(x, y) -> (vx, vy)` loaded from config (analytic fields for now: uniform, channel,
>   radial).
> - `uav.py` — kinematics (constant cruise speed, instant turns are acceptable, document
>   it), plus an energy model: `P = P_hover + k*v²`. A UAV triggers return-to-home when
>   remaining energy < `1.3 × E_required_to_reach_base`. The 1.3 goes in config with a
>   comment.
> - `engine.py` — fixed-timestep loop, deterministic given a seed, headless, no plotting.
>
> **Done when:** a 4-UAV 60-minute scenario runs in under 2 seconds on CPU; two runs with
> the same seed produce byte-identical event logs; and a test confirms a UAV given a
> base 5 km away at the RTH threshold actually reaches base with energy to spare.

---

## Phase 5 — Partitioning and coverage

> Implement `src/coordination/partition.py` and `src/coordination/coverage.py`.
>
> - `partition.py` — divide the AOI into N sectors. Implement two strategies behind one
>   interface: `grid` (naive baseline) and `weighted_voronoi` (seeded by prior hazard
>   probability, so sectors are equal-*workload* not equal-*area*).
> - `coverage.py` — boustrophedon coverage path within a sector, parameterised by camera
>   footprint width and desired sidelap.
>
> **Done when:** for 4 UAVs on a uniform-priority AOI, weighted Voronoi sector workloads
> are within 5% of each other; coverage paths achieve >99% cell visitation; and a test
> catches the classic bug where the last row of a boustrophedon sweep is skipped when
> sector height isn't an integer multiple of footprint width.

---

## Phase 6 — Dynamic task reallocation

> Implement `src/coordination/allocation.py`. This is the core novel contribution — treat
> it accordingly.
>
> Auction-based reallocation (Contract Net style). Triggers:
> 1. A UAV hits its RTH threshold and abandons unfinished cells → those cells go to auction.
> 2. The oracle returns a high-priority detection (e.g. `person`) → surrounding cells are
>    upweighted and may be re-auctioned.
>
> Bid = `α·(travel cost) + β·(remaining energy penalty) + γ·(cell priority)`. Weights in
> config. Idle-or-finished UAVs bid; the auctioneer assigns to the lowest bidder.
>
> Cite the mechanism in a module docstring (Contract Net Protocol; Gerkey & Matarić's
> MRTA taxonomy — this is ST-SR-TA). The examiner will ask where this came from.
>
> Implement three baselines behind the same interface, because the comparison *is* the
> result: `single_uav`, `static_partition_no_realloc`, `random_walk`.
>
> **Done when:** in a scripted scenario where UAV-2's battery dies at t=20min with 30% of
> its sector unsurveyed, an idle UAV picks up those exact cells, and total coverage still
> reaches >95%. `static_partition_no_realloc` must *fail* that same test — if it passes,
> the test is wrong.

---

## Phase 7 — Flood drift prediction

> Implement `src/drift/advect.py`.
>
> Lagrangian particle advection for a survivor in flowing water. Given a detection at
> `(lat, lon, t0)` and the flow field, project the position distribution at `t0 + Δt`:
>
> - `dx/dt = u(x,y) · leeway_factor + turbulent diffusion (random walk, K_h from config)`
> - Monte Carlo N particles → 2D probability density → containment polygons at 50/90%
>
> This is deliberately modelled on how maritime SAR planning works (USCG SAROPS uses
> leeway + Monte Carlo containment). Say so in the docstring and cite it — it is the
> difference between "I invented a thing" and "I adapted an established method to a new
> domain", and the second one marks higher.
>
> Output feeds routing: a drifting survivor's search polygon, not a point.
>
> **Done when:** zero diffusion + uniform flow displaces the particle to exactly `v·Δt`
> (analytic test); containment area grows monotonically with Δt; and 90% containment
> empirically contains ≥90% of particles across 1,000 seeds.

---

## Phase 8 — Hazard-weighted rescue routing

> Implement `src/routing/graph.py` and `src/routing/safe_path.py`.
>
> - `graph.py` — build a road graph with OSMnx for the scenario AOI. **Cache the extract
>   to disk**; do not hit the Overpass API on every run.
> - Apply detections to the graph: `road_blocked` detections remove edges;
>   `water`/`building_damaged` proximity raises edge risk.
> - `safe_path.py` — edge weight `w = length · (1 + λ · risk)`. Sweep λ to produce a
>   Pareto front of (distance, cumulative risk) rather than one arbitrary compromise.
>
> **Done when:** for one scenario I get a Pareto front plot with ≥5 non-dominated routes,
> a naive shortest-path baseline shown on the same axes, and a test proving a route never
> traverses an edge marked `road_blocked`.

---

## Phase 9 — Evaluation harness

> Implement `src/eval/runner.py` and `src/eval/metrics.py`. Nothing else should be added
> to the system until this exists.
>
> Metrics: coverage % vs time, redundant-coverage ratio, time-to-90%-coverage,
> survivors-found vs time, mission completion under battery failure, route length,
> cumulative route risk.
>
> `make sweep` runs the grid: {4 baselines} × {N=1,2,4,6 UAVs} × {30 seeds} × {3
> scenarios}, writes tidy Parquet, and emits mean ± 95% CI. Add an ablation switching off
> reallocation and off priority upweighting independently.
>
> **Done when:** `make sweep` completes on CPU in under 10 minutes and produces a results
> table where I can read off whether adaptive reallocation beats static partitioning, and
> by how much, with confidence intervals. If it doesn't beat it, tell me plainly — a
> negative result I can explain is worth more than a number I can't defend.

---

## Phase 10 — 3D reconstruction (do this last, see plan §Risks)

> Implement `src/recon3d/` as a **comparative study**, not a feature.
>
> Run COLMAP (photogrammetry baseline), Nerfacto, and Splatfacto via nerfstudio on one
> multi-view UAV sequence. Report PSNR / SSIM / LPIPS, training wall-clock, peak VRAM.
>
> Read the plan's §Objective 4 note before starting — near-nadir survey imagery is a known
> hard case for NeRF/3DGS and photogrammetry may well win. That is a legitimate finding.
> Frame the module to *measure* that, not to make 3DGS look good.
>
> **Done when:** a table of the three methods on one scene, with an honest statement of
> which capture geometries each method suits.
