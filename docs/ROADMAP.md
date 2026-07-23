# Roadmap & phase status

At-a-glance tracker of the plan (phases from `TASK_PROMPTS.md`, weeks from `PROJECT_PLAN.md`).
Legend: ✅ done · 🟡 partial · ⏳ next · ⬜ not started · ✂️ candidate to descope.

_Last updated: 2026-07-21._

| Phase | What | Week | Status |
|---|---|---|---|
| 0 | Repo scaffold, tooling, ADRs | 1 | ✅ done |
| 1 | Dataset unification (VisDrone, SARD, RescueNet, FloodNet → 2 unified sets) | 1 | ✅ done |
| 2a | Perception **training** (Model A detect, Model B segment) | 2 | ✅ done |
| 2b | Perception **evaluation** — SAHI vs full-frame, AP by object size, per-source | 3 | ✅ done |
| 3 | Detection cache + oracle (`detections.parquet`, ADR-001 bridge) | 4 | ✅ done |
| 4 | Simulator core (world, UAV energy model, engine) | 5 | ✅ done |
| 5 | Partitioning + coverage paths | 6 | ✅ done |
| 6 | **Auction reallocation + 3 baselines** (core contribution) | 7–8 | ✅ done |
| 7 | Flood survivor-drift prediction (most novel idea) | 10 | ✅ done |
| 8 | Hazard-weighted rescue routing (Pareto fronts) | 11 | ⏳ **next** |
| 9 | Evaluation harness + Monte Carlo sweep (mean ± 95% CI) | 9 | ✅ done |
| 10 | 3D reconstruction study (NeRF/3DGS vs photogrammetry) | 12 | ✂️ descope candidate |

## Done in detail

- **Phase 0–1:** `uv`/Hydra/pytest/ruff scaffold; two ADRs; taxonomy with totality tests;
  loaders for all four real dataset formats; `data/unified/{detect,seg}` built (detect: 126,897
  person + 205,663 vehicle; seg: 6,599 building_damaged / 2,643 road_blocked / 6,597 water).
- **Phase 2a:** local YOLO11s training on M4 MPS at 640px, 60 epochs each.
  Model A mAP@50 **0.674** / mAP@50-95 **0.392**; Model B mask mAP@50 **0.410** / mAP@50-95 **0.266**.
- **Phase 2b:** COCO evaluation of Model A. Size-stratified AP (small 0.26 / medium 0.59 /
  large 0.78) quantifies the small-object difficulty; SARD easier than VisDrone (AP@50 0.88 vs
  0.65); naive SAHI *reduced* AP (scale mismatch — needs slicing-aided fine-tuning). See
  PROGRESS_REPORT §6.
- **Phase 3:** detection cache + oracle (the ADR-001 bridge). `make cache-dets` runs both
  models over the flood_a scenario → `data/cache/detections.parquet` (768 detections / 36
  cells, synthetic georeferencing). `src/sim/oracle.py` serves them with seeded false-negative +
  latency noise; deterministic under seed; an AST test proves `src/sim` never imports the detector.
- **Phase 4:** simulator core — `world.py` (metric grid + analytic flow field: uniform/channel/
  radial), `uav.py` (P=P_hover+k·v² energy model + return-to-home at 1.3× energy-to-base),
  `engine.py` (deterministic fixed-timestep loop, event log, oracle survey hook). `make sim
  SCEN=flood_a SEED=0` runs 4 UAVs × 60 min in **0.59 s**, 100% coverage; same seed → byte-identical
  logs; RTH-from-5km lands safely. 48 tests green.
- **Phase 5:** partitioning + coverage. `partition.py` — `grid` baseline + `weighted_voronoi`
  (Lloyd + greedy boundary rebalance) that equalises sector *workload* to within 5% (vs the
  grid's spread of 10 on a priority field, Voronoi gets 2). `coverage.py` — boustrophedon sweep
  with an explicit last-row fix; >99% cell visitation. 58 tests green.
- **Phase 6 (the contribution):** `allocation.py` — auction reallocation (Contract Net /
  Gerkey & Matarić ST-SR-IA) + 3 baselines (`single_uav`, `static_partition_no_realloc`,
  `random_walk`) behind one `Coordinator` interface. Engine made coordinator-driven with a
  scripted-failure hook. **Acceptance test:** UAV-2 dies mid-sector → auction recovers to 100%
  coverage; `static_partition_no_realloc` stays <95% (loses the abandoned cells). 64 tests green.
- **Phase 7 (novel):** `drift/advect.py` — SAROPS-style Lagrangian advection of a detected
  survivor (flow × leeway + turbulent diffusion) → 50/90% containment polygons; `cells_in_region`
  maps a polygon to grid cells for drift-driven re-tasking (RQ4). Analytic test (zero-diffusion
  uniform flow → exact v·Δt), containment grows with horizon, 90% polygon holds ≥90% + generalises.
  74 tests green. _Next micro-step: wire the containment cells into the auction priority to run
  the RQ4 experiment._
- **Phase 9 (the evidence):** `eval/{metrics,runner}.py`. `make sweep` = 1800 runs in ~4 s.
  **Headline (6 UAVs, coverage mean ± 95% CI): adaptive auction beats static partitioning by
  +12.4 pts under one failure and +25.5 pts under two** (100% vs 87.6/74.5%); random_walk matches
  coverage but wastefully (redundancy 1.1–1.24), single_uav collapses. `auction_no_priority`
  ablation included. 69 tests green.

## Recommended focus (see PROGRESS_REPORT.md §6–7)

**Depth over breadth.** Prioritise the coordination contribution (Phase 6 + evaluation Phase 9)
and the drift re-tasking (Phase 7), plus the perception analysis (Phase 2b). Park Phase 10
(NeRF/3DGS), tracking, and multi-scenario breadth — they add breadth, not a contribution.

## Descope ladder (agree with supervisor)

Cut in this order if time runs short: tracking → NeRF → all 3D reconstruction → multi-scenario
→ drift (only if it never entered the approved proposal). **Never cut:** the coordination
baselines, the Monte-Carlo seeds, or the ablations — those _are_ the contribution.
