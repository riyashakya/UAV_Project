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
| 3 | Detection cache + oracle (`detections.parquet`, ADR-001 bridge) | 4 | ⏳ **next** |
| 4 | Simulator core (world, UAV energy model, engine) | 5 | ⬜ not started |
| 5 | Partitioning + coverage paths | 6 | ⬜ not started |
| 6 | **Auction reallocation + 3 baselines** (core contribution) | 7–8 | ⬜ not started |
| 7 | Flood survivor-drift prediction (most novel idea) | 10 | ⬜ not started |
| 8 | Hazard-weighted rescue routing (Pareto fronts) | 11 | ⬜ not started |
| 9 | Evaluation harness + Monte Carlo sweep (mean ± 95% CI) | 9 | ⬜ not started |
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

## Recommended focus (see PROGRESS_REPORT.md §6–7)

**Depth over breadth.** Prioritise the coordination contribution (Phase 6 + evaluation Phase 9)
and the drift re-tasking (Phase 7), plus the perception analysis (Phase 2b). Park Phase 10
(NeRF/3DGS), tracking, and multi-scenario breadth — they add breadth, not a contribution.

## Descope ladder (agree with supervisor)

Cut in this order if time runs short: tracking → NeRF → all 3D reconstruction → multi-scenario
→ drift (only if it never entered the approved proposal). **Never cut:** the coordination
baselines, the Monte-Carlo seeds, or the ablations — those _are_ the contribution.
