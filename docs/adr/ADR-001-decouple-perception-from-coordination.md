# ADR-001: Perception and coordination are decoupled — keep them that way

- **Status:** Accepted
- **Date:** 2026-07-17
- **Deciders:** Student 4437147 (with supervisor Anastasios Dagiuklas)

## Context

The obvious reading of the project brief is to build a photorealistic simulated world, fly
drones in it, and run YOLO on the rendered camera feed. That path implies AirSim/Gazebo, an
Unreal build, and days of environment setup — and at the end the detection numbers are
measured on *synthetic* imagery, which tells an examiner nothing about real-world detection
performance. It also couples every coordination experiment to a GPU render loop, making
large Monte Carlo studies infeasible on a laptop.

## Decision

Perception and coordination are **separate pipelines** joined by a single, narrow bridge.

```
   OFFLINE (GPU, once)                      SIM (CPU, 100s of seeds in minutes)
   Real labelled UAV imagery                Grid world + analytic flow field
     -> YOLO11 A (detect) + B (seg)           -> N UAVs, energy model
     -> data/cache/detections.parquet   ───►   -> partition -> survey -> auction realloc
        keyed by (scenario, cell_id)     oracle -> drift -> routing
     -> mAP vs real ground truth
```

- The simulator (`src/sim/`) **never** imports `ultralytics`, `torch`, or `sahi`.
- Detections are computed offline and cached to `data/cache/detections.parquet`, keyed by
  `(scenario, cell_id)`.
- `src/sim/oracle.py` is the **only** bridge: when a UAV enters a cell, the oracle returns
  that cell's cached detections, with configurable false-negative rate and latency noise.

## Consequences

**Positive**
- Two clean, separately-defensible contributions: perception scored on *real* data;
  coordination scored over hundreds of Monte Carlo seeds.
- Coordination experiments run on CPU with no GPU and no dataset present — the HPC queue
  can never block them.
- Perception is evaluated properly against real ground truth, not synthetic renders.

**Negative / accepted trade-offs**
- The simulator does not model perception–motion coupling (e.g. motion blur changing with
  UAV speed). The oracle's noise model is the deliberate, controllable stand-in.
- Requires discipline: the decoupling is **enforced by a test** (Phase 3) that walks the
  AST of `src/sim/` and fails if any import path reaches `ultralytics`.

## Enforcement

> If you are about to `import ultralytics` inside `src/sim/`, stop and reconsider.

`tests/test_sim_no_ultralytics.py` (Phase 3) makes this a build failure, not a convention.
