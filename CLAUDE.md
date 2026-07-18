# Multi-UAV Disaster Response Framework

MSc dissertation (LSBU, Level 7). Student 4437147. Supervisor: Anastasios Dagiuklas.
**Simulation only.** No physical drones, no flight hardware, no real-time constraints.

## What the system does

Partition a disaster area into sectors → assign N simulated UAVs (default 4) one sector
each → survey → YOLO11 detects survivors / damaged buildings / blocked roads / flood
water → georeference and log detections → reallocate UAVs dynamically (on battery abort
or newly-discovered high-priority cells) → project survivor drift in flowing water →
emit a hazard-weighted road graph and rescue routes.

## Non-goals — do not build these

- MAVLink, PX4, ArduPilot, or any flight-controller integration
- AirSim / Unreal / Gazebo / photorealistic rendering (see ADR-001)
- Real-time performance targets. This is offline batch evaluation.
- Web dashboards. Static matplotlib / folium output only.
- Anything needing a live network at run time (datasets and OSM extracts are cached to disk)

## ADR-001: perception and coordination are decoupled — keep them that way

**The simulator never calls YOLO.**

- Perception runs offline on real labelled UAV imagery. Detections are cached to
  `data/cache/detections.parquet`, keyed by `(scenario, cell_id)`.
- `src/sim/oracle.py` is the only bridge: when a UAV enters a cell, the oracle returns
  the cached detections for that cell, with configurable false-negative/latency noise.

Rationale: a photorealistic simulator produces *worse* perception numbers than real
labelled data and costs weeks of setup. Decoupling means coordination experiments run
500 Monte Carlo seeds in minutes on CPU, while perception is scored properly against
real ground truth. These are separate contributions in the write-up.

If you are about to `import ultralytics` inside `src/sim/`, stop and reconsider.

## ADR-002: two perception models, not one

No single dataset covers all classes, and the source label types differ.

- **Model A (`yolo11*.pt`, detect)** — `person`, `vehicle`. Trained on VisDrone + SARD.
  Always run with SAHI tiled inference; see "Small objects" below.
- **Model B (`yolo11*-seg.pt`, segment)** — `building_damaged`, `road_blocked`,
  `water`. Trained on RescueNet + FloodNet, whose labels are pixel masks.

Do not merge these into one model. Do not convert RescueNet's `water` masks to
bounding boxes — a box around a flood region is meaningless.

## Small objects: this is the core perception difficulty

In VisDrone, 68% of objects are under 32×32 px. In SARD/HERIDAL a person is ~0.1% of
frame area. Naive full-frame inference on 3000×4000 imagery will miss almost every
survivor.

- Always use SAHI for Model A: 640×640 slices, `overlap_ratio=0.2`.
- Report AP stratified by object size (COCO small/medium/large). Whole-image mAP alone
  hides the only result that matters here.

## Commands

```bash
make setup           # uv venv + deps, pinned
make test            # pytest -q, must pass before any commit
make lint            # ruff check + ruff format --check
make cache-dets      # offline YOLO pass -> data/cache/detections.parquet
make sim SCEN=flood_a SEED=0
make sweep           # full Monte Carlo grid, writes outputs/runs/<timestamp>/
```

## Conventions

- Python 3.10, `uv` for deps. Never `pip install` into the system interpreter.
- Config is Hydra YAML under `configs/`. **No magic numbers in `src/`** — if you need a
  constant, add it to a config with a comment on where the value came from.
- Every stochastic entry point takes an explicit `seed: int`. Never call the global
  `random`/`np.random` — pass a `np.random.Generator` through.
- Geospatial: EPSG:4326 for storage/interchange, local UTM for anything metric.
  Name variables `_wgs84` / `_utm` so mismatches are visible at a glance. Never compute
  distance in degrees.
- Long jobs write to `outputs/runs/<timestamp>/` with the resolved config dumped
  alongside results. Never overwrite a previous run.
- Type hints on public functions. `ruff` defaults, line length 100.

## Testing

Tests are the contract; write them before the implementation.

- Coordination and drift logic must be **testable without a GPU and without datasets**.
  Use synthetic fixtures in `tests/fixtures/`.
- Every algorithm needs one analytically-verifiable case: e.g. uniform flow field over
  10 min displaces a particle exactly `v*t`; four UAVs on a 4-sector grid with equal
  speeds finish within one timestep of each other.
- Mark GPU/dataset tests `@pytest.mark.slow` and exclude from `make test`.

## Data

- `data/` and `outputs/` are gitignored. Never commit imagery, weights, or run outputs.
- RescueNet is **CC BY-NC-ND**: use it, do not redistribute derived label files.
- Cite every dataset in `docs/datasets.md` as you add it. This feeds the dissertation.

## Working style

- Plan before implementing. For anything touching more than one module, propose the
  approach and wait for confirmation.
- Prefer stdlib + numpy/scipy/geopandas/networkx. Ask before adding a dependency.
- Stay inside the module named in the task. Do not refactor adjacent code opportunistically.
- If a task looks like it needs a change to an ADR above, say so instead of working around it.
