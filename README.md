# Multi-UAV AI Framework for Disaster Response

MSc dissertation (LSBU, Level 7) · Student 4437147 · Supervisor: Anastasios Dagiuklas
**Simulation only** — no flight hardware, no real-time constraints.

Partition a disaster area into sectors → assign *N* simulated UAVs one sector each →
survey → detect survivors / damaged buildings / blocked roads / flood water with YOLO11
(offline, on real imagery) → georeference and log detections → reallocate UAVs dynamically
→ project survivor drift in flowing water → emit a hazard-weighted road graph and rescue
routes.

## The one architectural decision (ADR-001)

Perception and coordination are **decoupled**. YOLO runs offline on real labelled imagery
and caches detections to `data/cache/detections.parquet`; the simulator reads them through
`src/sim/oracle.py` and never imports `ultralytics`. This gives two clean, separately-scored
contributions and lets coordination experiments run hundreds of Monte Carlo seeds on CPU in
minutes. See [`docs/adr/`](docs/adr/).

## Quickstart

```bash
make setup        # uv venv + core/dev deps, pinned via uv.lock
make test         # fast suite (excludes @pytest.mark.slow); must pass before any commit
make lint         # ruff check + format --check
make help         # list all targets
```

Heavier stacks are opt-in:

```bash
uv sync --extra perception   # torch + ultralytics + sahi (training machine only)
uv sync --extra geo          # geopandas / pyproj / osmnx (Phase 3 + Phase 8)
```

## Layout

| Path | Responsibility |
|---|---|
| `src/perception/` | Offline: dataset unification, YOLO11 train/eval, detection caching |
| `src/sim/` | Grid world, UAV energy model, fixed-timestep engine, detection **oracle** |
| `src/coordination/` | Partitioning, coverage paths, auction-based reallocation |
| `src/drift/` | Lagrangian survivor-drift advection (SAROPS-style) |
| `src/routing/` | Hazard-weighted road graph + safe-path search |
| `src/eval/` | Monte Carlo sweep runner + metrics |
| `configs/` | Hydra YAML — **no magic numbers in `src/`** |
| `docs/` | ADRs, dataset provenance, and the [build log](docs/BUILD_LOG.md) |
| `tests/` | Pytest; synthetic fixtures under `tests/fixtures/` (no GPU, no datasets) |

## Status

Built incrementally by phase (see [`TASK_PROMPTS.md`](TASK_PROMPTS.md) and
[`docs/BUILD_LOG.md`](docs/BUILD_LOG.md)). Phases 0–1 (scaffold + dataset unification) are
in place; the remaining modules are documented stubs.

## Data & licences

`data/` and `outputs/` are gitignored. Datasets are obtained separately and cited in
[`docs/datasets.md`](docs/datasets.md). **RescueNet is CC BY-NC-ND** — used for
non-commercial academic work only; derived label files are not redistributed.
