# Build log

Chronological engineering log — what was built, the decisions behind it, and open
follow-ups. Complements (does not duplicate) the ADRs in `docs/adr/`: ADRs record *why* an
architecture is the way it is; this log records *what happened when*. Newest entries at the
top.

Conventions: one entry per work session, dated. Each entry states the phase, what landed,
non-obvious decisions, any deviations from the plan, the acceptance-criteria status, and
follow-ups. Acceptance criteria come from `TASK_PROMPTS.md`; timeline from `PROJECT_PLAN.md`.

---

## 2026-07-17 — Week 1: Phase 0 (Scaffold) + Phase 1 (Dataset unification)

**Status:** ✅ complete · `make setup && make lint && make test` all green · 25 tests passing.

### Phase 0 — Scaffold

Built the repository skeleton from `TASK_PROMPTS.md` Phase 0.

- **Toolchain:** `uv` (installed via Homebrew, v0.11.29), Python 3.10.0 (`.python-version`),
  Hydra/OmegaConf config, pytest, ruff. `pyproject.toml` declares core deps and three opt-in
  extras: `geo` (geopandas/pyproj/osmnx), `perception` (torch/ultralytics/sahi), `dev`.
- **`make setup` is CPU-only by design:** it installs `core + dev` only — no torch, no GDAL —
  so coordination/dataset work never drags in the GPU stack (supports ADR-001). Heavier
  stacks are `make setup-all` or `uv sync --extra <name>`.
- **Package tree:** `src/{perception,sim,coordination,drift,routing,eval}/`, each with an
  `__init__.py` stating a single responsibility, and documented stubs that raise
  `NotImplementedError` when *called* (not at import — so tests can import freely).
- **ADRs** written up from CLAUDE.md: `docs/adr/ADR-001` (decouple perception/coordination)
  and `ADR-002` (two models), plus an ADR index.
- **Configs:** `configs/config.yaml` (root Hydra), `configs/datasets/{default,taxonomy}.yaml`.
- **Makefile** targets mirror CLAUDE.md's command list; everything runs via `uv run`.

**Acceptance (Phase 0):** `make setup && make lint && make test` green on a clean venv. ✔

### Phase 1 — Perception: dataset unification

Built `src/perception/datasets/` from `TASK_PROMPTS.md` Phase 1.

- **Taxonomy (`taxonomy.py` + `configs/datasets/taxonomy.yaml`):** single source of truth for
  the source→unified class mapping. Validated **total** — every native class is mapped or
  explicitly dropped exactly once; mapping to a wrong-task unified class, or leaving a class
  unaccounted for, raises `TaxonomyError`. This is the no-silent-drops guarantee.
- **Loaders (`loaders.py`):** four behind one `SourceLoader` interface — `VisDrone` (YOLO
  txt), `SARD` (Pascal VOC XML, pose kept as metadata), `RescueNet`/`FloodNet` (index masks).
  An undeclared class raises rather than being dropped.
- **Mask→polygon (`masks.py`):** RescueNet/FloodNet masks → YOLO-seg polygons via connected
  components + sub-pixel contours (scikit-image) + Douglas-Peucker simplify (shapely). Masks
  are **never** reduced to boxes (ADR-002).
- **Build CLI (`build.py`):** `--dry-run` prints a per-class instance-count table for both
  output sets (detect / seg); a real run writes YOLO / YOLO-seg labels + image symlinks +
  Ultralytics `data.yaml` under `data/unified/`.
- **`docs/datasets.md`:** per-source licence, size, resolution, platform, and the exact
  mapping — with citations. Class lists verified against the source papers (see below).

**Acceptance (Phase 1):** `build --dry-run` prints the instance-count table for both sets ✔;
tests verify the mapping is total (`tests/test_taxonomy.py`) ✔. Ran end-to-end on synthetic
fixtures (no datasets needed); ran the CLI against the real config with no data present — it
degrades gracefully, reporting every source as *absent* instead of crashing. ✔

### Decisions & deviations

- **Verified all source class lists against the papers** (WebFetch/WebSearch), correcting
  guesses: RescueNet index 3 is **Building-Minor-Damage** (not "Medium"); SARD has 6 poses
  incl. **Not-Defined**; FloodNet order confirmed. Encoded in `taxonomy.yaml` in on-disk index
  order so loaders translate index→class directly.
- **RescueNet `Vehicle` is dropped, not emitted.** It is a mask; `vehicle` is a detect (box)
  class. Emitting it via mask→bbox is deferred (documented, and listed as a drop so nothing is
  lost silently). PROJECT_PLAN §3.4 lists it as an option, not a requirement.
- **Ruff lint set = defaults + `I` (isort) + `UP` (pyupgrade).** CLAUDE.md says "ruff
  defaults"; I added two near-universal, low-noise rule groups for import hygiene. Trivial to
  revert to strict defaults if preferred.
- **Pinning via `uv.lock`, not `==` in `pyproject.toml`.** The committed lockfile is the
  reproducible pin (modern uv convention); `pyproject.toml` carries compatible lower bounds.
- **`min_polygon_points` default is 4** (≥3 distinct vertices). The `min_region_area_px=32`
  filter is the primary speckle guard; 4 keeps legitimate simple regions.
- **Phase 1 build path uses argparse (`--dry-run`)**, loading the Hydra YAML via OmegaConf,
  to honour the exact acceptance command in the task prompt (Hydra's `key=value` override
  style doesn't fit a `--dry-run` flag).

### Follow-ups / open items

- [ ] **Supervisor action (PROJECT_PLAN §1):** the **drift model** was suggested by the
      supervisor (per student, 2026-07-17); confirm it is written into the *approved proposal
      document*, and agree the **descope ladder** now, not in week 9.
- [ ] Confirm the actual submission date and re-map the §5 timeline (14-week assumption).
- [ ] Obtain datasets into `data/raw/`: VisDrone (auto-download), SARD (IEEE DataPort),
      RescueNet + FloodNet (BinaLab). Then run `make build-datasets` for the real
      instance-count table and paste it into `docs/datasets.md`.
- [ ] On first real mask: verify the palette index order matches `taxonomy.yaml` `native:`
      lists (one-line fix if not).
- [ ] Repo is **not yet a git repo** — `git init` + first commit pending your go-ahead
      (CLAUDE.md checklist wants "Repo + CLAUDE.md + ADRs committed").

### Next (Week 2 → Phase 2)

Train Model A (`yolo11s.pt`) and Model B (`yolo11s-seg.pt`) on the unified sets; first mAP
numbers. Needs `uv sync --extra perception` on a GPU machine and the datasets in place.
