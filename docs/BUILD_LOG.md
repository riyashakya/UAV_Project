# Build log

Chronological engineering log — what was built, the decisions behind it, and open
follow-ups. Complements (does not duplicate) the ADRs in `docs/adr/`: ADRs record *why* an
architecture is the way it is; this log records *what happened when*. Newest entries at the
top.

Conventions: one entry per work session, dated. Each entry states the phase, what landed,
non-obvious decisions, any deviations from the plan, the acceptance-criteria status, and
follow-ups. Acceptance criteria come from `TASK_PROMPTS.md`; timeline from `PROJECT_PLAN.md`.

---

## 2026-07-19 — Phase 1 hardening on real data + Phase 2 Colab training setup

**Status:** ✅ detect set built on real downloads · 26 tests green · Colab notebook ready.
**Trigger:** all four datasets downloaded; ran the "check the dataset before training" step.

### Dataset inspection findings (real downloads differ from the paper-based assumptions)

- **VisDrone** ships the **raw** annotation format (`annotations/*.txt`, comma-separated,
  absolute pixels, category id in field 6, ids 0–11) — not the pre-converted YOLO format the
  loader first assumed.
- **SARD** (Roboflow VOC export) collapses all poses to a single **`human`** class.
- **RescueNet** masks are proper **index masks** (0–10); originals sit in a sibling
  `*-org-img/` folder, not next to the masks.
- **FloodNet** ❌ incomplete: only the RGB **ColorMasks** were downloaded (1024², no original
  images). Unusable for training — needs re-download of `FloodNet-Supervised_v1.0`.

### Code changes (Phase 1 corrections)

- `taxonomy.yaml`: VisDrone → 12 raw categories (added `ignored-regions`, `others` as drops);
  SARD → accepts both `human` and the six poses (robust to either download).
- `loaders.py`: rewrote `VisDroneLoader` for the raw format (reads image size to normalise);
  mask loaders now find originals in the sibling `*-org-img/` folder and **raise a clear error
  on an RGB colour mask** (no-silent-drops philosophy). Added split inference (train/val/test)
  from the source folder.
- `build.py` / writer: real **`{images,labels}/<split>/`** layout, correct Ultralytics
  `data.yaml`, and a `--copy-images` flag (symlinks don't survive a Colab zip upload).
- Updated synthetic fixtures + tests to the real formats and split logic (26 tests).

### Built + verified

- `make build-datasets ARGS="--sources visdrone sard --copy-images"` → `data/unified/detect/`
  (2.8 GB): **126,897 person + 205,663 vehicle** across train 7,858 / val 944 / test 198.
- RescueNet mask→polygon measured at ~0.4 s/mask (≈30 min for the full set as a one-time job).

### Phase 2 — Colab training setup (train on GPU, use weights locally)

- `notebooks/train_yolo11_colab.ipynb`: one notebook, trains Model A or Model B; mounts Drive,
  unzips the dataset, fixes the `data.yaml` path for Colab, trains, validates, saves `best.pt`.
- `configs/perception/model_a.yaml` + `model_b.yaml`: hyperparameters with provenance
  (imgsz 1280 for the tiny-object detect set; 1024 for seg).

### Follow-ups

- [x] **Re-downloaded FloodNet** `FloodNet-Supervised_v1.0` (train 1445 / val 450 / test 448,
      index masks + originals, verified) — the loader handled it as-is.
- [x] **Segment set built** (`--sources rescuenet floodnet --copy-images --max-image-side 1280`):
      building_damaged 6,599 · road_blocked 2,643 · water 6,597; 6,387 images (2.9 GB).
      Added a `--max-image-side` resize option so the 3000×4000 originals ship small for Colab
      (labels are normalised, so resizing doesn't touch them).
- [ ] Train + **compare YOLO11 sizes (n/s/m/l)** for Model A and Model B on Colab (fair scale
      study: only size changes; 50-epoch sweep → retrain the winner at 100). Record the
      comparison tables here. Notebook + `configs/perception/{model_a,model_b}.yaml` updated.
- [ ] `road_blocked` is the minority seg class (2,643 vs ~6,600) — watch for class imbalance.
- [ ] SARD pose sub-labels unavailable in the Roboflow export — note in the limitations chapter.

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
- [x] Git initialised; first commit `7e668ed` on `main` (53 files; data/outputs/venv
      ignored). No remote yet — push to a private GitHub repo when off-machine backup wanted.

### Next (Week 2 → Phase 2)

Train Model A (`yolo11s.pt`) and Model B (`yolo11s-seg.pt`) on the unified sets; first mAP
numbers. Needs `uv sync --extra perception` on a GPU machine and the datasets in place.
