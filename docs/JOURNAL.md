# Work journal

A structured, chronological record of work. **Every task is journaled here _before_ it is
started**, for traceability and easy resumption. Newest entries at the top.

Entry format: **Request** · **Summary** · **Root cause / motivation** · **Solution** ·
**Why this solution** · **Files changed** · **Status**.

Related: [`ROADMAP.md`](ROADMAP.md) (phase status), [`BUILD_LOG.md`](BUILD_LOG.md) (earlier
detailed technical log), [`adr/`](adr/) (architecture decisions).

---

## 2026-07-21 — Phase 3: detection cache + oracle (the ADR-001 bridge)

- **Request:** start Phase 3 (user committing full-time).
- **Summary:** implement `src/perception/detect_cache.py` (runs both models over a scenario's
  imagery once → `data/cache/detections.parquet`), `src/sim/oracle.py` (the only bridge; serves
  cached detections with configurable false-negative + latency noise, deterministic under seed),
  and a test that walks the import graph of `src/sim/` to prove nothing reaches `ultralytics`.
- **Root cause / motivation:** ADR-001 — perception and coordination must stay decoupled; the
  oracle is the single, controllable interface the simulator uses to learn about the world.
- **Solution:** synthetic-grid scenario (no geotags in the datasets → `synthetic_geo=True`,
  flat-earth lat/lon from a config anchor); cache columns per spec; oracle reads parquet with
  pandas only (no torch). AST import-graph test enforces the firewall statically.
- **Why this solution:** keeps the heavy detector out of the sim (fast CPU Monte-Carlo later);
  georeferencing is honestly labelled synthetic; determinism via a threaded `np.random.Generator`.
- **Files changed:** `src/sim/oracle.py`, `src/perception/detect_cache.py`,
  `configs/scenario/flood_a.yaml`, `configs/sim/oracle.yaml`, `tests/test_oracle.py`,
  `tests/test_sim_isolation.py`, `tests/test_detect_cache.py`.
- **Status:** ✅ done. All 3 acceptance criteria met: cache builds (flood_a, 768 detections /
  36 cells), oracle deterministic under seed (verified), AST firewall test passes (src/sim
  can't reach ultralytics). 35 tests green.

## 2026-07-21 — Phase 2b: perception evaluation (SAHI ablation + size-stratified AP)

- **Request:** start Phase 2 evaluation on the trained Model A.
- **Summary:** implement `src/perception/eval.py` to compare **full-frame vs SAHI tiled
  inference**, report **AP stratified by object size** (COCO small/medium/large), and a
  **per-source** breakdown (VisDrone-val vs SARD-val) of the combined model.
- **Root cause / motivation:** training (2a) produced models but no *result*; the marked
  perception contribution is the SAHI/small-object comparison and the domain breakdown.
- **Solution:** run the model once full-frame and once with SAHI (640 slices, 0.2 overlap) over
  the detect val set; convert GT + predictions to COCO format and score with pycocotools to get
  AP_s/m/l; evaluate on all + per-source image subsets (no retraining needed).
- **Why this solution:** pycocotools is the standard for size-stratified AP; running inference
  once and slicing the eval by subset is cheap. Full train-on-A/test-on-B domain-gap ablation
  would need 2 more ~20 h trainings — deferred; the per-source-subset view gives the signal now.
- **Files changed:** `src/perception/eval.py` (impl), `configs/perception/eval.yaml` (new),
  `pyproject.toml` (add pycocotools), `docs/{ROADMAP,PROGRESS_REPORT}.md` (results).
- **Status:** ✅ done. Result: size-stratified AP small 0.26 / med 0.59 / large 0.78; SARD
  easier than VisDrone; **naive SAHI reduced AP** (train/inference scale mismatch — documented
  as an honest negative result, fix = slicing-aided fine-tuning). Full numbers in
  PROGRESS_REPORT §6.

## 2026-07-21 — Push repository to GitHub

- **Request:** push all work so far to `https://github.com/riyashakya/UAV_Project.git`.
- **Summary:** add the remote and push `main` (code + docs only; datasets/weights are gitignored).
- **Root cause / motivation:** off-machine backup and supervisor visibility; the repo had no remote.
- **Solution:** `git remote add origin <url>` then `git push -u origin main`.
- **Why this solution:** standard first push; nothing large or sensitive travels (data/ and *.pt
  are ignored). Credentials must come from the user's own git/GitHub auth — never entered here.
- **Files changed:** none (git metadata only; adds `origin` remote).
- **Status:** ✅ done — pushed to `origin/main` (13 commits). Future work: `git push` keeps it synced.

## 2026-07-21 — Project-tracking docs, progress report, and strategic review

- **Request:** (1) explain what segmentation is for; (2) create a rich change/decision log;
  (3) push everything to git; (4) a markdown file tracking finished plans; (5) journal work
  before starting it, going forward; (6) how good are the models; (7) my contribution vs
  existing work + how to improve; (8) the research question; (9) a dissertation progress
  report; (10-12) whether to focus/deepen features vs breadth, which ones, and whether the
  project is heading toward "mediocre".
- **Summary:** Established the journaling discipline (this file), added a phase-status tracker
  and an interim progress report, and gave an honest depth-vs-breadth assessment. Ensured all
  work is committed; documented the git-remote gap for pushing.
- **Root cause / motivation:** The project has produced solid setup + perception results but no
  novel *contribution* yet, and the student is (rightly) asking whether breadth across 7
  objectives risks a mediocre outcome. Also needs auditable history for the write-up.
- **Solution:** Created `docs/JOURNAL.md` (this), `docs/ROADMAP.md`, `docs/PROGRESS_REPORT.md`;
  saved the journal-before-work rule to memory; committed everything. Push to a GitHub remote
  is blocked (no remote, `gh` not installed) — handed off with exact commands.
- **Why this solution:** JOURNAL + ROADMAP separate "what happened / why" from "what's done vs
  pending"; the progress report consolidates research question, results, and the honest
  contribution analysis in one citable place. Kept `BUILD_LOG.md` for detailed history rather
  than deleting it.
- **Files changed:** `docs/JOURNAL.md` (new), `docs/ROADMAP.md` (new),
  `docs/PROGRESS_REPORT.md` (new), memory `journaling-workflow.md` (new).
- **Status:** ✅ done (docs); ⏳ push pending a remote (see ROADMAP / chat).

## Earlier work (pre-journal)

Phases 0–2 (scaffold, dataset unification, perception training) predate this journal and are
recorded in [`BUILD_LOG.md`](BUILD_LOG.md). Headline: both YOLO11 models trained —
Model A (detect) mAP@50 0.674, Model B (segment) mask mAP@50 0.410.
