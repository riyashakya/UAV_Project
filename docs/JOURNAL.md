# Work journal

A structured, chronological record of work. **Every task is journaled here _before_ it is
started**, for traceability and easy resumption. Newest entries at the top.

Entry format: **Request** · **Summary** · **Root cause / motivation** · **Solution** ·
**Why this solution** · **Files changed** · **Status**.

Related: [`ROADMAP.md`](ROADMAP.md) (phase status), [`BUILD_LOG.md`](BUILD_LOG.md) (earlier
detailed technical log), [`adr/`](adr/) (architecture decisions).

---

## 2026-08-06 — Re-look experiment: can coordination claw back the perception bottleneck?

- **Request:** build the re-look experiment (extend the §4.5 bottleneck finding). Also: user
  challenged the federated "privacy cost" / "feed FL detector into §4.5" angles — I **retracted**
  both (privacy-cost is the most standard FL result; the coupling-combo I hadn't verified). See
  [[novelty-positioning]].
- **Idea:** §4.5 says coordination can't fix a detection miss — but that assumed ONE look. Independent
  re-looks drop the miss rate (≈FN^k), so coordination *might* partly beat the detector, at a coverage
  cost. `src/eval/relook.py` (`make relook`) measures the trade-off. Honesty knob: `persistent_share`
  = fraction of misses that are STRUCTURAL (never recovered) → not the over-optimistic FN^k. Detection
  model `miss_after_looks` is pure + unit-tested. Reuses the benchmark survivor generator; the oracle
  already gives independent draws per visit, so no engine change.
- **Result (FN 0.4, persistent 0.3, budget 1.5x, 200 seeds) — conditional, honest:** re-look beats
  cover-more **only with a good prior**. good prior → k=2 finds 79% vs 60% (**+19 pts**); medium → +1.5;
  **poor prior → cover-more wins** (re-looks wasted on wrong cells). k≥3 hurts everywhere (coverage
  sacrifice). Refines the bottleneck message: coordination can partly beat the detector, but only by
  *spending a good prior* (which the drift model + hazard maps produce) — ties parts of the project
  together. Report §4.5a + figure.
- **Files:** src/eval/relook.py, configs/eval/relook.yaml, tests/test_relook.py (5 tests), Makefile,
  docs/dissertation_report.md. **114 tests green, lint clean.**
- **Status:** done.

---

## 2026-08-06 — Augmentation pinning, lighting robustness, and starting Flower (federated)

- **Request (3 parts):** (a) pin the training augmentation into the configs + report note; (b) build
  `make lighting-robustness` (mAP under bright/normal/dark); (c) start federated training with Flower
  (FedAvg/FedProx). Standing instruction reinforced: **research before any claim, no overclaims, plain
  words, improvement not novelty.** See [[novelty-positioning]].
- **(a) Augmentation — no retraining.** Read the runs' `args.yaml` (authoritative record): the models
  used the YOLO11 defaults (mosaic, hsv incl. `hsv_v=0.4` brightness, fliplr, scale, translate,
  erasing). Pinned those exact values into `model_a.yaml`/`model_b.yaml`; `train.py` now passes them so
  the config is the source of truth. Values equal what was applied ⇒ no retrain. Report §3.3 note added.
- **(b) Lighting robustness — real result.** `src/perception/lighting_eval.py`: re-scores the trained
  detector on the same val subset re-brightened. First run was SARD-biased (alphabetical limit);
  fixed with a deterministic shuffle → representative 300-img sample, **normal 68.0% == whole-dataset
  0.674** (consistency check). Finding: **robust to lighting** (dark −2.1, dim −0.6, bright −1.4) —
  the `hsv_v` augmentation pays off — with **glare (×1.8) the weakness at −5.1 pts**. Report §4.1 table.
- **(c) Flower — started, honestly scoped.** Web check first: FedAvg/FedProx-YOLO for UAV is
  **established → NOT novel**; framed as a privacy-preserving way to improve the *bottleneck* the
  project identified, on a real non-IID split (VisDrone 6471 vs SARD 1387). Built + tested the pure
  parts (`partition.py` non-IID split, `fedavg.py` weighted averaging). `fed_train.py` is a Flower
  scaffold, **not yet run** (needs `uv sync --extra federated` + GPU); FedProx proximal term into the
  YOLO loop is the flagged remaining work. Plan + honest positioning in `docs/federated_plan.md`.
- **Search-theory claim re-verified** (user asked): Koopman/ASWORG 1942–43, exponential detection
  function; Stone 1975 (Lanchester Prize) → SAROPS basis. My "1940s" claim holds; cited.
- **Files:** configs/perception/{model_a,model_b,lighting,federated}.yaml, src/perception/train.py,
  src/perception/lighting_eval.py, src/perception/federated/*, tests/test_lighting.py,
  tests/test_federated.py, Makefile, pyproject.toml (federated extra), docs/{dissertation_report,
  federated_plan}.md. **109 tests green, lint clean.**
- **Status:** (a) done · (b) done, real result · (c) started (scaffold + plan; full run needs GPU).

---

## 2026-08-05 — Vision-estimated flood current driving the drift model (the one "new" connection)

- **Request:** after dropping the field-tool idea on safety grounds, add something genuinely new.
  Chosen: estimate the flood **current from drone video** (image velocimetry) and drive the
  survivor-drift prediction from the *measured* flow instead of an *assumed* one. This is the single
  connection our own literature review flagged as un-made (the pieces exist; wiring them to
  survivor-drift does not appear in the reviewed sources). See [[novelty-positioning]] — claim only
  "novel integration / proof-of-concept", verify against literature before any stronger wording.
- **Motivation:** the drift model (`src/drift/advect.py`) takes a `FlowField = (x,y)->(vx,vy)`
  callable; today it is always an *assumed* analytic field (`make_flow_field`: uniform/channel/
  radial). Estimating the field from imagery upgrades the drift's weakest input from assumed to
  measured — a real methodological improvement, not packaging.
- **Approach:** `src/perception/flow.py` — **PIV** (particle image velocimetry via FFT
  cross-correlation between frames), **pure numpy, no OpenCV** (avoids a new dep); pixel→metric
  conversion (metres_per_pixel, fps); build a `FlowField` from the sparse vector grid by
  inverse-distance interpolation. A synthetic image-sequence generator with a **known ground-truth
  flow** makes PIV testable and keeps the demo honestly labelled synthetic (no real flood video with
  known current is available — the honest limitation). Offline/cached → respects ADR-001 (sim never
  runs perception; it just consumes the field, exactly like detections).
- **Deliverable:** `make flow-drift` — the **assumed vs measured vs true** drift comparison:
  PIV recovery error vs truth, and drift-region localisation error when the forecast is driven by an
  *assumed* (wrong) flow vs the *PIV-measured* flow. Tests-first in `tests/test_flow.py`.
- **Result (200 seeds, synthetic flood-channel truth):** PIV recovers the current at **RMSE 0.24 m/s**
  (mean true speed 2.0 m/s, 0° angular error) — and this matches real drone-LSPIV error (0.22–0.44 m/s,
  literature). Driving the drift forecast from the **assumed** current locates the survivor **0%** of
  the time (err 176±3 m); from the **PIV-measured** current, **82%** (err 28±2 m). Measuring the flow
  cuts localisation error by 148 m.
- **Honesty check (2 web searches):** drone PIV/LSPIV for currents is mature (hydrology/tidal energy);
  SAR drift tools (OpenDrift, ocean models) and flood-SAR (observer headings) take currents from
  *external* sources, not scene imagery. Did **not** find the exact PIV→victim-drift pipeline, but —
  per [[novelty-positioning]] and my overclaim history — framed **only as an engineering improvement**
  (assumed→measured flow), **not** a novel field contribution. Ground truth is synthetic (no real flood
  video with a known current) — proof-of-concept only.
- **Files:** `src/perception/flow.py`, `src/eval/flow_drift.py`, `configs/perception/flow.yaml`,
  `tests/test_flow.py` (5 tests, +100 total green), `Makefile`. Pure numpy — no new dependency.
- **Status:** done. Report future-work updated to reflect it is now prototyped.

---

## 2026-08-05 — Real-data grounding of the ablation (attacks the synthetic-scenario weakness)

- **Request:** ground one experiment in real georeferenced data. Assessed: a truly GPS-tagged SAR
  dataset is not readily available (rare, hours–days). Feasible now: use the **real detection cache**
  spatially. Time: ~30 min, no new requirements.
- **Summary:** `src/eval/benchmark_real.py` (`make benchmark-real`) reruns the static/auction/guided
  ablation on the **real** flood_a detection distribution — 439 real YOLO person detections, real
  per-cell density (concentrated in the bottom rows, not uniform) and confidences — with a **real,
  independent** search prior derived from the flood-water segmentation. Measured insight: survivors
  sit *away* from water (corr ≈ **−0.51**), i.e. they fled the flood, so the prior is the **inverted
  water map** (not the survivor locations → not circular).
- **Result (30 seeds, 1 UAV failure):** the honest decomposition **holds on real data** — static 70 %
  detected, auction **90 %** (reallocation **+20 pts**), auction+guided 90 % (guidance **+0** pts,
  **1.3× faster** to 50 %). Confirms the synthetic finding: reallocation does the work; guided search
  is a modest speed-up only.
- **Honest limits (kept):** georeferencing is still synthetic; still simulation-only.
- **Files changed:** `src/eval/benchmark.py` (prior + title params), `src/eval/benchmark_real.py`
  (new), `configs/eval/benchmark_real.yaml` (new), `Makefile`, `docs/`.
- **Note:** a `uv sync --extra perception` had dropped dev/geo from the venv; restored with
  `uv sync --all-extras`. **Fine-tune install should be `uv sync --all-extras`, not `--extra
  perception`** (the latter removes ruff/pytest/osmnx). 95 tests green.

## 2026-08-04 — Option A: controlled head-to-head benchmark (adaptive pipeline vs static baseline)

- **Request:** after the honest re-check (the project is integration-and-evaluation, not novel), make
  a defensible *performance* result. Option A: reproduce the static-SOTA class as a baseline in our
  own simulator and show the integrated adaptive pipeline beats it under realistic stress. (Option C —
  compound perception+failure robustness — re-checked and dropped: it is established, not a gap.)
- **Summary:** `src/eval/benchmark.py` (`make benchmark`) compares two systems on ONE controlled
  scenario (clustered survivors + imperfect prior + a UAV failure + detector FN): **baseline** =
  static partition + uniform sweep, no reallocation/guidance; **adaptive** = auction reallocation +
  probability-guided search. Reports **coverage**, **survivors detected**, and **time-to-locate 80 %**,
  mean ± 95 % CI, as a bar chart + table.
- **Root cause / motivation:** turns the scattered per-feature results (sweep/rq4/search-order) into a
  single apples-to-apples "our system vs a faithful static baseline under realistic conditions" story
  — the strongest honest claim an integration-and-evaluation MSc can make (not "beats the field").
- **Solution / why:** reuses the engine + Coordinator flags (`greedy_priority`, reallocation) + the
  clustered-survivor generator — no new mechanism. Baseline gets a uniform prior (no probability map);
  adaptive gets the imperfect prior. Dataset-free (synthetic detections) so the test runs anywhere.
- **Files changed:** `src/eval/benchmark.py` (new), `configs/eval/benchmark.yaml` (new),
  `tests/test_benchmark.py` (new), `Makefile`, `docs/`.
- **Status:** ✅ done. Headline (30 seeds, one UAV failure, clustered survivors): the adaptive
  pipeline beats the static baseline by **+22 coverage pts (100 % vs 78 %)**, **+21 survivors-detected
  pts (89 % vs 68 %)**, and locates 80 % of survivors **6.1× faster (4.2 vs 25.9 min)**. Honest nuance
  surfaced: the survivor-detection gap has wide variance because survivors are clustered — the
  baseline only loses survivors when the failure strikes the survivor-dense sector (so the test
  asserts detection is *never worse*, strict only on average). Coverage and speed advantages are
  robust. Dataset-free test; 95 tests total; no new mechanism (reuses `greedy_priority` + reallocation).

## 2026-08-04 — Probability-guided search ordering + experiment (addresses the SARCPPF gap)

- **Request:** a gap vs Wu et al. (2024) SARCPPF — their planner searches high-survivor-probability
  areas first over a probability map; ours surveys cells uniformly (FIFO within a sector). Add a
  **probability-guided coverage ordering** and an experiment measuring time-to-locate survivors vs a
  uniform sweep.
- **Summary:** (i) `Coordinator(greedy_priority=True)` — a guarded next-cell rule that, instead of
  FIFO, picks the **nearest high-priority** unvisited cell (lowest `_bid` = travel-discounted
  priority), so a UAV searches a supplied probability map first while respecting travel. Off by
  default → existing behaviour/tests unchanged. (ii) `src/eval/search_order.py` (`make search-order`):
  a single searcher over a grid with **clustered** survivors and an *imperfect* prior probability map
  (informative + noise); compares the survivors-found-over-time curve for guided vs uniform search,
  mean ± 95 % CI, and reports time-to-locate.
- **Root cause / motivation:** honest gap — the fleet does not currently exploit a survivor-likelihood
  prior to search the most-likely cells first; this is the main advantage of probability-map planners.
- **Solution / why:** reuses the existing bid function (`_bid`) as the guidance rule — no new planner,
  and it ties to the drift/containment map (a real probability prior). Prior is imperfect (not the
  ground truth) so the comparison is honest, not circular; the value shrinks as the prior degrades.
- **Files changed:** `src/coordination/allocation.py`, `src/eval/search_order.py` (new),
  `configs/eval/search.yaml` (new), `tests/test_allocation.py`, `tests/test_search_order.py` (new),
  `Makefile`, `docs/`.
- **Status:** ✅ done. Headline (`make search-order`, 69 clustered survivors, 30 seeds, imperfect
  prior): probability-guided search **locates 80 % of survivors in 9.6 ± 0.1 min vs 14.2 min for a
  uniform sweep — 1.5× faster**; the guided detection curve sits left of the sweep's. Value shrinks
  toward zero as the prior degrades (tested: no advantage with a uniform prior), so it is honest, not
  circular. `greedy_priority` is guarded (off by default) → the 91 prior tests are unchanged; 94
  total. Closes the SARCPPF gap: the fleet can now search the survivor-likelihood prior first.

## 2026-08-03 — Contribution B: perception × coordination sensitivity study

- **Request:** turn the honest positioning into a result — study how perception error propagates to
  coordination outcomes (the coupling most work isolates), using the decoupled oracle.
- **Summary:** `src/eval/sensitivity.py` (`make sensitivity`) sweeps the survivor **false-negative
  rate** as a controlled variable and, under a UAV failure, measures the end-to-end **survivor-
  detection rate** (found / ground-truth survivors) for adaptive auction vs static partitioning,
  mean ± 95 % CI over seeds → a line plot + CSV.
- **Root cause / motivation:** perception papers report detector AP; coordination papers assume
  perfect perception. The decoupled design lets the *measured* detector error be a knob over the
  coordination layer, exposing how the two failure modes (missed detections × lost coverage) compound
  and how much adaptive re-tasking mitigates the coverage half.
- **Solution / why:** reuses `engine.run` + `Oracle` (per-run `person` FN override) + the runner's
  failure helper — no engine change. Detection rate = `found_total["person"] / total person rows in
  the cache`, a clean, well-defined denominator.
- **Files changed:** `src/eval/sensitivity.py` (new), `configs/eval/sensitivity.yaml` (new),
  `tests/test_sensitivity.py` (new), `Makefile`, `docs/{JOURNAL,related_work}.md`.
- **Status:** ✅ done. Headline (360 runs, mean ± 95 % CI, 6 UAVs, 2-UAV failure): the **adaptive
  auction hugs the (1 − FN) ceiling** — 100 % of survivors at FN=0, ~51 % at FN=0.5 — because it
  recovers full coverage, so only perception error costs it survivors; **static partition sits a
  persistent ~15–29 pts below** (survivors lost in abandoned cells it never recovers). Cleanly
  separates the two loss factors: perception (irreducible, the ceiling) vs coordination (recovered
  by re-tasking). Test is dataset-free (synthetic detections); 91 tests total; no engine change.

## 2026-08-03 — Presentation batch: coverage-pattern comparison, geo pinpoints + Google Maps, pipeline map

- **Request:** (2) a web button to open the route in Google Maps; (3) compare spiral vs lawnmower
  coverage; (5) show survivor pinpoint (lat/lon) in the web and make results presentation-clear;
  (6) a doc mapping each pipeline step to files/`make` commands.
- **Summary:**
  - **(3)** `coverage.py` gains `spiral_path` + generic `path_length`/`path_coverage_fraction`;
    `src/eval/coverage_compare.py` (`make coverage-compare`) reports path length at equal coverage
    for lawnmower vs spiral over a sector, with a figure — answers "only one flying technique?".
  - **(2/5)** the sim is synthetic-geo but every detection is georeferenced (`origin_wgs84` anchor);
    the web server converts grid metres → WGS84 and returns lat/lon for each survivor cell + the
    drift target, plus a **Google Maps directions URL** through the recommended route's waypoints.
    UI shows the survivor coordinates and an **"Open route in Google Maps"** button.
  - **(6)** `docs/PIPELINE_MAP.md` — a table: pipeline step → file(s) → `make` command → output.
- **Why:** all reuse existing tested pieces; the geo is honest (synthetic anchor, clearly labelled),
  the coverage comparison is standalone (no engine change).
- **Files changed:** `src/coordination/coverage.py`, `src/eval/coverage_compare.py` (new),
  `configs/eval/coverage.yaml` (new), `webapp/{server.py,index.html}`, `tests/test_coverage.py`,
  `Makefile`, `docs/PIPELINE_MAP.md` (new), `docs/JOURNAL.md`.
- **Status:** ✅ done. (3) `make coverage-compare`: on an 8×8 sector both patterns cover 100 %;
  **lawnmower 15,780 m vs spiral 13,308 m** → spiral 16 % shorter here (a real, honest finding; the
  framework can now compare patterns). (2/5) web survivors table shows each pinpoint's **lat/lon**
  (georeferenced, clickable → Google Maps) and the response plan has an **"Open rescue route in
  Google Maps"** button through the safest route's waypoints; verified in-browser. (6)
  `docs/PIPELINE_MAP.md` maps all 15 stages → files → `make` command → output. 90 tests green.

## 2026-08-03 — RQ4 quantitative result: drift-aware search vs the stale sighting

- **Request:** a real quantitative result for the drift/re-tasking contribution (so far the RQ4 loop
  was proven only by unit tests / a mechanism, with no number).
- **Summary:** `src/eval/rq4.py` — a Monte-Carlo experiment that measures the value of re-tasking the
  search to the survivor's **predicted drift region** vs the **stale detection point**. Each seed:
  advect the survivor's *true* position with the flow (one particle); independently predict the drift
  region (`drift_search_region`, its own RNG so the predictor doesn't see the true draw); then score
  two policies — search the detection cell (stale) vs search the 90 % containment zone (drift-aware).
  Metrics over N seeds, mean ± 95 % CI: **localisation rate** (survivor actually inside the searched
  region) and **localisation error** (metres from the search target to the true position).
- **Root cause / motivation:** a survivor in floodwater moves; a stale sighting sends rescue to where
  they *were*. This quantifies how much the drift model helps — the missing RQ4 number.
- **Solution / why:** standalone experiment reusing only the tested Phase-7 advection — no change to
  the engine/coordinator (keeps the core + its 87 tests untouched). The true path and the prediction
  use independent RNG streams so the containment result is a genuine forecast, not circular.
- **Files changed:** `src/eval/rq4.py` (new), `configs/eval/rq4.yaml` (new), `tests/test_rq4.py`
  (new), `Makefile` (`rq4` target), `docs/` (ROADMAP/JOURNAL).
- **Status:** ✅ done. Headline (300 seeds, mean ± 95% CI): the survivor drifts **~893 m** from the
  sighting; **drift-aware search locates them 88% of the time within 108 ± 6 m**, while the **stale
  sighting locates 0%** (error 897 ± 10 m) — re-tasking to the drift zone cuts localisation error by
  **~790 m** and lifts the hit-rate by **88 pts**. Figure `outputs/runs/rq4_*/rq4.png` shows the
  true positions drifted east into the searched zone, far from the western sighting. 2 tests (exact
  zero-diffusion case; drift-aware beats stale on rate + error); 89 tests total; core untouched.

## 2026-08-02 — Interactive web mission visualiser (opt-in demo tool, ADR-003)

- **Request:** a web application to visualise the project in a browser. Flagged first that CLAUDE.md
  lists **"web dashboards"** as a non-goal; the user chose the live interactive app knowing it
  deviates, so it is built as a **separate, scoped demo tool**, not part of the evaluated pipeline.
- **Summary:** `webapp/server.py` — a **standard-library** HTTP server (no Flask, no new dependency)
  that runs the existing engine on `/api/run?strategy=&seed=&fail=…&drift=` and returns the mission
  as JSON; `webapp/index.html` animates it live on a canvas (grid, cells filling in, per-UAV trails,
  a UAV failing to an X with a "reauctioned" banner, playback + scrub + speed, and a result panel:
  coverage / survivors / cells-lost / per-UAV status). `make web` → http://127.0.0.1:8000.
- **Guardrails (why the dissertation framing is safe):** written up as `docs/adr/ADR-003`. Lives
  outside `src/`; produces **no** dissertation result (those still come from `make sweep`, offline +
  seeded); stdlib-only; not in `make test`/`make demo`; localhost-bound; ADR-001 intact (imports
  `src.sim`/`src.coordination`, reads cached detections, never the detector); deterministic.
- **Verified:** backend returns correct JSON (auction+fail → 100 %/0 lost; static+fail → 83 %/6
  lost); UI runs and animates live in a browser (checked in-pane), no console errors. Core 87 tests
  still green — the engine's only change (the guarded `record_trajectory`) predates this.
- **Files changed:** `webapp/{server.py,index.html}` (new), `docs/adr/ADR-003-web-demo-tool.md`
  (new), `.claude/launch.json` (preview), `Makefile` (`web` target), `docs/JOURNAL.md`.
- **Status:** ✅ done.
- **Follow-ups:** (1) UI now shows a banner if opened as a file / server unreachable, plus a
  seed-vs-scenario note (the seed changes only run noise, not the fixed `flood_a` disaster).
  (2) `make web` was crashing with **Error 139 (segfault)** — `ThreadingHTTPServer` ran numpy/pandas
  in spawned request threads, which segfaults on macOS. Switched to a single-threaded `HTTPServer`
  (a local single-user demo needs no concurrency) + swallow client-disconnect `BrokenPipeError`.
  Verified: 6 concurrent + all strategies + drift → all 200, process stable.
  (3) **Enriched the results** on request: a **Number of UAVs** control (engine already supports
  1–6, sweep uses [1,2,4,6]); a **survivors-detected** table with real YOLO **confidence** per cell
  (engine got a guarded `record_detections` flag; the oracle already carries confidence + lat/lon);
  and a **response plan** for the top survivor — its **drift** probability zones (50/90 % containment,
  drawn on the canvas) and the **fastest vs safest** rescue route from base to the *predicted drift
  cell* (reuses Phase 7 + Phase 8). Honest note surfaced in the UI: on `flood_a` the water is sparse
  so fastest == safest for that route ("already the safest here"). Clarified that UAVs are
  autonomously coordinated — you set the *count*, the auction decides positions; manual steering
  would defeat the contribution.
  (4) **Second scenario + dropdown.** `configs/scenario/flood_b.yaml` — a heavy-flood demo variant
  that **reuses flood_a's real detections** (`detections_scenario: flood_a`; same survivors + YOLO
  confidences — not a second perception run) but adds a **contiguous flood barrier** (`flood_overlay`)
  and a **stronger channel current**. Result: fastest vs safest routes now genuinely **diverge**
  (1400 m/risk 1947 through the flood vs 2200 m/risk 1356 around it) and drift is larger (target
  cell 17 vs 16). Server takes a `scenario` param, overlays the barrier in routing, honours the
  detection-source pointer, and merges a scenario flow override into the world; `/api/meta` lists
  scenarios; UI gets a scenario dropdown. Clearly labelled illustrative — the dissertation's
  evaluation still uses flood_a (real detections). Core 87 tests still green.

## 2026-07-30 — Visuals for the simulation: mission animation (GIF) + sweep results chart

- **Request:** the sim itself had no picture (only a terminal summary + `events.json`); add a
  **mission animation** (GIF of the UAVs flying, cells filling in, a failure + reallocation live)
  and a **sweep results chart** (adaptive vs baselines, coverage under 0/1/2 failures, 95 % CI bars).
- **Summary:** (1) `run(..., record_trajectory=True)` appends a per-timestep snapshot (each UAV's
  x/y/status + the surveyed-cell set) to the result — guarded, so normal runs are unchanged.
  `src/sim/animate.py` reruns a scripted-failure mission with that flag and renders a GIF
  (matplotlib `PillowWriter`, no extra dep): grid, surveyed cells shaded, UAVs coloured by status,
  survivor stars revealed as cells are surveyed, and a "UAV-k failed" caption at the failure.
  (2) `plot_sweep(summary, out_dir)` in the runner draws a grouped bar chart from `summary.csv` at
  the largest UAV count → `results.png`, written automatically by `make sweep`.
- **Solution / why:** both are config-driven + seeded, lazy-import matplotlib (Agg), and reuse the
  tested engine/metrics (no new coordination logic). Failure time/UAV for the animation live in a
  config (no magic numbers). Still CLI + static-file output — the GIF is a file, not a UI.
- **Files changed:** `src/sim/engine.py` (record_trajectory), `src/sim/animate.py` (new),
  `src/eval/runner.py` (plot_sweep), `configs/viz/mission.yaml` (new), `Makefile` (animate target),
  `docs/`.
- **Status:** ✅ done. `make animate` → `mission.gif` (27 frames, ~0.2 MB): survey fills in from
  1→36 cells over ~4 min, UAV-2 fails at 100 s (17/36 covered) and its cells are reauctioned so the
  fleet still finishes 100 %. `make sweep` now also writes `results.png` — grouped bars of coverage
  under 0/1/2 failures per strategy with 95 % CI, showing static partition collapse to ~88 %/~74 %
  while the auction holds 100 %. Tuning notes: failure fired early (the 4-UAV survey finishes ~300 s,
  so a 600 s failure never triggered) and the GIF tail is trimmed to a few frames past full coverage.
  87 tests green (record_trajectory is guarded → existing engine runs unchanged).

## 2026-07-30 — Demo ergonomics: a drift figure + a one-command walkthrough

- **Request:** add `make drift` (draw the survivor's drift/containment on a map — drift had no
  standalone figure, only re-tasking + tests) and `make demo` (run the CPU demo end-to-end). Noted
  there is **no UI** — the whole project is CLI + static matplotlib/CSV output (CLAUDE.md non-goal:
  no web dashboards), so the "demo" is a scripted command walkthrough that drops figures in outputs/.
- **Summary:** `src/drift/visualize.py` builds the world + flow from configs, runs
  `drift_search_region` for a configured survivor cell, and plots the particle cloud, the 50/90 %
  containment polygons, the detection point → drifted centroid, and the grid cells the auction
  would re-task toward (`cells_in_region`) → `outputs/drift/<ts>/drift.png`. `scripts/demo.sh`
  runs mission → sweep → drift → RQ4 A/B → routes → routes-osm with headers and (by default) a
  pause between steps.
- **Solution / why:** viz is config-driven + seeded, lazy-imports matplotlib (Agg), and delegates
  all maths to the already-tested Phase-7 functions, so there is no new untested logic (mirrors
  `plot_pareto`/`plot_osm_routes`, which are I/O-only). `make demo PAUSE=0` runs straight through.
- **Files changed:** `src/drift/visualize.py` (new), `configs/drift/default.yaml` (demo block),
  `scripts/demo.sh` (new), `Makefile` (drift, demo targets), `docs/`.
- **Status:** ✅ done. `make drift` → a figure with the detection point, eastward flow arrow, drift
  cloud, 50/90 % containment, drifted centroid, and the two re-task cells (survivor in cell 13
  drifts 541 m east over 30 min; 90 % ≈ 9.9 ha → cells [15, 16]). `make demo` (and `make demo
  PAUSE=0`) runs mission → sweep → drift → RQ4 A/B → routes → routes-osm end-to-end in ~7 s and
  prints where each figure landed. 87 tests green, lint clean.

## 2026-07-30 — RQ4: close the perception → drift → coordination loop

- **Request:** wire survivor-drift prediction (Phase 7) into the auction (Phase 6) so UAVs
  re-task toward where a survivor has *drifted*, not the stale detection point.
- **Summary:** on a `person` detection the auction currently boosts the four grid-neighbours of
  the surveyed cell. Add an opt-in mode where, instead, it projects the survivor's drift with the
  world flow field (`drift.advect.drift_search_region`), maps the containment polygon to cells
  (`cells_in_region`), and boosts *those* cells' priority — so the lowest-bid rule pulls UAVs
  downstream toward the drifting survivor.
- **Root cause / motivation:** the standout novelty is the closed loop; until now drift and the
  auction were built but never connected. RQ4 asks whether drift-aware re-tasking surveys the
  survivor's new location sooner / more often than neighbour-boosting.
- **Solution / why:** integration lives in `Coordinator` (both sides are CPU coordination-side;
  drift only imports numpy+shapely, so ADR-001 still holds — no detector import). Drift uses the
  Coordinator's **own** seeded `Generator`, so turning the mode on/off does **not** perturb the
  engine's oracle-noise stream — the A/B differs only in re-tasking decisions, keeping the
  comparison clean and runs reproducible. Off by default → all existing runs/tests unchanged.
- **Files changed:** `src/coordination/allocation.py`, `src/sim/engine.py` (config + `--drift-retask`
  CLI), `configs/coordination/default.yaml` (drift_retask block), `tests/test_allocation.py`
  (3 drift-retask tests), `docs/`.
- **Status:** ✅ integration done — 84 tests green (3 new: drift boosts only downstream/east cells,
  differs from the neighbour boost, reproducible under seed). ADR-001 still holds (drift pulls in
  only numpy+shapely, never the detector). Runnable via `make sim ... --drift-retask`; off by
  default so Phase 6 baselines are byte-identical.
- **Honest limitation (for the write-up, not hidden):** a *quantitative* RQ4 benefit can't be shown
  on `flood_a`. Two reasons: (1) priority only influences the auction under **reallocation**
  (a UAV death/RTH), and with 4 UAVs / 60 min the unconstrained mission reaches 100 % either way;
  (2) the cache has ~389 `person` detections spread across nearly every cell — a *saturated* field,
  not the single drifting survivor the drift model is for. A clean field experiment needs a
  **sparse-survivor, resource-constrained scenario** and an **oracle that re-detects the survivor at
  the drifted location** (today's oracle serves static per-cell detections). Flagged as the next
  scenario-design step; the mechanism itself is proven by the analytic tests.

## 2026-07-30 — Routing on a real OSM street network

- **Request:** run the hazard-weighted routing (Phase 8) on a real street map, not the synthetic
  lattice.
- **Summary:** use the existing lazy, disk-cached `build_osm_road_graph` (OSMnx) to fetch a real
  road network for a bbox once; convert the OSMnx `MultiDiGraph` to the simple weighted graph the
  Pareto search expects (collapse parallel edges to min length, carry node x/y); apply a spatial
  flood zone (raise risk on nodes inside a region) exactly as the synthetic corridor does; emit a
  distance-vs-risk Pareto front + figure for the real network.
- **Root cause / motivation:** shows the routing generalises off the toy grid — a stronger
  write-up figure. `geo` extra (`osmnx>=1.9`) was already declared; installing it is a sync, and
  it is a **one-time offline cache** step (CLAUDE.md non-goal is *live network at run time*).
- **Solution / why:** conversion + spatial-hazard helpers in `src/routing/graph.py`; a
  `routes-osm` demo in `safe_path.py` gated on the cache; a network-free test drives a synthetic
  OSMnx-shaped `MultiDiGraph` so the graph logic is tested without hitting Overpass.
- **Files changed:** `src/routing/graph.py` (osmnx 2.x bbox fix + `simple_graph_from_osmnx`,
  `apply_flood_zone`, `nearest_node`), `src/routing/safe_path.py` (`main_osm` + `plot_osm_routes`),
  `configs/routing/osm.yaml`, `tests/test_routing.py` (3 network-free OSM tests), `Makefile`
  (`routes-osm`), `docs/datasets.md` (OSM/ODbL citation).
- **Status:** ✅ done. `make routes-osm` fetched a real 219-node / 295-edge London street network
  (cached once, then offline) and produced an **8-point** Pareto front: naive shortest **1141 m at
  risk 3558** (through the flood) → safest **2367 m at risk 0** (detours west around it), plus a map
  figure showing both routes. osmnx 1.x→2.x changed the bbox order (now `(west, south, east,
  north)`) — fixed. 87 tests green; the 3 OSM graph tests run on a synthetic MultiDiGraph so CI
  never touches Overpass.

## 2026-07-28 — Phase 8: hazard-weighted rescue routing (Pareto fronts)

- **Request:** start Phase 8 (last build phase).
- **Summary:** implement `src/routing/graph.py` (road graph + apply detections: `road_blocked`
  removes edges, `water`/`building_damaged` raise edge risk; OSMnx builder cached to disk for
  real data) and `src/routing/safe_path.py` (edge weight = length·(1+λ·risk); sweep λ → Pareto
  front of distance vs cumulative risk, with a naive shortest-path baseline + a matplotlib plot).
- **Root cause / motivation:** RQ6 — rescue routes should trade distance for safety, not take
  one arbitrary compromise; segmentation hazards feed the graph.
- **Solution / why:** synthetic road lattice over the (synthetic-geo) grid keeps it offline +
  testable; OSMnx path is lazy + disk-cached (CLAUDE.md non-goal: no live network at run time).
  networkx already available (via scikit-image). Blocked edges are removed → provably untraversable.
- **Files changed:** `src/routing/{graph,safe_path}.py`, `configs/routing/default.yaml`,
  `tests/test_routing.py`, `pyproject.toml` (matplotlib), `Makefile` (routes target).
- **Note on the demo scenario:** scattered point-detections on a dense grid don't constrain
  routing (a free equal-length detour always exists), so the demo overlays a contiguous flood
  *corridor* — a realistic barrier. Its crossing risk decays **geometrically** north→south
  (`decay**row`), which makes the distance-vs-risk trade-off *convex* so a weighted-sum λ-sweep
  recovers every crossing as a distinct Pareto vertex (a linear gradient makes the points
  collinear and the front collapses to its two endpoints — a documented limitation of
  weighted-sum scalarisation). Front points are deduped by trade-off point, not just by path.
- **Status:** ✅ done. 7 routing tests green (81 total); acceptance holds — a blocked cell's
  edges are removed so no route can traverse it, and the flood barrier yields a ≥5-point
  non-dominated front on the 6×6 test grid. `make routes` writes a clean 3-point Pareto
  trade-off menu for `flood_a` (1000 m/2400 risk → 1400 m/1810 → 2600 m/1401) with the naive
  shortest path pinned to the highest-risk corner. Perception stays decoupled (ADR-001):
  routing reads hazards from the cache, never the detector.

## 2026-07-22 — Phase 7: flood survivor-drift prediction (SAROPS-style, the novel piece)

- **Request:** start Phase 7.
- **Summary:** implement `src/drift/advect.py` — Lagrangian particle advection of a detected
  survivor in flowing water (flow × leeway + turbulent diffusion), Monte-Carlo cloud → 50/90%
  containment polygons; a helper mapping a polygon to grid cells (for drift-driven re-tasking).
- **Root cause / motivation:** RQ4 — the perception→drift→coordination loop is the standout
  novelty; routing/search should target a drifting survivor's *polygon*, not a stale point.
- **Solution / why:** adapts USCG SAROPS (leeway + Monte-Carlo containment) to UAV re-tasking;
  containment via distance-peel convex hull (non-Gaussian-robust, no KDE dependency).
- **Files changed:** `src/drift/advect.py`, `configs/drift/default.yaml`, `tests/test_drift.py`.
- **Status:** ✅ done. All 4 acceptance criteria pass: zero-diffusion uniform flow → exact v·Δt;
  containment area grows with horizon; 90% containment holds ≥90% and generalises to a fresh
  cloud; region→cells mapping works. 74 tests green. RQ4 loop (feed containment cells into the
  auction) is the next small integration.

## 2026-07-22 — Research questions, novelty, and related-work positioning

- **Request:** expand to >3 research questions; explain how area allocation works and how flood
  drift can be predicted from video; what novel additions are possible; which projects to
  compare against.
- **Summary:** wrote `docs/research_questions.md` — 6 research questions (1 primary + 5 sub),
  a "novelty / what more can be added" list, and a related-work/positioning section.
- **Root cause / motivation:** dissertation needs a richer RQ set and an explicit comparison
  frame; the student is scoping the write-up.
- **Solution / why:** consolidated planning artefact so RQs, novelty and related work live in
  one citable place, mapped to which phase answers each.
- **Files changed:** `docs/research_questions.md` (new), `docs/PROGRESS_REPORT.md` (RQ pointer).
- **Status:** ✅ done (doc); explanations of allocation + drift given in chat.

## 2026-07-21 — Phase 9: evaluation harness + Monte-Carlo sweep (the evidence)

- **Request:** start Phase 9.
- **Summary:** implement `src/eval/metrics.py` (coordination metrics from an event log) and
  `src/eval/runner.py` (`make sweep`): grid of {5 strategies incl. ablations} × {1,2,4,6 UAVs} ×
  {none/one/two failures} × {30 seeds}, tidy Parquet + mean ± 95% CI summary answering whether
  adaptive reallocation beats static partitioning.
- **Root cause / motivation:** the Phase-6 mechanism needs *evidence over many seeds/conditions*
  to be a research result, not one scripted case.
- **Solution:** per-seed randomised UAV failures create the abandonment that differentiates the
  strategies; metrics: coverage, time-to-90%, redundant-coverage ratio, survivors-found,
  distance, completion, lost cells. Priority-upweight ablation (`auction_no_priority`); the
  static baseline doubles as the reallocation-off ablation.
- **Why this solution:** the sim is deterministic per seed, so randomised failures + oracle noise
  give honest confidence intervals; the whole grid runs on CPU in seconds.
- **Files changed:** `src/eval/{metrics,runner}.py`, `configs/eval/sweep.yaml`,
  `tests/test_metrics.py`, `tests/test_runner.py`.
- **Status:** ✅ done. `make sweep` = 1800 runs in ~4 s. **Headline: adaptive auction beats
  static partitioning by +12.4 pts (one failure) and +25.5 pts (two failures) in coverage,
  6 UAVs, mean ± 95% CI (100% vs 87.6% / 74.5%).** random_walk matches coverage but wastefully;
  single_uav collapses. 69 tests green.

## 2026-07-21 — Phase 6: dynamic task reallocation (auction) + baselines — THE CONTRIBUTION

- **Request:** start Phase 6.
- **Summary:** implement `src/coordination/allocation.py` — a `Coordinator` with an
  auction (Contract Net) reallocation policy and three baselines behind one interface
  (`single_uav`, `static_partition_no_realloc`, `random_walk`); wire reallocation triggers
  (UAV death/RTH abandons cells → re-auction; high-priority detection → upweight neighbours)
  into the engine.
- **Root cause / motivation:** this is the novel contribution — does adaptive reallocation beat
  static partitioning? Needs the mechanism + the baselines it is measured against.
- **Solution:** bid = a·travel + b·energy_penalty − c·priority (config weights); abandoned cells
  auctioned to still-flying UAVs (lowest bid wins); idle auction UAVs loiter to stay available.
  Engine refactored to be coordinator-driven with a scripted-failure hook (`fail_at`) for the
  acceptance test; backward-compatible with the Phase-4 static `plan`.
- **Why this solution:** Contract Net / Gerkey & Mataric ST-SR-IA applied online is the standard,
  citable mechanism; the baselines share the interface so the comparison is fair.
- **Files changed:** `src/coordination/allocation.py`, `src/sim/engine.py` (coordinator-driven),
  `configs/coordination/default.yaml` (allocation), `tests/test_allocation.py`.
- **Status:** ✅ done. Acceptance test passes: UAV-2 dies mid-sector → auction recovers to 100%
  coverage while `static_partition_no_realloc` stays <95% (loses the cells). All 4 strategies run
  on the real flood_a scenario; random_walk's redundant revisits show up as inflated found-counts
  (a metric Phase 9 will report). 64 tests green; ADR-001 isolation still holds.

## 2026-07-21 — Phase 5: partitioning + coverage paths

- **Request:** start Phase 5.
- **Summary:** implement `src/coordination/partition.py` (divide AOI into N sectors: `grid`
  baseline + `weighted_voronoi` balanced by prior hazard priority) and
  `src/coordination/coverage.py` (boustrophedon sweep within a sector, parameterised by camera
  footprint width and sidelap).
- **Root cause / motivation:** the coordination study needs a fair *static* partitioning +
  coverage baseline to compare the adaptive reallocation (Phase 6) against.
- **Solution:** weighted Voronoi via Lloyd relaxation + greedy boundary rebalancing to equalise
  per-sector workload; boustrophedon with an explicit last-row fix so the final strip is never
  skipped when sector height isn't a multiple of the footprint step.
- **Why this solution:** greedy rebalancing guarantees the <5% workload balance criterion;
  the last-row fix targets exactly the classic coverage bug the acceptance test checks.
- **Files changed:** `src/coordination/{partition,coverage}.py`,
  `configs/coordination/default.yaml`, `tests/test_partition.py`, `tests/test_coverage.py`.
- **Status:** ✅ done. Weighted-Voronoi workload within 5% (uniform priority; spread 2 vs grid's
  10 on a clustered-priority field); coverage >99%; last-row bug caught by test. 58 tests green.

## 2026-07-21 — Remove AI attribution from git history

- **Request:** remove Claude from the contributors list; no AI assistant traced in GitHub.
- **Summary:** stripped `Co-Authored-By: Claude` trailers from all 17 commits and force-pushed;
  stopped adding them going forward.
- **Root cause / motivation:** user does not want AI attribution in the repo history.
- **Solution:** `git filter-branch --msg-filter` to drop the trailer lines, cleaned backup refs
  + gc, `git push --force`. Saved the preference to memory so future commits omit the trailer.
- **Why this solution:** filter-branch is available (git-filter-repo is not) and rewrites all
  messages in one pass; author/committer were already the user, so only the trailer needed removal.
- **Files changed:** none (git metadata only). Flagged for the user: `CLAUDE.md` and
  `PROJECT_PLAN.md`/`TASK_PROMPTS.md` still name Claude Code — their call whether to rename/scrub.
- **Status:** ✅ done — remote history has 0 AI trailers.

## 2026-07-21 — Phase 4: simulator core (world, UAV, engine)

- **Request:** start Phase 4 with a full explanation of choices and alternatives.
- **Summary:** implement `src/sim/world.py` (grid + analytic flow field), `src/sim/uav.py`
  (kinematics + P=P_hover+k·v² energy model + return-to-home), `src/sim/engine.py` (deterministic
  fixed-timestep loop, headless, event log, optional oracle survey hook).
- **Root cause / motivation:** the coordination contribution needs a fast, seed-deterministic
  CPU simulator to run hundreds of Monte-Carlo trials (ADR-001).
- **Solution:** metric grid reusing the scenario geometry; analytic flow fields (uniform/channel/
  radial) from config; energy-aware RTH at `rth_margin × energy_to_base`; engine threads one
  `np.random.Generator` so runs are reproducible; surveys query the oracle from Phase 3.
- **Why this solution:** analytic flow (not CFD) and a point-mass energy model keep it CPU-cheap
  and analytically testable; config-driven constants (no magic numbers).
- **Files changed:** `src/sim/{world,uav,engine}.py`, `configs/sim/{world,uav}.yaml`,
  `tests/test_{world,uav,engine}.py`, `Makefile` (sim target).
- **Status:** ✅ done. 4 UAVs × 60 min in 0.59 s (<2 s target); byte-identical logs under a seed;
  RTH-from-5km lands with energy to spare; 48 tests green.

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

## 2026-07-28 — Two-slide progress deck for presentation

- **Request:** progress report in two slides for a presentation.
- **Summary:** built `docs/progress_slides.pptx` (python-pptx) — Slide 1: project + pipeline +
  status; Slide 2: results (stat callouts + native auction-vs-static coverage chart) + novelty +
  next. Content-QA clean; no local LibreOffice so no rendered preview.
- **Files changed:** `docs/progress_slides.pptx` (new).
- **Status:** ✅ done.
