# Results and Evaluation (draft)

> Draft for the dissertation results chapter. Numbers are the real outputs of the commands named in
> each section (regenerate to insert the figures). Novelty is positioned per
> [`related_work.md`](related_work.md): auction reallocation, coverage sweeps, and SAROPS drift are
> **established methods** — the contribution is the integrated framework, the controlled evaluation,
> and the perception-decoupling methodology. Do not restate them as novel algorithms.

## 1. Evaluation approach

The system is evaluated in simulation, offline and batch (no real drones, no real-time targets).
Two properties make the results defensible:

- **Reproducibility.** Every stochastic entry point threads an explicit seeded generator, so a given
  seed reproduces a run exactly. Coordination results are reported as **mean ± 95 % confidence
  interval** over 30 seeds per configuration.
- **Decoupled perception (RQ5).** Perception is scored on real labelled imagery and cached; the
  simulator reads those detections through a single oracle interface (ADR-001). This lets the
  *measured* detector error be treated as a controlled variable over the coordination layer
  (Section 6), which a perception-in-the-loop simulator could not do cleanly.

| Research question | Answered by | Command |
|---|---|---|
| RQ1 — does adaptive reallocation improve coverage/response? | §3 | `make sweep` |
| RQ2 — robustness to UAV count and failures | §3–§4 | `make sweep` |
| RQ3 — perception domain gap & small-object difficulty | §2 | `make eval-perception` |
| RQ4 — drift-driven re-tasking | §5 | `make rq4` |
| RQ5 — decoupling as methodology (enables §6) | §1, §6 | `make sensitivity` |
| Probability-guided vs uniform search (SARCPPF comparison) | §7 | `make search-order` |
| RQ6 — hazard-aware routing | §8 | `make routes`, `make routes-osm` |

### Headline benchmark — ablation of the adaptive components (`make benchmark`)

The consolidating result, **decomposed** (an earlier single "adaptive vs static" comparison conflated
two effects and used an unfair capped metric; this replaces it). Three systems run under a UAV failure
with **sparse survivor hotspots**, reporting the detection curve (survivors located vs time), 30 seeds
(`outputs/runs/benchmark_*/benchmark.png`):

| System | Survivors detected | Time to locate 50 % |
|---|--:|--:|
| static (no reallocation) | 68 % | 4.3 min |
| auction (reallocation) | **89 %** | 4.3 min |
| auction + guided search | 89 % | **2.8 min** |

**Honest decomposition:**
- **Reallocation** does the heavy lifting: recovering the failed UAV's coverage lifts detection
  68 % → 89 % (**+21 pts**). But reallocation is a *generic, established* capability — not specific to
  this project.
- The project-specific **probability-guided search adds no extra detections** (both auction variants
  eventually cover the area and reach 89 %); its only benefit is **speed** — it locates the first 50 %
  of survivors ≈ **1.5× sooner** (2.8 vs 4.3 min).

So the measured advantage of the *adaptive* pipeline is mostly generic reallocation, with a modest
time-to-locate gain from guidance. This is stated plainly rather than headlined as a single large
number. It is a controlled comparison against a reproduced static baseline — **not** a claim to beat
the state of the art.

**Confirmed on real data (`make benchmark-real`).** The same ablation, rerun on the **real detection
distribution** (439 real YOLO person detections, real per-cell density and confidences) with a real,
independent prior derived from the flood-water map (survivors were measured to sit *away* from water,
corr ≈ −0.51 — they fled the flood), reproduces the decomposition: static 70 % → auction **90 %**
(reallocation +20 pts) → auction+guided 90 % (guidance +0 pts, 1.3× faster). This grounds the finding
in real perception outputs rather than an invented distribution. *Limitations kept: the
georeferencing is synthetic and the study is simulation-only.*

## 2. Perception (RQ3)

Two YOLO11s models were trained locally (Apple-Silicon M4, 640 px, 60 epochs): **Model A** (detect:
`person`, `vehicle`) and **Model B** (segment: `water`, `building_damaged`, `road_blocked`).

| Model | Task | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|--:|--:|--:|--:|
| A | detect | 0.674 | 0.392 | 0.79 | 0.61 |
| B | segment (mask) | 0.410 | 0.266 | 0.67 | 0.43 |

**Small objects are the core difficulty.** COCO size-stratified AP for Model A is **small 0.26 /
medium 0.59 / large 0.78** — detection quality falls sharply for the small targets that dominate UAV
imagery (≈68 % of VisDrone objects are < 32×32 px), which is precisely the survivor-detection regime.

**Domain gap (RQ3).** `person` accuracy is markedly higher on dedicated SAR imagery than on generic
UAV imagery (AP@50 ≈ 0.88 on SARD vs ≈ 0.65 on VisDrone). Because no disaster dataset labels
survivors, `person` is trained on ordinary/SAR imagery and transferred into the disaster domain; this
transfer cost is measured, not hidden.

**Honest negative finding.** Naive SAHI tiled inference *reduced* AP rather than improving it — a
train/inference scale mismatch (the detector was trained on whole 640 px frames, not slices).
Slicing-aided fine-tuning is the indicated fix and is left as future work. This is reported as a
result in its own right.

## 3. Coordination under UAV failure (RQ1, RQ2)

`make sweep` runs a 1,800-run grid: {5 strategies} × {1, 2, 4, 6 UAVs} × {0, 1, 2 failures} × {30
seeds}. The headline compares the **adaptive auction** against **static partitioning** at 6 UAVs
(coverage = fraction of the area surveyed; figure: `outputs/runs/sweep_*/results.png`).

| Failures | Adaptive auction | Static partition | Δ (pts) |
|---|--:|--:|--:|
| 0 | 100.0 % | 100.0 % | +0.0 |
| 1 | 100.0 % | 87.6 ± 2.3 % | **+12.4** |
| 2 | 100.0 % | 74.5 ± 2.9 % | **+25.5** |

With no failure both reach full coverage. Under failure, static partitioning permanently abandons the
failed UAV's cells, while the auction re-allocates them and still completes coverage. The advantage
**grows with the number of failures** (RQ2). Baselines: `random_walk` also reaches full coverage but
wastefully (survey redundancy 1.1–1.24×); `single_uav` collapses; the `auction_no_priority` ablation
isolates the reallocation from the priority-boost.

*Positioning:* auction/Contract-Net reallocation of a failed robot's coverage is an established
method; the contribution here is the systematic, seeded, CI-backed quantification of *when and by how
much* it helps in this disaster-response setting.

## 4. Robustness (RQ2)

The advantage is not an artefact of one fleet size: the same qualitative pattern holds across 1–6
UAVs (smaller fleets show a larger absolute gap because each lost UAV is a larger fraction of the
fleet). Section 6 extends the robustness analysis to *perception* error.

## 5. Drift-aware search re-tasking (RQ4)

A survivor detected in floodwater drifts; the question is whether re-tasking search to the
**predicted drift region** beats searching the **stale sighting**. `make rq4` advects the survivor's
true position with the flow and, with an *independent* generator (so the forecast never sees the true
draw), predicts the 90 % drift-containment zone; 300 seeds (figure: `outputs/runs/rq4_*/rq4.png`).

| Search policy | Locates survivor | Localisation error |
|---|--:|--:|
| Stale sighting | 0 % | 897 ± 10 m |
| Drift-aware (90 % zone) | 88 % | 108 ± 6 m |

The survivor drifts ≈ 893 m from the sighting, so searching the last-known point locates essentially
no one, while searching the drift zone locates 88 % (the containment calibration holding up) within
≈ 108 m. Re-tasking to the drift region cuts localisation error by ≈ 790 m.

*Positioning:* leeway/Monte-Carlo drift (USCG SAROPS) is standard maritime SAR; the contribution is
its **adaptation to inland-flood UAV re-tasking**, evaluated quantitatively. **Limitation:** the flow
field is an assumed analytic current, not estimated from the imagery (see §8).

## 6. Perception × coordination sensitivity (RQ2, RQ5)

Enabled by the decoupled design (RQ5): the survivor false-negative rate is swept as a controlled
variable and the end-to-end **survivor-detection rate** (found ÷ ground-truth survivors) is measured
under a 2-UAV failure, for adaptive vs static coordination — 360 runs (figure:
`outputs/runs/sensitivity_*/sensitivity.png`).

| FN rate | Adaptive auction | Static partition |
|--:|--:|--:|
| 0.0 | 100 % | 71 % |
| 0.5 | 51 % | 36 % |

The adaptive auction **tracks the (1 − FN) ceiling**: because it recovers full coverage, the only
survivors it loses are those the detector misses. Static partitioning sits a persistent ≈ 15–29 pts
below the ceiling — survivors stranded in the cells it never re-surveys. This **separates two loss
factors**: perception error is an irreducible floor no coordination can remove, whereas coverage loss
is a coordination problem that re-tasking recovers. *(Honesty note: sweeping a detector false-negative
rate against multi-robot detection/coordination is **not new** — see `related_work.md`. Present this
as a competent, scenario-specific sensitivity analysis grounded in this project's own measured
detector error, not as a novel study.)*

## 7. Probability-guided search (comparison with SARCPPF-style planners)

Probability-map planners for SAR (e.g. Wu et al., 2024, deep-RL persons-in-water coverage) search the
most-likely-to-contain-a-survivor cells first; a uniform sweep does not. This project adds a
lightweight alternative — a guided ordering that visits the nearest high-prior cell first (reusing the
auction bid) — and quantifies its value against a travel-efficient uniform boustrophedon sweep. One
searcher, clustered survivors, an **imperfect** prior probability map, 30 seeds
(`make search-order`; figure `outputs/runs/search_*/search_order.png`).

| Policy | Time to locate 80 % of survivors |
|---|--:|
| Uniform sweep | 14.2 min |
| **Probability-guided** | **9.6 ± 0.1 min** |

Guided search locates survivors ≈ **1.5× faster** when an informative prior is available; critically,
the advantage **vanishes as the prior degrades** (with a uniform prior it is no better than the sweep),
so the result measures the *value of prior information* rather than a rigged comparison. This answers
the main advantage of learned probability-map planners without a learned policy, and connects
directly to the drift model — the drift-containment zone (§5) is itself a survivor-likelihood prior.

## 8. Hazard-aware routing (RQ6)

Segmentation hazards fold into a road graph (`road_blocked` removes edges; `water`/`building_damaged`
raise edge risk); sweeping the risk weight traces the **Pareto front** of distance vs cumulative
risk — a menu of routes, not one compromise.

- **Synthetic grid** (`make routes`): a 3-point front from 1000 m (through hazard) to 2600 m
  (safest), the naive shortest path pinned to the highest-risk corner.
- **Real street network** (`make routes-osm`): on a cached 219-node OpenStreetMap extract, an 8-point
  front from **1141 m / risk 3558** (crossing the flood) to **2367 m / risk 0** (detouring around it)
  — demonstrating the method transfers off the synthetic lattice onto real roads
  (`outputs/routing/osm_*/map.png`).

*Positioning:* hazard-weighted shortest-path routing is standard; the contribution is folding the
*segmentation outputs* into the cost and reporting the trade-off explicitly.

## 9. Threats to validity / limitations (honest)

- **Assumed flow field.** Drift uses an analytic current from config, not one estimated from the
  drone imagery — the largest modelling assumption (estimating it from video is the proposed
  extension, Contribution A).
- **Synthetic georeferencing.** `flood_a` places real imagery on a synthetic grid anchored near
  Houston; coordinates are illustrative. `flood_b` is a deliberately-labelled heavy-flood *demo*
  variant reusing `flood_a`'s real detections — not a second perception pass.
- **Saturated survivor field.** `flood_a` has survivors in nearly every cell, which suits the
  aggregate detection-rate metric (§6) but limits single-survivor re-tasking demonstrations in the
  full sim; §5 therefore evaluates drift in a controlled setting.
- **Perception domain gap.** `person` is transferred from non-disaster imagery (§2); disaster-domain
  survivor accuracy would differ.
- **No flight dynamics.** UAVs are point-mass with an energy model; a single lawnmower coverage
  pattern is used (a spiral alternative is compared for path efficiency, `make coverage-compare`:
  15,780 m vs 13,308 m on an 8×8 sector, but this does not model aerodynamics).
- **Established methods.** The coordination, coverage, drift, and routing methods are not novel (see
  [`related_work.md`](related_work.md)); the contribution is integration, controlled evaluation, and
  the decoupling methodology.

## 10. Summary

| RQ | Finding |
|---|---|
| RQ1 | Adaptive reallocation holds 100 % coverage under failure vs static's 87.6 % (1 fail) / 74.5 % (2 fail): **+12.4 / +25.5 pts**. |
| RQ2 | The advantage grows with failures and with smaller fleets; and it recovers the *coverage* half of the loss under detector error (§6). |
| RQ3 | Model A mAP@50 0.674; small-object AP 0.26; a measured `person` domain gap; naive SAHI reduced AP. |
| RQ4 | Drift-aware search locates the survivor **88 % vs 0 %** (error 108 m vs 897 m). |
| RQ5 | Decoupling enables the perception × coordination sensitivity study — the auction tracks the (1 − FN) ceiling. |
| RQ6 | Hazard-weighted routing yields a distance-vs-risk Pareto front on both synthetic and **real OSM** road networks. |
| SARCPPF | Probability-guided search locates 80 % of survivors **1.5× faster** than a uniform sweep (9.6 vs 14.2 min), with the gain scaling with prior quality. |
