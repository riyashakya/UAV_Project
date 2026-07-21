# Interim progress report — Multi-UAV AI framework for disaster response

MSc dissertation (LSBU, Level 7) · Student 4437147 · Supervisor: Anastasios Dagiuklas
Simulation-based · _Report date: 2026-07-21_

---

## 1. Status in one paragraph

The engineering foundation and the perception subsystem are complete: a reproducible,
config-driven codebase; four real UAV datasets unified into two training sets under one
taxonomy; two YOLO11 models trained and evaluated on real labelled imagery; and the
perception→simulator bridge (the cached detection oracle, Phase 3) in place. **The coordination,
simulation, drift, and routing themselves are not built yet** — which is exactly where the
project's research contribution is meant to come from. The project is at a decision point (§7).

## 2. Research questions

**Primary (coordination — the intended contribution):**
> Does *dynamic, detection- and drift-informed task reallocation* among multiple UAVs improve
> disaster-area survey outcomes — coverage %, time-to-90%-coverage, survivors-found-over-time,
> and mission completion under battery failure — compared with static partitioning and simpler
> baselines, and under what conditions (UAV count, hazard density, failure rate)?

**Secondary (perception — supporting empirical study):**
> How well does survivor (`person`) detection transfer from ordinary/SAR aerial imagery to
> post-disaster imagery (the domain gap), and how much does tiled **SAHI** inference recover
> small-object accuracy versus full-frame inference?

## 3. Objectives and scope

The approved proposal spans six objectives (coordination, detection, tracking, NeRF/3DGS, GIS
routing, risk analysis) plus a seventh in the build prompt (flood-drift prediction). This is a
very wide scope for one MSc — see §7. Current position on the **descope ladder**: 3D
reconstruction (Phase 10) and tracking are parked; coordination baselines, Monte-Carlo seeds,
and ablations are protected and must not be cut.

## 4. Method / architecture

Two decisions govern the design (full write-ups in [`adr/`](adr/)):

- **ADR-001 — perception and coordination are decoupled.** YOLO runs offline on real imagery and
  caches detections; the simulator reads them through a single `oracle` and never imports the
  detector. This yields two independently-scored contributions and lets coordination run
  hundreds of Monte-Carlo seeds on CPU. _This decoupling is itself a methodological choice worth
  defending in the viva._
- **ADR-002 — two models, not one.** Boxes for countable objects (person, vehicle); polygon
  masks for spread-out hazards (damaged buildings, blocked roads, water). A bounding box around
  a flood is meaningless, so the two label types are never merged.

## 5. Data

Four real datasets unified into two YOLO datasets (provenance + licences in
[`datasets.md`](datasets.md)):

- **Detect set** (VisDrone + SARD): 9,000 images — **126,897 person**, **205,663 vehicle**.
- **Segment set** (RescueNet + FloodNet): 6,387 images — **building_damaged 6,599**,
  **road_blocked 2,643**, **water 6,597** (road_blocked is a minority class).

Key honest limitation, already designed-for: **no disaster dataset labels survivors**, so
`person` is trained on ordinary/SAR imagery and transferred into the disaster domain. This
domain gap is to be measured and reported, not hidden.

### What the segmentation is *for* (answering a common question)

Detection (Model A) finds **who/what to rescue** — people and vehicles. Segmentation (Model B)
maps **the hazards and terrain that constrain rescue** — its outputs feed later phases:
`road_blocked` masks remove/penalise edges in the rescue road-graph (Phase 8); `water` and
`building_damaged` proximity raise edge risk; and hazard state seeds the simulator's world
(Phase 4) and the flood-drift region (Phase 7). Masks (not boxes) because these are extents,
not countable objects.

## 6. Perception results — how good are the models?

Trained locally on Apple-Silicon (M4, MPS), YOLO11s, 640px, 60 epochs each.

| Model | Task | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|--:|--:|--:|--:|
| A | detect (person, vehicle) | **0.674** | **0.392** | 0.79 | 0.61 |
| B | segment — mask | **0.410** | **0.266** | 0.67 | 0.43 |

**Interpretation (honest):**
- **Model A is a solid baseline.** UAV small-object detection is genuinely hard (in VisDrone,
  ~68% of objects are under 32×32 px); getting ~0.39 mAP@50-95 from a *small* model at *640px*
  is respectable, and two levers we have not yet used — **SAHI tiled inference** and a larger
  image size — should raise it. It is a competent baseline, not state-of-the-art.
- **Model B is moderate and the weaker link.** Disaster-class instance segmentation is hard,
  and `road_blocked` (thin, rare) drags the mean down. 0.41 mask mAP@50 is usable for feeding
  the routing/hazard layer but has clear room to improve (class-balanced sampling, larger model,
  higher resolution).
- Both models were **still improving at epoch 60** (no early stop), so they are under-trained
  by choice (local-compute cap at 640px). A cloud run at 1280px would lift both.
- **These numbers are not yet a research result.** The marked perception result is the
  *comparison* still owed in Phase 2b: SAHI-vs-full-frame on small objects, AP stratified by
  object size, and the VisDrone-only/SARD-only/combined domain-gap study.

### Phase 2b — evaluation results (Model A, 944 val images, COCO metrics)

**Size-stratified AP (full-frame)** — quantifies the core small-object difficulty:

| view | AP@[.5:.95] | AP@50 | AP_small | AP_medium | AP_large |
|---|--:|--:|--:|--:|--:|
| all | 0.389 | 0.665 | **0.258** | 0.585 | **0.778** |
| VisDrone-val | 0.376 | 0.645 | 0.256 | 0.567 | 0.805 |
| SARD-val | 0.479 | 0.876 | 0.276 | 0.528 | 0.696 |

- **Small objects are ~3× harder than large ones** (AP 0.26 vs 0.78) — the headline perception
  finding, and exactly what the small-object literature predicts.
- **SARD is easier than VisDrone** (AP@50 0.88 vs 0.65): SARD persons are larger and scenes less
  dense. This per-source gap is the domain/scene-difficulty signal.

**SAHI vs full-frame ablation — an honest negative result:** naive SAHI *reduced* AP
(all 0.175 vs 0.389; AP_small 0.111 vs 0.258, −0.15). This is a **train/inference object-scale
mismatch**: the detector was trained on full frames downscaled to 640px, so when SAHI feeds it
native-resolution 640px *slices*, objects appear at an unfamiliar (larger) scale. It matches the
known caveat that SAHI's large gains require **slicing-aided fine-tuning** (training on slices),
which we have not done. VisDrone-val (~1400px) is also below SAHI's ideal very-large-image
regime; SAHI is expected to matter more on the true 3000×4000 disaster imagery (Phase 3), *with*
slice-aware training. **Recommended follow-up:** slicing-aided fine-tune, then re-run this
ablation — a clean, well-evidenced result either way.

## 7. Contribution and novelty — the honest assessment

**As of today, the project contains no novel research contribution.** Unifying public datasets
and training YOLO11 is careful, correct engineering, but it applies existing methods that many
projects already do. This is normal at this stage — the contribution is *designed* to come from
the parts not yet built:

1. **Adaptive multi-UAV task reallocation** (Phase 6) evaluated against three baselines with
   Monte-Carlo statistics (Phase 9) — *the* core contribution. The result "adaptive beats static
   by X% (95% CI …), and here is when it does not" is publishable-shaped and marks well.
2. **Survivor-drift-driven re-tasking** (Phase 7) — adapting maritime SAR planning (USCG SAROPS:
   leeway + Monte-Carlo containment) to UAV re-tasking is the most novel idea in the whole plan
   and is what would lift this above "another coverage-planning project."
3. **The decoupled evaluation methodology** (ADR-001) — scoring perception on real data while
   running coordination over hundreds of CPU seeds — is a defensible methodological contribution.
4. **The perception domain-gap + SAHI study** (Phase 2b) — a smaller, clean empirical result.

**To turn setup into contribution:** build (1) and (2) and evaluate them rigorously (baselines,
seeds, confidence intervals, ablations). Without the baselines and statistics, the system is a
demo, not research.

## 8. Is this heading toward "mediocre"? Depth vs breadth

**Direct answer: it is at a fork, and breadth is the road to mediocre.** A project that touches
all six/seven objectives thinly — some detection, some tracking, some NeRF, some routing — reads
as a broad integration exercise and tends to land as a *pass*, not a distinction. The plan's own
scope warning says the same: any **one** of the objectives, done rigorously, is a full MSc.

The route to a **strong** dissertation is **depth on one or two things**, done with baselines,
ablations, and statistics:

- **Go deep on:** multi-UAV adaptive coordination (Phase 6) + its rigorous evaluation (Phase 9)
  — this is the contribution. Amplify novelty by coupling it to **drift-driven re-tasking**
  (Phase 7): a detected survivor drifts in flood water → the search region updates → UAVs
  re-auction. That pairing is genuinely novel and directly ties perception → prediction →
  coordination.
- **Keep as a supporting chapter:** the perception SAHI + domain-gap study (Phase 2b) — you
  already have the models, so this is cheap and clean.
- **Cut / park (breadth, not depth):** NeRF/3DGS (Phase 10), object tracking, and multi-scenario
  sprawl. These are the riskiest, add the least contribution per week, and are first on the
  descope ladder.

Right now the project is **well-engineered but pre-contribution** — neither mediocre nor strong
yet. Pivoting to depth on coordination + drift now makes it strong; adding more breadth (more
datasets, NeRF, tracking) pushes it toward mediocre.

## 9. Limitations (current)

- Perception capped at 640px by local compute (a cloud 1280px run would improve both models).
- SARD pose sub-labels lost in the Roboflow export (single `person` class).
- `road_blocked` class imbalance in the segment set.
- Survivor domain gap is unavoidable and must be reported, not hidden.
- Sim-to-real: this is design-space exploration, not a deployment claim — state this first.

## 10. Next steps (in priority order)

1. **Phase 2b evaluation** — SAHI-vs-full-frame, AP by object size, domain-gap study (turns the
   trained models into the perception chapter's headline figures). Cheap; do it next.
2. **Agree the focus with the supervisor** — confirm depth on coordination + drift, park NeRF.
3. **Phase 4 → 6 → 9** — the simulator core, then the auction reallocation with baselines, then
   the Monte-Carlo evaluation harness. This is the contribution. (Phase 3, the detection
   cache + oracle bridge, is now done.)
4. **Phase 7 drift** — once coordination works, add drift-driven re-tasking.
