# Related work & honest positioning

**Purpose:** state plainly what in this project is *established prior work* and what the *defensible
contribution* is. Written after a literature check (Aug 2026). **Do not claim novelty for the
auction reallocation, the coverage sweep, or the drift model individually — each is established.**
An MSc (Level 7) dissertation does not require a novel algorithm; it requires competent application,
rigorous evaluation, and honest positioning. This document is the honest positioning.

## What is NOT novel (established prior work — cite, don't claim)

| Component in this project | Established body of work |
|---|---|
| Auction / Contract-Net reallocation of tasks | Smith (1980) Contract Net; Gerkey & Matarić (2004) MRTA taxonomy; [ACM survey on MRTA (2024)](https://dl.acm.org/doi/10.1145/3700591) |
| **Reallocating a failed robot's coverage to neighbours** | [Resilient Multi-Robot Coverage Path Redistribution using Boustrophedon Decomposition](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644315/) — almost the exact mechanism, same lawnmower sweep |
| Market/auction reallocation on UAV failure (SAR) | [Market-Based Replanning for Safety-Critical UAV Swarms in SAR](https://arxiv.org/html/2606.01970) (reverse-auction on node loss); [Auction- & Pheromone-Based Multi-UAV SAR](https://doi.org/10.3390/drones9110794) |
| Boustrophedon ("lawnmower") coverage; grid vs Voronoi partition | Classic coverage-path-planning literature |
| Survivor **drift** via leeway + Monte-Carlo containment | USCG **SAROPS** (Kratzke, Stone & Frost, 2010); [leeway field method](https://arxiv.org/pdf/1111.0750) — maritime SAR standard |
| **Drift-aware / moving-target multi-UAV search with dynamic re-tasking** (RQ4's mechanism) | [AUV coverage path planning from target-drift prediction (2026)](https://onlinelibrary.wiley.com/doi/10.1002/rob.70053); [multi-UAV moving-target search + dynamic reallocation](https://www.mdpi.com/2504-446X/8/8/378); [persons-in-water coverage via drift-trajectory RL](https://www.sciencedirect.com/science/article/abs/pii/S0029801823027877); [dynamic target-location probability path-finding (2024)](https://www.tandfonline.com/doi/full/10.1080/0305215X.2024.2446590) |
| Prioritising UAV waypoints from survivor info vs lawnmower | [Weight-Based Exploration for UAV survivor search](https://arxiv.org/pdf/2012.11131) |
| Estimating water current from drone video | [Surface flow from drones](https://www.researchgate.net/publication/303941066_Surface_Flow_Measurements_From_Drones); [UAV river velocity via optical flow](https://ascelibrary.org/doi/10.1061/9780784483060.018) (hydrology/LSPIV) |
| Perception uncertainty affecting coordination | [Multi-Robot Coordination with Adversarial Perception (2025)](https://arxiv.org/abs/2504.09047) (mostly adversarial / collaborative-perception fusion) |

## Where the honest contribution lies (integration + evaluation + methodology)

The individual methods are known; what is less common is the **specific combination and the
controlled evaluation**. Candidate contributions, ranked by how defensible they are:

- **(B) Perception × coordination sensitivity study.** The decoupled cached-oracle design lets the
  detector's *measured* false-negative rate be swept as a controlled variable. **Honesty correction
  (Aug 2026 re-check):** sweeping a detector FN rate and studying its effect on multi-robot
  detection/coordination *is* done in prior work (FN rates of ~0.2 are used, and detection
  probability is known to fall ~linearly in FN). So this is **not a distinctive idea**; at most, the
  modest angle is grounding FN in *this project's own measured* detector statistics and the
  adaptive-vs-static coverage-recovery framing. Present as a competent sensitivity analysis, not a gap.
- **(A) Video-estimated flow → drift.** Replace the *assumed* flow field with a current estimated
  from the drone imagery (optical flow / LSPIV). **Honesty correction:** drone-video current
  estimation is well established (hydrology/LSPIV; a 2026 drone-wave ocean-current method), UAV
  real-time water segmentation + surface-velocity sensors exist (RivAIr), and SAR already
  *conceptually* uses current + elapsed time to predict a drowning victim's location. So this is
  **less novel than earlier claimed** — at best an under-explored *engineering integration* into a
  UAV re-tasking loop, and a proof-of-concept only. Verify carefully before any novelty claim.
- **(C) Adapting maritime SAROPS drift to *inland flood* UAV re-tasking** — an integration/adaptation
  claim, evidenced by the RQ4 result (`make rq4`: search the 90% drift zone locates 88% vs 0% for
  the stale sighting). Position as adaptation, not a new drift algorithm.
- **(D) Honest small-object finding** — naive SAHI tiling *reduced* AP (train/inference scale
  mismatch); a reportable empirical result for survivor detection.
- **(E) Reproducible benchmark** — the decoupled, seeded, CPU-only Monte-Carlo harness as an open,
  reusable benchmark for coordination-under-failure with realistic perception (tools/reproducibility).

## Closest prior art (found on Aug 2026 re-check — READ THESE; cite as the nearest neighbours)

An honest re-check found integrated frameworks very close to this project. Do not describe the
integration itself as distinctive.

- **[AI-Enhanced UAV Clusters for SAR in Natural Disasters (Algorithms, 2026)](https://doi.org/10.3390/a19010031)**
  — the nearest neighbour: integrates **YOLOv8 human detection** (VisDrone weights — the same data
  this project uses) + **multi-UAV routing/coordination** + **lawnmower coverage** in **simulation**
  (100 % coverage of a 17.6 km² area with 16 UAVs). This is essentially the same detection +
  coordination + coverage/routing pipeline. The project's honest deltas versus it are *incremental*:
  the **drift → re-tasking** loop, the **hazard-weighted routing** and **decoupled controlled
  perception-error study**, the **inland-flood** framing, and the honest negative findings.
- **[Comparative Evaluation of YOLO for Human Position Recognition with UAVs During a Flood](https://doi.org/10.3390/asi9010006)**;
  **[Real-Time SAR small-object YOLO detection (Drones, 2025)](https://doi.org/10.3390/drones9080514)**
  — flood/UAV YOLO survivor detection is well trodden.
- **[Multi-UAV Flood Monitoring coverage control (2025)](https://arxiv.org/pdf/2510.19548)** — multi-UAV
  flood-region coverage.
- FN-rate sensitivity in multi-robot search/coordination: see the multi-robot performance-prediction
  and adversarial-perception literature (§ table).

**Implication for positioning:** this is best framed as an **integration-and-evaluation study** that
combines and rigorously compares established components in an inland-flood UAV context, with a few
incremental additions — *not* as a novel framework. An MSc can absolutely be that; the marks come
from rigour, critical analysis, and honest positioning, not from a novelty claim.

## How to phrase it in the dissertation

> "This work does not propose a new task-allocation algorithm; auction-based reallocation of a
> failed robot's coverage is established [refs]. The contribution is (i) a decoupled framework that
> lets real, offline-measured perception error be treated as a controlled variable over the
> coordination layer, and (ii) an empirical evaluation of how that perception error, and UAV
> failures, propagate to disaster-response outcomes, together with an adaptation of maritime drift
> modelling to inland-flood UAV re-tasking."

## Next step

Do a focused reading pass on the closest refs above (especially the boustrophedon redistribution
paper and the reverse-auction SAR swarm) and expand each row into a paragraph. Verify (A) and (B)
against a dedicated search before making any novelty claim.
