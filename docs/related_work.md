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
| Prioritising UAV waypoints from survivor info vs lawnmower | [Weight-Based Exploration for UAV survivor search](https://arxiv.org/pdf/2012.11131) |
| Estimating water current from drone video | [Surface flow from drones](https://www.researchgate.net/publication/303941066_Surface_Flow_Measurements_From_Drones); [UAV river velocity via optical flow](https://ascelibrary.org/doi/10.1061/9780784483060.018) (hydrology/LSPIV) |
| Perception uncertainty affecting coordination | [Multi-Robot Coordination with Adversarial Perception (2025)](https://arxiv.org/abs/2504.09047) (mostly adversarial / collaborative-perception fusion) |

## Where the honest contribution lies (integration + evaluation + methodology)

The individual methods are known; what is less common is the **specific combination and the
controlled evaluation**. Candidate contributions, ranked by how defensible they are:

- **(B) Perception × coordination sensitivity study — strongest & most buildable.** The decoupled
  cached-oracle design lets the detector's *measured* false-negative rate be swept as a controlled
  variable, quantifying how perception error propagates to coverage / survivors-found / time. The
  literature typically isolates perception *or* coordination; studying the *coupling* with real
  detector statistics is an empirical gap. Uses the existing `make sweep` + oracle noise.
- **(A) Video-estimated flow → drift — most novel-leaning.** Replace the *assumed* flow field with a
  current estimated from the drone imagery (optical flow / LSPIV), bridging UAV hydrometry and SAR
  drift. The two fields exist separately; connecting them for survivor drift + UAV re-tasking is not
  in the results found. Realistically a proof-of-concept (needs water-surface video) + honest limits.
- **(C) Adapting maritime SAROPS drift to *inland flood* UAV re-tasking** — an integration/adaptation
  claim, evidenced by the RQ4 result (`make rq4`: search the 90% drift zone locates 88% vs 0% for
  the stale sighting). Position as adaptation, not a new drift algorithm.
- **(D) Honest small-object finding** — naive SAHI tiling *reduced* AP (train/inference scale
  mismatch); a reportable empirical result for survivor detection.
- **(E) Reproducible benchmark** — the decoupled, seeded, CPU-only Monte-Carlo harness as an open,
  reusable benchmark for coordination-under-failure with realistic perception (tools/reproducibility).

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
