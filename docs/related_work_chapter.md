# Related Work (draft chapter)

> Draft prose for the dissertation's related-work chapter, expanded from
> [`related_work.md`](related_work.md). Citations marked with a URL are the sources located during
> the literature check (Aug 2026); **complete the author/year/venue for each from the linked source
> before submission**, and do not present any method below as novel to this work (see §7).

## 1. Scope

This project sits at the intersection of four established areas: multi-robot/multi-UAV task
allocation, fault-tolerant area coverage, aerial search-and-rescue (SAR) perception, and
search-object drift modelling. This chapter reviews each, then positions the present work against
them. The recurring finding is that the individual mechanisms used here are well established; the
contribution lies in their integration, the controlled evaluation, and the perception-decoupling
methodology (§7).

## 2. Multi-robot task allocation and market-based methods

The assignment of tasks to a team of robots is a mature field. Smith's **Contract Net Protocol**
(Smith, 1980) introduced the announce–bid–award pattern that underpins auction-based allocation, and
Gerkey and Matarić (2004) provided the taxonomy — single-task/single-robot, instantaneous vs
time-extended assignment — that classifies the scheme used here (ST-SR applied repeatedly online). A
recent systematic review of multi-robot task allocation confirms the breadth of the field and, in
particular, that fault tolerance has historically received comparatively little attention within it
([Systematic Literature Review on MRTA, ACM Computing Surveys, 2024](https://dl.acm.org/doi/10.1145/3700591)).
Market-based methods are repeatedly identified as strong in distributed, real-time adaptation, at the
cost of communication overhead ([Multi-UAV Task Assignment in Dynamic Environments](https://www.researchgate.net/publication/388211666_Multi-UAV_Task_Assignment_in_Dynamic_Environments_Current_Trends_and_Future_Directions)).
The auction reallocation implemented in this project is a direct application of this line of work.

## 3. Fault-tolerant and resilient coverage

Reallocating work when a robot fails is a specific, well-studied sub-problem. Dynamic task-reassignment
frameworks redistribute a failed robot's tasks among survivors using recovery-driven reallocation and
adaptive scheduling ([Fault-Tolerant Framework for Dynamic Task Reassignment](https://www.mdpi.com/2673-4591/120/1/22)).
Most directly comparable to the present system, resilient coverage-path redistribution iteratively
reassigns a *failed robot's coverage path to its neighbours* using **boustrophedon decomposition** —
the same lawnmower sweep used here — to keep monitoring continuous
([Resilient Multi-Robot Coverage Path Redistribution](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644315/)).
This is effectively the mechanism this project implements, and it must be cited as prior art rather
than claimed as new.

## 4. Aerial search and rescue

UAV-based SAR is an active application area. Survivor-detection pipelines run detectors on aerial
imagery in near real time ([UAV-Based Real-Time Survivor Detection](https://www.researchgate.net/publication/351857034_UAV-Based_Real-Time_Survivor_Detection_System_in_Post-Disaster_Search_and_Rescue_Operations)),
and coordination for SAR swarms specifically addresses agent failure: market-based replanning turns
the loss of a UAV into a localized **reverse-auction** reallocation
([Market-Based Replanning for Safety-Critical UAV Swarms in SAR](https://arxiv.org/html/2606.01970)),
and auction/pheromone hybrids coordinate multi-UAV maritime SAR
([Auction- and Pheromone-Based Multi-UAV Cooperative SAR](https://doi.org/10.3390/drones9110794)).
Prior work also shows that prioritising UAV waypoints from survivor information outperforms a plain
lawnmower search ([Weight-Based Exploration for UAV survivor search](https://arxiv.org/pdf/2012.11131)) —
close in spirit to the priority-boosting used here. The auction-on-failure and priority-driven
re-tasking in this project therefore extend, rather than originate, this literature.

## 5. Search-object drift modelling

Predicting where a person drifts is standard in maritime SAR. The **leeway** method quantifies
wind/current-induced drift of objects at sea ([leeway field method](https://arxiv.org/pdf/1111.0750)),
and the US Coast Guard's **SAROPS** advects Monte-Carlo particles under current and leeway to compute
probabilistic containment areas (Kratzke, Stone & Frost, 2010,
[SAROPS](https://www.researchgate.net/publication/224218783_Search_and_Rescue_Optimal_Planning_System)).
The drift model in this project is a direct adaptation of this established method; its
application to *inland-flood UAV re-tasking* — using the drift zone to redirect the search — is the
integration this work contributes (§7), not the drift method itself.

## 6. Perception, its uncertainty, and current estimation

Small-object detection in UAV imagery is the recognised difficulty for aerial survivor detection, and
the benchmark datasets used here — VisDrone (Zhu et al., 2021), SARD (Sambolek & Ivašić-Kos, 2021),
RescueNet (Chowdhury et al., 2023) and FloodNet (Rahnemoonfar et al., 2021) — define that regime. The
effect of perception error on coordination is studied mainly under *adversarial* conditions —
misclassification, mislocalisation and latency as attack models
([Multi-Robot Coordination with Adversarial Perception](https://arxiv.org/abs/2504.09047)) — and via
collaborative-perception fusion, rather than as a controlled study of a benign detector's measured
error rate propagating to coordination outcomes, which is the angle taken in this project's
sensitivity study. Separately, estimating water surface velocity from drone video via optical flow /
large-scale particle image velocimetry is an established hydrology technique
([Surface Flow from Drones](https://www.researchgate.net/publication/303941066_Surface_Flow_Measurements_From_Drones);
[UAV river velocity via optical flow](https://ascelibrary.org/doi/10.1061/9780784483060.018)) — but the
reviewed sources do not connect it to survivor-drift prediction, which is the gap the proposed
video-estimated-flow extension would address.

## 7. Positioning of this work

Against the above, this project makes **no claim to a novel task-allocation, coverage, drift, or
routing algorithm** — each is established (§§2–6). Its contribution is threefold and honest:

1. **A decoupled, reproducible evaluation framework** in which perception is scored on real labelled
   data and then treated as a *controlled input* to a fast, seeded coordination simulator — enabling
   experiments (notably the perception × coordination sensitivity study) that perception-in-the-loop
   simulators cannot run cleanly.
2. **A systematic, statistical evaluation** of adaptive reallocation under UAV failure (mean ± 95 % CI
   over 1,800 runs), quantifying *when and by how much* it helps and when it does not.
3. **An integration** of maritime drift modelling into inland-flood UAV re-tasking, evaluated
   quantitatively.

An MSc-level contribution does not require a new algorithm; it requires competent application,
rigorous evaluation and honest positioning. The above framing is consistent with that standard and
must be maintained throughout the dissertation — the results chapter and abstract should not describe
the coordination mechanism as novel.

## References (complete before submission)

- Smith, R. G. (1980). The Contract Net Protocol. *IEEE Transactions on Computers*.
- Gerkey, B. P., & Matarić, M. J. (2004). A formal analysis and taxonomy of task allocation in
  multi-robot systems. *IJRR*.
- Systematic Literature Review on MRTA (2024), *ACM Computing Surveys* — https://dl.acm.org/doi/10.1145/3700591
- Resilient Multi-Robot Coverage Path Redistribution (Boustrophedon) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644315/
- Fault-Tolerant Framework for Dynamic Task Reassignment — https://www.mdpi.com/2673-4591/120/1/22
- Market-Based Replanning for Safety-Critical UAV Swarms in SAR — https://arxiv.org/html/2606.01970
- Auction- and Pheromone-Based Multi-UAV Cooperative SAR — https://doi.org/10.3390/drones9110794
- Multi-UAV Task Assignment in Dynamic Environments (survey) — https://www.researchgate.net/publication/388211666
- Weight-Based Exploration for UAV survivor search — https://arxiv.org/pdf/2012.11131
- UAV-Based Real-Time Survivor Detection — https://www.researchgate.net/publication/351857034
- Leeway field method — https://arxiv.org/pdf/1111.0750
- Kratzke, Stone & Frost (2010), SAROPS — https://www.researchgate.net/publication/224218783
- Multi-Robot Coordination with Adversarial Perception — https://arxiv.org/abs/2504.09047
- Surface Flow Measurements from Drones — https://www.researchgate.net/publication/303941066
- Remote sensing of river velocity via drone video and optical flow — https://ascelibrary.org/doi/10.1061/9780784483060.018
- Datasets: Zhu et al. (2021) VisDrone; Sambolek & Ivašić-Kos (2021) SARD; Chowdhury et al. (2023)
  RescueNet; Rahnemoonfar et al. (2021) FloodNet (see `docs/datasets.md`).
