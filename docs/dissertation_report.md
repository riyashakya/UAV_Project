# A Simulation Framework for Coordinated Multi-UAV Disaster Response: Perception-Decoupled Coordination, Survivor-Drift Re-tasking and Hazard-Aware Routing

**MSc Dissertation — London South Bank University (Level 7)**
Student 4437147 · Supervisor: Anastasios Dagiuklas

> Draft report built from the project's own code, configs and generated results. Numbers are the
> outputs of the named `make` commands; regenerate to insert the figures. The positioning is
> deliberately honest: the methods used are established, and the contribution is one of integration,
> rigorous evaluation and an honest methodology, not a novel algorithm. Complete every reference
> marked "to confirm" against its source before submission.

---

## Abstract

This dissertation builds and evaluates a simulation framework for coordinating a small fleet of
uncrewed aerial vehicles (UAVs) in a flood disaster. The system partitions an area of interest into
sectors, assigns one UAV per sector, surveys each sector, detects survivors and hazards using two
YOLO11 models trained on real labelled imagery, reallocates work when a UAV fails, predicts where a
survivor in floodwater will drift, and emits hazard-weighted rescue routes. Perception and
coordination are deliberately decoupled: detections are computed offline on real imagery and cached,
and the simulator reads them through a single oracle interface. This design lets a measured detector
error rate be treated as a controlled variable over the coordination layer, and lets hundreds of
seeded Monte-Carlo runs execute in minutes on a CPU.

The work makes no claim to a novel algorithm. Auction-based task reallocation, boustrophedon
coverage, leeway-based drift prediction and hazard-weighted routing are all established, and
integrated UAV search-and-rescue frameworks that combine detection with coordination already exist.
The contribution is the integration, the controlled and honest evaluation, and the decoupled
methodology. Across 1,800 seeded runs the adaptive auction holds full area coverage under UAV failure
where a static partition falls to 87.6% (one failure) and 74.5% (two failures). A three-way ablation
shows this advantage is driven almost entirely by generic reallocation; the added probability-guided
search contributes no extra detections and only a modest time-to-locate speed-up, and this
decomposition holds when the ablation is grounded in the real detection distribution. Drift-aware
search locates a drifted survivor 88% of the time against 0% for the stale sighting. Naive tiled
inference reduced detection accuracy rather than improving it. The report states these results, and
their limitations, plainly.

---

## Table of Contents

- Chapter 1: Introduction
  - 1.1 Problem Motivation
  - 1.2 Background and Context
  - 1.3 The Identified Gap
  - 1.4 Aim, Objectives and Research Questions
  - 1.4a Statement of Contributions
  - 1.5 Methodology Overview
  - 1.6 Headline Findings
  - 1.7 Project Plan (Gantt Chart)
  - 1.8 Risk Management Table
  - 1.9 Dissertation Structure
- Chapter 2: Literature Review
  - 2.1 Introduction
  - 2.2 UAV Perception for Disaster Imagery and Small-Object Detection
  - 2.3 Multi-Robot Task Allocation and Auction Methods
  - 2.4 Fault-Tolerant Coordination and Area Coverage
  - 2.5 Survivor Drift Prediction and Moving-Target Search
  - 2.6 Hazard-Aware Routing and Decoupled Evaluation
  - 2.7 Legal, Social, Ethical and Professional Issues
  - 2.8 Summary and Identified Research Gaps
  - 2.9 Literature Review Summary Table
- Chapter 3: Methodology
  - 3.1 Overview of Research Design
  - 3.2 Datasets and Synthetic Scenario Generation
  - 3.3 The Perception Pipeline (Two Models)
  - 3.4 Coordination: Partitioning, Coverage and Auction Reallocation
  - 3.5 Survivor Drift Prediction and Re-tasking
  - 3.6 Hazard-Aware Routing
  - 3.7 The Oracle Bridge, Software Verification and Reproducibility
  - 3.8 Evaluation Methodology
- Chapter 4: Results and Analysis
  - 4.1 Perception Accuracy and the Domain Gap
  - 4.2 Coordination under UAV Failure
  - 4.3 Decomposing the Advantage: A Three-Way Ablation
  - 4.4 Drift-Aware Search Re-tasking
  - 4.4a Measured versus Assumed Current for the Drift Forecast
  - 4.5 Perception × Coordination Sensitivity
  - 4.5a Can Coordination Claw Back the Bottleneck by Looking Twice?
  - 4.6 Hazard-Aware Routing (Synthetic and Real Networks)
  - 4.7 Coverage-Pattern Comparison
  - 4.8 Synthesis
  - 4.9 Comparison with Reported Systems and Positioning
- Chapter 5: Discussion and Future Work
  - 5.1 Overview
  - 5.2 Achievement Against Research Questions
  - 5.3 Limitations
  - 5.4 Future Work
  - 5.4a What the Coupling Result Means, and What It Does Not
  - 5.5 Broader Significance
- Chapter 6: Conclusion
  - 6.1 What This Dissertation Set Out to Do
  - 6.2 What Was Achieved
  - 6.3 Reflection on Learning
  - 6.4 Final Statement
- Appendices
- References

---

# Chapter 1: Introduction

## 1.1 Problem Motivation

Floods are the most frequent natural disaster worldwide, and the first hours after a flood are the
period in which most survivors can still be reached. Ground teams are slowed by blocked roads,
standing water and damaged buildings, and often cannot see where survivors are. UAVs help because
they cover ground quickly and send back imagery from above, and a fleet of several UAVs can cover a
large area in parallel. A fleet is only useful, though, if the UAVs coordinate: if they simply fly
fixed routes, then one drone running out of battery or failing mid-mission leaves its part of the
area unsearched, and there is no mechanism to send more effort towards places where survivors have
actually been found.

Three specific difficulties motivate this project. First, survivors in aerial imagery are small —
often a handful of pixels — so a detector that works on ordinary photographs can miss them. Second,
drones fail; a plan that assumes every drone completes its route is not realistic. Third, a person in
moving floodwater does not stay where they were first seen, so a rescue route computed to the
original sighting sends help to the wrong place.

This project addresses these as a coordination and evaluation problem rather than a hardware problem.
It is a simulation study. There are no physical drones, no flight controller, and no real-time
target. The reason is practical: a coordination policy can be tested across hundreds of failure
scenarios in a simulator in minutes, whereas doing so with real hardware is slow, expensive and
unsafe. The perception side is trained and scored on real labelled imagery so that the detection
numbers are honest, and the two are connected through a single, well-defined interface. The choice to
simulate is therefore a choice about where rigour is affordable, not an attempt to avoid the hard
parts of the problem.

## 1.2 Background and Context

The system sits at the meeting point of four established areas. Object detection on UAV imagery
provides the "who and what" — people, vehicles, and the hazards (water, damaged buildings, blocked
roads) that constrain a rescue. Multi-robot task allocation provides the machinery for dividing work
among a fleet and reassigning it when circumstances change. Search-and-rescue drift modelling, taken
from maritime practice, predicts where a person in water will move. Path planning on a hazard-weighted
graph provides rescue routes that trade distance against danger.

Each of these areas is mature. The detection stack uses YOLO11, a current single-stage detector.
Task allocation follows the Contract Net Protocol (Smith, 1980) and the multi-robot task-allocation
taxonomy of Gerkey and Matarić (2004); in that taxonomy the scheme here is single-task, single-robot,
instantaneous assignment applied repeatedly online. Drift modelling adapts the ideas behind the US
Coast Guard's Search and Rescue Optimal Planning System (Kratzke, Stone and Frost, 2010), which
advects Monte-Carlo particles under a current and a leeway factor. Routing uses weighted-sum
shortest-path search over a road graph, with the road graph coming either from a synthetic grid or a
real OpenStreetMap extract (Boeing, 2017).

The context also includes a deliberate design decision that shapes the whole project. Rather than run
the detector inside the simulator, perception is run once, offline, and its outputs are cached. The
simulator reads those cached detections when a UAV enters a cell. This "decoupling" is discussed at
length in Chapter 3; the short version is that it separates two questions that are usually tangled —
how good is the detector, and how good is the coordination — so that each can be measured properly,
and it makes the coordination experiments cheap enough to run at the scale needed for confidence
intervals.

## 1.3 The Identified Gap

An honest account of the gap is necessary, because it is not a gap in the sense of a missing
algorithm. A literature review carried out during the project (Chapter 2, and the comparison in
Section 2.9) found that every individual component is established, and that integrated UAV
search-and-rescue systems combining detection with multi-UAV coordination and coverage already
exist — the closest being an AI-enhanced UAV-cluster framework that pairs YOLOv8 detection with
multi-UAV routing in simulation (Karystinos and colleagues, 2026; full citation to be confirmed).
Fault-tolerant reallocation on UAV failure, drift-aware search, probability-guided search and the
effect of perception error on coordination have all been studied.

The gap is easier to see through a concrete example. There are only two reasons a survivor is not
found in a drone search. Either a drone flew over them but the detector did not recognise them — a
perception failure — or no drone ever reached them, for instance because one drone failed and its
area was abandoned — a coverage failure. This distinction is not new, and it would be wrong to
present it as such: it is the foundation of classical search theory, where the probability of finding
a target is the probability of searching its location multiplied by the probability of detecting it
once searched (Koopman, 1980; Stone, 1975). Coverage planning with an explicit, imperfect probability
of detection is an established sub-field, and the effect of perception error on multi-robot
coordination has been studied directly. So the coupling is well understood in theory. What is less
common, and is where this project sits, is to drive that detection term not from an assumed analytical
sensor model but from the measured error profile of a modern deep object detector, and to sweep it
over a fault-tolerant coordination policy in a reproducible testbed. The project's role is therefore
to instantiate and measure a known coupling for a modern learned-perception, market-based-coordination
pipeline, and to separate how much of a missed survivor is each side's fault — not to claim the
coupling as a discovery.

The gap this project can honestly claim is narrow, and it is one of evaluation rather than of theory.
Applied UAV search-and-rescue systems tend to report detector accuracy while assuming coordination is
perfect, or report coordination performance while assuming perception is perfect; search theory couples
the two but usually with an assumed, analytical detection model. The decoupled cached-oracle design
used here lets the detector's *measured* false-negative rate be swept as a controlled input to the
coordination layer, which the applied systems surveyed here rarely do with a real detector rate.
Alongside that, the project provides a reproducible, seeded, CPU-only testbed that runs the full
pipeline (detection through coordination-under-failure to drift and routing) and reports results with
confidence intervals and ablations. The gap, then, is a gap in
rigorous, controlled, reproducible evaluation of the perception–coordination coupling, not a gap in
method. Naming the gap this precisely is itself part of the contribution, because it protects the
work from the more common failure of claiming novelty that a reviewer can quickly disprove.

## 1.4 Aim, Objectives and Research Questions

The aim is to build and rigorously evaluate an integrated, reproducible simulation framework for
coordinated multi-UAV flood response, and to quantify honestly what its components do and do not
contribute.

The objectives are:

1. Train and evaluate two perception models on real labelled imagery, and report accuracy stratified
   by object size and dataset, including any negative findings.
2. Build a deterministic, seeded simulator with an energy model and a scripted-failure mechanism.
3. Implement auction-based reallocation and three baselines behind one interface.
4. Adapt a maritime drift model to inland-flood UAV re-tasking, and quantify its value for locating a
   drifted survivor.
5. Implement hazard-weighted routing on both a synthetic grid and a real road network.
6. Evaluate the coordination layer with Monte-Carlo simulation, confidence intervals and ablations,
   including a controlled study of how perception error propagates to coordination outcomes.

Underneath all of this is one practical motivation: after a flood, how can a small team of drones
help find survivors faster and get rescuers to them, and what makes the biggest difference to that
outcome? The research questions are the concrete, answerable pieces of that motivation. They are
phrased as problems a rescue team would recognise rather than as the methods used to answer them, so
that each could honestly have come out negative.

- **RQ1 (coverage under failure).** When a team of drones is searching and one fails partway through,
  does its area simply go unsearched, or can the fleet reorganise so the whole area is still covered —
  and how much difference does that make to survivors found?
- **RQ2 (robustness).** How does that difference change with the number of drones, the number of
  failures, and how often the detector misses?
- **RQ3 (seeing survivors).** How reliably can survivors be spotted in drone imagery, where does
  detection break down, and does slicing large images into tiles help or hurt?
- **RQ4 (a moving survivor).** If a survivor is carried by floodwater, is it better to search where
  they were last seen or where they are predicted to have drifted to? This is framed as an adaptation
  of maritime drift modelling, not as a new method.
- **RQ5 (where the effort should go).** When survivors are missed, how much of the loss is the
  detector failing versus the drones not covering the ground — and can keeping perception separate
  from coordination make that split measurable in the first place?
- **RQ6 (reaching them safely).** Once a survivor's location is known, what is the safest route to
  reach them across flooded, partly blocked roads, and how much longer is safe than shortest?

## 1.4a Statement of Contributions

The project claims no new algorithm; every method it uses is established. Its contributions are of
the integration, evaluation and reproducibility kind, which is the appropriate register for this
work. Stated plainly, they are:

1. **A controlled measurement of the perception–coordination coupling for a modern pipeline.** The
   coupling itself is classical — search theory has separated coverage from probability of detection
   since the 1940s — so the contribution is the instantiation, not the idea: using the decoupled
   oracle, the *measured* error profile of a real deep detector is swept as an input to a
   fault-tolerant coordination policy, and the loss of survivors is split into a perception part and a
   coverage part (Section 4.5). The value lies in grounding the detection term empirically and making
   the split reproducible, not in the coupling, which is well established.
2. **A quantified, honest evaluation of fault-tolerant coordination.** Auction reallocation is
   measured against a static partition and baselines across 1,800 seeded runs with confidence
   intervals, and an ablation decomposes the advantage into reallocation (large) and added guided
   search (negligible for detection totals) — Sections 4.2 and 4.3.
3. **Two reported negative findings.** Tiled inference reduced accuracy without fine-tuning, and
   guided search added no detections. Trustworthy negative results are a genuine contribution.
4. **An end-to-end integrated pipeline** spanning detection, coordination under failure, drift
   prediction and hazard-aware routing — the widest coverage among the compared systems (Section 2.9),
   even though no single component is new.
5. **A reproducible, seeded, CPU-only testbed** — configuration-driven, one hundred tests, running
   the full pipeline in minutes — that others could reuse to measure the same coupling.
6. **An engineering improvement to the drift model:** replacing its assumed current with one measured
   from the drone imagery by particle image velocimetry, shown to cut localisation error in a proof of
   concept (Section 4.4a).

## 1.5 Methodology Overview

The work is quantitative and comparative. Perception is trained with Ultralytics YOLO11 on a unified
dataset and scored against ground truth. Coordination is evaluated by running a deterministic
simulator many times under controlled conditions and comparing an adaptive method against baselines,
reporting means with 95% confidence intervals over independent seeds. Configuration lives in YAML
files, so there are no hard-coded parameters in the source; every stochastic entry point takes an
explicit seeded random generator, so a run reproduces exactly. The perception–coordination interface
is a cached detection table read through an oracle that adds configurable noise. Chapter 3 gives the
detail.

## 1.6 Headline Findings

- Model A (person, vehicle) reached mAP@50 of 0.674; size-stratified average precision was 0.26 for
  small objects, which is the survivor-detection regime, against 0.78 for large objects.
- Under one UAV failure the adaptive auction held 100% coverage against a static partition's 87.6% ±
  2.3%; under two failures, 100% against 74.5% ± 2.9% (six UAVs, 1,800 runs).
- A three-way ablation showed this gain is almost entirely generic reallocation. The added
  probability-guided search produced no extra detections and only a 1.3–1.5× time-to-locate speed-up.
  The decomposition held on the real detection distribution as well as the synthetic one.
- Searching the predicted 90% drift zone located a drifted survivor 88% of the time within about
  108 m; searching the stale sighting located them 0% of the time (the survivor drifts about 893 m).
- Naive tiled (SAHI) inference reduced detection accuracy, a train/inference scale mismatch reported
  as a negative finding rather than hidden.

## 1.7 Project Plan (Gantt Chart)

The project ran over roughly twelve weeks. The plan below is indicative; the ordering reflects the
build log in the repository.

| Week | Activity |
|---|---|
| 1 | Repository scaffold, tooling, architecture decisions, dataset survey |
| 1–2 | Dataset unification (VisDrone, SARD, RescueNet, FloodNet) |
| 2–3 | Perception training (Model A detect, Model B segment) |
| 3 | Perception evaluation: size-stratified AP, domain gap, tiled inference |
| 4 | Detection cache and oracle bridge |
| 5 | Simulator core: world, UAV energy model, engine |
| 6 | Partitioning and coverage paths |
| 7–8 | Auction reallocation and baselines |
| 9 | Monte-Carlo evaluation harness |
| 10 | Survivor-drift prediction |
| 11 | Hazard-weighted routing (synthetic and OpenStreetMap) |
| 11–12 | Ablations, real-data grounding, literature positioning, write-up |

## 1.8 Risk Management Table

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No dataset labels survivors in the disaster domain | High | High | Transfer `person` from SAR/UAV imagery; measure and report the domain gap rather than hide it |
| Local compute insufficient for training | Medium | Medium | Train a small model (YOLO11s) at 640 px on Apple-Silicon MPS; keep epochs modest |
| Scope creep across too many components | High | Medium | Prioritise coordination and its evaluation; treat 3D reconstruction as a descope candidate |
| Overclaiming novelty in a saturated field | Medium | High | Literature check; position as integration and evaluation; state established methods as such |
| Simulation results not reproducible | Medium | High | Seed every random source; pin dependencies; write tests as a contract |
| Synthetic scenarios biasing the result | Medium | Medium | Ground one experiment in the real detection distribution; document the limitation |

## 1.9 Dissertation Structure

Chapter 2 reviews the relevant literature and states the gap. Chapter 3 describes the design: the
datasets, the perception pipeline, the simulator and coordination, the drift and routing components,
and the decoupled evaluation. Chapter 4 presents the results with their confidence intervals and the
ablation that decomposes the advantage. Chapter 5 discusses the results against the research
questions, states the limitations frankly, and sets out future work. Chapter 6 concludes.

---

# Chapter 2: Literature Review

## 2.1 Introduction

This review covers the four areas the system integrates — UAV perception, multi-robot task
allocation, drift-based search, and hazard-aware routing — and the methodological question of how to
evaluate a coupled perception–coordination system. The purpose is twofold: to place the work in
context, and to be explicit about what is already established, so that no part of the system is
presented as novel when it is not. Each section states what the established methods are, what recent
work adds, and what the project borrows. The review closes with a summary table comparing the project
against the ten most similar systems and with a statement of the gap.

## 2.2 UAV Perception for Disaster Imagery and Small-Object Detection

The central perception difficulty in aerial search is object size. In the VisDrone benchmark (Zhu et
al., 2021), a large fraction of annotated objects are smaller than 32×32 pixels, and a person seen
from altitude occupies a small share of the frame. Small objects are hard for convolutional detectors
because repeated downsampling in the backbone leaves them with few pixels by the time features reach
the detection head, and because their signal is easily lost against textured backgrounds such as
rubble or water. This is the reason whole-image mean average precision can look acceptable while the
detector is failing on exactly the objects a search cares about, and it is why size-stratified
reporting matters.

Single-stage detectors of the YOLO family are the common choice for UAV human detection because they
run fast and detect small objects reasonably when trained appropriately. The family has moved from the
original single-scale design through the addition of feature-pyramid necks, anchor-free heads and
architectural refinements, and YOLO11 is a current member of that line. Recent applied work has taken
these detectors into disaster and search settings, reporting strong accuracy on curated flood and
person-in-water test sets. Two-stage detectors and transformer detectors reach higher accuracy on
some benchmarks at a higher computational cost; for a coordination study that only needs plausible
detection outputs, the single-stage choice is adequate, and the detector here is used to produce a
realistic error profile rather than to compete on a leaderboard.

Several techniques target the small-object problem specifically. Feature-pyramid and
path-aggregation necks combine fine and coarse features so that small objects survive to the head.
Super-resolution and context-aggregation modules have been proposed to recover small-object detail.
The most directly relevant technique here is tiled, or sliced, inference: a large image is cut into
overlapping tiles that are run through the detector separately and the results merged, so that small
objects appear larger relative to each tile. The Slicing-Aided Hyper Inference method popularised this
for aerial imagery and reports gains when the detector has been trained or fine-tuned on tiles. The
crucial condition is that training and inference operate at the same object scale; if the detector was
trained on whole downscaled frames and then run on tiles, the objects appear at a scale it never saw
in training, and accuracy can fall. The present work observed exactly this and reports it as a
negative finding, then addresses it with a slicing-aided fine-tuning step described in Chapter 3.

Data is the other constraint. Search-and-rescue datasets such as SARD (Sambolek and Ivašić-Kos, 2021)
contain people in outdoor search settings, from UAV viewpoints, and yield higher person-detection
accuracy than generic UAV imagery, but they are not disaster imagery. Disaster segmentation datasets
such as RescueNet (Chowdhury et al., 2023) and FloodNet (Rahnemoonfar et al., 2021) label hazards and
terrain — flood water, damaged and undamaged buildings, roads — as pixel masks, not survivors. The
practical consequence is a domain gap: a survivor detector must be trained on non-disaster imagery and
transferred, and that transfer cost should be measured rather than assumed away. Domain adaptation
methods exist to reduce such gaps, but they need target-domain data, which for labelled survivors in
flood scenes does not exist at scale; this is why the project measures the gap and treats it as a
limitation rather than claiming to have closed it.

## 2.3 Multi-Robot Task Allocation and Auction Methods

Dividing tasks among a team of robots is a mature field, surveyed comprehensively in recent work — a
2024 systematic review in ACM Computing Surveys catalogues the problem variants and methods. The
foundational framing is the Contract Net Protocol (Smith, 1980), which introduced the
announce–bid–award pattern in which a manager announces a task, agents bid according to their fitness,
and the task is awarded to the best bidder. Gerkey and Matarić (2004) provided the taxonomy that
classifies allocation problems along three axes: single-task versus multi-task robots, single-robot
versus multi-robot tasks, and instantaneous versus time-extended assignment. In that taxonomy the
scheme used here is single-task, single-robot, instantaneous assignment applied repeatedly online,
which is the simplest useful class and the one for which greedy auctions are known to be effective.

Allocation methods divide broadly into optimal, centralised solvers and distributed, market-based
heuristics. The Hungarian algorithm solves the one-shot assignment optimally in cubic time, but it
needs a central coordinator and a static problem, which does not fit a mission where UAVs fail and
priorities change. Distributed methods trade some optimality for responsiveness. Auction and
market-based methods are repeatedly identified as strong for distributed, real-time reallocation, at
the cost of communication overhead for the bidding rounds. The consensus-based bundle algorithm
extends single-item auctions to bundles of tasks with a consensus phase that resolves conflicting bids
without a central auctioneer, and is widely used for UAV task allocation. For safety-critical UAV
swarms, market-based replanning reassigns tasks when an agent is lost, triggering a localised
re-auction rather than a global replan. Auction and pheromone hybrids have been applied to maritime
search, combining the responsiveness of bidding with a stigmergic map that draws agents toward likely
target regions.

The scheme in this project is a direct, deliberately simple application of this line of work: a
repeated single-item auction with a linear bid, triggered by UAV failure and by new detections. It is
not a new allocation method, and it does not need to be; its role is to provide a correct, standard
adaptive baseline whose behaviour under failure can be measured cleanly against a static partition and
an ablation that removes the priority mechanism.

## 2.4 Fault-Tolerant Coordination and Area Coverage

Reallocating work when a robot fails is a specific and well-studied sub-problem. Fault-tolerant
task-allocation frameworks detect a failure and redistribute the failed robot's tasks among the
survivors, using recovery-driven reassignment so that the mission degrades gracefully rather than
losing a whole sector. The most directly comparable line of work is resilient coverage-path
redistribution, in which a failed robot's coverage path is reassigned to its neighbours after a
boustrophedon decomposition of the area — the same lawnmower sweep this project uses within each
sector. The presence of this prior work is one reason the coordination result here is framed as a
quantification of a known capability rather than a new mechanism.

Coverage-path planning is itself standard, and the choice of sweep pattern matters. Boustrophedon, or
lawnmower, sweeps decompose a region into strips and traverse them in a back-and-forth pattern; they
cover any shape reliably, which is why they are the default for arbitrary or non-convex regions.
Spanning-tree coverage guarantees complete coverage on a grid with a bounded path length. Spiral
patterns cover compact convex regions efficiently but degrade on thin or irregular regions, because a
spiral assumes the region contracts smoothly toward a centre. This project uses a workload-balanced
weighted-Voronoi partition to divide the area and a boustrophedon sweep within each sector, both
standard choices, and it includes a direct spiral-versus-lawnmower comparison to justify the default
rather than assert it (Chapter 4).

The literature also studies combined failure and perception degradation. Work on failure-aware
coordination and on resilience through reconfiguration handles both sensor degradation and agent
failure, and there is a body of work on how perception uncertainty affects coordination, much of it
under adversarial conditions where an agent's observations may be corrupted or spoofed. This matters
for positioning: the coupling between perception error and coordination outcome is not unstudied, so
the sensitivity study in this project is framed as a controlled analysis using a real measured error
rate over an established coordination method, not as the discovery of the coupling.

The deepest prior art for that coupling is classical search theory, which predates the robotics
literature by decades. Koopman's wartime analysis of search operations (Koopman, 1980) and Stone's
theory of optimal search (Stone, 1975) express the probability of finding a target as the product of a
coverage term and a probability of detection, the latter typically an exponential function of search
effort. Separating a coverage failure from a detection failure — the framing this project uses in
Chapter 4 — is therefore inherent in search theory, and coverage-path planning with an explicit,
imperfect probability of detection is an established extension of it. The honest consequence for this
project is that neither the coupling nor its decomposition can be claimed as novel; what differs here
is only that the detection term is taken from the measured error profile of a modern deep detector
rather than an assumed analytical model, and is swept over a fault-tolerant coordination policy in a
reproducible testbed. The contribution is an empirical instantiation, not a theoretical one.

## 2.5 Survivor Drift Prediction and Moving-Target Search

Predicting where a person in water will move is standard in maritime search and rescue. The leeway
method quantifies the wind- and current-induced drift of a floating object, calibrated per object
class from field experiments, and the US Coast Guard's Search and Rescue Optimal Planning System
(Kratzke, Stone and Frost, 2010) operationalises it by advecting a cloud of Monte-Carlo particles
under a current field and a leeway factor to produce a probability distribution over the object's
location and, from it, containment areas that bound the search. The distribution is updated as the
search proceeds, so that unsuccessful searching of a region lowers its probability and redirects
effort. This is the standard framing of search for a drifting target, and it is the one this project
adapts.

More recent work applies drift prediction directly to coverage planning. A 2026 study plans an
autonomous underwater vehicle's coverage path from a target-drift prediction, so that the vehicle
searches where the target is expected to be rather than where it was last seen. Several multi-UAV and
multi-agent studies search for moving targets by maintaining a probability map and reallocating agents
toward high-probability cells, and deep-reinforcement-learning approaches have been trained to plan
coverage for a person drifting in water. The common theme is that ignoring drift wastes the search on
a stale position, and that a probability map derived from a drift model concentrates the search where
it pays off.

The drift model in this project is a direct adaptation of the SAROPS idea to the inland-flood UAV
setting: particles are advected under a configured flow field with a leeway factor and a
turbulent-diffusion term, and the resulting cloud is reduced to containment polygons that feed the
coordination layer's priorities. Drift-aware moving-target search is therefore established prior art;
the contribution here is the adaptation to inland flooding and the controlled evaluation of its value
against a stale-sighting baseline, not the drift method itself. The flow field is a configured
assumption rather than an estimate from imagery, which is a limitation discussed later and a clear
opening for future work.

## 2.6 Hazard-Aware Routing and Decoupled Evaluation

Routing on a weighted graph is textbook shortest-path search; the design choice specific to this
setting is what goes into the edge weights. Folding segmentation hazards into the weights lets a route
trade distance against risk: a road cell that the segmenter marks as blocked is removed from the graph
so no route can use it, while water and building-damage raise the risk on nearby edges without
forbidding them. Because a single scalar weight collapses distance and risk into one number, the
informative object is not one route but the Pareto front of route distance against cumulative risk,
traced by sweeping the relative weight of risk from zero upward. Presenting the front rather than one
arbitrary compromise is the standard way to expose a trade-off. A known limitation of weighted-sum
scalarisation is that it recovers only the points on the convex hull of the Pareto front, so non-convex
trade-offs can be missed; the demonstration hazard here is graded so that the recovered front is
non-degenerate, and this limitation is stated rather than hidden. Real road networks are obtained from
OpenStreetMap through OSMnx (Boeing, 2017), which downloads and cleans a routable graph for a place.

On evaluation, a recurring theme in the recent multi-robot literature is the shortage of standardised,
reproducible benchmarks that pair realistic perception with coordination. New platforms and benchmarks
have begun to appear, but they tend to focus on perception reasoning, or on evaluating language-model
planners, rather than on the coupling between a measured detector error rate and a coordination
outcome. The decoupled cached-oracle design used here is aligned with this concern: by turning the
detector's error rate into a controllable input to the coordination layer, it provides a lightweight,
reproducible way to study that coupling directly, which is the one place the project does something few
of the surveyed systems do. Separately, estimating a water current from drone video is an established
technique in hydrology, using image velocimetry, but the reviewed sources do not connect it to
survivor-drift prediction; that connection is noted as future work rather than claimed here.

## 2.7 Legal, Social, Ethical and Professional Issues

Because the project is simulation-only, several operational risks do not arise, but the ethical and
professional context still matters and would bind any real deployment of a system like this one.

Privacy is the most direct concern. Aerial imagery over a disaster zone captures identifiable people
who have not consented and cannot consent, potentially in distress or in private circumstances. Under
the UK General Data Protection Regulation this is personal data, and its processing would need a lawful
basis, data minimisation, a defined retention period and secure handling. A responsible design would
avoid retaining raw imagery beyond the immediate rescue purpose, restrict access, and consider on-board
processing that keeps only detections rather than full frames. The decoupled design here is compatible
with that stance, since the coordination layer needs only detection records, not imagery.

Safety and the duty not to mislead responders are the ethical core. An automated survivor detector
that misses people — a false negative — can cost a life if responders treat "no detection" as "no
survivor". The domain gap and the weak small-object accuracy measured in Chapter 4 are exactly the
kind of limitation that must be communicated to any operator, so that automated detections are treated
as advisory rather than authoritative and areas are not written off on the strength of a silent
detector. Overclaiming the system's reliability would itself be an ethical failure, which is part of
why this report is careful to state what the system does not do. The professional codes that govern
computing work, such as the BCS Code of Conduct, place the public interest and the honest
representation of capability above the promotion of one's own work, and this dissertation tries to
follow that in its framing.

Dual-use and misuse deserve a mention. Coordinated UAV search-and-detection technology has obvious
surveillance and military applications, and a framework that detects and tracks people is not neutral.
The intended use here is humanitarian, and the simulation-only scope means nothing built is
field-deployable, but the reuse risk is real and worth naming. Finally, data licensing was respected
throughout. RescueNet is distributed under a non-commercial, no-derivatives licence, so no derived
label files are redistributed and the dataset is cited; OpenStreetMap data is used under the Open
Database Licence with attribution; no personal data was collected, and the imagery comes from public
research datasets. The simulation uses synthetic georeferencing, so no real location of a real person
is ever implied.

## 2.8 Summary and Identified Research Gaps

The review shows a mature field. Perception, task allocation, fault-tolerant coverage, drift-based
search and hazard-aware routing are each established, and integrated UAV search-and-rescue frameworks
that combine detection with multi-UAV coordination already exist. There is no clean opening for a
novel algorithm, and it would be dishonest to claim one. What the review does surface is a narrower,
defensible gap. First, few systems study the coupling between a real, measured detector error rate and
a coordination outcome as a controlled variable, which the decoupled design here is built to do.
Second, the recent literature repeatedly notes the shortage of lightweight, reproducible testbeds that
run the full detection-to-coordination pipeline with seeded, statistically reported results. Third, the
specific integration of an imagery-estimated flood current into a drift model is absent from the
reviewed sources, and although this project does not implement it either, naming it precisely marks a
concrete direction. The project positions itself in the first two of these, and against the closest
neighbours in the table below, honestly.

## 2.9 Literature Review Summary Table

The table compares this project against ten similar systems on capability dimensions. Marks were
inferred from abstracts and indexing and should be verified against the full papers; a tick means the
capability is present, a tilde partial, a cross absent, and a question mark not determinable from the
abstract.

| System | Domain | Real detector in loop | Fault-tolerant | Drift | Hazard routing | Perception-error study | Rigorous evaluation | Reproducible |
|---|---|---|---|---|---|---|---|---|
| This project | inland flood (sim) | ~ (cached, decoupled) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AI-Enhanced UAV Clusters (2026) | disaster | ✓ | ✗ | ✗ | ~ | ✗ | ~ | ? |
| SARCPPF (2024) | maritime | ✗ | ✗ | ✓ | ✗ | ✗ | ~ | ? |
| Market-Based Replanning (2026) | SAR | ✗ | ✓ | ✗ | ✗ | ✗ | ~ | ? |
| Resilient Coverage Redistribution | monitoring | ✗ | ✓ | ✗ | ✗ | ✗ | ~ | ? |
| Auction+Pheromone Maritime SAR | maritime | ? | ? | ~ | ✗ | ✗ | ~ | ? |
| Weight-Based Exploration (2020) | wilderness | ~ | ✗ | ✗ | ✗ | ✗ | ~ | ? |
| Bio-inspired Swarm (2025) | SAR | ✓ (thermal) | ~ | ✗ | ✗ | ✗ | ~ | ? |
| Multi-UAV Flood CVT (2025) | flood | ✗ | ~ | ✗ | ✗ | ✗ | ~ | ? |
| AUV Drift Coverage (2026) | maritime | ✗ | ✗ | ✓ | ✗ | ✗ | ~ | ? |
| MultiUAV-Plat (2026) | platform | ✗ | ~ | ✗ | ✗ | ✗ | ~ | ✓ |

No single competitor spans the full pipeline; this project's row is the widest across detection,
fault-tolerance, drift, routing and evaluation. The table also shows where others are clearly
stronger — a live detector and a real pilot area (AI-Enhanced UAV Clusters), a learned planner
(SARCPPF), high-fidelity physics simulation (Bio-inspired Swarm), and a released open platform
(MultiUAV-Plat). The distinctiveness here is breadth, evaluation rigour and reproducibility, not any
single feature, and reading the table honestly means acknowledging the columns where the project is
weaker as much as the row where it is wider.

---

# Chapter 3: Methodology

## 3.1 Overview of Research Design

The research design is simulation-based, offline and batch. It is quantitative: claims are supported by
many runs under controlled conditions with confidence intervals, and by baselines and ablations rather
than by single illustrative runs. The philosophy is that a coordination policy is a hypothesis about
behaviour under uncertainty, and the way to test such a hypothesis is to hold everything else fixed,
vary one factor, and repeat across enough independent random seeds to separate signal from sampling
noise. Three architecture decisions, recorded formally in the repository as decision records, shape
everything else.

The first (ADR-001) is that perception and coordination are decoupled — the simulator never runs the
detector. Perception is computed once, offline, on real imagery and cached to a table keyed by scenario
and cell; the simulator reads that table through one oracle. This is enforced by a test that fails the
build if the simulator package imports the detector, so the decoupling is a checked invariant, not a
convention. The second (ADR-002) is that there are two perception models, because no single dataset
covers all classes and the label types differ: a detector for people and vehicles, whose labels are
bounding boxes, and a segmenter for water, damaged buildings and blocked roads, whose labels are pixel
masks. A bounding box around a flood region is not meaningful, so the two are kept separate. The third
(ADR-003) is that the interactive browser demonstration built for the viva is an optional aid outside
the evaluated pipeline, so that a convenience feature does not blur the boundary of what is being
measured.

Reproducibility is treated as a first-class requirement rather than an afterthought. Every constant
lives in a configuration file with a comment on its provenance, so the source contains no hard-coded
numbers that a reader would have to reverse-engineer. Every stochastic entry point takes an explicit
seeded random generator, and the global generator is never used, so a run reproduces byte-for-byte from
its seed and two experiments cannot silently share state. Dependencies are pinned to exact versions,
and the environment is created with a single command. Long runs write their resolved configuration
alongside their outputs into a timestamped directory, so a result can always be traced back to the
exact parameters that produced it.

## 3.2 Datasets and Synthetic Scenario Generation

Two kinds of data are used, and it is important to be clear about which is real and which is
constructed, because the honesty of the whole study depends on that distinction.

Perception uses real imagery. Four public datasets are unified into two training sets. The detect set
holds 9,000 images with 126,897 person and 205,663 vehicle instances, drawn from VisDrone and SARD;
VisDrone provides dense, small-object urban aerial scenes, and SARD provides people in outdoor search
settings from UAV viewpoints. The segment set holds 6,387 images with 6,599 building-damaged, 2,643
road-blocked and 6,597 water instances, drawn from RescueNet and FloodNet, both of which label
post-disaster aerial scenes as pixel masks. Per-source loaders normalise each dataset's native
annotation format into the two unified training sets, remapping the source classes onto the project's
label set and discarding classes that fall outside it. The honest constraint, stated plainly, is that
no disaster dataset labels survivors, so the person class is trained on ordinary and search-and-rescue
imagery and transferred into the disaster domain; the resulting domain gap is measured in Chapter 4
rather than assumed to be zero.

Coordination uses constructed scenarios, and here the georeferencing is synthetic. The main scenario,
flood_a, is a six-by-six grid of 200-metre cells onto which real detections are composited to build the
cached table, so that each cell carries realistic detection counts and confidences. The grid is
anchored at an arbitrary point, so the resulting latitude and longitude are illustrative rather than the
real location of a real person; distances are computed in a local metric projection, never in degrees,
so that anything measured in metres is correct even though the absolute position is notional. For the
ablation experiments, survivor distributions are generated procedurally as a small number of Gaussian
hotspots placed in different sectors of the grid, with most cells left empty, so that the distribution
is sparse and controllable and reallocation and guided search have something to be measured against. A
distribution with a survivor in every cell would make every strategy look equally good and hide the
effect being studied. This procedural generation is a modelling choice and a limitation, and to reduce
the reliance on it, one experiment was re-grounded on the real detection distribution taken directly
from the cache (Section 4.3).

## 3.3 The Perception Pipeline (Two Models)

Two YOLO11s models were trained locally on Apple-Silicon hardware, using the Metal Performance Shaders
backend, at 640-pixel input for 60 epochs each, with early stopping on validation mean average
precision so that training halts when the held-out score stops improving. Model A is a detector for
person and vehicle; Model B is a segmenter for water, building-damaged and road-blocked. The "s" variant
is the small member of the family, chosen so that training and inference fit comfortably on a
laptop-class accelerator, which keeps the whole pipeline reproducible on ordinary hardware at the cost
of some accuracy relative to larger variants.

Evaluation reports whole-image mean average precision at an intersection-over-union threshold of 0.5
(mAP@50) and averaged over thresholds from 0.5 to 0.95 (mAP@50-95), the standard COCO measures, and,
importantly, average precision stratified by object size using the COCO small, medium and large bands.
The stratification is the point: a single whole-image number averages over object sizes and hides the
small-object regime that survivor detection lives in, so reporting it alone would be misleading for this
application. Precision and recall are reported at the operating threshold, with recall being the more
consequential of the two here because a missed survivor is worse than a false alarm.

Data augmentation during training is the Ultralytics YOLO11 default pipeline, and its exact values are
pinned in the model configurations rather than left implicit, so the report can state precisely what was
applied: mosaic compositing (disabled for the final ten epochs), horizontal flipping, random scaling and
translation, random erasing, and HSV colour jitter — the last including a value (brightness) variation of
±40 percent, which matters because it is what gives the detector any robustness to lighting. Vertical
flipping, rotation, shear and mix-up were off. Test-time augmentation was not used. Because the pinned
values equal the defaults that were actually applied, documenting them required no retraining.

Tiled inference was evaluated as a candidate improvement, at 640-pixel slices with a 0.2 overlap ratio
so that objects on slice boundaries appear whole in at least one tile. The result was negative and is
reported as such: because the detector was trained on whole downscaled frames rather than on tiles, at
inference the tiled objects appear at a scale the detector never saw, and accuracy fell rather than
rose. To address this scale mismatch, a slicing-aided fine-tuning pipeline was built. A script slices
the detect set into 640-pixel tiles, keeping the original frames as well so the detector still sees
whole-image context, and a fine-tune configuration continues training from the existing checkpoint on
the sliced set so that the detector learns objects at slice scale. This fine-tuning was set up to run on
the author's machine; its outcome, if completed, would update Section 4.1, and the pipeline is included
so that the negative finding comes with a concrete remedy rather than an excuse.

## 3.4 Coordination: Partitioning, Coverage and Auction Reallocation

The coordination layer divides the area among the UAVs, sweeps each part, and reassigns work online
when circumstances change. It is built from three interacting pieces: partitioning, coverage and
allocation.

Partitioning offers a grid baseline and a workload-balanced weighted-Voronoi method. The weighted method
places one site per UAV and assigns each cell to its nearest site under a weighted distance, then applies
Lloyd relaxation — repeatedly moving each site to the centroid of its assigned cells — to even out the
sectors, followed by a greedy boundary-rebalancing pass that shifts boundary cells from heavier to
lighter sectors until the workloads are close to equal. Balancing the workload matters because a mission
finishes only when the busiest UAV finishes, so an unbalanced partition wastes the idle capacity of the
lighter UAVs.

Coverage within a sector uses a boustrophedon sweep. The sweep lines are spaced by the camera footprint
reduced by the sidelap, so that adjacent passes overlap and no ground is left unseen between them; with
footprint f and fractional sidelap s the line spacing is

    Δ = f · (1 − s).

An explicit final sweep line flush with the far edge of the sector is added, which guards against the
classic coverage bug in which the last strip is narrower than the spacing and would otherwise be left
uncovered. This detail is exactly the kind of thing the acceptance tests check.

Allocation is auction-based. When a piece of work — a cell or a set of cells — needs an owner, each
eligible UAV computes a bid, and the lowest bidder wins. The bid for UAV i on cell c is a linear
combination of the travel cost to reach the cell, an energy penalty that discourages a UAV low on
battery from taking on more work, and the negative of the cell's priority so that high-priority cells
attract lower (more competitive) bids:

    bid_i(c) = w_t · travel_i(c) + w_e · energy_penalty_i − w_p · priority(c),

with the weights w_t, w_e and w_p set in configuration. A nearby UAV with plenty of energy bidding on a
high-priority cell therefore produces the lowest bid and wins. Reallocation is triggered in two ways.
When a UAV reaches its return-to-home battery threshold or fails outright, its unfinished cells are
returned to the pool and re-auctioned among the UAVs still flying, which is the mechanism that recovers
coverage under failure. When a survivor is detected, the priority of nearby cells is multiplied by a
boost factor,

    priority(c') ← priority(c') · (1 + boost)   for c' near the detection,

so that those cells attract stronger bids and the fleet concentrates effort where survivors have
actually been found.

Three baselines share the same interface so that comparisons are fair: a single UAV that must cover
everything, a static partition that assigns sectors once and never reallocates, and a random walk that
moves without a plan. An ablation removes the priority up-weighting while keeping the reallocation, so
that the effect of reallocation can be separated from the effect of priority. A probability-guided
search option was added later, in which a UAV chooses its next cell as the nearest high-priority
unvisited cell rather than sweeping in fixed order; it is off by default, so the baseline results are
unaffected by its presence, and its contribution is isolated in the three-way ablation of Section 4.3.

## 3.5 Survivor Drift Prediction and Re-tasking

The drift model adapts the SAROPS Monte-Carlo approach to the inland-flood UAV setting. From a
detection point, a cloud of particles is seeded and advected forward for a prediction horizon under the
world's flow field. At each time step of length dt, a particle at position x moves under the flow
velocity u(x), scaled by a leeway factor L that captures how strongly the person is carried by the
water, plus a random turbulent-diffusion term that spreads the cloud:

    x_{t+dt} = x_t + L · u(x_t) · dt + √(2 D dt) · ξ,   ξ ~ N(0, I),

where D is a horizontal diffusivity and ξ is a standard bivariate normal draw. The deterministic term
carries the cloud downstream; the stochastic term widens it over time, so the predicted region grows
with the horizon, which is the expected behaviour of a search that becomes less certain the longer a
survivor has been adrift.

The particle cloud is reduced to 50% and 90% containment polygons using a distance-peel convex hull:
particles are ordered by distance from the cloud centre and the outermost fraction is peeled away before
taking the convex hull, which gives a region that contains the required share of the probability mass.
This method is robust to a non-Gaussian, skewed cloud and needs no kernel-density estimation dependency,
keeping the implementation light and testable. The 90% polygon is mapped to grid cells, and those cells
can have their auction priority raised so that UAVs re-task toward the predicted region rather than the
stale sighting. The flow field, the leeway factor and the diffusivity are documented, illustrative
assumptions in the configuration; the flow is not estimated from imagery, which is the model's main
limitation and is stated as such.

## 3.6 Hazard-Aware Routing

The routing component turns segmentation outputs into rescue routes. Segmentation hazards are folded
into a road graph as follows: a cell the segmenter marks as road-blocked has its edges removed entirely,
so no route can pass through it, while water and building-damage raise a per-cell risk that is applied to
nearby edges. Each edge then carries a weight that combines its length with its risk,

    w(e) = length(e) · (1 + λ · risk(e)),

where λ controls how strongly risk is penalised relative to distance. Sweeping λ from zero upward traces
the Pareto front of route distance against cumulative route risk: at λ = 0 the route is the plain
shortest path, which may cut through hazard; as λ grows the route detours to avoid risk at the cost of
extra distance. Presenting the whole front lets a human choose the operating point rather than having one
baked in.

The road graph comes from one of two sources. For the synthetic scenario it is a four-connected lattice
over the grid, which keeps the geometry simple and analytically checkable. For a realistic demonstration
it is a real OpenStreetMap extract fetched once through OSMnx and cached to disk, so the network is never
hit at run time, in keeping with the offline design. The OSMnx multigraph, which may carry several
parallel edges between two nodes, is collapsed to a simple weighted graph by keeping the shortest of any
parallel edges, and hazards are then applied to edges by their spatial position. This lets the same
routing method run unchanged on both a toy lattice and a real street network.

## 3.7 The Oracle Bridge, Software Verification and Reproducibility

The oracle is the single bridge from perception to coordination, and confining the coupling to one place
is what makes the decoupled design work. The oracle reads the cached detection table, and when a UAV
enters a cell it returns that cell's detections, applying a configurable per-class false-negative rate —
so that a set fraction of true detections are dropped, modelling a detector that misses objects — and a
reporting latency drawn from a uniform range, all deterministically under the run's seed. Because the
oracle is the only coupling, the detector's error rate becomes a controlled input that can be swept,
which is what the sensitivity study in Section 4.5 exploits. The same interface also means the oracle
could be replaced by a live detector that produces the same kind of records without changing anything
downstream, so the decoupling is a deployment path as well as an evaluation convenience.

Software verification is by an automated suite (over one hundred tests) that runs without a GPU or datasets
in a few seconds, so it runs before every change. Most tests were written before the code they check. They
include analytic acceptance cases with a known answer (zero-diffusion drift displaces a particle by exactly
velocity times time; a blocked edge is untraversable); an end-to-end test of the core claim (a UAV fails and
the auction must recover full coverage while the static partition must not); an architecture-guarding test
that fails the build if the simulator imports the detector (ADR-001); a determinism test; and per-module
unit tests. Formatting and linting gate every change. The suite is the evidence that the results are
reproducible; categories are listed in Appendix B.

## 3.8 Evaluation Methodology

The coordination layer is evaluated by Monte-Carlo simulation. The main sweep runs five strategies by four
UAV counts (one, two, four, six) by three failure conditions (none, one, two failures) by thirty seeds —
1,800 runs — completing in a few seconds on a CPU because perception is cached. Failures are injected per
seed (which UAVs fail, and when within a fixed window), so timing varies across seeds but reproduces within
one. Metrics are coverage, survivors detected, survey redundancy (wasteful revisiting) and time-to-locate.
Results are reported as a mean with a 95% confidence interval, `mean ± 1.96 · s / √n`, so differences can be
read as significant or not rather than asserted from a single run.

Separate, focused experiments address the individual research questions. The drift experiment compares
searching the predicted 90% zone against searching the stale sighting over 300 seeds. The
perception–coordination experiment sweeps the detector's survivor false-negative rate under a two-UAV
failure and measures the end-to-end detection rate for the adaptive and static strategies. The
coverage-pattern experiment compares lawnmower and spiral sweeps at equal coverage on sectors of
different shapes. The head-to-head ablation compares static, auction and auction-plus-guided search on a
sparse survivor distribution, reporting the detection curve over time and a reachable threshold, and a
second version of it runs on the real detection distribution to reduce the dependence on constructed
scenarios. Each experiment writes its resolved configuration and outputs to a timestamped directory, so
every number in Chapter 4 is traceable to the run that produced it.

---

# Chapter 4: Results and Analysis

## 4.1 Perception Accuracy and the Domain Gap

Model A, the person-and-vehicle detector, reached a mean average precision at 0.5 intersection-over-union
of 0.674 and a COCO-style mAP over 0.5–0.95 of 0.392, with precision 0.79 and recall 0.61. Model B, the
hazard segmenter, reached a mask mAP@50 of 0.410 and mAP@50-95 of 0.266, with precision 0.67 and recall
0.43. These are reasonable numbers for small models trained at modest resolution for a limited number of
epochs on hard aerial imagery, and they are not strong. The segmenter is the weaker of the two, which is
expected given the difficulty of delineating flood-water and damage boundaries and the smaller segment
training set.

| Model | Task | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|---|
| A | detect (person, vehicle) | 0.674 | 0.392 | 0.79 | 0.61 |
| B | segment (water, damage, road) | 0.410 | 0.266 | 0.67 | 0.43 |

The most informative view is the size breakdown for Model A. Average precision was 0.26 for small
objects, 0.59 for medium and 0.78 for large.

| Object size (COCO band) | Average precision |
|---|---|
| small (< 32×32 px) | 0.26 |
| medium | 0.59 |
| large | 0.78 |

Since a survivor seen from altitude is a small object, the 0.26 figure is the one that matters for the
application, and it is honest to say the detector is weak in exactly the regime the whole system depends
on. Reporting only the 0.674 whole-image number would have hidden this. The domain gap is visible in a
per-source comparison of person detection: accuracy was markedly higher on the search-and-rescue imagery
than on the generic urban UAV imagery, roughly 0.88 against 0.65 at 0.5 IoU. Because no disaster dataset
labels survivors, the person class is trained on this non-disaster imagery and transferred, and that
difference is the transfer cost the application pays.

Tiled inference reduced accuracy rather than improving it, which is a train/inference scale mismatch: the
detector saw whole downscaled frames in training but 640-pixel slices at inference, so the objects
appeared at a scale it had not learned. This negative result is reported rather than buried, and it
motivated the slicing-aided fine-tuning pipeline described in Section 3.3. At the time of writing that
fine-tuning was still running, so its outcome is not included; when it completes, the expected effect is a
recovery of small-object accuracy under tiled inference, and this section will be updated with the
measured numbers rather than the expectation.

A separate robustness check (`make lighting-robustness`) re-scores the detector on the same validation
subset re-rendered at several brightness levels, to test whether the brightness augmentation actually
buys day-and-night robustness. It does. Against a normal-light baseline of 68.0% mAP@50 on a
representative 300-image sample (consistent with the whole-dataset figure), darkening the images by half
cost only 2.1 points (65.9%) and moderate brightening 1.4 points (66.6%); the detector held up well
across ordinary lighting. The one real weakness was over-exposure: a 1.8× glare cost 5.1 points (62.9%),
the largest drop, because saturated highlights wash out the small objects the detector already struggles
with. The practical reading is that the augmentation gives usable low-light robustness but that glare, not
darkness, is the lighting condition to worry about.

| Lighting (brightness ×) | mAP@50 | Change vs normal |
|---|---|---|
| dark (×0.5) | 65.9% | −2.1 pts |
| dim (×0.75) | 67.4% | −0.6 pts |
| normal (×1.0) | 68.0% | — |
| bright (×1.4) | 66.6% | −1.4 pts |
| glare (×1.8) | 62.9% | −5.1 pts |

## 4.2 Coordination under UAV Failure

The headline coordination result concerns coverage under failure at the largest fleet size of six UAVs,
over the 1,800-run sweep. With no failure, both the adaptive auction and the static partition reach full
coverage, as they should. The difference appears under failure and grows with the number of failures.

| Condition | Auction coverage | Static coverage | Difference |
|---|---|---|---|
| no failure | 100% | 100% | 0.0 pts |
| one failure | 100% | 87.6% ± 2.3% | +12.4 pts |
| two failures | 100% | 74.5% ± 2.9% | +25.5 pts |

Under one failure the auction holds full coverage while the static partition falls to 87.6%; under two
failures it holds full coverage while the static partition falls to 74.5%. The reason is direct: a static
partition permanently abandons a failed UAV's cells, so each failure removes a fixed share of the area,
whereas the auction returns those cells to the pool and the surviving UAVs bid for them. The confidence
intervals are narrow enough that the differences are unambiguous.

The baselines behave as expected and serve as sanity checks. The random walk also reaches full coverage
but wastefully, with a survey redundancy between 1.1 and 1.24, meaning cells are revisited rather than
covered once, so it "succeeds" only by spending far more effort. The single-UAV strategy collapses because
one drone cannot cover the area within the mission time. The priority-removed ablation isolates the
reallocation from the priority up-weighting and behaves like the auction on coverage, which foreshadows
the finding in the next section that the priority mechanism adds little to detection. The honest reading is
that this result quantifies a real benefit of reallocation under failure, using an established method, with
confidence intervals; it is not the demonstration of a new mechanism.

## 4.3 Decomposing the Advantage: A Three-Way Ablation

A single "adaptive versus static" comparison conflates two different things: the reallocation of a failed
UAV's work, and the added probability-guided search that steers a UAV toward high-priority cells. An
initial version of the head-to-head benchmark did exactly this, and it also used a time-to-locate metric
with an unfair denominator that inflated the apparent advantage. Both problems were corrected. The
corrected experiment is a three-way ablation — static, auction, and auction with guided search — run under
a UAV failure on a sparse survivor distribution, reporting the detection curve of survivors located
against time and a threshold that all strategies can reach.

| System (sparse synthetic) | Survivors located | Gain vs previous |
|---|---|---|
| static (no realloc) | 68% | — |
| auction (realloc) | 89% | +21 pts (reallocation) |
| auction + guided | 89% | +0 pts, ~1.5× faster to 50% |

The decomposition is clear. The reallocation gain, from static to auction, is 21 points; the guidance
gain, from auction to auction-plus-guided, is zero points and only about a 1.5× speed-up to the halfway
mark. The reallocation does the work, and the added guided search contributes no extra detections and only
a modest time advantage, because both auction variants eventually cover the area and locate the same
survivors — guidance changes the order, not the total. This is not the result the project set out to find,
and reporting it is the point of doing an honest ablation.

The experiment was then re-run on the real detection distribution taken from the cache — 439 real person
detections with their real per-cell density and confidences — with a search prior derived from the real
flood-water segmentation rather than from the survivor locations. A measured observation supports that
prior: survivors sit away from the water, with a correlation of about −0.51 between survivor density and
water density, consistent with people having moved to higher ground, so the prior is the inverted water
map, which keeps it independent of the thing being searched for and avoids a circular experiment.

| System (real detection distribution) | Survivors located | Gain vs previous |
|---|---|---|
| static (no realloc) | 70% | — |
| auction (realloc) | 90% | +20 pts (reallocation) |
| auction + guided | 90% | +0 pts, ~1.3× faster to 50% |

On this real distribution, static located 70%, auction 90% and auction-plus-guided 90%; the reallocation
gain is 20 points and the guidance gain is again zero points with about a 1.3× speed-up. The decomposition
therefore holds on real data as well as synthetic, which strengthens the finding and grounds it in real
perception outputs rather than an invented distribution.

## 4.4 Drift-Aware Search Re-tasking

The drift experiment asks whether searching a survivor's predicted drift region beats searching the stale
sighting. For each of 300 seeds, the survivor's true final position is advected under the flow, and — with
an independent random stream, so the prediction never sees the true draw — the 90% drift zone is predicted
from the drift model. The survivor drifts about 893 m from the original sighting over the horizon, which is
far enough that the initial position is useless as a search target.

| Search target | Located | Mean localisation error |
|---|---|---|
| predicted 90% drift zone | 88% | 108 ± 6 m |
| stale sighting | 0% | 897 ± 10 m |

Searching the predicted zone located the survivor 88% of the time within about 108 m; searching the stale
sighting located them 0% of the time, with an error equal to the drift distance because the survivor is
simply no longer there. The result is strong, but it should be read with care. The 88% figure sits close
to the 90% containment level by construction, so part of what it confirms is that the containment estimate
is well calibrated — the model puts 90% of the probability where about 88% of the survivors actually end
up. The substantive point is the contrast with the stale sighting, which shows concretely that ignoring
drift sends the search to where the survivor is not. This is an adaptation of established maritime
drift-based search to the inland-flood UAV setting, evaluated quantitatively; it is not a new drift
method, and the flow field is assumed rather than estimated from imagery, so the absolute drift distance is
a property of the configured scenario.

## 4.4a Measured versus Assumed Current for the Drift Forecast

The drift result above depends on the flow field, which was *assumed*. A follow-on experiment
(`make flow-drift`) instead *measures* the current from the drone imagery by particle image velocimetry
(PIV — FFT cross-correlation of interrogation windows between frames) and drives the same forecast from the
measured field. No real flood video with a known current was available, so it runs on a synthetic clip whose
texture is advected by a known current (which also gives the ground truth) — a proof of concept, not a
field result. PIV recovered the current at 0.24 m/s RMSE against a 2.0 m/s mean speed with no directional
error, in line with published drone velocimetry (~0.22–0.44 m/s). The effect is large: the *assumed* current
located the survivor in 0% of 200 seeds (176 m off); the *measured* current located them 82% of the time
(28 m off).

| Current driving the forecast | Located | Mean localisation error |
|---|---|---|
| assumed (hand-set, wrong) | 0% | 176 ± 3 m |
| PIV-measured (from the video) | 82% | 28 ± 2 m |

The honest reading is narrow: this is an engineering improvement to the drift component — swapping an assumed
input for a measured one — not a new method, since both PIV and drift-based search are established. Its value
is that it removes the drift model's least-defensible assumption and quantifies what that assumption was
costing. The limitations (synthetic ground truth, no comparison yet against the external current forecasts
operational tools use) are carried into the future work.

## 4.5 Perception × Coordination Sensitivity

This experiment uses the decoupled design to do something the other experiments cannot: sweep the
detector's survivor false-negative rate as a controlled variable and measure the end-to-end
survivor-detection rate under a two-UAV failure, for the adaptive auction and the static partition, over
360 runs.

| False-negative rate | Auction detected | Static detected |
|---|---|---|
| 0.0 | 100% | 71% |
| 0.25 | ~75% | ~55% |
| 0.5 | ~51% | ~36% |

The adaptive auction tracks the ceiling set by perception: it detects essentially every survivor the
detector can see, so at a zero false-negative rate it locates 100% and at a 0.5 rate it locates about 51%,
close to the (1 − false-negative-rate) ceiling, because it recovers full coverage and therefore loses only
the survivors the detector itself misses. The static partition sits a persistent 15 to 29 points below that
ceiling across the sweep, because it loses coverage on top of losing detections. The experiment separates
the two loss factors cleanly: perception error is an irreducible floor that no coordination policy can
remove, whereas coverage loss under failure is a coordination problem that reallocation recovers. The
split itself is not new — classical search theory already writes the probability of finding a target as
coverage multiplied by a probability of detection (Koopman, 1980), so separating a coverage failure from
a detection failure is inherent in that framing. What the decoupled design adds is empirical: the
detection term is the *measured* false-negative rate of a real deep detector rather than an assumed
analytical model, swept over a fault-tolerant coordination policy, which the applied UAV search-and-rescue
systems surveyed here rarely do.

The result is best read as a diagnosis of where the bottleneck lies: because the adaptive system already
sits on the perception ceiling, improving coordination further cannot raise the detection rate — improve
the coordination and the number does not move, improve the detector and it does. In this regime the
detector is the bottleneck and coordination has done its job. Section 5.4a develops what this does and does
not imply; the honest addition here is only that the split is measured with a real detector's error profile,
not that the coupling is new.

## 4.5a Can Coordination Claw Back the Bottleneck by Looking Twice?

Section 4.5 says coordination cannot recover a survivor the detector misses — but that assumes a single
look. If a UAV takes a *second* look at an uncertain cell, two independent looks cut the miss rate to
roughly its square, so coordination can attack the perception floor after all, at the cost of covering
less ground and only for misses that are bad luck rather than structural. A follow-on experiment
(`make relook`) measures the trade-off, with an honest detail: a tunable share of misses is *persistent*
(occluded, too small, under water — never recovered), so the model is not the over-optimistic miss-rate
to the power of the number of looks. The effect is classical search theory; the point is to measure it
against this project's bottleneck result.

The finding is conditional, and the condition is the useful part: with a weak detector (40% miss rate)
and a fixed look budget, re-looking beats covering more only when the search prior is good enough to know
which cells to look at twice.

| Search prior quality | Cover-more (1 look) | Best re-look | Verdict |
|---|---|---|---|
| good (noise 0) | 59.8% found | 79.0% (k=2) | re-look wins by +19 pts |
| medium (noise 0.3) | 59.8% found | 61.3% (k=2) | marginal, +1.5 pts |
| poor (noise 0.6) | 59.5% found | ~59.8% | cover-more wins |

A good prior lifts survivors found from 60% to 79%; a poor prior wastes the looks and the coverage
sacrificed is not repaid; three or four looks hurt in every case. So coordination can partly beat the
detector, but only by spending a good prior — exactly what the drift model and hazard maps produce — which
makes re-looking's value inseparable from the quality of the information guiding it.

## 4.6 Hazard-Aware Routing (Synthetic and Real Networks)

On the synthetic grid, folding hazards into the road graph and sweeping the risk weight produced a
three-point Pareto front, ranging from a 1,000 m route that passes through hazard to a 2,600 m route that
avoids it, with the naive shortest path sitting at the highest-risk corner of the front. The front is small
because the toy lattice offers few distinct detours, but it demonstrates the mechanism on a graph whose
geometry can be checked by hand.

| Network | Route | Distance | Cumulative risk |
|---|---|---|---|
| synthetic grid | shortest (through hazard) | 1,000 m | high |
| synthetic grid | safest | 2,600 m | low |
| London OSM (219 nodes) | shortest (crosses flood) | 1,141 m | 3,558 |
| London OSM (219 nodes) | safest (detours) | 2,367 m | 0 |

On a real OpenStreetMap network — a cached 219-node, 295-edge extract of central London — the same method
produced an eight-point front, from a 1,141 m route at risk 3,558 that crosses the flooded streets to a
2,367 m route at risk 0 that detours around them. The map figure shows the naive shortest path threading
through the flood and the safe route going around it. This demonstrates that the routing transfers
unchanged from the synthetic lattice to a real road network; it is a standard hazard-weighted
shortest-path method applied to segmentation outputs, and its value is the clean distance-versus-risk
trade-off it exposes rather than any algorithmic novelty.

## 4.7 Coverage-Pattern Comparison

To answer whether the project relies on a single flight pattern without justification, a lawnmower sweep
was compared against a spiral on an eight-by-eight sector, and then on other shapes. On the square sector
both covered 100% of the cells, but the spiral path was shorter, 13,308 m against the lawnmower's 15,780 m,
so the spiral was about 16% more efficient there.

| Sector shape | Pattern | Coverage | Path length |
|---|---|---|---|
| 8×8 square | lawnmower | 100% | 15,780 m |
| 8×8 square | spiral | 100% | 13,308 m |
| 1×20 thin | lawnmower | 100% | — |
| 1×20 thin | spiral | 10% | — |

Extending the comparison changed the conclusion. On a one-by-twenty thin sector the spiral covered only
10% of the cells and effectively failed, because a spiral assumes the region contracts toward a centre and
a thin strip does not, while the lawnmower covered it fully. The honest reading is that a spiral can be
more efficient on compact sectors but is not robust to thin or irregular ones, which is why the
boustrophedon sweep is the safe default when a partition can hand back sectors of any shape. The 16% figure
is real but conditional, and it would have been misleading to report it without the thin-sector
counter-example.

## 4.8 Synthesis

The results support a modest, honest story. Reallocation recovers coverage under failure (quantified with
confidence intervals); the project-specific guided search adds little; the drift zone beats the stale
sighting; the sensitivity study separates perception error from coverage loss; the perception is weak on
small objects and tiled inference did not help without fine-tuning; and the routing exposes a clean
distance-versus-risk trade-off on a real network. None of these claims to beat the state of the art — each
is a controlled measurement against internal baselines, which is what the project set out to provide.

## 4.9 Comparison with Reported Systems and Positioning

A direct numerical comparison against prior systems is **not possible**, and saying so plainly is more
honest than forcing one. The comparable systems report different metrics, on different domains and
datasets, with no shared benchmark, and most release no code, so their headline numbers are not
comparable to each other or to this work. The table below lists each system's *own* reported headline on
its *own* setup; it should be read as evidence that the field lacks a common benchmark, not as a ranking.

| System | Reported headline | Setup | Comparable? |
|---|---|---|---|
| This project | coverage under 2-UAV failure 100% vs 74.5%; drift-zone localisation 88% vs 0%; detector mAP@50 0.674 (small-object AP 0.26) | flood_a sim, 1,800 seeded runs, cached YOLO11 | — |
| AI-Enhanced UAV Clusters (2026) | YOLOv8 mAP@50 98.4%, area coverage 100% | real imagery + sim, 17.6 km², 16 UAVs | no — different task/data |
| Market-Based Replanning (2026) | 93% mission success at 25% UAV loss | SAR swarm sim | no |
| Weight-Based Exploration (2020) | ~215% search-time reduction vs lawnmower (sim) | wilderness UAV team | no |
| Bio-inspired Swarm (2025) | exploration score 0.67 (PSO) | PX4 + Gazebo | no |
| SARCPPF (2024) | qualitative: prioritises high-probability regions | maritime, deep RL | no |

The honest positioning follows from this. On any single capability the project does not lead: the
detector is weaker than the accuracy figures others report on curated sets, and every method used is
established. Where the project is distinctive is not one feature but the combination of three — breadth of
integration, evaluation rigour, and reproducibility — and the specific, trustworthy findings that follow,
including negative ones. That contrast is set out below.

| Dimension | Typical comparable UAV-SAR work | This project |
|---|---|---|
| End-to-end breadth | usually one or two stages | widest: detection → coordination-under-failure → drift → routing |
| Evaluation | a single headline number | seeded sweeps, 95% confidence intervals, ablations, negative findings, real-data grounding |
| Perception–coordination coupling | each side assumes the other is perfect | a real measured detector miss-rate swept through coordination (§4.5) |
| Robustness reported | clean-condition accuracy only | lighting degradation measured (glare the weakness, §4.1) |
| Drift input | current from external forecasts or assumed | current measured from the drone video (§4.4a) |
| Reproducibility | rarely reproducible | seeded, CPU, config-driven, one-command experiments, full test suite |
| Novel algorithm | sometimes claimed | none — stated honestly |

In one sentence: the detector is not the best and no method is new, but the pipeline is integrated more
broadly than the comparable systems, evaluated more honestly and reproducibly, and it yields specific,
trustworthy findings — including negative ones — that single-metric studies do not report. That is the
sense in which the work is "better", and it is the sense appropriate to the level.

---

# Chapter 5: Discussion and Future Work

## 5.1 Overview

This chapter reads the results against the research questions, states the limitations without softening
them, and sets out future work. The through-line is that the value of the project is in correct
integration and honest, controlled evaluation, not in a new method or a performance record, and the
discussion is written to make that claim defensible rather than to inflate it.

## 5.2 Achievement Against Research Questions

RQ1, on whether auction reallocation improves outcomes under failure, is answered affirmatively and
quantitatively: coverage held at 100% against a static partition's 87.6% under one failure and 74.5% under
two failures at six UAVs, and the three-way ablation attributes this to the reallocation rather than to the
priority or guided-search additions. The qualifier is that reallocation under failure is a known
capability, so the answer is a measurement of a known effect, not a new claim.

RQ2, on how the advantage depends on fleet size, failures and detector error, is partly answered. The
advantage grows with the number of failures, from 12.4 to 25.5 points between one and two failures. The
sensitivity study shows that reallocation recovers the coverage component of the loss under detector error
but cannot recover the perception component, so the advantage over static persists across the
false-negative sweep while the absolute detection rate falls with perception quality. A fuller answer would
map the advantage across all fleet sizes and failure counts jointly, which the sweep data support but this
report summarises only at the headline size.

RQ3, on the perception domain gap, is answered with a clear result: small-object average precision of 0.26
against 0.78 for large objects, a per-source gap of roughly 0.88 to 0.65 between search-and-rescue and
generic imagery, and a reported negative result for naive tiled inference. The gap is measured rather than
closed, and the fine-tuning remedy is set up but not yet evaluated.

RQ4, on drift-driven re-tasking, is answered: searching the predicted 90% zone located a drifted survivor
88% of the time against 0% for the stale sighting. The honest caveats are that the 88% is partly a
confirmation that the containment estimate is calibrated, and that the flow driving the drift is assumed
rather than estimated.

RQ5, on where the effort should go and whether the split is even measurable, is answered by the sensitivity
study of Section 4.5, which only the decoupled design makes possible. It splits a missed survivor into a
perception part and a coverage part, shows that reallocation recovers the coverage part but nothing recovers
the perception part, and so identifies the detector as the bottleneck in this regime — a diagnosis of where
improvement pays off, not merely a demonstration that the two are decoupled.

RQ6, on hazard-aware routing, is answered on both a synthetic grid and a real London street network,
producing usable distance-versus-risk Pareto fronts in both cases. Across all six questions, the recurring
qualifier is that the methods are established and the claims are scoped to internal baselines rather than to
the state of the art.

## 5.3 Limitations

The limitations are substantial and are stated plainly, because a study whose contribution is honesty of
evaluation cannot be selective about them.

The study is simulation-only. There is no flight hardware, no real-time behaviour and no field trial, so no
claim of real-world performance is made or implied. The simulation is deliberately low-fidelity: UAVs are
point masses with an energy model rather than aerodynamic bodies, wind and sensor noise beyond the modelled
false-negative rate are absent, and the environment is a grid rather than a rendered scene. These choices
buy the scale and reproducibility the study depends on, but they mean the numbers describe the model, not
the world.

The coordination experiments use constructed survivor distributions. The ablation, drift and search
experiments use procedurally generated hotspots, and although one experiment was re-grounded on the real
detection distribution, the scenario parameters — grid size, cell size, hotspot placement, failure timing —
remain choices that affect the result. The georeferencing is synthetic throughout, so the coordinates are
illustrative and the absolute drift distances belong to the configured scenario rather than to a real
flood.

The perception is weak in the regime that matters: small-object average precision of 0.26 is the
survivor-detection regime, the person class is transferred across a measured domain gap, and tiled inference
did not help without fine-tuning (result still pending). The coordination advantage is generic — reallocation
under failure is established and the ablation shows guided search adds little. Most importantly, the project
claims no novel method: every component is established, so the contribution is integration, rigorous
evaluation and the decoupled methodology, and the report is written to keep that claim honest.

## 5.4 Future Work

Several directions would strengthen the work, ordered here roughly by how directly they follow from what
exists. The most immediate is the perception fine-tuning already set up: completing it should turn the
negative tiled-inference finding into a measured small-object improvement and close the loop on RQ3. Beyond
that, grounding more of the coordination experiments on real georeferenced data — ideally a dataset that
carries real GPS locations rather than the synthetic anchoring used here — would reduce the dependence on
constructed scenarios that is the study's main internal-validity limitation.

The vision-estimated current prototyped in Section 4.4a should next be run on real flood footage and
compared against the external current forecasts operational tools use. Replacing the greedy auction with a
consensus-based bundle algorithm would give a stronger, still-standard coordination baseline. The re-look
study (Section 4.5a) could be run inside the full engine rather than as a focused model. Finally, packaging
the framework as a small, open, reproducible benchmark for the perception–coordination coupling would let
others measure the same coupling on their own methods.

## 5.4a What the Coupling Result Means, and What It Does Not

The "detector is the bottleneck" finding should not be read as the trivial advice "improve the detector".
Two things make it more. First, "the coordination has done its job" is a *measured* outcome, not an
assumption: the study tested coordination across failures, found reallocation reaches full coverage cheaply,
and only then attributed the remaining loss to perception — the coverage problem was shown solved before the
perception problem was named. Second, the value is the split and what it implies for effort: the sweep says,
for a given fleet, failure rate and detector quality, how many more survivors each kind of improvement buys,
and here it says further coordination work is largely wasted while the detector is where the return is. It
also cautions the coordination literature that a coverage method scored against a perfect detector reports a
gain that shrinks once a real detector is in front of it. The contribution is not the bottleneck idea, which
is elementary, but the clean, reproducible measurement of the split with a real detector error rate.

## 5.5 Broader Significance

The broader significance is methodological rather than operational: decoupling perception from coordination
lets the two be evaluated together, with a real measured error rate as an input, which most work in either
field does not do. It also shows the value of honest ablation — the initial single-number benchmark
overstated the advantage, and decomposing it revealed the project-specific component adds little. In a
disaster-response setting, where an overstated capability could cost lives if an operator over-trusts it, a
trustworthy deflating result is more useful than an impressive but fragile one.

---

# Chapter 6: Conclusion

## 6.1 What This Dissertation Set Out to Do

The dissertation set out to build and rigorously evaluate an integrated simulation framework for
coordinated multi-UAV flood response, spanning perception, coordination under failure, drift prediction and
routing, and to state honestly what each part contributes and what it does not. The emphasis on honesty was
not incidental: the field is saturated, and the responsible course was to integrate established methods
correctly and to measure them well, rather than to manufacture a novelty claim that a reviewer could
dismantle.

## 6.2 What Was Achieved

An end-to-end, reproducible, seeded framework was built and tested: two perception models scored on real
imagery with accuracy broken down by object size and source; an auction reallocation scheme and three
baselines whose behaviour under failure was quantified across 1,800 runs with confidence intervals; a
three-way ablation that decomposed the advantage into reallocation (large) and guided search (negligible),
confirmed on real data; a drift model that locates a drifted survivor far better than a stale sighting; and
hazard-aware routing on a synthetic grid and a real London network. The system was positioned honestly
against the closest prior art, with limitations stated in full.

## 6.3 Reflection on Learning

The most valuable lesson was about honesty in evaluation. Early framing overstated the novelty and, in one
case, the size of a result; checking the literature and decomposing the benchmark corrected both, and doing
so improved the work rather than diminishing it. Building the software to a testable, seeded,
configuration-driven standard made those corrections possible, because a result could be regenerated and
trusted rather than defended from memory — and finding that a carefully built component contributed little
was itself instructive, since that only surfaces when the evaluation is honest enough to admit it.

## 6.4 Final Statement

This is an integration-and-evaluation study. It does not propose a new method and does not beat the state
of the art, and it does not need to. Its value is a correctly integrated, rigorously and honestly evaluated
simulation of coordinated multi-UAV flood response, with a methodology that measures the
perception–coordination coupling directly, and with its limitations stated plainly enough that a reader can
trust the parts that do hold.

---

# Appendices

**Appendix A — Reproducing the results.** Each result is regenerated by a named command: the coordination
sweep (`make sweep`); the drift experiment (`make rq4`); the perception–coordination sensitivity study
(`make sensitivity`); the coverage-pattern comparison (`make coverage-compare`); routing on synthetic and
real networks (`make routes`, `make routes-osm`); the three-way ablation (`make benchmark`) and its
real-data version (`make benchmark-real`); and a single mission run (`make sim`). Every run writes its
resolved configuration alongside its outputs into a timestamped directory, and every random source is
seeded, so any figure can be reproduced from the recorded configuration and seed.

**Appendix B — Software verification (test suite).** The suite (over one hundred tests) runs without a GPU
or datasets in a few seconds. Categories: analytic acceptance cases (drift displacement, coverage
completeness, blocked-edge untraversability, the re-look and PIV detection models); the core failure-recovery
acceptance test; the architecture-isolation test that forbids the simulator from importing the detector; a
determinism test; and per-module unit tests. Linting and formatting gate every commit.

**Appendix C — Configuration.** All parameters live in YAML files under `configs/` with provenance
comments, including the grid geometry, the UAV energy model, the oracle noise rates, the auction bid
weights and priority boost, the drift constants (leeway, diffusivity and horizon) and the routing risk
weight. This is what "no magic numbers in the source" means in practice, and it is what makes a run
traceable to the exact values that produced it.

---

# References

> Complete each entry against its source (authors, year, venue, DOI) before submission; entries marked
> "to confirm" were located by title and link during the project's literature search and need their full
> bibliographic details verified. Nothing below has had authors or dates invented where they were not known.

- Boeing, G. (2017) 'OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks', *Computers, Environment and Urban Systems*, 65, pp. 126–139.
- Chowdhury, T. et al. (2023) 'RescueNet: A High Resolution UAV Semantic Segmentation Dataset for Natural Disaster Damage Assessment', *Scientific Data*, 10. arXiv:2202.12361.
- Gerkey, B.P. and Matarić, M.J. (2004) 'A Formal Analysis and Taxonomy of Task Allocation in Multi-Robot Systems', *The International Journal of Robotics Research*, 23(9), pp. 939–954.
- Koopman, B.O. (1980) *Search and Screening: General Principles with Historical Applications*. Pergamon Press. (foundational search theory; probability of detection — to confirm edition/pages)
- Kratzke, T.M., Stone, L.D. and Frost, J.R. (2010) 'Search and Rescue Optimal Planning System', *13th International Conference on Information Fusion*. (to confirm)
- Stone, L.D. (1975) *Theory of Optimal Search*. Academic Press. (coverage × probability of detection — to confirm edition/pages)
- Rahnemoonfar, M. et al. (2021) 'FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding', *IEEE Access*. arXiv:2012.02951.
- Sambolek, S. and Ivašić-Kos, M. (2021) 'Automatic Person Detection in Search and Rescue Operations Using Deep CNN Detectors', *IEEE Access*.
- Smith, R.G. (1980) 'The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver', *IEEE Transactions on Computers*, C-29(12), pp. 1104–1113.
- Zhu, P. et al. (2021) 'Detection and Tracking Meet Drones Challenge (VisDrone)', *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

Closest and related systems (verify full citations):

- AI-Enhanced UAV Clusters for Search and Rescue in Natural Disasters (2026), *Algorithms*. https://doi.org/10.3390/a19010031 (to confirm)
- SARCPPF: Autonomous Coverage Path Planning for Persons-in-Water via Deep Reinforcement Learning (2024), *Ocean Engineering*. https://www.sciencedirect.com/science/article/abs/pii/S0029801823027877 (to confirm)
- Market-Based Replanning for Safety-Critical UAV Swarms in Search and Rescue (2026). https://arxiv.org/html/2606.01970 (to confirm)
- Resilient Multi-Robot Coverage Path Redistribution using Boustrophedon Decomposition. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644315/ (to confirm)
- Auction- and Pheromone-Based Multi-UAV Cooperative Search and Rescue in Maritime Environments, *Drones*. https://doi.org/10.3390/drones9110794 (to confirm)
- Weight-Based Exploration for Unmanned Aerial Teams Searching for Multiple Survivors (2020). https://arxiv.org/pdf/2012.11131 (to confirm)
- Bio-inspired Swarm UAV Framework Integrating Thermal Sensing and Optimization-based Coordination (2025), *Scientific Reports*. https://www.nature.com/articles/s41598-025-33223-z (to confirm)
- Multi-UAV Flood Monitoring via CVT with Gaussian Mixture Coverage Control (2025). https://arxiv.org/pdf/2510.19548 (to confirm)
- Boundary-Adaptive Neural-Network Coverage Path Planning for AUVs Based on Target Drift Prediction (2026), *Journal of Field Robotics*. https://onlinelibrary.wiley.com/doi/10.1002/rob.70053 (to confirm)
- MultiUAV-Plat: An LLM-Oriented Platform, Benchmark and Framework for Multi-UAV Collaborative Task Planning (2026). https://arxiv.org/abs/2606.31073v1 (to confirm)
- A Systematic Literature Review on Multi-Robot Task Allocation (2024), *ACM Computing Surveys*. https://dl.acm.org/doi/10.1145/3700591 (to confirm)
- Multi-Robot Coordination with Adversarial Perception (2025). https://arxiv.org/abs/2504.09047 (to confirm)

*(A fuller, grouped list with links is maintained in `docs/references.md`.)*
