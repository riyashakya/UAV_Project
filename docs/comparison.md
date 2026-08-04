# Comparison with the 10 most similar systems (for the related-work chapter)

> **Read this first.** Feature marks are **inferred from abstracts/indexing, not full-text reads** —
> **verify every mark by reading the paper before citing it.** `?` = not determinable from the
> abstract. This matrix is a *positioning aid*: it shows this project's distinctiveness is **breadth
> across the pipeline + evaluation rigour + reproducibility**, not any single novel feature — and it
> shows honestly where other systems are *stronger* (real-world pilots, higher-fidelity simulation,
> learned planners). This project is **simulation-only with synthetic scenarios and modest
> perception**; those are real weaknesses the table does not hide.

**Legend:** ✓ present · ~ partial / implicit · ✗ absent · ? unclear from abstract.

| # | System | Domain | Real detector (in loop) | Coordination | Fault-tolerant (UAV failure) | Drift prediction | Hazard-aware routing | Guided / priority search | Perception-error as controlled variable | Eval: Monte-Carlo + CIs + ablation | Reproducible / open code | Sim fidelity / field test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | **This project** | inland flood (sim) | ~ (real YOLO11, but **cached/decoupled**, not live) | auction/CNP + 3 baselines | ✓ | ✓ (SAROPS) | ✓ (Pareto + real OSM) | ✓ | ✓ (distinctive) | ✓ (1,800-run, CIs, ablation) | ✓ (seeded, 95 tests, CPU) | ✗ low (analytic grid); no field test |
| 1 | [AI-Enhanced UAV Clusters for SAR (Algorithms, 2026)](https://doi.org/10.3390/a19010031) | general disaster | ✓ (YOLOv8, live) | multi-UAV routing (lawnmower) | ✗ | ✗ | ~ (routing, not hazard-weighted) | ✗ | ✗ | ~ (perception metrics; little coord stats) | ? | ~ (sim + 17.6 km² pilot area) |
| 2 | [SARCPPF (Wu et al., 2024, Ocean Eng.)](https://ui.adsabs.harvard.edu/abs/2024OcEng.29116403W/abstract) | maritime | ✗ (assumed) | **DRL** coverage planning | ✗ | ✓ | ✗ | ✓ (probability map) | ✗ | ~ | ? | sim |
| 3 | [Market-Based Replanning for UAV Swarms in SAR (2026)](https://arxiv.org/html/2606.01970) | general SAR | ✗ | reverse-auction | ✓ (93% at 25% loss) | ✗ | ✗ | ✗ | ✗ | ~ | ? | sim |
| 4 | [Resilient Coverage Redistribution (Boustrophedon)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644315/) | env. monitoring | ✗ | coverage redistribution | ✓ | ✗ | ✗ | ✗ | ✗ | ~ | ? | sim |
| 5 | [Auction- & Pheromone-Based Multi-UAV Maritime SAR (Drones)](https://doi.org/10.3390/drones9110794) | maritime | ? | auction + pheromone | ? | ~ | ✗ | ~ | ✗ | ~ | ? | sim |
| 6 | [Weight-Based Exploration for UAV Survivor Search (2020)](https://arxiv.org/pdf/2012.11131) | wilderness | ~ (survivor reports) | multi-UAV team | ✗ | ✗ | ✗ | ✓ | ✗ | ~ | ? | sim + physical test |
| 7 | [Bio-inspired Swarm UAV (thermal, Sci. Reports 2025)](https://www.nature.com/articles/s41598-025-33223-z) | general SAR | ✓ (thermal) | PSO / GWO / ACO | ~ | ✗ | ✗ | ~ | ✗ | ~ (metaheuristic scores) | ? | ✓ high (PX4 + Gazebo) |
| 8 | [Multi-UAV Flood Monitoring via CVT (2025)](https://arxiv.org/pdf/2510.19548) | flood (monitoring) | ✗ (flood extent, not survivors) | CVT / Voronoi coverage | ~ | ✗ | ✗ | ~ (density-weighted) | ✗ | ~ | ? | sim |
| 9 | [AUV Coverage from Target-Drift Prediction (Zhou, 2026, JFR)](https://onlinelibrary.wiley.com/doi/10.1002/rob.70053) | maritime (AUV) | ✗ | boundary-adaptive NN coverage | ✗ | ✓ | ✗ | ~ | ✗ | ~ | ? | sim |
| 10 | [MultiUAV-Plat (LLM benchmark, 2026)](https://arxiv.org/abs/2606.31073v1) | general (platform) | ✗ (abstracted) | LLM-driven planning | ~ | ✗ | ✗ | ~ | ✗ | ~ | ✓ (open platform) | sim (lightweight) |

## How to read this (the honest takeaway)

- **No single system spans the whole pipeline.** Each row is deep on one or two components; this
  project's row has the most ✓ across *detection → coordination → fault-tolerance → drift → routing →
  evaluation*. That **breadth of correct integration**, plus the **evaluation rigour** (Monte-Carlo +
  CIs + a self-corrected ablation) and **reproducibility** (seeded, tested, CPU-only), is the honest
  distinctiveness — **not novelty of any single feature.**
- **The perception-error-as-controlled-variable column** is the one place this project does something
  the others in this set do not directly do (enabled by the decoupled cached oracle). State it as a
  *methodological* point, modestly.
- **Where others are clearly stronger** (state these too, for credibility): #1 has a real detector in
  the loop and a large pilot area; #2 has a *learned* planner; #7 runs in high-fidelity PX4/Gazebo;
  #10 is a released open platform. This project is **simulation-only, low-fidelity, synthetic
  scenarios, modest perception, no field test, and no novel method** — do not hide these.

## Suggested columns to keep in the dissertation version

For the final chapter, a smaller matrix reads better: **Domain · Real detector · Fault-tolerant ·
Drift · Hazard routing · Rigorous evaluation · Reproducible.** Fill each mark **after reading the
paper**, and add a sentence per closest neighbour (#1, #2, #3) in the prose.
