# Reported headline results of similar systems (NOT a head-to-head)

> **Critical honesty note.** A true benchmark comparison across these systems is **not possible** —
> they use **different domains, datasets, metrics, and no shared testbed**, and most do not release
> code. The numbers below are each paper's *own reported headline*, on its *own* setup; they are
> **not comparable to each other or to this project.** This table's real message is that the field
> **has no common benchmark**, which is exactly why no system (including this one) can claim to
> "outperform" the others numerically. Verify every number against the paper before citing; `?` =
> not found in the abstract.

| System | Reported headline metric | Value | On what setup | Comparable to ours? |
|---|---|---|---|---|
| **This project** | coverage under 2-UAV failure (auction vs static) | **100% vs 74.5%** | flood_a sim, 1,800 MC runs, cached YOLO11 | — |
| **This project** | survivor localisation (drift zone vs stale sighting) | **88% vs 0%** | sim, 300 seeds | — |
| **This project** | detector Model A mAP@50 / small-object AP | **0.674 / 0.26** | VisDrone+SARD val | — |
| [1 AI-Enhanced UAV Clusters (2026)](https://doi.org/10.3390/a19010031) | YOLOv8 mAP@0.5 · area coverage | 98.4% · 100% | real imagery + sim; 17.6 km², 16 UAVs | ✗ different task/data |
| [2 SARCPPF (Wu, 2024)](https://ui.adsabs.harvard.edu/abs/2024OcEng.29116403W/abstract) | qualitative: prioritises high-probability regions, avoids repeat coverage | — | maritime, DRL | ✗ |
| [3 Market-Based Replanning (2026)](https://arxiv.org/html/2606.01970) | mission success at 25% UAV loss | 93% | sim SAR swarm | ✗ |
| [4 Resilient Coverage Redistribution](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644315/) | continuous coverage after failure (no single figure) | ? | env-monitoring sim | ✗ |
| [5 Auction+Pheromone Maritime SAR](https://doi.org/10.3390/drones9110794) | search efficiency vs baselines | ? | maritime sim | ✗ |
| [6 Weight-Based Exploration (2020)](https://arxiv.org/pdf/2012.11131) | search-time reduction vs lawnmower | ~215% (sim) / 75% (physical) | wilderness UAV team | ✗ |
| [7 Bio-inspired Swarm (2025)](https://www.nature.com/articles/s41598-025-33223-z) | exploration score (PSO best) | 0.67 (vs GWO 0.62, ACO 0.59) | PX4 + Gazebo | ✗ |
| [8 Multi-UAV Flood CVT (2025)](https://arxiv.org/pdf/2510.19548) | coverage-control convergence | ? | time-varying flood sim | ✗ |
| [9 AUV Drift Coverage (2026)](https://onlinelibrary.wiley.com/doi/10.1002/rob.70053) | search-path efficiency from drift prediction | ? | maritime AUV sim | ✗ |
| [10 MultiUAV-Plat (2026)](https://arxiv.org/abs/2606.31073v1) | benchmark platform (scores LLM planners) | n/a | lightweight sim platform | ✗ |

## What to write from this

State plainly: *"A direct numerical comparison with prior systems is not possible — they report
different metrics on different domains and datasets, with no shared benchmark and mostly no released
code. This work is therefore evaluated against internal, reproduced baselines under controlled
conditions, not against the reported figures of other systems."* That sentence is both true and
protective.
