# Federated training of the detector — plan and honest positioning

Status: **started.** The pieces that can be built and verified on a laptop are done and unit-tested
(non-IID partitioning, FedAvg aggregation). The Flower training loop is scaffolded but **not yet run** —
it needs `uv sync --extra federated` and a GPU. This document is the plan and, importantly, the honest
framing, so nothing here gets overstated.

## Why this is here (motivation, not a novelty claim)

A prior literature check (recorded in the session memory) found that **federated learning for UAV
object detection with FedAvg and FedProx already exists** — for example *Federated Learning for Object
Detection: Enabling Collaborative Drone Learning Without Centralizing Data* (2026), and UAV-detection
work on data-heterogeneity mitigation. So federated learning here is **not new**, and this plan does
not claim it is. Two honest reasons it still earns a place in the project:

1. **It attacks the bottleneck the project identified.** The sensitivity study (§4.5) showed the
   detector is the limiting factor once coordination is good. Federated learning is a realistic,
   *privacy-preserving* way to improve that detector: several drone operators or agencies can jointly
   train one model **without sharing raw disaster imagery**, which the project's own ethics section
   (§2.7, UK GDPR) argues they may be unable to share. It connects a training method to a problem the
   project measured, rather than bolting on an unrelated technique.

2. **The data gives a genuinely non-IID, imbalanced split for free.** The detect set is 6,471 VisDrone
   images (urban/generic aerial) and 1,387 SARD images (search-and-rescue persons). Treating each source
   as one client is a real heterogeneous, imbalanced federation — precisely the setting **FedProx** was
   designed for (Li et al., 2020) and where it is expected to beat **FedAvg** (McMahan et al., 2017).
   So the FedAvg-vs-FedProx comparison is meaningful here, not a toy.

The contribution, therefore, is an **evaluation/instantiation** in this project's SAR context, tied to
its bottleneck finding and its privacy argument — not a new algorithm.

## The experiment

- **Clients:** one per source (VisDrone, SARD). Non-IID and imbalanced by construction.
- **Baselines to compare:** (a) centralised training on the pooled data — what the project already has;
  (b) **FedAvg**; (c) **FedProx**. Also a per-client *local-only* model, as a lower bound.
- **Metric:** mAP@50 and size-stratified AP on the shared held-out val split — the same measures used
  everywhere else, so the numbers are comparable to the rest of the report.
- **Question:** how much accuracy does federation give up versus pooling the data (the privacy cost),
  and does FedProx recover some of that gap over FedAvg on this heterogeneous split?

## What is built vs what remains

| Piece | State |
|---|---|
| Non-IID client partition (`partition.py`) | done, unit-tested |
| FedAvg aggregation core (`fedavg.py`) | done, unit-tested |
| Flower client + simulation (`fed_train.py`) | scaffold, **not yet run** (needs `--extra federated` + GPU) |
| FedProx proximal term in the YOLO loss | **not done** — the main remaining engineering |
| Actual federated runs + comparison table/figure | to do (GPU time) |

## The honest caveat on FedProx

FedProx differs from FedAvg by adding a proximal term to each client's *local* loss, keeping local
weights close to the global model. Ultralytics' `model.train()` is a closed high-level loop, so
injecting that term needs a custom trainer or callback. Until that is wired in, the `strategy: fedprox`
path uses FedProx **server aggregation** but not the client-side proximal regulariser — so it should not
be reported as a true FedProx result yet. This is stated in `fed_train.py` and must be fixed before any
FedProx claim is made.

## Compute reality

Federated training re-trains a model across many rounds, so it is far heavier than the CPU experiments
in the rest of the project. Start with `yolo11n` and ~10 rounds on a GPU; move to `yolo11s` for the
final comparison. This is run by the author, like the perception fine-tuning, not inside `make test`.
