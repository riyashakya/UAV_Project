"""Federated training of the detector (Flower) — privacy-preserving, non-IID by data source.

Motivation, not novelty: federated learning for UAV object detection with FedAvg/FedProx is
established (see docs/federated_plan.md). The honest angle here is (1) a *privacy-preserving* way to
improve the detector — the component this project measured as the bottleneck — without pooling raw
disaster imagery across operators, and (2) a genuinely **non-IID, imbalanced** client split that
falls straight out of the data (VisDrone vs SARD), which is the setting FedProx is designed for.

Perception-only subpackage: may import flwr / torch / ultralytics; src/sim never does (ADR-001).
The partitioning and the FedAvg averaging are pure and unit-tested; the Flower training loop is a
scaffold that needs `uv sync --extra federated` and a GPU to run.
"""
