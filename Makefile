# Multi-UAV Disaster Response Framework — developer tasks.
# Every Python command runs through `uv run`, so the locked venv (uv.lock) is always used.
# Targets mirror the command list in CLAUDE.md.

.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help setup setup-all test test-all lint fmt clean \
        cache-dets sim sweep drift animate rq4 coverage-compare routes routes-osm demo web \
        eval-perception build-datasets

PAUSE ?= 1

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv + install core & dev deps (no torch/GDAL), pinned via uv.lock
	$(UV) sync --extra dev

setup-all:  ## Install everything: core, dev, geo (GDAL) and perception (torch)
	$(UV) sync --all-extras

test:  ## Run the fast test suite (excludes @pytest.mark.slow)
	$(UV) run pytest -m "not slow"

test-all:  ## Run every test, including slow GPU/dataset tests
	$(UV) run pytest

lint:  ## ruff check + format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

fmt:  ## Auto-format and auto-fix with ruff
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

build-datasets:  ## Phase 1: unify source datasets -> data/unified/ (add --dry-run to preview)
	$(UV) run python -m src.perception.datasets.build $(ARGS)

train-a:  ## Phase 2: train Model A (detect) locally -> outputs/perception/
	$(UV) run python -m src.perception.train --config configs/perception/model_a.yaml $(ARGS)

slice-dets:  ## Phase 2b: slice the detect set into 640 tiles for slicing-aided fine-tuning
	$(UV) run python -m src.perception.slice_dataset $(ARGS)

train-b:  ## Phase 2: train Model B (segment) locally -> outputs/perception/
	$(UV) run python -m src.perception.train --config configs/perception/model_b.yaml $(ARGS)

progress:  ## Show progress/ETA of the latest perception training run
	$(UV) run python scripts/train_progress.py

cache-dets:  ## Phase 3: offline YOLO pass -> data/cache/detections.parquet
	$(UV) run python -m src.perception.detect_cache

eval-perception:  ## Phase 2: perception eval tables -> outputs/perception/<timestamp>/
	$(UV) run python -m src.perception.eval

sim:  ## Phase 4: one simulation run, e.g. `make sim SCEN=flood_a SEED=0`
	$(UV) run python -m src.sim.engine --scenario configs/scenario/$(SCEN).yaml --seed $(SEED)

sweep:  ## Phase 9: full Monte Carlo grid -> outputs/runs/<timestamp>/
	$(UV) run python -m src.eval.runner

drift:  ## Phase 7: draw the survivor-drift projection + containment -> outputs/drift/<timestamp>/
	$(UV) run python -m src.drift.visualize

rq4:  ## RQ4: quantify drift-aware search vs the stale sighting -> outputs/runs/rq4_<timestamp>/
	$(UV) run python -m src.eval.rq4

flow-drift:  ## Vision-estimated (PIV) vs assumed current for drift forecasting -> outputs/runs/flow_drift_*/
	$(UV) run python -m src.eval.flow_drift

relook:  ## Re-look vs cover-more: can coordination claw back the detection bottleneck? -> outputs/runs/relook_*/
	$(UV) run python -m src.eval.relook

lighting-robustness:  ## Detector mAP under bright/normal/dark val images -> outputs/runs/lighting_*/
	$(UV) run python -m src.perception.lighting_eval

sahi-recall:  ## SAHI vs full-frame recall/precision at a realistic threshold -> outputs/runs/sahi_*/
	$(UV) run python -m src.perception.sahi_eval

federated-train:  ## Federated detector training (FedAvg/FedProx, non-IID by source). Needs: uv sync --extra federated
	$(UV) run python -m src.perception.federated.fed_train

sensitivity:  ## Contribution B: perception FN-rate x coordination -> outputs/runs/sensitivity_*/
	$(UV) run python -m src.eval.sensitivity

search-order:  ## Probability-guided vs uniform search: time-to-locate -> outputs/runs/search_*/
	$(UV) run python -m src.eval.search_order

benchmark-real:  ## Ablation on the REAL detection distribution + flood-derived prior -> outputs/runs/
	$(UV) run python -m src.eval.benchmark_real

benchmark:  ## Option A: adaptive pipeline vs static baseline head-to-head -> outputs/runs/benchmark_*/
	$(UV) run python -m src.eval.benchmark

coverage-compare:  ## Phase 5: lawnmower vs spiral coverage path length -> outputs/runs/coverage_*/
	$(UV) run python -m src.eval.coverage_compare

animate:  ## Phase 6: GIF of the mission (UAVs, coverage, a failure + reallocation) -> outputs/runs/
	$(UV) run python -m src.sim.animate

routes:  ## Phase 8: hazard-weighted rescue-route Pareto front -> outputs/routing/
	$(UV) run python -m src.routing.safe_path

routes-osm:  ## Phase 8: routing on a REAL OSM street network (needs `geo` extra) -> outputs/routing/
	$(UV) run python -m src.routing.safe_path osm

demo:  ## Run the whole CPU demo end-to-end (mission->sweep->drift->routing). PAUSE=0 for no pauses
	UV="$(UV)" PAUSE="$(PAUSE)" bash scripts/demo.sh

web:  ## Opt-in browser mission visualiser (ADR-003, stdlib only) -> http://127.0.0.1:8000
	$(UV) run python webapp/server.py

clean:  ## Remove tooling caches and build artifacts (never touches data/ or outputs/)
	rm -rf .pytest_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
