# Multi-UAV Disaster Response Framework — developer tasks.
# Every Python command runs through `uv run`, so the locked venv (uv.lock) is always used.
# Targets mirror the command list in CLAUDE.md.

.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help setup setup-all test test-all lint fmt clean \
        cache-dets sim sweep eval-perception build-datasets

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

train-b:  ## Phase 2: train Model B (segment) locally -> outputs/perception/
	$(UV) run python -m src.perception.train --config configs/perception/model_b.yaml $(ARGS)

cache-dets:  ## Phase 3: offline YOLO pass -> data/cache/detections.parquet
	$(UV) run python -m src.perception.detect_cache

eval-perception:  ## Phase 2: perception eval tables -> outputs/perception/<timestamp>/
	$(UV) run python -m src.perception.eval

sim:  ## Phase 4: one simulation run, e.g. `make sim SCEN=flood_a SEED=0`
	$(UV) run python -m src.sim.engine scenario=$(SCEN) seed=$(SEED)

sweep:  ## Phase 9: full Monte Carlo grid -> outputs/runs/<timestamp>/
	$(UV) run python -m src.eval.runner

clean:  ## Remove tooling caches and build artifacts (never touches data/ or outputs/)
	rm -rf .pytest_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
