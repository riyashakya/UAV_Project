"""Shared pytest fixtures (synthetic datasets + configs). No GPU, no downloads."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from src.perception.datasets.taxonomy import load_taxonomy

from tests.fixtures import synthetic


@pytest.fixture
def taxonomy():
    """The real, validated project taxonomy from configs/datasets/taxonomy.yaml."""
    return load_taxonomy()


@pytest.fixture
def datasets_root(tmp_path: Path) -> Path:
    """A temp dir populated with all four synthetic source datasets."""
    synthetic.make_all(tmp_path / "raw")
    return tmp_path


@pytest.fixture
def datasets_cfg(datasets_root: Path):
    """An OmegaConf datasets config pointing at the synthetic tree (roots + output + build)."""
    raw = datasets_root / "raw"
    return OmegaConf.create(
        {
            "roots": {
                "visdrone": str(raw / "visdrone"),
                "sard": str(raw / "sard"),
                "rescuenet": str(raw / "rescuenet"),
                "floodnet": str(raw / "floodnet"),
            },
            "output": {
                "detect_dir": str(datasets_root / "unified" / "detect"),
                "seg_dir": str(datasets_root / "unified" / "seg"),
            },
            "build": {
                "polygon_simplify_px": 1.0,
                "min_polygon_points": 4,
                "min_region_area_px": 32,
            },
        }
    )
