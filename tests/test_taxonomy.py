"""Taxonomy validation: the mapping must be total (ADR-002, Phase 1 acceptance)."""

from __future__ import annotations

import pytest
from src.perception.datasets.taxonomy import (
    SourceMapping,
    TaxonomyError,
    UnifiedTaxonomy,
    load_taxonomy,
    validate_taxonomy,
)


def test_real_taxonomy_loads_and_validates():
    tax = load_taxonomy()
    assert tax.detect == ("person", "vehicle")
    assert tax.segment == ("building_damaged", "road_blocked", "water")
    assert set(tax.sources) == {"visdrone", "sard", "rescuenet", "floodnet"}


@pytest.mark.parametrize("source", ["visdrone", "sard", "rescuenet", "floodnet"])
def test_mapping_is_total_and_disjoint(source):
    """Every native class is mapped or dropped exactly once — no silent drops."""
    m = load_taxonomy().sources[source]
    native = set(m.native)
    mapped, dropped = set(m.map), set(m.drop)

    assert len(m.native) == len(native), "duplicate native classes"
    assert mapped.isdisjoint(dropped), "class both mapped and dropped"
    assert mapped | dropped == native, "native classes neither mapped nor dropped"


def test_map_targets_are_valid_unified_classes():
    tax = load_taxonomy()
    for m in tax.sources.values():
        assert set(m.map.values()) <= set(tax.classes(m.task))


def test_unified_for_raises_on_unknown_class():
    m = load_taxonomy().sources["visdrone"]
    assert m.unified_for("pedestrian") == "person"
    assert m.unified_for("bicycle") is None  # explicitly dropped
    with pytest.raises(TaxonomyError):
        m.unified_for("spaceship")  # undeclared -> loud failure, not a silent drop


def test_index_to_unified_uses_native_order():
    m = load_taxonomy().sources["rescuenet"]
    idx = m.index_to_unified()
    # native index 1 == water, 4 == building-major-damage, 8 == road-blocked
    assert idx[1] == "water"
    assert idx[4] == "building_damaged"
    assert idx[8] == "road_blocked"
    assert 0 not in idx  # background is dropped


def test_silent_drop_is_rejected():
    """A source that leaves a native class unaccounted for must fail validation."""
    bad = UnifiedTaxonomy(
        detect=("person",),
        segment=("water",),
        sources={
            "x": SourceMapping(
                name="x", task="detect", native=("a", "b"), map={"a": "person"}, drop=()
            )
        },
    )
    with pytest.raises(TaxonomyError, match="silent drop"):
        validate_taxonomy(bad)


def test_mapped_and_dropped_overlap_is_rejected():
    bad = UnifiedTaxonomy(
        detect=("person",),
        segment=("water",),
        sources={
            "x": SourceMapping(
                name="x", task="detect", native=("a",), map={"a": "person"}, drop=("a",)
            )
        },
    )
    with pytest.raises(TaxonomyError, match="both mapped and dropped"):
        validate_taxonomy(bad)
