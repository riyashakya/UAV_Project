"""Unified taxonomy + per-source class mapping (loads ``configs/datasets/taxonomy.yaml``).

This module is the code side of ADR-002. It loads the mapping table and validates it is
**total** — every native source class is mapped or explicitly dropped, exactly once, with no
silent drops. ``build`` refuses to run on an unmapped class, so a dataset that ships an
unexpected label fails loudly instead of quietly losing annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "configs" / "datasets" / "taxonomy.yaml"

VALID_TASKS = ("detect", "segment")


class TaxonomyError(ValueError):
    """Raised when the taxonomy config is inconsistent or a class is unaccounted for."""


@dataclass(frozen=True)
class SourceMapping:
    """How one source dataset's native classes map into the unified taxonomy."""

    name: str
    task: str
    native: tuple[str, ...]
    map: dict[str, str]  # native class -> unified class
    drop: tuple[str, ...]
    keep_metadata: str | None = None

    def unified_for(self, native_class: str) -> str | None:
        """Unified class for ``native_class``, or ``None`` if it is explicitly dropped.

        Raises :class:`TaxonomyError` for any class not declared in ``native`` — this is the
        no-silent-drops guarantee that loaders rely on.
        """
        if native_class not in self.native:
            raise TaxonomyError(
                f"{self.name}: encountered class {native_class!r} not declared in the "
                f"taxonomy (native={list(self.native)}). Refusing to silently drop it."
            )
        return self.map.get(native_class)

    def index_to_unified(self) -> dict[int, str]:
        """Map native palette/id index -> unified class, for mapped classes only.

        Used by mask/YOLO loaders where annotations are integer-indexed.
        """
        return {i: self.map[name] for i, name in enumerate(self.native) if name in self.map}


@dataclass(frozen=True)
class UnifiedTaxonomy:
    """The full taxonomy: unified class lists per task + every source mapping."""

    detect: tuple[str, ...]
    segment: tuple[str, ...]
    sources: dict[str, SourceMapping]

    def classes(self, task: str) -> tuple[str, ...]:
        if task == "detect":
            return self.detect
        if task == "segment":
            return self.segment
        raise TaxonomyError(f"Unknown task {task!r}; expected one of {VALID_TASKS}.")

    def class_id(self, task: str, cls: str) -> int:
        """YOLO class index of a unified class within its task's ordered list."""
        classes = self.classes(task)
        if cls not in classes:
            raise TaxonomyError(f"{cls!r} is not a unified {task} class ({list(classes)}).")
        return classes.index(cls)


def load_taxonomy(path: Path | str | None = None) -> UnifiedTaxonomy:
    """Load and validate the taxonomy config. Raises :class:`TaxonomyError` on any problem."""
    path = Path(path) if path is not None else DEFAULT_TAXONOMY_PATH
    if not path.exists():
        raise TaxonomyError(f"Taxonomy config not found: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    unified = raw.get("unified", {})
    detect = tuple(unified.get("detect", []))
    segment = tuple(unified.get("segment", []))

    sources: dict[str, SourceMapping] = {}
    for name, spec in (raw.get("sources") or {}).items():
        sources[name] = SourceMapping(
            name=name,
            task=spec["task"],
            native=tuple(spec.get("native", [])),
            map=dict(spec.get("map", {})),
            drop=tuple(spec.get("drop", [])),
            keep_metadata=spec.get("keep_metadata"),
        )

    tax = UnifiedTaxonomy(detect=detect, segment=segment, sources=sources)
    validate_taxonomy(tax)
    return tax


def validate_taxonomy(tax: UnifiedTaxonomy) -> None:
    """Enforce the ADR-002 invariants. Raises :class:`TaxonomyError` on the first violation."""
    for task, classes in (("detect", tax.detect), ("segment", tax.segment)):
        if not classes:
            raise TaxonomyError(f"Unified {task} class list is empty.")
        if len(set(classes)) != len(classes):
            raise TaxonomyError(f"Duplicate unified {task} classes: {classes}")

    for name, m in tax.sources.items():
        if m.task not in VALID_TASKS:
            raise TaxonomyError(f"{name}: invalid task {m.task!r} (expected {VALID_TASKS}).")

        native_set = set(m.native)
        if len(native_set) != len(m.native):
            raise TaxonomyError(f"{name}: duplicate entries in `native` ({list(m.native)}).")

        map_keys = set(m.map)
        drop_set = set(m.drop)

        unknown = (map_keys | drop_set) - native_set
        if unknown:
            raise TaxonomyError(
                f"{name}: `map`/`drop` reference classes not in `native`: {sorted(unknown)}"
            )

        overlap = map_keys & drop_set
        if overlap:
            raise TaxonomyError(f"{name}: classes both mapped and dropped: {sorted(overlap)}")

        unaccounted = native_set - map_keys - drop_set
        if unaccounted:
            raise TaxonomyError(
                f"{name}: native classes neither mapped nor dropped (silent drop): "
                f"{sorted(unaccounted)}"
            )

        valid_targets = set(tax.classes(m.task))
        bad_targets = set(m.map.values()) - valid_targets
        if bad_targets:
            raise TaxonomyError(
                f"{name}: maps to non-{m.task} unified classes: {sorted(bad_targets)}"
            )
