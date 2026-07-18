"""Build the detection cache — Phase 3.

Single responsibility: run both trained models over a scenario's imagery **once** and write
``data/cache/detections.parquet`` with columns
``scenario, cell_id, class, confidence, lat, lon, bbox_utm, source_image, model``.

Georeference using image-footprint metadata; where a dataset lacks geotags, synthesise a
plausible grid layout and record that in a ``synthetic_geo: bool`` column — never silently
fake georeferencing.

This is the last GPU stage; everything downstream is CPU (ADR-001).

Stub: implemented in Phase 3.
"""

from __future__ import annotations


def main() -> None:
    """CLI entry point (``make cache-dets``)."""
    raise NotImplementedError("Phase 3: detection caching is not implemented yet.")


if __name__ == "__main__":
    main()
