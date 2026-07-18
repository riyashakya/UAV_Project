"""Multi-UAV AI framework for disaster response (simulation-based).

Two decoupled pipelines joined by a single bridge (ADR-001):

* ``src.perception`` — offline, GPU: turn real labelled UAV imagery into cached detections.
* ``src.sim`` / ``src.coordination`` / ``src.drift`` / ``src.routing`` / ``src.eval`` —
  CPU: read cached detections through ``src.sim.oracle`` and run the coordination study.

The simulator never imports ``ultralytics`` (enforced by test in Phase 3).
"""

__version__ = "0.1.0"
