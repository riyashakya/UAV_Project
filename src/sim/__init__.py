"""Simulator core (CPU, headless, deterministic).

Single responsibility: advance a grid world of UAVs over discrete time, given only the
world state and the detection oracle. Fixed-timestep, seed-deterministic, no plotting.

ADR-001: nothing in this subpackage may import ``ultralytics``/``torch``/``sahi``. The world
is learned exclusively through ``src.sim.oracle`` reading the cached detections.
"""
