"""Fixed-timestep simulation engine — Phase 4.

Single responsibility: drive the world + UAVs through a deterministic, fixed-timestep loop
and emit an event log. Headless, no plotting. Given the same seed, two runs must produce
byte-identical event logs. Every stochastic draw threads an explicit ``np.random.Generator``
(never the global RNG).

Stub: implemented in Phase 4.
"""

from __future__ import annotations


def main() -> None:
    """CLI entry point (``make sim SCEN=... SEED=...``)."""
    raise NotImplementedError("Phase 4: simulation engine is not implemented yet.")


if __name__ == "__main__":
    main()
