"""Monte Carlo sweep runner — Phase 9.

Single responsibility: execute the grid {baselines} x {N=1,2,4,6 UAVs} x {seeds} x
{scenarios}, plus ablations (reallocation off; priority-upweighting off), and write tidy
Parquet to ``outputs/runs/<timestamp>/`` with the resolved config dumped alongside. Never
overwrite a previous run.

Stub: implemented in Phase 9.
"""

from __future__ import annotations


def main() -> None:
    """CLI entry point (``make sweep``)."""
    raise NotImplementedError("Phase 9: sweep runner is not implemented yet.")


if __name__ == "__main__":
    main()
