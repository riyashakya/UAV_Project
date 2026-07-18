"""Flood survivor-drift prediction (CPU).

Single responsibility: project the position distribution of a survivor drifting in flowing
water, so routing searches a polygon rather than a stale point. Method adapted from maritime
SAR planning (USCG SAROPS: leeway + Monte Carlo containment) — Phase 7.

Scope note: the drift idea was suggested by the supervisor (per the student, 2026-07-17).
PROJECT_PLAN §1 flags that it is not in the *written approved proposal* — so confirm it is
captured there before relying on it in the write-up.
"""
