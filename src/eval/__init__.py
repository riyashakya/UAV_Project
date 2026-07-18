"""Evaluation harness (CPU).

Single responsibility: run the Monte Carlo sweep across baselines x UAV counts x seeds x
scenarios, compute metrics, and emit tidy Parquet with mean +/- 95 % CI. Build this the
moment coordination works (Phase 9), not at the end — every later result is then measurable
the day it lands.
"""
