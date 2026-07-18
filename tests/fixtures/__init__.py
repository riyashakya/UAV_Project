"""Synthetic dataset fixtures — tiny on-disk stand-ins shaped like the real datasets.

No GPU, no downloads, no real imagery (CLAUDE.md Testing). Each builder writes a handful of
files in the native format of one source dataset with *known* instance counts, so loader and
build tests can assert exact numbers.
"""
