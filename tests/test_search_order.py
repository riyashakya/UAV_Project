"""Probability-guided vs uniform search ordering (self-contained: synthetic clustered survivors)."""

from __future__ import annotations

from omegaconf import OmegaConf
from src.eval.search_order import run_search_order


def _cfg(**over):
    d = dict(
        grid={"rows": 6, "cols": 6},
        cell_size_m=200,
        cluster_center=[4, 4],
        cluster_sigma=1.2,
        survivors_total=40,
        prior_noise=0.15,
        person_fn=0.1,
        n_seeds=6,
        duration_min=90,
    )
    d.update(over)
    return OmegaConf.create(d)


def test_informative_prior_locates_survivors_faster():
    """With an informative prior, guided search locates 80% of survivors meaningfully sooner."""
    res = run_search_order(_cfg(prior_noise=0.1))
    g = res["policies"]["guided"]["t80_mean"]
    u = res["policies"]["uniform"]["t80_mean"]
    assert g < u  # guided reaches 80% first
    assert u / g >= 1.15  # ...and by a meaningful margin


def test_uninformative_prior_gives_no_big_advantage():
    """With a useless (uniform) prior, guided has no large advantage — the benefit is the prior."""
    res = run_search_order(_cfg(prior_noise=1.0))
    g = res["policies"]["guided"]["t80_mean"]
    u = res["policies"]["uniform"]["t80_mean"]
    assert u / max(g, 1e-9) < 1.15  # comparable, not a big speedup
