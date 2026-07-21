"""Coverage paths: full visitation incl. the last-row bug (Phase 5 acceptance)."""

from __future__ import annotations

from src.coordination.coverage import coverage_fraction, coverage_path, sweep_line_ys
from src.sim.world import World


def test_sweep_lines_cover_the_full_extent():
    # 0..250 with 100 m footprint, no sidelap -> stride 100, does NOT divide evenly.
    ys = sweep_line_ys(0.0, 250.0, footprint_width_m=100.0, sidelap=0.0)
    assert max(ys) + 50.0 >= 250.0 - 1e-9  # far edge is covered (the last-row fix)
    assert min(ys) - 50.0 <= 0.0 + 1e-9  # near edge covered too


def test_last_row_not_skipped_when_height_not_multiple_of_footprint():
    """The classic bug: a naive sweep would leave the top strip uncovered. Ours must not."""
    # 5 rows x 2 cols, 100 m cells -> sector height 500 m; footprint 120, sidelap 0 -> stride 120,
    # which does not divide 500 -> the naive loop stops short of the top row.
    world = World(rows=5, cols=2, cell_size_m=100.0)
    sector = list(range(world.n_cells))
    frac = coverage_fraction(sector, world, footprint_width_m=120.0, sidelap=0.0)
    assert frac == 1.0  # every cell, including the last row, is covered


def test_coverage_exceeds_99pct_on_a_sector():
    world = World(rows=6, cols=6, cell_size_m=200.0)
    sector = list(range(world.n_cells))
    assert coverage_fraction(sector, world, footprint_width_m=220.0, sidelap=0.2) > 0.99


def test_coverage_path_is_boustrophedon():
    world = World(rows=4, cols=4, cell_size_m=100.0)
    path = coverage_path(list(range(16)), world, footprint_width_m=100.0, sidelap=0.0)
    assert len(path) >= 4  # at least a couple of sweep legs
    # consecutive legs reverse direction: leg 0 goes +x, leg 1 goes -x
    assert path[0][0] < path[1][0]  # first leg left -> right
    assert path[2][0] > path[3][0]  # second leg right -> left


def test_thin_sector_gets_a_single_centred_line():
    ys = sweep_line_ys(0.0, 50.0, footprint_width_m=200.0, sidelap=0.2)
    assert ys == [25.0]  # thinner than one footprint -> one line at the centre
