"""Vision-estimated flood current — drive the drift model from a *measured* flow, not an assumed.

The survivor-drift model (`src/drift/advect.py`) advects particles under a
``FlowField = (x, y) -> (vx, vy)``. Until now that field was always *assumed* (an analytic
uniform / channel / radial guess). This module *estimates* it from drone imagery by **particle
image velocimetry** (PIV): FFT cross-correlation of interrogation windows between consecutive
frames gives a per-window pixel displacement, which converts to a metric velocity grid and then to
a ``FlowField`` the drift model consumes unchanged.

This is the one connection the literature review flagged as un-made (image velocimetry and
survivor-drift both exist; wiring one into the other does not appear in the reviewed sources). It is
offered as a *novel integration / proof of concept*, not a novel method — and it stays offline and
cached, so ADR-001 holds (the simulator never runs perception; it only consumes the field).

Honest limitation: no real flood video with a known current was available, so the demo runs on a
**synthetic** clip with a KNOWN ground-truth flow (also what makes PIV testable). Pure numpy — no
OpenCV; every stochastic call threads an explicit ``np.random.Generator``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

FlowField = Callable[[float, float], tuple[float, float]]


# ------------------------------------------------------------------------- synthesis (known truth)
def synthesize_flow_video(
    truth_flow: FlowField,
    *,
    n_frames: int,
    frame_px: tuple[int, int],
    metres_per_pixel: float,
    fps: float,
    rng: np.random.Generator,
    density: float = 0.03,
    noise: float = 0.0,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """A synthetic overhead 'water' clip whose surface texture is advected by a ``truth_flow``.

    Each frame resamples one pristine speckle texture by the cumulative displacement implied by the
    flow, so the motion between consecutive frames is exactly the flow (in px/frame) and the texture
    never blurs. Returns ``(n_frames, H, W)`` float32 in [0, 1].
    """
    h, w = int(frame_px[0]), int(frame_px[1])
    ox, oy = origin_xy

    # pristine speckle texture (bright dots on dark water)
    base = np.zeros((h, w), dtype=np.float32)
    n_dots = max(1, int(density * h * w))
    dr = rng.integers(0, h, size=n_dots)
    dc = rng.integers(0, w, size=n_dots)
    base[dr, dc] = 1.0

    # per-pixel displacement (px/frame): east = column, south = row
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_world = cc * metres_per_pixel + ox  # east
    y_world = rr * metres_per_pixel + oy  # south
    vf = np.vectorize(lambda x, y: truth_flow(float(x), float(y)), otypes=[float, float])
    vx, vy = vf(x_world, y_world)  # m/s
    d_col = vx / (metres_per_pixel * fps)  # px/frame east
    d_row = vy / (metres_per_pixel * fps)  # px/frame south

    frames = np.empty((int(n_frames), h, w), dtype=np.float32)
    for t in range(int(n_frames)):
        src_r = np.mod(np.round(rr - t * d_row).astype(int), h)
        src_c = np.mod(np.round(cc - t * d_col).astype(int), w)
        frame = base[src_r, src_c]
        if noise > 0.0:
            frame = frame + rng.normal(0.0, noise, size=frame.shape).astype(np.float32)
        frames[t] = np.clip(frame, 0.0, 1.0)
    return frames


# ------------------------------------------------------------------------------------- PIV estimate
def _xcorr_shift(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple[float, float]:
    """Integer (dy, dx) content shift from window ``a`` -> ``b`` via FFT cross-correlation.

    ``b ≈ roll(a, (dy, dx))`` at the correlation peak; searched within ``±max_shift`` of zero.
    """
    a = a - a.mean()
    b = b - b.mean()
    r = np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))).real
    r = np.fft.fftshift(r)  # zero shift -> centre
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    ms = int(min(max_shift, cy, cx))
    sub = r[cy - ms : cy + ms + 1, cx - ms : cx + ms + 1]
    py, px = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return float(py - ms), float(px - ms)


def estimate_flow_piv(
    frames: np.ndarray,
    *,
    window_px: int,
    step_px: int,
    max_shift_px: int | None = None,
) -> dict:
    """PIV over a clip. Returns ``{'centers_px': (M,2)[row,col], 'disp_px': (M,2)[dY,dX]}`` — the
    displacement is the mean per-frame content shift over consecutive frame pairs (px/frame)."""
    frames = np.asarray(frames, dtype=float)
    n, h, w = frames.shape
    win, step = int(window_px), int(step_px)
    max_shift = int(max_shift_px) if max_shift_px is not None else win // 2

    centers, disps = [], []
    for r0 in range(0, h - win + 1, step):
        for c0 in range(0, w - win + 1, step):
            pair_shifts = []
            for t in range(n - 1):
                a = frames[t, r0 : r0 + win, c0 : c0 + win]
                b = frames[t + 1, r0 : r0 + win, c0 : c0 + win]
                pair_shifts.append(_xcorr_shift(a, b, max_shift))
            centers.append((r0 + win / 2.0, c0 + win / 2.0))
            disps.append(np.mean(pair_shifts, axis=0))
    return {"centers_px": np.asarray(centers), "disp_px": np.asarray(disps)}


# ------------------------------------------------------------------- PIV -> metric FlowField (IDW)
def piv_to_flowfield(
    piv: dict,
    *,
    metres_per_pixel: float,
    fps: float,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[FlowField, dict]:
    """Convert PIV pixel displacements to a metric ``FlowField`` by inverse-distance interpolation.

    ``velocity = disp_px · metres_per_pixel · fps``. East = column (dX), south = row (dY).
    Returns the callable field and ``{'points_xy': (M,2), 'vel': (M,2)}`` (world metres, m/s).
    """
    centers = np.asarray(piv["centers_px"], dtype=float)  # [row, col]
    disp = np.asarray(piv["disp_px"], dtype=float)  # [dY, dX]
    ox, oy = origin_xy
    xs = centers[:, 1] * metres_per_pixel + ox  # east
    ys = centers[:, 0] * metres_per_pixel + oy  # south
    vx = disp[:, 1] * metres_per_pixel * fps  # east m/s
    vy = disp[:, 0] * metres_per_pixel * fps  # south m/s

    def flow(x: float, y: float) -> tuple[float, float]:
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        j = int(np.argmin(d2))
        if d2[j] < 1e-9:  # sample sits on a measured point
            return float(vx[j]), float(vy[j])
        wgt = 1.0 / d2  # inverse-distance-squared weighting
        wsum = wgt.sum()
        return float((wgt * vx).sum() / wsum), float((wgt * vy).sum() / wsum)

    return flow, {"points_xy": np.column_stack([xs, ys]), "vel": np.column_stack([vx, vy])}


# ----------------------------------------------------------------------------- cache (offline art)
def save_flow_field(path: Path, sampled: dict, meta: dict | None = None) -> None:
    """Persist the estimated field (world points + velocities) so the sim can consume it offline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        points_xy=sampled["points_xy"],
        vel=sampled["vel"],
        meta=np.array([str(meta or {})], dtype=object),
    )


def load_flow_field(path: Path) -> FlowField:
    """Rebuild an IDW ``FlowField`` from a cached ``save_flow_field`` artifact."""
    data = np.load(Path(path), allow_pickle=True)
    xs, ys = data["points_xy"][:, 0], data["points_xy"][:, 1]
    vx, vy = data["vel"][:, 0], data["vel"][:, 1]

    def flow(x: float, y: float) -> tuple[float, float]:
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        j = int(np.argmin(d2))
        if d2[j] < 1e-9:
            return float(vx[j]), float(vy[j])
        wgt = 1.0 / d2
        wsum = wgt.sum()
        return float((wgt * vx).sum() / wsum), float((wgt * vy).sum() / wsum)

    return flow
