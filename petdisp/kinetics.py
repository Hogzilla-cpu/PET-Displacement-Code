"""Core kinetic machinery: time grids, frames, input functions and ODE solvers.

All times are in minutes. Activity concentrations are decay corrected and in
kBq/mL unless stated otherwise.

The tissue models in this package are all of the form

    dx/dt = a(t) - b(t) x(t),   x(0) = 0

with a time-varying efflux b(t). This linear ODE is solved exactly with an
integrating factor evaluated by the trapezoid rule on a uniform fine grid,
which is fully vectorised (no Python loop over time steps).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAMBDA_C11 = np.log(2.0) / 20.364  # 1/min
LAMBDA_F18 = np.log(2.0) / 109.77  # 1/min


# --------------------------------------------------------------------------
# Grids and frames
# --------------------------------------------------------------------------
def fine_grid(t_end: float, dt: float = 0.02) -> np.ndarray:
    """Uniform fine time grid from 0 to ``t_end`` (inclusive) in minutes."""
    n = int(round(t_end / dt))
    return np.linspace(0.0, n * dt, n + 1)


def cumtrapz0(y: np.ndarray, dt: float) -> np.ndarray:
    """Cumulative trapezoid integral on a uniform grid, starting at 0."""
    out = np.empty_like(y, dtype=float)
    out[0] = 0.0
    np.cumsum((y[1:] + y[:-1]) * (0.5 * dt), out=out[1:])
    return out


@dataclass(frozen=True)
class Frames:
    """PET frame definition (minutes)."""

    start: np.ndarray
    end: np.ndarray

    @classmethod
    def from_durations(cls, durations_sec) -> "Frames":
        d = np.asarray(durations_sec, dtype=float) / 60.0
        end = np.cumsum(d)
        return cls(start=end - d, end=end)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def dur(self) -> np.ndarray:
        return self.end - self.start

    @property
    def n(self) -> int:
        return len(self.start)

    @property
    def t_end(self) -> float:
        return float(self.end[-1])


def default_frames(scan_min: float = 150.0, late_frame_sec: float = 300.0) -> Frames:
    """6x10s, 6x20s, 4x60s, 4x120s, then ``late_frame_sec`` frames to the end."""
    early = [10] * 6 + [20] * 6 + [60] * 4 + [120] * 4  # 15 min
    used = sum(early) / 60.0
    n_late = int(np.ceil((scan_min - used) * 60.0 / late_frame_sec - 1e-9))
    late = [late_frame_sec] * n_late
    total = (sum(early) + sum(late)) / 60.0
    if total > scan_min + 1e-9:  # trim last frame
        late[-1] -= (total - scan_min) * 60.0
    return Frames.from_durations(early + late)


def frame_average(t: np.ndarray, y: np.ndarray, frames: Frames) -> np.ndarray:
    """Average a fine-grid curve over each frame."""
    dt = t[1] - t[0]
    integ = cumtrapz0(y, dt)
    return (np.interp(frames.end, t, integ) - np.interp(frames.start, t, integ)) / frames.dur


def frames_to_grid(frames: Frames, values: np.ndarray, t: np.ndarray, n_iter: int = 6) -> np.ndarray:
    """Piecewise-linear curve on the fine grid whose frame averages reproduce
    the framed data (nodes at frame mid-times, 0 at t=0).

    Plain linear interpolation of frame averages biases a convex/concave curve
    (about 1% in BPND for 5-min frames); a few additive correction passes on
    the node values remove most of it. Used for the reference region in
    reference-tissue fits.
    """
    values = np.asarray(values, dtype=float)
    tm = np.concatenate([[0.0], frames.mid])
    nodes = values.copy()
    for _ in range(n_iter):
        g = np.interp(t, tm, np.concatenate([[0.0], nodes]))
        nodes = nodes + (values - frame_average(t, g, frames))
    return np.interp(t, tm, np.concatenate([[0.0], nodes]))


# --------------------------------------------------------------------------
# Input function
# --------------------------------------------------------------------------
def feng_input(
    t: np.ndarray,
    A1: float = 851.1225,
    A2: float = 21.8798,
    A3: float = 20.8113,
    l1: float = -4.1339,
    l2: float = -0.1191,
    l3: float = -0.0104,
    delay: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Feng et al. (1993) plasma input function model (decay corrected).

    Defaults are Feng's published parameters; ``scale`` rescales amplitude.
    """
    tt = np.clip(np.asarray(t, dtype=float) - delay, 0.0, None)
    cp = (A1 * tt - A2 - A3) * np.exp(l1 * tt) + A2 * np.exp(l2 * tt) + A3 * np.exp(l3 * tt)
    cp = np.where(np.asarray(t) >= delay, cp, 0.0)
    return scale * np.clip(cp, 0.0, None)


# --------------------------------------------------------------------------
# ODE solver
# --------------------------------------------------------------------------
def solve_linear(a: np.ndarray, b: np.ndarray, dt: float, max_exp: float = 300.0) -> np.ndarray:
    """Solve dx/dt = a(t) - b(t) x on a uniform grid with x(0) = 0.

    Uses x(t) = exp(-B(t)) * int_0^t a(s) exp(B(s)) ds with B = int b. The grid
    is split into chunks whenever B grows by more than ``max_exp`` so that the
    exponentials never overflow.
    """
    a = np.asarray(a, dtype=float)
    b = np.broadcast_to(np.asarray(b, dtype=float), a.shape)
    B = cumtrapz0(b, dt)
    if B[-1] <= max_exp:
        eB = np.exp(B)
        return cumtrapz0(a * eB, dt) / eB
    # chunked version
    x = np.empty_like(a)
    i0, x0 = 0, 0.0
    n = len(a)
    while i0 < n:
        i1 = int(np.searchsorted(B, B[i0] + max_exp, side="right"))
        i1 = max(min(i1, n), i0 + 2) if i0 + 1 < n else n
        Bs = B[i0:i1] - B[i0]
        eB = np.exp(Bs)
        x[i0:i1] = (x0 + cumtrapz0(a[i0:i1] * eB, dt)) / eB
        if i1 >= n:
            break
        x0 = x[i1 - 1]
        i0 = i1 - 1
    return x


# --------------------------------------------------------------------------
# Tissue models with time-varying binding
# --------------------------------------------------------------------------
def tissue_1tc(t: np.ndarray, cp: np.ndarray, K1: float, k2: float, bp_t) -> np.ndarray:
    """Plasma-input 1TC displacement model (Laurell et al. 2023, Eq. 2 style).

    Specific and non-displaceable compartments are lumped (fast exchange), so
    the apparent efflux is k2a(t) = k2 / (1 + BPND(t)), where k2 = K1 / VND and
    BPND(t) = BPND0 * (1 - occ(t)).
    """
    dt = t[1] - t[0]
    k2a = k2 / (1.0 + np.asarray(bp_t))
    return solve_linear(K1 * cp, k2a, dt)


def tissue_srtm(t: np.ndarray, cr: np.ndarray, R1: float, k2: float, bp_t) -> np.ndarray:
    """Time-varying simplified reference tissue model.

    Exact consequence of 1TC target and reference regions with equal VND:

        dCt/dt = R1 dCr/dt + k2 Cr - k2a(t) Ct,  k2a(t) = k2 / (1 + BPND(t))

    Solved via y = Ct - R1 Cr:  dy/dt = (k2 - R1 k2a(t)) Cr - k2a(t) y,
    which avoids differentiating the (possibly noisy) reference curve.
    """
    dt = t[1] - t[0]
    k2a = k2 / (1.0 + np.asarray(bp_t))
    y = solve_linear((k2 - R1 * k2a) * cr, k2a, dt)
    return y + R1 * cr


def pet_signal(ct: np.ndarray, cb: np.ndarray, vB: float) -> np.ndarray:
    """Mix tissue and whole blood: (1 - vB) Ct + vB Cb."""
    return (1.0 - vB) * ct + vB * cb
