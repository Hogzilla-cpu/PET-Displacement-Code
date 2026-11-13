"""Model fitting: SRTM, time-varying SRTM, plasma-input 1TC displacement model
(single and multi-region, Laurell et al. 2023), and the Lassen plot.

All fits use weighted non-linear least squares (scipy ``least_squares``,
trust-region reflective with bounds) with frame weights w_k = dt_k exp(-lambda t_k)
unless explicit weights are given. Approximate parameter covariance is
returned from the Jacobian at the solution: cov = s^2 (J^T J)^-1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .kinetics import (Frames, fine_grid, frame_average, frames_to_grid, pet_signal,
                       tissue_1tc, tissue_srtm)
from .occupancy import OccupancyModel
from .simulate import frame_weights


@dataclass
class FitResult:
    params: dict
    derived: dict
    x: np.ndarray
    names: List[str]
    cov: Optional[np.ndarray]
    cost: float
    success: bool
    fitted: np.ndarray
    extra: dict = field(default_factory=dict)

    def se(self) -> dict:
        if self.cov is None:
            return {n: np.nan for n in self.names}
        d = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        return dict(zip(self.names, d))


def _lsq(resid, x0s, lo, hi, **kw):
    best = None
    for x0 in x0s:
        x0 = np.clip(np.asarray(x0, float), lo + 1e-9 * (hi - lo + 1), hi - 1e-9 * (hi - lo + 1))
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", **kw)
        except Exception:  # pragma: no cover - robustness in MC loops
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def _cov(res, n_obs):
    J = res.jac
    dof = max(n_obs - len(res.x), 1)
    s2 = 2.0 * res.cost / dof
    try:
        return s2 * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:  # pragma: no cover
        return None


def _mask(frames: Frames, t_max: Optional[float]):
    return np.ones(frames.n, bool) if t_max is None else frames.end <= t_max + 1e-9


def _grid(frames: Frames, dt: float):
    return fine_grid(frames.t_end, dt)


# --------------------------------------------------------------------------
# Reference-tissue models
# --------------------------------------------------------------------------
def fit_srtm(frames: Frames, ct, cr, lam: float, t_max: Optional[float] = None,
             weights=None, dt: float = 0.05) -> FitResult:
    """Standard SRTM (constant BPND). With ``t_max`` only frames ending before
    t_max are used, e.g. the pre-displacement part of the scan."""
    t = _grid(frames, dt)
    crf = frames_to_grid(frames, cr, t)
    m = _mask(frames, t_max)
    w = np.sqrt(frame_weights(frames, lam) if weights is None else weights)[m]
    ct = np.asarray(ct)[m]

    def model(x):
        return frame_average(t, tissue_srtm(t, crf, x[0], x[1], x[2]), frames)

    def resid(x):
        return w * (model(x)[m] - ct)

    lo, hi = np.array([0.05, 1e-4, 0.0]), np.array([5.0, 2.0, 20.0])
    res = _lsq(resid, [[1.0, 0.1, 1.0], [1.0, 0.05, 3.0]], lo, hi)
    names = ["R1", "k2", "BP0"]
    return FitResult(dict(zip(names, res.x)), {"BP0": float(res.x[2])}, res.x, names,
                     _cov(res, m.sum()), float(res.cost), bool(res.success), model(res.x))


def fit_tv_srtm(frames: Frames, ct, cr, occ_model: OccupancyModel, lam: float,
                weights=None, dt: float = 0.05, x0_occ: Optional[Sequence] = None) -> FitResult:
    """Time-varying SRTM: BPND(t) = BP0 (1 - occ(t; theta)).

    Parameters: R1, k2, BP0 and the occupancy model's parameters. With a
    :class:`~petdisp.occupancy.DrugDrivenOccupancy`, theta = log10(EC50).
    """
    t = _grid(frames, dt)
    crf = frames_to_grid(frames, cr, t)
    w = np.sqrt(frame_weights(frames, lam) if weights is None else weights)
    ct = np.asarray(ct)
    n_occ = len(occ_model.names)

    def model(x):
        bp_t = x[2] * (1.0 - occ_model(t, x[3:]))
        return frame_average(t, tissue_srtm(t, crf, x[0], x[1], bp_t), frames)

    def resid(x):
        return w * (model(x) - ct)

    olo, ohi = occ_model.bounds()
    lo = np.concatenate([[0.05, 1e-4, 0.0], olo])
    hi = np.concatenate([[5.0, 2.0, 20.0], ohi])
    occ_starts = [occ_model.x0()] if x0_occ is None else [np.asarray(x0_occ)]
    # extra starts for the leading occupancy parameter
    base = occ_starts[0]
    for v in ([0.2, 0.8] if occ_model.names[0] == "occ_max" else [base[0] - 1.0, base[0] + 1.0]):
        alt = base.copy()
        alt[0] = v
        occ_starts.append(alt)
    x0s = [np.concatenate([[1.0, 0.08, bp], o]) for o in occ_starts for bp in (1.5, 4.0)]
    res = _lsq(resid, x0s, lo, hi)
    names = ["R1", "k2", "BP0", *occ_model.names]
    derived = {"BP0": float(res.x[2]), **occ_model.summary(t, res.x[3:])}
    return FitResult(dict(zip(names, res.x)), derived, res.x, names, _cov(res, frames.n),
                     float(res.cost), bool(res.success), model(res.x))


def fit_tv_srtm_multi(frames: Frames, cts, cr, occ_model: OccupancyModel, lam: float,
                      weights=None, dt: float = 0.05) -> FitResult:
    """Multi-region time-varying SRTM with shared reference efflux k2' (so that
    k2_i = R1_i k2', as in SRTM2) and shared occupancy parameters; per-region
    R1_i and BP0_i. This mirrors the paper's shared-occupancy strategy for a
    reference-tissue setting."""
    t = _grid(frames, dt)
    crf = frames_to_grid(frames, cr, t)
    cts = np.atleast_2d(cts)
    nr = cts.shape[0]
    w = np.sqrt(frame_weights(frames, lam) if weights is None else weights)
    n_occ = len(occ_model.names)

    def unpack(x):
        k2r, th = x[0], x[1:1 + n_occ]
        rest = x[1 + n_occ:].reshape(nr, 2)
        return k2r, th, rest

    def model(x):
        k2r, th, rest = unpack(x)
        occ = occ_model(t, th)
        out = []
        for R1, bp0 in rest:
            out.append(frame_average(t, tissue_srtm(t, crf, R1, R1 * k2r, bp0 * (1 - occ)), frames))
        return np.array(out)

    def resid(x):
        return (w * (model(x) - cts)).ravel()

    olo, ohi = occ_model.bounds()
    lo = np.concatenate([[1e-3], olo, np.tile([0.05, 0.0], nr)])
    hi = np.concatenate([[2.0], ohi, np.tile([5.0, 20.0], nr)])
    base = occ_model.x0()
    starts = [base]
    alt = base.copy()
    alt[0] = 0.8 if occ_model.names[0] == "occ_max" else base[0] + 1.0
    starts.append(alt)
    x0s = [np.concatenate([[0.08], o, np.tile([1.0, 2.5], nr)]) for o in starts]
    res = _lsq(resid, x0s, lo, hi)
    names = ["k2_ref", *occ_model.names] + [f"{p}_{i}" for i in range(nr) for p in ("R1", "BP0")]
    k2r, th, rest = unpack(res.x)
    derived = {"BP0": rest[:, 1].copy(), "R1": rest[:, 0].copy(), "k2_ref": float(k2r),
               **occ_model.summary(t, th)}
    return FitResult(dict(zip(names, res.x)), derived, res.x, names, _cov(res, cts.size),
                     float(res.cost), bool(res.success), model(res.x))


# --------------------------------------------------------------------------
# Plasma-input models
# --------------------------------------------------------------------------
def fit_1tcm(frames: Frames, ct, t: np.ndarray, cp: np.ndarray, lam: float,
             t_max: Optional[float] = None, fit_vb: bool = True, weights=None) -> FitResult:
    """Standard 1TCM with blood volume. Returns K1, k2, vB and VT = K1/k2.
    ``t``/``cp`` are the fine grid and plasma input (whole blood = plasma)."""
    m = _mask(frames, t_max)
    w = np.sqrt(frame_weights(frames, lam) if weights is None else weights)[m]
    ct = np.asarray(ct)[m]

    def model(x):
        vb = x[2] if fit_vb else 0.0
        return frame_average(t, pet_signal(tissue_1tc(t, cp, x[0], x[1], 0.0), cp, vb), frames)

    def resid(x):
        return w * (model(x)[m] - ct)

    lo = np.array([1e-3, 1e-4, 0.0])[: 3 if fit_vb else 2]
    hi = np.array([3.0, 3.0, 0.3])[: 3 if fit_vb else 2]
    x0s = [np.array([0.3, 0.02, 0.05])[: len(lo)], np.array([0.3, 0.1, 0.05])[: len(lo)]]
    res = _lsq(resid, x0s, lo, hi)
    names = ["K1", "k2", "vB"][: len(lo)]
    VT = res.x[0] / res.x[1]
    return FitResult(dict(zip(names, res.x)), {"VT": float(VT)}, res.x, names,
                     _cov(res, m.sum()), float(res.cost), bool(res.success), model(res.x))


def fit_tv_1tc(frames: Frames, cts, t: np.ndarray, cp: np.ndarray, occ_model: OccupancyModel,
               lam: float, weights=None, fit_vb: bool = True) -> FitResult:
    """Laurell et al. (2023) displacement model, multi-region.

    Global: VND and the occupancy parameters (occ_max and t_e for the beta
    growth model, or occ_max and t_s for the single-step model, or EC50 for
    the drug-driven model). Per region: K1_i, BP0_i (and vB_i).

        dC_i/dt = K1_i C_p - (K1_i / VND) / (1 + BP0_i (1 - occ(t))) C_i

    The paper used a nested optimisation (outer: global, inner: regional);
    here all parameters are estimated jointly, which minimises the same
    objective. Pass a single TAC for a single-region fit (VND then only weakly
    identifiable, as the paper notes).
    """
    cts = np.atleast_2d(cts)
    nr = cts.shape[0]
    w = np.sqrt(frame_weights(frames, lam) if weights is None else weights)
    n_occ = len(occ_model.names)
    npr = 3 if fit_vb else 2

    def unpack(x):
        return x[0], x[1:1 + n_occ], x[1 + n_occ:].reshape(nr, npr)

    def model(x):
        vnd, th, rest = unpack(x)
        occ = occ_model(t, th)
        out = []
        for row in rest:
            K1, bp0 = row[0], row[1]
            vb = row[2] if fit_vb else 0.0
            ct = tissue_1tc(t, cp, K1, K1 / vnd, bp0 * (1 - occ))
            out.append(frame_average(t, pet_signal(ct, cp, vb), frames))
        return np.array(out)

    def resid(x):
        return (w * (model(x) - cts)).ravel()

    olo, ohi = occ_model.bounds()
    rlo, rhi = np.array([1e-3, 0.0, 0.0])[:npr], np.array([3.0, 30.0, 0.3])[:npr]
    lo = np.concatenate([[0.1], olo, np.tile(rlo, nr)])
    hi = np.concatenate([[50.0], ohi, np.tile(rhi, nr)])
    base = occ_model.x0()
    starts = [base]
    alt = base.copy()
    alt[0] = 0.8 if occ_model.names[0] == "occ_max" else base[0] + 1.0
    starts.append(alt)
    x0s = [np.concatenate([[v], o, np.tile(np.array([0.3, 2.5, 0.05])[:npr], nr)])
           for o in starts for v in (3.0, 6.0)]
    res = _lsq(resid, x0s, lo, hi)
    pnames = ("K1", "BP0", "vB")[:npr]
    names = ["VND", *occ_model.names] + [f"{p}_{i}" for i in range(nr) for p in pnames]
    vnd, th, rest = unpack(res.x)
    derived = {"VND": float(vnd), "BP0": rest[:, 1].copy(), "K1": rest[:, 0].copy(),
               "VT": vnd * (1 + rest[:, 1]), **occ_model.summary(t, th)}
    return FitResult(dict(zip(names, res.x)), derived, res.x, names, _cov(res, cts.size),
                     float(res.cost), bool(res.success), model(res.x))


# --------------------------------------------------------------------------
# Lassen plot
# --------------------------------------------------------------------------
def lassen(vt_base, vt_block) -> dict:
    """Lassen (occupancy) plot: VT_base - VT_block = occ (VT_base - VND).

    Ordinary least squares of y = VT_base - VT_block on x = VT_base gives
    slope = occupancy and x-intercept = VND.
    """
    x = np.asarray(vt_base, float)
    y = x - np.asarray(vt_block, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
    vnd = -icpt / slope if slope != 0 else np.nan
    return {"occ": float(slope), "VND": float(vnd), "intercept": float(icpt)}


def occupancy_from_bp(bp_base, bp_block) -> dict:
    """Reference-region analogue: occ_i = 1 - BP_block_i/BP_base_i, plus the
    through-origin regression slope of (BP_base - BP_block) on BP_base."""
    b = np.asarray(bp_base, float)
    k = np.asarray(bp_block, float)
    slope = float(np.sum(b * (b - k)) / np.sum(b * b))
    return {"occ": slope, "occ_regional": 1.0 - k / b}
