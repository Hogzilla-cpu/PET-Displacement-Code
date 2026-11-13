"""Occupancy time-course models used inside the time-varying kinetic models.

Each model maps a parameter vector ``theta`` to occ(t) on the fine grid, where
occ(t) is the fraction of *baseline-available* receptors blocked by the
displacing drug, so that BPND(t) = BPND0 * (1 - occ(t)).

* :class:`StepOccupancy`       - instantaneous change at t_s (paper's
                                  "single-step" approximation).
* :class:`BetaGrowthOccupancy` - sigmoidal growth from t_b to t_e (paper's
                                  occupancy function, Yin et al. 2003 beta
                                  growth function).
* :class:`DrugDrivenOccupancy` - occupancy computed from a known brain drug
                                  concentration profile via a competitive
                                  binding (Emax / Hill) relation with a single
                                  unknown EC50. This is the extension asked
                                  for: occupancy tied to the drug's measured or
                                  modelled brain PK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


class OccupancyModel:
    """Base class. Subclasses define names, x0, lower, upper and __call__."""

    names: tuple = ()

    def x0(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def bounds(self):  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, t: np.ndarray, theta: Sequence[float]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def summary(self, t: np.ndarray, theta: Sequence[float]) -> dict:
        """Derived occupancy values reported after a fit."""
        occ = self(t, theta)
        return {"occ_end": float(occ[-1]), "occ_peak": float(occ.max())}


@dataclass
class StepOccupancy(OccupancyModel):
    """occ(t) = occ_max for t >= t_s, else 0 (with a one-sample linear ramp so
    the objective is continuous in t_s)."""

    t_s0: float = 60.0
    t_s_bounds: tuple = (0.0, 1e9)
    fit_ts: bool = True
    names: tuple = field(default=("occ_max", "t_s"), init=False)

    def x0(self):
        return np.array([0.5, self.t_s0])

    def bounds(self):
        return np.array([0.0, self.t_s_bounds[0]]), np.array([1.0, self.t_s_bounds[1]])

    def __call__(self, t, theta):
        occ_max, ts = theta
        dt = t[1] - t[0]
        return occ_max * np.clip((t - ts) / dt + 0.5, 0.0, 1.0)

    def summary(self, t, theta):
        return {"occ_max": float(theta[0]), "occ_end": float(theta[0]), "occ_peak": float(theta[0])}


def beta_growth(t: np.ndarray, t_b: float, t_e: float, tm_frac: float = 0.5) -> np.ndarray:
    """Yin et al. (2003) beta growth function, normalised to 0..1.

    f(t) = (1 + (t_e - t)/(t_e - t_m)) * ((t - t_b)/(t_e - t_b))^((t_e - t_b)/(t_e - t_m))

    for t_b <= t <= t_e, 0 before t_b and 1 after t_e, with
    t_m = t_b + tm_frac (t_e - t_b) the time of the steepest rise.
    """
    span = max(t_e - t_b, 1e-9)
    tau = np.clip((t - t_b) / span, 0.0, 1.0)
    c = 1.0 / (1.0 - tm_frac)
    return (1.0 + (1.0 - tau) * c) * tau**c


@dataclass
class BetaGrowthOccupancy(OccupancyModel):
    """Laurell et al. (2023) sigmoidal occupancy: occ(t) = occ_max * f(t).

    ``t_b`` (drug administration time) is fixed and known. ``t_e`` (time when
    occ_max is reached) is estimated. ``tm_frac`` fixes the shape (0.5 =
    symmetric); set ``fit_shape=True`` to also estimate it.
    """

    t_b: float = 60.0
    te0: float = 75.0
    te_max: float = 1e9
    tm_frac: float = 0.5
    fit_shape: bool = False

    @property
    def names(self):
        return ("occ_max", "t_e", "tm_frac") if self.fit_shape else ("occ_max", "t_e")

    def x0(self):
        x = [0.5, max(self.te0, self.t_b + 1.0)]
        if self.fit_shape:
            x.append(self.tm_frac)
        return np.array(x)

    def bounds(self):
        lo = [0.0, self.t_b + 0.25]
        hi = [1.0, self.te_max]
        if self.fit_shape:
            lo.append(0.05)
            hi.append(0.95)
        return np.array(lo), np.array(hi)

    def __call__(self, t, theta):
        occ_max, te = theta[0], theta[1]
        frac = theta[2] if self.fit_shape else self.tm_frac
        return occ_max * beta_growth(t, self.t_b, te, frac)

    def summary(self, t, theta):
        return {"occ_max": float(theta[0]), "t_e": float(theta[1]),
                "occ_end": float(self(t, theta)[-1]), "occ_peak": float(theta[0])}


def hill_occupancy(conc, ec50, emax: float = 1.0, hill: float = 1.0):
    """Emax / Hill occupancy, safe for conc = 0."""
    c = np.clip(np.asarray(conc, dtype=float), 0.0, None) ** hill
    return emax * c / (c + ec50**hill)


@dataclass
class DrugDrivenOccupancy(OccupancyModel):
    """Occupancy driven by a known brain concentration profile.

    occ(t) = Emax * C(t)^h / (C(t)^h + EC50^h),  C(t) = conc_fn(t)

    ``conc_fn`` returns the (free) brain concentration of the displacing drug
    on the fine grid, already including dose and administration time. With
    competitive binding, the *apparent* EC50 = Ki_drug * (1 + E / K_E), where
    E is the endogenous competitor concentration and K_E its affinity; see
    :mod:`petdisp.endogenous`. Parameter: log10(EC50) (and optionally Emax).
    """

    conc_fn: Callable[[np.ndarray], np.ndarray] = None
    ec50_0: float = 1.0
    log10_ec50_bounds: tuple = (-6.0, 6.0)
    fit_emax: bool = False
    hill: float = 1.0
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def names(self):
        return ("log10_ec50", "emax") if self.fit_emax else ("log10_ec50",)

    def _conc(self, t):
        key = (len(t), float(t[0]), float(t[-1]))
        if key not in self._cache:
            self._cache.clear()
            self._cache[key] = np.asarray(self.conc_fn(t), dtype=float)
        return self._cache[key]

    def x0(self):
        x = [np.log10(self.ec50_0)]
        if self.fit_emax:
            x.append(0.9)
        return np.array(x)

    def bounds(self):
        lo = [self.log10_ec50_bounds[0]]
        hi = [self.log10_ec50_bounds[1]]
        if self.fit_emax:
            lo.append(0.0)
            hi.append(1.0)
        return np.array(lo), np.array(hi)

    def __call__(self, t, theta):
        ec50 = 10.0 ** theta[0]
        emax = theta[1] if self.fit_emax else 1.0
        return hill_occupancy(self._conc(t), ec50, emax, self.hill)

    def summary(self, t, theta):
        occ = self(t, theta)
        out = {"ec50": float(10.0 ** theta[0]), "occ_peak": float(occ.max()),
               "occ_end": float(occ[-1]), "t_peak": float(t[int(np.argmax(occ))])}
        if self.fit_emax:
            out["emax"] = float(theta[1])
        return out
