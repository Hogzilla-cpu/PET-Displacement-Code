"""Brain concentration profiles of the displacing drug.

Two ways to describe the drug's brain time course:

* :class:`DrugPK` - a compact parametric model: IV infusion (bolus when the
  infusion is short) into a one- or two-exponential plasma disposition,
  followed by first-order equilibration of the free brain concentration with
  free plasma (effect-compartment rate ``keq``). Linear in dose.
* :class:`TabulatedProfile` - a measured or externally simulated brain
  concentration curve for a reference dose, shifted to the administration time
  and scaled linearly with dose.

Both expose ``brain(t, dose, t_inj)`` returning concentration on ``t`` in the
units the user chooses (e.g. ng/mL or nM); EC50 / Ki must use the same units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .kinetics import solve_linear


@dataclass
class DrugPK:
    """Parametric plasma -> brain PK for the displacing drug.

    Plasma unit impulse response: h(t) = (1/V) [f1 exp(-k1 t) + (1-f1) exp(-k2 t)].
    Free brain: dCb/dt = keq (Kp_uu * fu * Cp - Cb).

    Parameters
    ----------
    V : volume of the central compartment (L). Dose in mg gives mg/L = ug/mL.
    k1, k2 : disposition rate constants (1/min). Set f1 = 1 for one exponential.
    keq : plasma-brain equilibration rate (1/min). ln(2)/keq is the brain
        equilibration half-life; large keq = "fast" drug.
    Kp_uu, fu : unbound brain/plasma partition and plasma free fraction.
    infusion_min : infusion duration (min); 0.33 min ~ the paper's 20 s bolus.
    """

    V: float = 50.0
    k1: float = 0.1
    k2: float = 0.005
    f1: float = 0.5
    keq: float = 0.3
    Kp_uu: float = 1.0
    fu: float = 1.0
    infusion_min: float = 1.0 / 3.0

    def plasma(self, t: np.ndarray, dose: float, t_inj: float) -> np.ndarray:
        tau = np.asarray(t, dtype=float) - t_inj
        T = max(self.infusion_min, 1e-6)
        rate = dose / T
        out = np.zeros_like(tau)
        for frac, k in ((self.f1, self.k1), (1.0 - self.f1, self.k2)):
            if frac == 0.0:
                continue
            during = (tau >= 0) & (tau < T)
            after = tau >= T
            part = np.zeros_like(tau)
            part[during] = (1.0 - np.exp(-k * tau[during])) / k
            part[after] = (1.0 - np.exp(-k * T)) / k * np.exp(-k * (tau[after] - T))
            out += frac * rate / self.V * part
        return out

    def brain(self, t: np.ndarray, dose: float, t_inj: float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        dt = t[1] - t[0]
        cp_free = self.Kp_uu * self.fu * self.plasma(t, dose, t_inj)
        return solve_linear(self.keq * cp_free, np.full_like(t, self.keq), dt)

    def conc_fn(self, dose: float, t_inj: float):
        """Closure for :class:`~petdisp.occupancy.DrugDrivenOccupancy`."""
        return lambda t: self.brain(t, dose, t_inj)


@dataclass
class TabulatedProfile:
    """Brain concentration measured/simulated for ``ref_dose`` at times
    ``t_rel`` (minutes after administration). Linear interpolation, zero before
    administration, last value held (or decayed with ``tail_k``) afterwards."""

    t_rel: Sequence[float]
    conc: Sequence[float]
    ref_dose: float = 1.0
    tail_k: float = 0.0

    def brain(self, t: np.ndarray, dose: float, t_inj: float) -> np.ndarray:
        tr = np.asarray(self.t_rel, dtype=float)
        c = np.asarray(self.conc, dtype=float)
        tau = np.asarray(t, dtype=float) - t_inj
        out = np.interp(tau, tr, c, left=0.0, right=c[-1])
        if self.tail_k > 0:
            late = tau > tr[-1]
            out[late] = c[-1] * np.exp(-self.tail_k * (tau[late] - tr[-1]))
        out[tau < 0] = 0.0
        return out * (dose / self.ref_dose)

    def conc_fn(self, dose: float, t_inj: float):
        return lambda t: self.brain(t, dose, t_inj)
