"""Forward simulation of single-scan displacement studies (and the classic
baseline + block two-scan design used for Lassen plots).

The data-generating model:

* Plasma input C_p(t) (Feng model by default), whole blood = plasma.
* Every region follows a 1TC model with lumped specific binding (Laurell et al.
  2023), dC_T/dt = K1 C_p - k2/(1 + BPND(t)) C_T, k2 = K1 / VND.
* BPND(t) comes from competitive binding between the tracer, the displacing
  drug (known brain profile C(t), affinity Ki), a constant endogenous
  competitor (E/K_E) and, optionally, the tracer's own mass (molar activity):

      BPND(t) = fND Bmax / (Kd (1 + E/K_E + C(t)/Ki + F_tracer(t)/Kd))

* A reference region with Bmax = 0 for reference-tissue (SRTM) analyses.
* Gaussian frame noise with variance proportional to C exp(lambda t) / dt.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Union

import numpy as np

from .endogenous import apparent_ec50, bp_from_receptors, competitive_bp
from .kinetics import (LAMBDA_C11, Frames, default_frames, feng_input, fine_grid,
                       frame_average, pet_signal, tissue_1tc)
from .pk import DrugPK, TabulatedProfile


# --------------------------------------------------------------------------
# Scenario description
# --------------------------------------------------------------------------
@dataclass
class Tracer:
    """PET tracer properties.

    kd : equilibrium dissociation constant (nM).
    fnd : free fraction in the non-displaceable compartment.
    half_life : radionuclide half-life (min).
    molar_activity : GBq/umol at injection; None = true tracer dose (no mass
        effect). kBq/mL divided by GBq/umol gives nM.
    """

    kd: float = 3.0
    fnd: float = 0.3
    half_life: float = 20.364
    molar_activity: Optional[float] = None

    @property
    def lam(self) -> float:
        return np.log(2.0) / self.half_life


@dataclass
class Region:
    """A brain region: delivery K1 (mL/cm3/min), VND (mL/cm3), Bmax (nM)."""

    name: str
    K1: float
    VND: float
    bmax: float = 0.0

    @property
    def k2(self) -> float:
        return self.K1 / self.VND

    @classmethod
    def from_bp(cls, name, K1, VND, bp0, tracer: Tracer, endo_ratio: float = 0.0):
        """Construct from a desired baseline BPND (in presence of endogenous)."""
        bmax = bp0 * tracer.kd * (1.0 + endo_ratio) / tracer.fnd
        return cls(name, K1, VND, bmax)


@dataclass
class Drug:
    """Displacing drug: brain PK, affinity (same units as PK output), dose and
    administration time (min after tracer injection)."""

    pk: Union[DrugPK, TabulatedProfile] = field(default_factory=DrugPK)
    ki: float = 0.05
    dose: float = 10.0
    t_inj: float = 60.0

    def conc(self, t):
        if self.dose == 0:
            return np.zeros_like(t)
        return self.pk.brain(t, self.dose, self.t_inj)

    def conc_fn(self):
        return self.pk.conc_fn(self.dose, self.t_inj)


@dataclass
class Scenario:
    tracer: Tracer = field(default_factory=Tracer)
    regions: List[Region] = field(default_factory=list)
    reference: Region = field(default_factory=lambda: Region("reference", 0.3, 4.0, 0.0))
    drug: Drug = field(default_factory=Drug)
    endo_ratio: float = 0.0  # E / K_E of the endogenous competitor
    k_endo: Optional[float] = None  # K_E, to convert the ratio to a concentration
    frames: Frames = field(default_factory=default_frames)
    input_fn: Callable[[np.ndarray], np.ndarray] = feng_input
    noise_scale: float = 0.5
    vB: float = 0.05
    dt: float = 0.02

    def with_drug(self, **kw) -> "Scenario":
        return replace(self, drug=replace(self.drug, **kw))

    @property
    def ec50_app(self) -> float:
        return apparent_ec50(self.drug.ki, self.endo_ratio)

    def bp0(self, region: Region) -> float:
        return float(bp_from_receptors(region.bmax, self.tracer.kd, self.tracer.fnd, self.endo_ratio))


def example_scenario(endo_ratio: float = 0.5, **overrides) -> Scenario:
    """[11C]UCB-J-like tracer (VND = 4, baseline BPND 1.5-4.6) with 4 target
    regions and a reference region, Feng input, 150-min scan, drug at 60 min,
    endogenous competitor occupying E/K_E = 0.5 (a third of receptors)."""
    tracer = Tracer()
    regions = [
        Region.from_bp("putamen", 0.30, 4.0, 4.6, tracer, endo_ratio),
        Region.from_bp("frontal", 0.32, 4.0, 3.5, tracer, endo_ratio),
        Region.from_bp("thalamus", 0.35, 4.0, 2.6, tracer, endo_ratio),
        Region.from_bp("hippocampus", 0.25, 4.0, 1.5, tracer, endo_ratio),
    ]
    sc = Scenario(tracer=tracer, regions=regions, endo_ratio=endo_ratio)
    return replace(sc, **overrides)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------
def frame_sd(frames: Frames, tac: np.ndarray, noise_scale: float, lam: float) -> np.ndarray:
    """SD_k = scale * sqrt(C_k exp(lambda t_k) / dt_k[s])."""
    return noise_scale * np.sqrt(np.clip(tac, 1e-6, None) * np.exp(lam * frames.mid) / (frames.dur * 60.0))


def add_noise(frames, tac, noise_scale, lam, rng):
    return tac + frame_sd(frames, tac, noise_scale, lam) * rng.standard_normal(np.shape(tac))


def frame_weights(frames: Frames, lam: float) -> np.ndarray:
    """Standard fit weights w_k = dt_k exp(-lambda t_k), normalised to mean 1."""
    w = frames.dur * np.exp(-lam * frames.mid)
    return w / w.mean()


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    t: np.ndarray  # fine grid
    frames: Frames
    cp: np.ndarray  # fine plasma input
    ref_fine: np.ndarray
    ref_true: np.ndarray  # framed
    ref: np.ndarray  # framed, noisy
    tac_true: np.ndarray  # (n_regions, n_frames)
    tac: np.ndarray  # noisy
    bp_t: np.ndarray  # (n_regions, n_fine) true BPND(t)
    occ_t: np.ndarray  # (n_regions, n_fine) true occupancy vs baseline
    drug_conc: np.ndarray  # fine
    truth: dict


def _region_curve(sc: Scenario, region: Region, t, cp, drug_conc, n_iter=3):
    tr = sc.tracer
    bp0 = sc.bp0(region)
    free = 0.0
    for _ in range(n_iter if tr.molar_activity else 1):
        bp_t = competitive_bp(region.bmax, tr.kd, tr.fnd, sc.endo_ratio, drug_conc, sc.drug.ki, free)
        ct = tissue_1tc(t, cp, region.K1, region.k2, bp_t)
        if tr.molar_activity:
            # free tracer (nM): fND * C_ND, C_ND = C_T / (1 + BP)
            free = tr.fnd * ct / (1.0 + bp_t) / tr.molar_activity
    occ = 1.0 - bp_t / bp0 if bp0 > 0 else np.zeros_like(t)
    return ct, bp_t, occ


def simulate(sc: Scenario, rng: Optional[np.random.Generator] = None, noisy: bool = True,
             drug_conc_override: Optional[np.ndarray] = None) -> SimResult:
    """Simulate one scan. ``drug_conc_override`` (on the fine grid) replaces the
    drug's PK profile, e.g. a constant for a pre-dosed block scan."""
    rng = np.random.default_rng() if rng is None else rng
    t = fine_grid(sc.frames.t_end, sc.dt)
    cp = sc.input_fn(t)
    drug_conc = sc.drug.conc(t) if drug_conc_override is None else np.broadcast_to(drug_conc_override, t.shape).astype(float)
    lam = sc.tracer.lam

    ref_ct, _, _ = _region_curve(sc, sc.reference, t, cp, drug_conc)
    ref_fine = pet_signal(ref_ct, cp, sc.vB)
    ref_true = frame_average(t, ref_fine, sc.frames)

    tacs, bps, occs = [], [], []
    for r in sc.regions:
        ct, bp_t, occ = _region_curve(sc, r, t, cp, drug_conc)
        tacs.append(frame_average(t, pet_signal(ct, cp, sc.vB), sc.frames))
        bps.append(bp_t)
        occs.append(occ)
    tac_true = np.array(tacs)
    if noisy:
        tac = add_noise(sc.frames, tac_true, sc.noise_scale, lam, rng)
        ref = add_noise(sc.frames, ref_true, sc.noise_scale, lam, rng)
    else:
        tac, ref = tac_true.copy(), ref_true.copy()

    occ_arr = np.array(occs)
    truth = {
        "bp0": np.array([sc.bp0(r) for r in sc.regions]),
        "VT": np.array([r.VND * (1 + sc.bp0(r)) for r in sc.regions]),
        "VND": np.array([r.VND for r in sc.regions]),
        "ec50_app": sc.ec50_app,
        "endo_ratio": sc.endo_ratio,
        "endo_conc": None if sc.k_endo is None else sc.endo_ratio * sc.k_endo,
        "occ_peak": float(occ_arr.max()) if occ_arr.size else 0.0,
        "occ_end": float(occ_arr[:, -1].mean()) if occ_arr.size else 0.0,
        "drug_peak_conc": float(drug_conc.max()),
        "t_peak_conc": float(t[int(np.argmax(drug_conc))]),
    }
    return SimResult(t, sc.frames, cp, ref_fine, ref_true, ref, tac_true, tac,
                     np.array(bps), occ_arr, drug_conc, truth)


def simulate_two_scan(sc: Scenario, block_conc: float, rng=None, noisy=True):
    """Baseline (no drug) and block (constant brain drug conc) scans, as used
    for the Lassen plot. Returns (baseline SimResult, block SimResult, true occ)."""
    rng = np.random.default_rng() if rng is None else rng
    base = simulate(sc, rng, noisy, drug_conc_override=0.0)
    block = simulate(sc, rng, noisy, drug_conc_override=block_conc)
    true_occ = block_conc / (block_conc + sc.ec50_app)
    return base, block, true_occ
