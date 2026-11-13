"""Receptor-level relations linking BPND, tracer Kd, drug Ki and an endogenous
competitor (e.g. a neurotransmitter or endogenous ligand).

Competitive binding at equilibrium (tracer at tracer dose) gives

    BPND0      = fND * Bmax / (Kd_tracer * (1 + E/K_E))          (baseline)
    BPND(t)    = fND * Bmax / (Kd_tracer * (1 + E/K_E + C(t)/Ki)) (with drug)
    occ(t)     = 1 - BPND(t)/BPND0 = C(t) / (C(t) + EC50_app)
    EC50_app   = Ki * (1 + E/K_E)                                (Cheng-Prusoff)

so the endogenous occupancy ratio E/K_E can be recovered in two independent
ways from a single displacement scan:

1. from the apparent EC50 and a known in-vitro Ki of the drug
   (:func:`endogenous_from_ec50`), or
2. from BPND0 and known Bmax, Kd_tracer and fND
   (:func:`endogenous_from_bp`).

Absolute E requires K_E (the endogenous ligand's affinity). Units: E and K_E
the same; C, Ki and EC50 the same; Bmax and Kd the same (e.g. nM).
"""
from __future__ import annotations

import numpy as np


def bp_from_receptors(bmax, kd_tracer, fnd=1.0, endo_ratio=0.0):
    """BPND = fND * Bmax / (Kd (1 + E/K_E))."""
    return fnd * np.asarray(bmax) / (kd_tracer * (1.0 + endo_ratio))


def bavail_from_bp(bp, kd_tracer, fnd=1.0):
    """Available receptor density B_avail = BPND * Kd / fND."""
    return np.asarray(bp) * kd_tracer / fnd


def apparent_ec50(ki_drug, endo_ratio=0.0):
    """Cheng-Prusoff: EC50_app = Ki (1 + E/K_E)."""
    return ki_drug * (1.0 + endo_ratio)


def endogenous_from_ec50(ec50_app, ki_drug, k_endo=None):
    """E/K_E (and E if K_E given) from the fitted apparent EC50 and known Ki."""
    ratio = np.asarray(ec50_app) / ki_drug - 1.0
    return {"endo_ratio": ratio, "endo_conc": None if k_endo is None else ratio * k_endo,
            "endo_occupancy": ratio / (1.0 + ratio)}


def endogenous_from_bp(bp0, bmax, kd_tracer, fnd=1.0, k_endo=None):
    """E/K_E (and E if K_E given) from baseline BPND and known Bmax, Kd, fND."""
    ratio = fnd * np.asarray(bmax) / (kd_tracer * np.asarray(bp0)) - 1.0
    return {"endo_ratio": ratio, "endo_conc": None if k_endo is None else ratio * k_endo,
            "endo_occupancy": ratio / (1.0 + ratio)}


def competitive_bp(bmax, kd_tracer, fnd, endo_ratio, drug_conc, ki_drug, tracer_free=0.0):
    """Time-varying BPND under competition from drug, endogenous ligand and the
    tracer's own mass (``tracer_free`` = free tracer concentration, same units
    as Kd_tracer)."""
    denom = 1.0 + endo_ratio + np.asarray(drug_conc) / ki_drug + np.asarray(tracer_free) / kd_tracer
    return fnd * bmax / (kd_tracer * denom)
