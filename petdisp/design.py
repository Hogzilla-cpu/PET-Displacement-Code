"""Study design: which displacing-drug dose and administration time give the
most reliable estimates?

Two complementary tools:

* :func:`fisher_design_map` - fast, noise-model-based prediction of the
  precision (CV%) of BP0, EC50 (hence endogenous competitor level) and peak
  occupancy for every (t_inj, dose) on a grid, from the Fisher information of
  the drug-driven time-varying SRTM. Cheap enough for dense grids.
* :func:`evaluate_design` - Monte Carlo: simulate noisy scans, fit the models,
  and report bias, CV and RMSE for each estimate. Use it to confirm the Fisher
  map at a few candidate designs (it also captures reference-region noise,
  bounds and model misspecification, which the Fisher map ignores).

:func:`lassen_benchmark` gives the same statistics for the classic two-scan
(baseline + block) Lassen plot, for comparison.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from .endogenous import endogenous_from_bp, endogenous_from_ec50
from .fitting import fit_1tcm, fit_srtm, fit_tv_1tc, fit_tv_srtm, fit_tv_srtm_multi, lassen
from .kinetics import fine_grid, frame_average, tissue_srtm
from .occupancy import BetaGrowthOccupancy, DrugDrivenOccupancy, hill_occupancy
from .simulate import Scenario, frame_sd, simulate, simulate_two_scan


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _stats(est: np.ndarray, true: float) -> dict:
    est = np.asarray(est, float)
    est = est[np.isfinite(est)]
    if est.size == 0:
        return {"true": true, "mean": np.nan, "median": np.nan, "bias_pct": np.nan,
                "cv_pct": np.nan, "rmse_pct": np.nan, "n": 0}
    denom = abs(true) if true != 0 else 1.0
    return {
        "true": float(true),
        "mean": float(est.mean()),
        "median": float(np.median(est)),
        "bias_pct": float(100 * (est.mean() - true) / denom),
        "cv_pct": float(100 * est.std(ddof=1) / abs(est.mean())) if est.size > 1 and est.mean() != 0 else np.nan,
        "rmse_pct": float(100 * np.sqrt(np.mean((est - true) ** 2)) / denom),
        "n": int(est.size),
    }


# --------------------------------------------------------------------------
# Fisher-information design map
# --------------------------------------------------------------------------
def _fisher_one(sc: Scenario, regions: Sequence[int], dt: float, rel_step: float = 1e-4):
    """Covariance of (k2', log10 EC50, [R1_i, BP0_i]) for the drug-driven
    multi-region tv-SRTM at the true parameters (single region if one index)."""
    sim = simulate(sc, noisy=False)
    t = fine_grid(sc.frames.t_end, dt)
    cr = np.interp(t, sim.t, sim.ref_fine)
    conc = np.interp(t, sim.t, sim.drug_conc)
    k2r = sc.reference.k2
    R1 = [sc.regions[i].K1 / sc.reference.K1 for i in regions]
    bp0 = [sc.bp0(sc.regions[i]) for i in regions]
    theta = np.concatenate([[k2r, np.log10(sc.ec50_app)], np.ravel(np.column_stack([R1, bp0]))])

    def model(th):
        occ = hill_occupancy(conc, 10 ** th[1])
        out = []
        for j in range(len(regions)):
            r1, b = th[2 + 2 * j], th[3 + 2 * j]
            out.append(frame_average(t, tissue_srtm(t, cr, r1, r1 * th[0], b * (1 - occ)), sc.frames))
        return np.concatenate(out)

    y0 = model(theta)
    sd = frame_sd(sc.frames, sim.tac_true[list(regions)], sc.noise_scale, sc.tracer.lam).ravel()
    J = np.empty((y0.size, theta.size))
    for k in range(theta.size):
        h = rel_step * max(abs(theta[k]), 1e-3)
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        J[:, k] = (model(tp) - model(tm)) / (2 * h)
    Jw = J / sd[:, None]
    F = Jw.T @ Jw
    cov = np.linalg.pinv(F)
    return theta, cov, sim


def fisher_design_map(sc: Scenario, t_inj_grid: Iterable[float], dose_grid: Iterable[float],
                      regions: Sequence[int] = (0,), dt: float = 0.05) -> Dict[str, np.ndarray]:
    """Predicted precision for every (t_inj, dose) combination.

    Returns arrays of shape (len(t_inj_grid), len(dose_grid)) with keys
    ``cv_bp0`` (first listed region), ``cv_ec50``, ``se_occ_peak`` (absolute,
    occupancy units), ``cv_endo_ratio`` (E/K_E from EC50 with known Ki),
    ``occ_peak`` (true), and a combined ``score`` (see below).

    ``score`` = sqrt(cv_bp0^2 + cv_ec50^2 + (100*se_occ_peak)^2): a single
    figure of merit; the best design minimises it. Inspect the individual maps
    when one parameter matters most.
    """
    t_inj_grid = np.asarray(list(t_inj_grid), float)
    dose_grid = np.asarray(list(dose_grid), float)
    shape = (t_inj_grid.size, dose_grid.size)
    out = {k: np.full(shape, np.nan) for k in
           ("cv_bp0", "cv_ec50", "se_occ_peak", "cv_endo_ratio", "occ_peak", "score")}
    for i, ti in enumerate(t_inj_grid):
        for j, d in enumerate(dose_grid):
            s = sc.with_drug(t_inj=float(ti), dose=float(d))
            theta, cov, sim = _fisher_one(s, regions, dt)
            ec50 = 10 ** theta[1]
            se_log = np.sqrt(max(cov[1, 1], 0))
            se_ec50 = ec50 * np.log(10) * se_log
            cpk = sim.drug_conc.max()
            occ_pk = cpk / (cpk + ec50)
            # d occ / d log10 EC50 = -ln10 * occ (1 - occ)
            se_occ = np.log(10) * occ_pk * (1 - occ_pk) * se_log
            cv_bp = 100 * np.sqrt(max(cov[3, 3], 0)) / theta[3]
            cv_ec = 100 * se_ec50 / ec50
            ratio = s.endo_ratio
            ki = s.drug.ki
            cv_endo = 100 * (se_ec50 / ki) / ratio if ratio > 0 else np.nan
            out["cv_bp0"][i, j] = cv_bp
            out["cv_ec50"][i, j] = cv_ec
            out["se_occ_peak"][i, j] = se_occ
            out["cv_endo_ratio"][i, j] = cv_endo
            out["occ_peak"][i, j] = occ_pk
            out["score"][i, j] = np.sqrt(cv_bp**2 + cv_ec**2 + (100 * se_occ) ** 2)
    out["t_inj_grid"] = t_inj_grid
    out["dose_grid"] = dose_grid
    k = np.unravel_index(np.nanargmin(out["score"]), shape)
    out["best"] = {"t_inj": float(t_inj_grid[k[0]]), "dose": float(dose_grid[k[1]]),
                   "occ_peak": float(out["occ_peak"][k]), "score": float(out["score"][k])}
    return out


# --------------------------------------------------------------------------
# Monte Carlo evaluation
# --------------------------------------------------------------------------
def evaluate_design(sc: Scenario, t_inj: float, dose: float, n_rep: int = 50,
                    region: int = 0, methods: Sequence[str] = ("drug_srtm", "beta_srtm", "srtm_pre"),
                    seed: int = 0, dt_fit: float = 0.05) -> Dict[str, dict]:
    """Monte Carlo bias / CV / RMSE of the estimates for one design.

    methods:
      ``drug_srtm``   time-varying SRTM with occupancy from the known brain
                      concentration profile (fits R1, k2, BP0, EC50)
      ``drug_srtm_multi`` the same, all regions jointly with shared k2', EC50
      ``beta_srtm``   time-varying SRTM with the paper's sigmoidal occupancy
                      (fits R1, k2, BP0, occ_max, t_e)
      ``beta_1tc``    paper's plasma-input multi-region model (shared VND,
                      occ_max, t_e)
      ``srtm_pre``    standard SRTM on pre-drug frames (baseline BP0 only)

    Returns {method: {quantity: stats}}. Endogenous ratio E/K_E is estimated
    from EC50 (needs Ki) and, where Bmax is known, from BP0.
    """
    s = sc.with_drug(t_inj=float(t_inj), dose=float(dose))
    rng = np.random.default_rng(seed)
    lam = s.tracer.lam
    truth_sim = simulate(s, noisy=False)
    tr = truth_sim.truth
    reg = s.regions[region]
    res: Dict[str, Dict[str, list]] = {m: {} for m in methods}

    def push(m, k, v):
        res[m].setdefault(k, []).append(v)

    drug_occ = DrugDrivenOccupancy(conc_fn=s.drug.conc_fn(), ec50_0=s.ec50_app * 2)
    beta_occ = BetaGrowthOccupancy(t_b=s.drug.t_inj, te0=s.drug.t_inj + 10, te_max=s.frames.t_end)

    for _ in range(n_rep):
        sim = simulate(s, rng)
        if "drug_srtm" in methods:
            f = fit_tv_srtm(sim.frames, sim.tac[region], sim.ref, drug_occ, lam, dt=dt_fit)
            push("drug_srtm", "BP0", f.derived["BP0"])
            push("drug_srtm", "ec50", f.derived["ec50"])
            push("drug_srtm", "occ_peak", f.derived["occ_peak"])
            push("drug_srtm", "endo_ratio_from_ec50", float(endogenous_from_ec50(f.derived["ec50"], s.drug.ki)["endo_ratio"]))
            push("drug_srtm", "endo_ratio_from_bp", float(endogenous_from_bp(f.derived["BP0"], reg.bmax, s.tracer.kd, s.tracer.fnd)["endo_ratio"]))
        if "drug_srtm_multi" in methods:
            f = fit_tv_srtm_multi(sim.frames, sim.tac, sim.ref, drug_occ, lam, dt=dt_fit)
            push("drug_srtm_multi", "BP0", f.derived["BP0"][region])
            push("drug_srtm_multi", "ec50", f.derived["ec50"])
            push("drug_srtm_multi", "occ_peak", f.derived["occ_peak"])
            push("drug_srtm_multi", "endo_ratio_from_ec50", float(endogenous_from_ec50(f.derived["ec50"], s.drug.ki)["endo_ratio"]))
        if "beta_srtm" in methods:
            f = fit_tv_srtm(sim.frames, sim.tac[region], sim.ref, beta_occ, lam, dt=dt_fit)
            push("beta_srtm", "BP0", f.derived["BP0"])
            push("beta_srtm", "occ_peak", f.derived["occ_max"])
        if "beta_1tc" in methods:
            f = fit_tv_1tc(sim.frames, sim.tac, sim.t, sim.cp, beta_occ, lam)
            push("beta_1tc", "BP0", f.derived["BP0"][region])
            push("beta_1tc", "occ_peak", f.derived["occ_max"])
            push("beta_1tc", "VND", f.derived["VND"])
        if "srtm_pre" in methods:
            f = fit_srtm(sim.frames, sim.tac[region], sim.ref, lam, t_max=s.drug.t_inj, dt=dt_fit)
            push("srtm_pre", "BP0", f.derived["BP0"])

    true = {"BP0": tr["bp0"][region], "ec50": tr["ec50_app"], "occ_peak": float(truth_sim.occ_t[region].max()),
            "endo_ratio_from_ec50": s.endo_ratio, "endo_ratio_from_bp": s.endo_ratio,
            "VND": float(reg.VND)}
    out = {m: {k: _stats(v, true[k]) for k, v in q.items()} for m, q in res.items()}
    out["_design"] = {"t_inj": float(t_inj), "dose": float(dose), "true_occ_peak": true["occ_peak"],
                      "true_peak_conc": tr["drug_peak_conc"], "n_rep": n_rep}
    return out


def lassen_benchmark(sc: Scenario, block_conc: float, n_rep: int = 50, seed: int = 0,
                     scan_min: Optional[float] = None) -> Dict[str, dict]:
    """Two-scan design: baseline and block scans fitted with 1TCM, occupancy
    and VND from the Lassen plot. ``block_conc`` = steady brain drug conc."""
    rng = np.random.default_rng(seed)
    lam = sc.tracer.lam
    occ, vnd = [], []
    for _ in range(n_rep):
        base, block, true_occ = simulate_two_scan(sc, block_conc, rng)
        vt_b = [fit_1tcm(base.frames, y, base.t, base.cp, lam, t_max=scan_min).derived["VT"] for y in base.tac]
        vt_k = [fit_1tcm(block.frames, y, block.t, block.cp, lam, t_max=scan_min).derived["VT"] for y in block.tac]
        L = lassen(vt_b, vt_k)
        occ.append(L["occ"])
        vnd.append(L["VND"])
    return {"lassen": {"occ": _stats(occ, true_occ), "VND": _stats(vnd, sc.regions[0].VND)}}


def design_table(results: Sequence[dict]) -> list:
    """Flatten a list of :func:`evaluate_design` outputs into rows (dicts),
    convenient for ``pandas.DataFrame(rows)``."""
    rows = []
    for r in results:
        d = r["_design"]
        for m, q in r.items():
            if m.startswith("_"):
                continue
            for k, st in q.items():
                rows.append({"t_inj": d["t_inj"], "dose": d["dose"], "true_occ_peak": d["true_occ_peak"],
                             "method": m, "quantity": k, **st})
    return rows
