"""How confident is the baseline BPND as a function of how long the TAC runs
before the displacing drug is injected?

For each drug injection time t_inj (min after tracer injection) we simulate
noisy single-scan displacement studies with ``petdisp`` and estimate the
baseline BPND (BP0) with

* drug-driven time-varying SRTM fitted to the whole scan (occupancy from the
  known brain drug profile, EC50 free), and
* standard SRTM fitted only to the frames before t_inj.

For reference, a drug-free scan of the same length fitted with SRTM gives the
best achievable BP0 precision (dashed line).

Outputs: CV%, bias% and the 95% interval of the BP0 estimates, each with
bootstrap 95% confidence bands, for a high- and a low-binding region.

Usage:  python scripts/bp0_vs_pre_injection_time.py [--n-rep 500] [--out figures/...]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from petdisp.fitting import fit_srtm, fit_tv_srtm
from petdisp.occupancy import DrugDrivenOccupancy
from petdisp.simulate import example_scenario, simulate

REGIONS = (0, 3)  # putamen (BP0 4.6), hippocampus (BP0 1.5)
DOSE = 23.0  # ~81% peak occupancy in the example scenario
DT_FIT = 0.05


def run_one(args):
    t_inj, n_rep, seed = args
    sc = example_scenario().with_drug(t_inj=float(t_inj), dose=DOSE)
    lam = sc.tracer.lam
    rng = np.random.default_rng(seed)
    occ = DrugDrivenOccupancy(conc_fn=sc.drug.conc_fn(), ec50_0=sc.ec50_app * 2)
    est = {m: {r: [] for r in REGIONS} for m in ("tv_srtm", "srtm_pre")}
    for _ in range(n_rep):
        sim = simulate(sc, rng)
        for r in REGIONS:
            try:
                est["tv_srtm"][r].append(fit_tv_srtm(sim.frames, sim.tac[r], sim.ref, occ, lam, dt=DT_FIT).derived["BP0"])
            except Exception:
                est["tv_srtm"][r].append(np.nan)
            try:
                est["srtm_pre"][r].append(fit_srtm(sim.frames, sim.tac[r], sim.ref, lam, t_max=t_inj, dt=DT_FIT).derived["BP0"])
            except Exception:
                est["srtm_pre"][r].append(np.nan)
    sim0 = simulate(sc, noisy=False)
    return t_inj, est, float(sim0.occ_t[0].max())


def run_baseline(n_rep, seed):
    """Drug-free scan of full length, standard SRTM: the precision ceiling."""
    sc = example_scenario().with_drug(dose=0.0)
    rng = np.random.default_rng(seed)
    est = {r: [] for r in REGIONS}
    for _ in range(n_rep):
        sim = simulate(sc, rng)
        for r in REGIONS:
            est[r].append(fit_srtm(sim.frames, sim.tac[r], sim.ref, sc.tracer.lam, dt=DT_FIT).derived["BP0"])
    return est


def summarise(x, true, n_boot=2000, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    cv = lambda a: 100 * a.std(ddof=1, axis=-1) / np.abs(a.mean(axis=-1))
    bias = lambda a: 100 * (a.mean(axis=-1) - true) / true
    boot = x[rng.integers(0, x.size, (n_boot, x.size))]
    return {
        "n": int(x.size),
        "cv": float(cv(x)), "cv_lo": float(np.percentile(cv(boot), 2.5)), "cv_hi": float(np.percentile(cv(boot), 97.5)),
        "bias": float(bias(x)), "bias_lo": float(np.percentile(bias(boot), 2.5)), "bias_hi": float(np.percentile(bias(boot), 97.5)),
        "median": float(np.median(x)), "p2.5": float(np.percentile(x, 2.5)), "p97.5": float(np.percentile(x, 97.5)),
        "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-rep", type=int, default=500)
    ap.add_argument("--t-inj", type=float, nargs="+", default=list(range(15, 141, 5)))
    ap.add_argument("--out", default="figures/bp0_vs_pre_injection_time.png")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true", help="redraw from the saved .json")
    a = ap.parse_args()
    if a.plot_only:
        out = Path(a.out)
        plot(json.loads(out.with_suffix(".json").read_text()), example_scenario(), out)
        return

    sc = example_scenario()
    truth = {r: sc.bp0(sc.regions[r]) for r in REGIONS}
    jobs = [(t, a.n_rep, 1000 + i) for i, t in enumerate(a.t_inj)]
    with ProcessPoolExecutor(a.workers) as ex:
        fut_base = ex.submit(run_baseline, a.n_rep, 7)
        results = sorted(ex.map(run_one, jobs), key=lambda z: z[0])
        base = fut_base.result()

    summ = {"t_inj": [z[0] for z in results], "occ_peak": [z[2] for z in results],
            "truth": {sc.regions[r].name: truth[r] for r in REGIONS}, "n_rep": a.n_rep, "dose": DOSE}
    for r in REGIONS:
        name = sc.regions[r].name
        summ[name] = {m: [summarise(z[1][m][r], truth[r]) for z in results] for m in ("tv_srtm", "srtm_pre")}
        summ[name]["drug_free_full_scan"] = summarise(base[r], truth[r])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    plot(summ, sc, out)
    print("wrote", out, out.with_suffix(".json"))


# ---------------------------------------------------------------------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"tv_srtm": "#2a78d6", "srtm_pre": "#eb6834"}
LAB = {"tv_srtm": "tv-SRTM, whole scan (drug-driven occupancy)", "srtm_pre": "SRTM, pre-drug frames only"}


def plot(summ, sc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
    t = np.array(summ["t_inj"])
    names = [sc.regions[r].name for r in REGIONS]
    fig, axes = plt.subplots(len(names), 3, figsize=(15, 8.6), sharex=True)
    t_end = sc.frames.t_end

    for i, name in enumerate(names):
        true = summ["truth"][name]
        ref = summ[name]["drug_free_full_scan"]
        g = lambda m, k: np.array([s[k] for s in summ[name][m]])
        axA, axB, axC = axes[i]
        for m in ("srtm_pre", "tv_srtm"):
            c = COL[m]
            axA.fill_between(t, g(m, "p2.5"), g(m, "p97.5"), color=c, alpha=0.15, lw=0)
            axA.fill_between(t, g(m, "p25"), g(m, "p75"), color=c, alpha=0.30, lw=0)
            axA.plot(t, g(m, "median"), color=c, lw=2, label=LAB[m])
            axB.fill_between(t, g(m, "cv_lo"), g(m, "cv_hi"), color=c, alpha=0.25, lw=0)
            axB.plot(t, g(m, "cv"), color=c, lw=2, marker="o", ms=4, label=LAB[m])
            axC.fill_between(t, g(m, "bias_lo"), g(m, "bias_hi"), color=c, alpha=0.25, lw=0)
            axC.plot(t, g(m, "bias"), color=c, lw=2, marker="o", ms=4, label=LAB[m])
        axA.axhline(true, color=INK, lw=1, ls=":")
        axA.text(t[-1], true, "true ", ha="right", va="bottom", color=INK2, fontsize=9)
        axA.set_ylabel(f"{name}\nestimated baseline BP$_{{ND}}$")
        axA.set_ylim(true * 0.6, true * 1.4)
        axB.axhline(ref["cv"], color=INK2, lw=1.2, ls="--")
        axB.text(t[0], ref["cv"] * 1.08, f"drug-free {t_end:.0f}-min scan, SRTM: {ref['cv']:.1f}%",
                 ha="left", va="bottom", color=INK2, fontsize=9)
        axB.set_ylabel("CV of BP$_{ND}$ (%)")
        axB.set_yscale("log")
        axB.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axC.axhline(0, color=INK, lw=1, ls=":")
        axC.set_ylabel("bias of BP$_{ND}$ (%)")
        axC.set_ylim(-15, 15)
        for ax in (axA, axB, axC):
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel("time from tracer injection to drug injection (min)")
    axes[0, 0].set_title("Median, IQR and 95% range of estimates", loc="left", color=INK, fontsize=11)
    axes[0, 1].set_title("Precision (CV) with bootstrap 95% CI", loc="left", color=INK, fontsize=11)
    axes[0, 2].set_title("Bias with bootstrap 95% CI", loc="left", color=INK, fontsize=11)
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("How long a baseline before the displacing drug?  Baseline BP$_{ND}$ reliability",
                 x=0.01, ha="left", fontsize=13, color=INK, y=0.985)

    tr, ref, d = sc.tracer, sc.reference, sc.drug
    pk = d.pk
    occ = summ["occ_peak"]
    txt = (f"Simulation: {summ['n_rep']} noise realisations per point ({tr.half_life:.1f}-min half-life, $^{{11}}$C UCB-J-like), "
           f"{t_end:.0f}-min scan, Feng plasma input, noise α = {sc.noise_scale}, vB = {sc.vB}.  "
           f"Tracer Kd = {tr.kd} nM, fND = {tr.fnd}; VND = 4 mL/cm³; reference K1 = {ref.K1}, VND = {ref.VND}.\n"
           f"Regions: putamen K1 = 0.30, BP$_{{ND}}$ = {summ['truth']['putamen']:.1f};  hippocampus K1 = 0.25, "
           f"BP$_{{ND}}$ = {summ['truth']['hippocampus']:.1f}.  Endogenous competitor E/K$_E$ = {sc.endo_ratio}.  "
           f"Drug: dose {summ['dose']:g}, Ki = {d.ki}, bolus, V = {pk.V} L, k1/k2 = {pk.k1}/{pk.k2} min⁻¹, "
           f"brain keq = {pk.keq} min⁻¹ → peak occupancy {100*np.mean(occ):.0f}%.  "
           f"Fits weighted by frame duration × decay.")
    fig.text(0.01, 0.005, txt, ha="left", va="bottom", fontsize=8.3, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.955))
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
