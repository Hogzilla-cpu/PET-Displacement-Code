# petdisp: single-scan PET displacement occupancy models

Python implementation of the kinetic models in

> Laurell GL, Plavén-Sigray P, Johansen A, et al. *Kinetic models for estimating occupancy from
> single-scan PET displacement studies.* J Cereb Blood Flow Metab. 2023;43(9):1544–56.
> doi:10.1177/0271678X231168591

extended with a **time-varying SRTM whose occupancy is driven by the displacing drug's brain
concentration profile**, a receptor-level simulator (tracer Kd, Bmax, endogenous competitor, drug Ki),
and tools to choose the **dose and injection time** of the displacing drug that give the most reliable
estimates of:

* baseline BPND of the tracer (before displacement),
* displacement-scan occupancy (peak / time course) and apparent EC50,
* Lassen-plot occupancy and VND (two-scan comparison, and the paper's shared-VND multi-region fit),
* the endogenous competitor level, E/K_E (and E when K_E is known).

## Install

```bash
pip install -e .[notebook,test]
pytest
jupyter lab notebooks/displacement_design_example.ipynb
```

## Models

All regions follow a one-tissue model with specific binding lumped into the tissue compartment, so the
apparent efflux changes as receptors are blocked:

| Model | Equation | Function |
|---|---|---|
| Plasma-input displacement 1TC (paper) | dC/dt = K1·Cp − (K1/VND)/(1 + BP0(1 − occ(t)))·C | `tissue_1tc`, `fit_tv_1tc` |
| Time-varying SRTM | dCt/dt = R1·dCr/dt + k2·Cr − k2/(1 + BP0(1 − occ(t)))·Ct | `tissue_srtm`, `fit_tv_srtm`, `fit_tv_srtm_multi` |
| Standard SRTM / 1TCM | constant BP | `fit_srtm`, `fit_1tcm` |

Occupancy time courses (`petdisp.occupancy`):

* `BetaGrowthOccupancy`: the paper's sigmoid (Yin et al. 2003 beta growth function), occ_max reached at t_e.
* `StepOccupancy`: the paper's single-step approximation with free step time t_s.
* `DrugDrivenOccupancy`: occ(t) = C(t)/(C(t) + EC50) with C(t) the drug's free brain concentration
  (from `DrugPK` or a measured `TabulatedProfile`). Only EC50 is estimated.

`fit_tv_1tc` follows the paper's multi-region strategy: VND and the occupancy parameters are shared across
regions, K1, BP0 and vB are regional. The paper used a nested optimiser; here all parameters are estimated
jointly (same objective). `fit_tv_srtm_multi` is the reference-tissue analogue (shared k2′ and occupancy).

### Receptor level and the endogenous competitor

With competitive binding (`petdisp.endogenous`):

```
BP0     = fND·Bmax / (Kd·(1 + E/K_E))
BP(t)   = fND·Bmax / (Kd·(1 + E/K_E + C(t)/Ki + F_tracer(t)/Kd))
EC50app = Ki·(1 + E/K_E)          (Cheng–Prusoff)
```

so one displacement scan gives E/K_E either from EC50 and the drug's in-vitro Ki
(`endogenous_from_ec50`) or from BP0 and known Bmax, Kd, fND (`endogenous_from_bp`). Setting
`Tracer.molar_activity` adds the tracer's own mass effect through its Kd.

### Simulation and noise

`simulate(scenario)` generates noise-free and noisy framed TACs for target and reference regions from a
Feng plasma input. Frame noise SD = α·√(C·e^{λt}/Δt[s]) (α = `noise_scale`). `simulate_two_scan` gives a
baseline + block pair for Lassen plots (`lassen`).

### Study design

* `fisher_design_map(scenario, t_inj_grid, dose_grid, regions)`: predicted CV of BP0, EC50, E/K_E and SE of
  peak occupancy for every design, from the Fisher information; returns the best design under a combined score.
* `evaluate_design(scenario, t_inj, dose, n_rep, methods)`: Monte Carlo bias, CV and RMSE per estimate and
  method; `design_table` flattens results for pandas. `lassen_benchmark` does the same for the two-scan design.

## Findings with the example scenario

[11C]UCB-J-like tracer, 150-min scan, fast drug (brain equilibration half-life about 2 min), E/K_E = 0.5,
noise α = 0.5. Monte Carlo, 40 replicates, putamen, drug-driven tv-SRTM, RMSE in %:

| inj. time | dose | true peak occ | BP0 | peak occ | EC50 | E/K_E |
|---|---|---|---|---|---|---|
| 40 min | 10 | 65% | 7.1 | 4.7 | 13.8 | 41 |
| 60 min | 3 | 36% | 3.5 | 8.1 | 13.7 | 41 |
| 60 min | 10 | 65% | 3.5 | 2.5 | 7.4 | 22 |
| 60 min | 30 | 85% | 3.5 | 0.9 | 5.9 | 18 |
| 80 min | 23 | 81% | 2.5 | 1.1 | 5.7 | 17 |
| 100 min | 30 | 85% | 2.0 | 0.9 | 6.1 | 18 |

* Aim for a dose giving roughly 75–90% peak occupancy; below about 40% the occupancy and EC50 are poorly
  determined (consistent with the paper's warning about ≤25% occupancy).
* Inject the drug late enough to estimate baseline BP0 from the pre-drug frames (about 80 min of a 150-min
  scan here) while leaving 50–70 min for the displacement.
* The paper's beta-growth model assumes occupancy reaches a plateau. When the drug washes out during the
  scan it underestimates peak occupancy (−12% at 65% occupancy here); using the known brain profile avoids this.
* Fitting all regions jointly roughly halves the CV of EC50.
* The endogenous ratio from a single scan is the least precise output (about 17% RMSE at best for E/K_E = 0.5)
  because it is a small difference of EC50/Ki from 1. It improves as E/K_E grows.
* The Fisher map ignores reference-region noise, so it is optimistic; confirm designs with `evaluate_design`.

These numbers depend on the scenario. Re-run the notebook with your tracer, regions and drug PK.

### ¹¹C versus ¹⁸F (notebook section 7)

Same kinetics and starting count level, only the half-life differs. With a 150-min scan the best drug injection
time is about 80 min for ¹¹C (70–100 min near-optimal) and about 100 min for ¹⁸F (90–110 min), at the same dose
(about 80% peak occupancy). ¹⁸F cuts the error of every estimate by roughly 2–3×. A 240-min ¹⁸F scan makes the
timing forgiving (about 120–170 min) and brings the E/K_E error down to about 4%.
