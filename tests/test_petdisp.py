import numpy as np
import pytest

import petdisp as pd
from petdisp.kinetics import solve_linear, fine_grid, frame_average, tissue_1tc, tissue_srtm


# ---------------------------------------------------------------- kinetics
def test_solve_linear_matches_analytic_convolution():
    t = fine_grid(100, 0.005)
    a_rate, k = 0.2, 0.05
    a = np.exp(-a_rate * t)
    x = solve_linear(a, np.full_like(t, k), t[1] - t[0])
    exact = (np.exp(-k * t) - np.exp(-a_rate * t)) / (a_rate - k)
    assert np.max(np.abs(x - exact)) < 1e-5


def test_solve_linear_chunked_equals_unchunked():
    t = fine_grid(150, 0.02)
    a = pd.feng_input(t)
    b = 0.05 + 0.02 * np.sin(t / 10) ** 2
    x1 = solve_linear(a, b, 0.02)
    x2 = solve_linear(a, b, 0.02, max_exp=1.0)
    assert np.allclose(x1, x2, rtol=1e-9, atol=1e-9)


def test_default_frames_cover_scan():
    fr = pd.default_frames(150)
    assert fr.t_end == pytest.approx(150)
    assert np.all(fr.dur > 0)
    assert np.allclose(fr.start[1:], fr.end[:-1])


def test_frames_to_grid_preserves_frame_means():
    fr = pd.default_frames(150)
    t = fine_grid(150, 0.02)
    y = 50 * np.exp(-0.02 * t) * (1 - np.exp(-2 * t))
    framed = frame_average(t, y, fr)
    g = pd.frames_to_grid(fr, framed, t)
    assert np.allclose(frame_average(t, g, fr), framed, rtol=2e-3)


def test_srtm_equals_1tc_when_vnd_equal():
    t = fine_grid(150, 0.01)
    cp = pd.feng_input(t)
    occ = pd.beta_growth(t, 60, 80) * 0.6
    bp_t = 3.0 * (1 - occ)
    K1r, VND = 0.3, 4.0
    K1 = 0.36
    cr = tissue_1tc(t, cp, K1r, K1r / VND, 0.0)
    ct = tissue_1tc(t, cp, K1, K1 / VND, bp_t)
    ct_srtm = tissue_srtm(t, cr, K1 / K1r, K1 / VND, bp_t)
    assert np.max(np.abs(ct - ct_srtm)) / ct.max() < 1e-3


# ---------------------------------------------------------------- occupancy
def test_beta_growth_shape():
    t = np.linspace(0, 150, 1501)
    f = pd.beta_growth(t, 60, 90, 0.5)
    assert f[t < 60].max() == 0
    assert np.allclose(f[t >= 90], 1)
    assert np.all(np.diff(f) >= -1e-12)


def test_drug_driven_occupancy_hill():
    occ = pd.DrugDrivenOccupancy(conc_fn=lambda t: np.full_like(t, 2.0))
    t = np.linspace(0, 10, 11)
    assert np.allclose(occ(t, [np.log10(2.0)]), 0.5)


def test_pk_linear_in_dose_and_zero_before_injection():
    pk = pd.DrugPK()
    t = fine_grid(150, 0.02)
    c1, c2 = pk.brain(t, 1.0, 60), pk.brain(t, 3.0, 60)
    assert np.allclose(c2, 3 * c1)
    assert c1[t < 60].max() == 0
    tab = pd.TabulatedProfile([0, 10, 60], [0, 1.0, 0.5], ref_dose=2.0)
    c = tab.brain(t, 4.0, 30)
    assert c[np.argmin(abs(t - 40))] == pytest.approx(2.0)


# ---------------------------------------------------------------- endogenous
def test_endogenous_roundtrip():
    bmax, kd, fnd, ratio, ki = 50.0, 3.0, 0.3, 0.7, 0.05
    bp0 = pd.bp_from_receptors(bmax, kd, fnd, ratio)
    assert pd.endogenous_from_bp(bp0, bmax, kd, fnd)["endo_ratio"] == pytest.approx(ratio)
    ec50 = pd.apparent_ec50(ki, ratio)
    e = pd.endogenous_from_ec50(ec50, ki, k_endo=10.0)
    assert e["endo_ratio"] == pytest.approx(ratio)
    assert e["endo_conc"] == pytest.approx(7.0)
    # competitive model consistency: occupancy vs baseline follows EC50_app
    c = 0.2
    bp = pd.competitive_bp(bmax, kd, fnd, ratio, c, ki)
    assert 1 - bp / bp0 == pytest.approx(c / (c + ec50))


def test_mass_effect_lowers_binding():
    sc = pd.example_scenario()
    a = pd.simulate(sc, noisy=False)
    sc.tracer.molar_activity = 0.5  # very low molar activity
    b = pd.simulate(sc, noisy=False)
    assert b.bp_t[0].min() < a.bp_t[0].min()


# ---------------------------------------------------------------- fitting
@pytest.fixture(scope="module")
def noise_free():
    sc = pd.example_scenario(vB=0.0)
    return sc, pd.simulate(sc, noisy=False)


def test_tv_srtm_drug_recovers_truth(noise_free):
    sc, sim = noise_free
    occ = pd.DrugDrivenOccupancy(conc_fn=sc.drug.conc_fn(), ec50_0=1.0)
    f = pd.fit_tv_srtm(sim.frames, sim.tac[0], sim.ref, occ, sc.tracer.lam)
    assert f.derived["BP0"] == pytest.approx(sim.truth["bp0"][0], rel=5e-3)
    assert f.derived["ec50"] == pytest.approx(sc.ec50_app, rel=1e-2)
    assert f.derived["occ_peak"] == pytest.approx(sim.occ_t[0].max(), abs=5e-3)


def test_tv_srtm_multi_recovers_truth(noise_free):
    sc, sim = noise_free
    occ = pd.DrugDrivenOccupancy(conc_fn=sc.drug.conc_fn(), ec50_0=1.0)
    f = pd.fit_tv_srtm_multi(sim.frames, sim.tac, sim.ref, occ, sc.tracer.lam)
    assert np.allclose(f.derived["BP0"], sim.truth["bp0"], rtol=1e-2)
    assert f.derived["ec50"] == pytest.approx(sc.ec50_app, rel=2e-2)


def test_srtm_pre_injection(noise_free):
    sc, sim = noise_free
    f = pd.fit_srtm(sim.frames, sim.tac[1], sim.ref, sc.tracer.lam, t_max=sc.drug.t_inj)
    assert f.derived["BP0"] == pytest.approx(sim.truth["bp0"][1], rel=1e-2)


def _beta_data(occ_max=0.6, te=75.0):
    sc = pd.example_scenario()
    t = fine_grid(sc.frames.t_end, sc.dt)
    cp = sc.input_fn(t)
    occ = occ_max * pd.beta_growth(t, 60, te)
    tacs = []
    for r in sc.regions:
        ct = tissue_1tc(t, cp, r.K1, r.k2, sc.bp0(r) * (1 - occ))
        tacs.append(frame_average(t, pd.pet_signal(ct, cp, sc.vB), sc.frames))
    return sc, t, cp, np.array(tacs)


def test_paper_1tc_multiregion_beta_recovers_truth():
    sc, t, cp, tacs = _beta_data()
    occm = pd.BetaGrowthOccupancy(t_b=60, te0=70, te_max=150)
    f = pd.fit_tv_1tc(sc.frames, tacs, t, cp, occm, sc.tracer.lam)
    assert f.derived["occ_max"] == pytest.approx(0.6, abs=5e-3)
    assert f.derived["VND"] == pytest.approx(4.0, rel=1e-2)
    assert f.derived["t_e"] == pytest.approx(75.0, abs=1.0)


def test_paper_single_step_close_for_fast_drug():
    sc, t, cp, tacs = _beta_data(te=62.0)
    occm = pd.StepOccupancy(t_s0=61.0, t_s_bounds=(55, 80))
    f = pd.fit_tv_1tc(sc.frames, tacs, t, cp, occm, sc.tracer.lam)
    assert f.derived["occ_max"] == pytest.approx(0.6, abs=0.03)


def test_lassen_noise_free():
    sc = pd.example_scenario(vB=0.0)
    base, block, true_occ = pd.simulate_two_scan(sc, block_conc=0.1, noisy=False)
    lam = sc.tracer.lam
    vb = [pd.fit_1tcm(base.frames, y, base.t, base.cp, lam).derived["VT"] for y in base.tac]
    vk = [pd.fit_1tcm(block.frames, y, block.t, block.cp, lam).derived["VT"] for y in block.tac]
    L = pd.lassen(vb, vk)
    assert L["occ"] == pytest.approx(true_occ, abs=5e-3)
    assert L["VND"] == pytest.approx(4.0, rel=2e-2)


# ---------------------------------------------------------------- design
def test_fisher_map_shapes_and_trends():
    sc = pd.example_scenario()
    m = pd.fisher_design_map(sc, [40, 80], [1, 30])
    assert m["cv_ec50"].shape == (2, 2)
    assert np.all(np.isfinite(m["score"]))
    # later injection -> more baseline frames -> better BP0
    assert m["cv_bp0"][1, 0] < m["cv_bp0"][0, 0]
    # very low occupancy -> poorer EC50
    assert m["cv_ec50"][1, 0] > m["cv_ec50"][1, 1]


def test_evaluate_design_runs():
    sc = pd.example_scenario()
    r = pd.evaluate_design(sc, 60, 10, n_rep=3, methods=("drug_srtm", "srtm_pre"))
    assert r["drug_srtm"]["ec50"]["n"] == 3
    rows = pd.design_table([r])
    assert any(row["quantity"] == "endo_ratio_from_ec50" for row in rows)
