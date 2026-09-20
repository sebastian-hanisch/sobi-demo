import numpy as np
import pytest

import sobi_algorithm as alg
import sobi_constants as C
import sobi_evaluation as ev
import sobi_ica as ica
import sobi_scenario as sc


def _ar_sources(phis, T=20000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for phi in phis:
        e = rng.standard_normal(T)
        s = np.empty(T)
        s[0] = e[0]
        for t in range(1, T):
            s[t] = phi * s[t - 1] + e[t]
        out.append((s - s.mean()) / s.std())
    return np.array(out)


def _mixture(phis, n=None, seed=0):
    S = _ar_sources(phis, seed=seed)
    n = n or len(phis)
    A = np.random.default_rng(seed + 1).standard_normal((n, len(phis)))
    return S, A, A @ S


# --- Verzögerte Kovarianzen -----------------------------------------------------------------------------------------------------------


def test_lagged_covariances_are_symmetric_and_match_a_hand_computation():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((3, 500))
    R = alg.lagged_covariances(Z, (1, 5))
    assert R.shape == (2, 3, 3) and np.allclose(R, R.transpose(0, 2, 1))
    manual = Z[:, :-1] @ Z[:, 1:].T / 499
    assert np.allclose(R[0], 0.5 * (manual + manual.T))


def test_lag_set_and_constant_sets():
    assert alg.lag_set(3, 4) == (4, 8, 12)
    assert C.LAG_SETS["geometric"][-1] == 256 and len(C.LAG_SETS["geometric"]) == 9 and C.LAG_SETS["medium"] == tuple(range(2, 21, 2))
    assert set(C.LAG_SETS) == set(C.LAG_SET_LABELS) and C.DEFAULT_LAG_SET in C.LAG_SETS


def test_off_diagonality_hand_instances():
    assert alg.off_diagonality(np.array([np.diag([1.0, 2.0, 3.0])])) == 0.0
    assert abs(alg.off_diagonality(np.ones((1, 2, 2))) - 0.5) < 1e-12


# --- Gemeinsame Diagonalisierung ----------------------------------------------------------------------------------------------------------


def _diagonalizable_stack(n=4, K=6, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    D = rng.standard_normal((K, n))
    return Q, np.array([Q @ np.diag(d) @ Q.T for d in D])


def test_joint_diagonalization_recovers_an_exactly_diagonalizable_stack():
    Q, M = _diagonalizable_stack()
    V, history, angles, converged, rotations = alg.joint_diagonalize(M)
    assert converged and np.allclose(V.T @ V, np.eye(4), atol=1e-12)
    assert history[-1] < 1e-20 and alg.off_diagonality(np.einsum("ji,kjl,lm->kim", V, M, V)) < 1e-20
    assert np.allclose(np.abs(V.T @ Q).max(axis=1), 1.0, atol=1e-8)                      # V gleicht Q bis auf Reihenfolge und Vorzeichen
    assert len(rotations) == len(history) == len(angles) + 1 and np.allclose(rotations[0], np.eye(4)) and np.allclose(rotations[-1], V)


def test_joint_diagonalization_never_increases_the_off_diagonal_energy():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((5, 5, 5))
    M = base + base.transpose(0, 2, 1)
    V, history, angles, converged, _ = alg.joint_diagonalize(M)
    assert all(b <= a + 1e-12 for a, b in zip(history, history[1:])) and history[-1] < history[0]
    assert np.allclose(V.T @ V, np.eye(5), atol=1e-12)


def test_single_matrix_joint_diagonalization_equals_the_eigen_decomposition():
    """Kreuzprüfung gegen numpy.linalg.eigh: für eine einzige Matrix ist die gemeinsame Diagonalisierung die Eigenzerlegung."""
    rng = np.random.default_rng(3)
    B = rng.standard_normal((4, 4))
    R = B + B.T
    V, *_ = alg.joint_diagonalize(R[None])
    values, vectors = np.linalg.eigh(R)
    assert np.allclose(np.abs(V.T @ vectors).max(axis=1), 1.0, atol=1e-8)
    assert np.allclose(np.sort(np.diag(V.T @ R @ V)), values, atol=1e-8)


def test_joint_diagonalization_edge_cases():
    V, history, angles, converged, rotations = alg.joint_diagonalize(np.ones((3, 1, 1)))
    assert converged and V.shape == (1, 1) and angles == []
    M = np.array([np.diag([1.0, 2.0])] * 3)
    V, history, angles, converged, _ = alg.joint_diagonalize(M)
    assert converged and np.allclose(np.abs(V), np.eye(2))


# --- AMUSE und SOBI -----------------------------------------------------------------------------------------------------------------------


def test_sobi_separates_a_noise_free_coloured_gaussian_mixture_almost_exactly():
    S, A, X = _mixture((0.95, 0.5, -0.5))
    m = alg.fit_sobi(X, 3, alg.lag_set(10, 2))
    assert m.converged and np.allclose(m.W @ m.W.T, np.eye(3), atol=1e-10) and np.allclose(m.transform(X), m.sources, atol=1e-8)
    assert ev.matched(S, m.sources)[1].min() > 0.99
    Z = m.sources
    assert np.allclose(Z @ Z.T / Z.shape[1], np.eye(3), atol=1e-8)


def test_amuse_separates_when_the_autocorrelations_are_far_apart_and_sobi_does_at_least_as_well():
    S, A, X = _mixture((0.95, 0.3), n=4)
    am = alg.fit_amuse(X, 2, 3)
    sb = alg.fit_sobi(X, 2, alg.lag_set(8, 2))
    assert am.kind == "amuse" and am.converged and np.allclose(am.W @ am.W.T, np.eye(2), atol=1e-10)
    assert ev.matched(S, am.sources)[1].min() > 0.99 and ev.matched(S, sb.sources)[1].min() > 0.99


def test_equal_spectra_leave_the_pair_unresolved_but_the_third_source_and_the_subspace_are_found():
    """Zwei Quellen mit demselben AR-Koeffizienten: die Drehung im Paar ist nicht durch die R_tau bestimmt, der Unterraum schon."""
    S, A, X = _mixture((0.9, 0.9, 0.3))
    m = alg.fit_sobi(X, 3, alg.lag_set(10, 2))
    cm = ev.correlation_matrix(S, m.sources)
    third = int(np.argmax(cm[2]))
    assert cm[2, third] > 0.99                                                                    # die Quelle mit anderem Spektrum ist getrennt
    rest = [j for j in range(3) if j != third]
    assert (cm[0, rest] ** 2).sum() > 0.98 and (cm[1, rest] ** 2).sum() > 0.98                       # das Paar liegt vollständig im übrigen 2-D-Unterraum ...
    assert cm[0, rest].max() < 0.999 and cm[1, rest].max() < 0.999                                # ... ist darin aber nicht sauber getrennt


def test_sobi_history_and_rotations_are_consistent_with_the_final_sources():
    ds = ev.make_dataset()
    a = ev.analyse(ds, ev.Settings())
    m = a.models["sobi"]
    assert len(m.history) == len(m.angles) + 1 == len(m.rotations) and m.history[-1] < m.history[0] and np.allclose(m.rotations[-1].T, m.W)
    curve = ev.sobi_curve(a)
    assert len(curve) == len(m.rotations) and abs(curve[-1] - a.methods["sobi"].neuron_corr) < 1e-12 and curve[0] < curve[-1]
    before, after = ev.diagonalized(m)
    assert alg.off_diagonality(after) < alg.off_diagonality(before) and np.allclose(before, m.covariances)


def test_autocorrelation_of_ar1_and_sinusoid_match_theory():
    S = _ar_sources((0.95, 0.5), T=40000)
    acf = alg.autocorrelation(S, 20)
    assert np.allclose(acf[:, 0], 1.0) and abs(acf[0, 10] - 0.95 ** 10) < 0.08 and abs(acf[1, 2] - 0.25) < 0.03
    t = np.arange(20000) / C.SAMPLE_RATE
    sine = np.sin(2 * np.pi * 10.0 * t)[None]
    assert abs(alg.autocorrelation((sine - sine.mean()) / sine.std(), 100)[0, 50] - np.cos(2 * np.pi * 10.0 * 50 / C.SAMPLE_RATE)) < 0.01


def test_power_spectrum_peaks_at_the_rhythm_frequency():
    ds = ev.make_dataset(kind="rhythm", spacing=0.45)
    freqs, spec = alg.power_spectrum(ds.S)
    assert spec.shape[0] == ds.S.shape[0] and abs(freqs[np.argmax(spec[4][1:]) + 1] - 10.0) < 30 and abs(freqs[np.argmax(spec[5][1:]) + 1] - 23.0) < 30


# --- Vergleichsverfahren aus ica-demo (eingefroren) ----------------------------------------------------------------------------------


def test_whitening_copy_has_identity_covariance():
    ds = ev.make_dataset(g=1)
    wh = ica.whiten(ds.X, 5)
    assert np.allclose(wh.Z @ wh.Z.T / ds.X.shape[1], np.eye(5), atol=1e-9)


def test_fastica_copy_reproduces_the_frozen_reference():
    """Eingefrorener Wert aus ica-demo (4 Neuronen, 6 Elektroden, Rauschen 0.05, Seed 7): Neuronen-Korrelation 0.981."""
    ds = ev.make_dataset(g=0)
    m = ica.fit_ica(ds.X, 4, "logcosh", "symmetric", 1)
    assert m.converged and abs(ev.matched(ds.S, m.sources)[1].mean() - 0.981) < 2e-3


# --- Szenario -----------------------------------------------------------------------------------------------------------------------


def test_scenario_is_the_one_of_ica_demo_for_the_old_hintergrund_settings():
    """Eingefroren aus ica-demo (Gauß-AR 0.95/0.5, Seed 7, Rauschen 0.05): Neuronen und Elektrodensignale identisch."""
    ds = sc.make_dataset(4, 6, 2, "gauss", 0.45, 0.05, 20000, 7)
    assert abs(np.abs(ds.S[0][:3000]).sum() - 436.61272090571026) < 1e-6 and abs(np.abs(ds.S[4][:2000]).sum() - 1644.2975358383712) < 1e-6
    assert abs(np.abs(ds.S[5][:2000]).sum() - 1620.632875442106) < 1e-6 and abs(np.abs(ds.X[:, :2000]).sum() - 12101.142712296238) < 1e-6
    assert ds.phis == pytest.approx((0.95, 0.5))


def test_background_parameters_and_spacing():
    assert sc.background_parameters("gauss", 2, 0.45) == pytest.approx((0.95, 0.5)) and sc.background_parameters("gauss", 2, 0.0) == (0.95, 0.95)
    assert sc.background_parameters("rhythm", 2, 0.45) == pytest.approx((10.0, 23.0)) and sc.background_parameters("rhythm", 2, 0.0) == (10.0, 10.0) and sc.background_parameters("gauss", 0, 0.3) == ()


def test_sources_have_unit_variance_and_the_expected_autocorrelation():
    ds = ev.make_dataset(g=2, kind="gauss", spacing=0.3, n_samples=40000)
    assert np.allclose(ds.S.std(axis=1), 1.0) and np.allclose(ds.S.mean(axis=1), 0.0, atol=1e-12)
    acf = alg.autocorrelation(ds.S[4:], 5)
    assert abs(acf[0, 3] - 0.95 ** 3) < 0.05 and abs(acf[1, 3] - 0.65 ** 3) < 0.05


def test_clean_mixture_is_exact_and_noise_is_independent_of_the_background():
    ds = ev.make_dataset(noise=0.0, g=2)
    assert np.allclose(ds.X, ds.A @ ds.S) and ds.noise_sigma == 0.0
    assert abs(ev.make_dataset(noise=0.5, g=0).noise_sigma - ev.make_dataset(noise=0.5, g=2).noise_sigma) < 1e-12


def test_streams_do_not_depend_on_the_other_settings():
    a = ev.make_dataset(m=2, n=3, g=1, seed=11)
    b = ev.make_dataset(m=5, n=8, g=2, spacing=0.2, seed=11)
    assert np.array_equal(a.spike_times[0], b.spike_times[0]) and np.allclose(a.S[0], b.S[0]) and np.allclose(a.S[1], b.S[1])
    assert np.allclose(ev.make_dataset(g=2, spacing=0.45).S[4], ev.make_dataset(g=2, spacing=0.1).S[4])


def test_constants_are_consistent():
    assert len(C.NEURON_SIGMAS) == len(C.NEURON_RATES) == len(C.NEURON_AMPLITUDES) == len(C.NEURON_POSITIONS) == C.N_NEURONS_MAX
    assert C.N_BACKGROUND_MAX == 2 and max(max(v) for v in C.LAG_SETS.values()) < C.N_SAMPLES_MIN // 4
