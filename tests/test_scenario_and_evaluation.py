"""Auswertung und die Aussagen der App als Tests: jede Zahl in den Hilfetexten, Tabellen und Presets ist hier über die festen Sweep-Datensätze belegt (Toleranzen bewusst weit).
Vorsicht bei Aussagen über die ICA auf Gauß'schen Quellen: ihr Ergebnis ist je Datensatz Zufall - dort nur Mittel und Spannen prüfen."""

import numpy as np
import pytest

import sobi_algorithm as alg
import sobi_constants as C
import sobi_evaluation as ev


def _analyses(seeds=C.SWEEP_SEEDS, settings=ev.Settings(), **kw):
    return [ev.analyse(ev.make_dataset(seed=s, **kw), settings) for s in seeds]


def _mean(analyses, method, key):
    return float(np.mean([getattr(a.methods[method], key) for a in analyses]))


def test_analysis_has_the_expected_structure_on_the_default_scene():
    a = ev.analyse(ev.make_dataset(), ev.Settings())
    assert set(a.methods) == {"pca", "ica", "amuse", "sobi"} and set(a.models) == {"ica", "sobi", "amuse"}
    assert a.methods["sobi"].aligned.shape == a.ds.S.shape and 0.4 < a.ref_geometric < 0.9 and a.ref_clean > 0.99
    assert all(np.isfinite(m.bg_margin) for m in a.methods.values())
    n = ev.analyse(ev.make_dataset(g=0, noise=0.0), ev.Settings())
    assert np.isnan(n.methods["sobi"].bg_margin) and np.isnan(n.ref_geometric) and np.isnan(n.ref_clean)


def test_geometric_reference_is_skipped_when_the_geometric_set_is_selected():
    a = ev.analyse(ev.make_dataset(), ev.Settings(lag_set="geometric"))
    assert np.isnan(a.ref_geometric)


def test_chance_level_of_the_pair_margin_matches_a_simulation():
    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.pi, 400_000)
    hi, lo = np.maximum(abs(np.cos(th)), abs(np.sin(th))), np.minimum(abs(np.cos(th)), abs(np.sin(th)))
    assert abs(((hi - lo) / (hi + lo)).mean() - ev.CHANCE_PAIR_MARGIN) < 0.005 and abs(ev.CHANCE_PAIR_MARGIN - 0.441) < 0.001
    assert ev.BG_FAILED < ev.BG_SEPARATED and ev.CHANCE_PAIR_MARGIN < ev.BG_FAILED


def test_bg_margin_hand_instances():
    cm = np.array([[0.9, 0.1, 0.0], [0.1, 0.7, 0.7], [0.0, 0.5, 0.5]])
    assert abs(ev.bg_margin(cm, 1) - (((0.7 - 0.7) / 1.4) + ((0.5 - 0.5) / 1.0)) / 2) < 1e-12
    assert ev.bg_margin(np.array([[1.0, 0.0], [0.6, 0.0]]), 1) == pytest.approx(1.0) and np.isnan(ev.bg_margin(cm, 3))


def test_scene_lag_and_sweep_tables_have_the_expected_shape_and_are_deterministic():
    rows = ev.sweep("spacing", values=(0.0, 0.45))
    assert rows == ev.sweep("spacing", values=(0.0, 0.45)) and [r["x"] for r in rows] == [0.0, 0.45] and rows[1]["sobi_bg"] > rows[0]["sobi_bg"] + 0.2
    assert all(set(r) >= {"x", "ica_neuron", "sobi_bg", "amuse_bg", "sobi_bg_std"} for r in rows)
    lag = ev.lag_table()
    assert [r["lag_set"] for r in lag] == list(C.LAG_SETS) and all(r["sweeps"] >= 1 for r in lag) and lag[3]["sweeps"] > lag[1]["sweeps"]
    scenes = ev.scene_table()
    assert [r["scene"] for r in scenes] == [s for s, _ in ev.SCENES] and np.isnan(scenes[0]["ica_bg"]) and all(r["sobi_neuron"] > 0.8 for r in scenes)


# --- Verdict-Codes ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,lag_set,codes", [
    (dict(), "medium", {"sobi_wins", "both_work"}),                             # ICA trennt das Gauß-Paar in einzelnen Datensätzen zufällig
    (dict(kind="rhythm"), "medium", {"lags_mismatch"}),
    (dict(kind="rhythm"), "geometric", {"both_work"}),
    (dict(kind="rhythm", spacing=0.05), "geometric", {"ica_wins_bg"}),
    (dict(g=1), "medium", {"both_work"}),
    (dict(g=0), "medium", {"neutral"}),
    (dict(noise=0.4), "medium", {"noise"}),
    (dict(n=5), "medium", {"underdetermined"}),
    (dict(n=1), "medium", {"underdetermined"}),
])
def test_verdict_codes_hold_on_several_datasets(kwargs, lag_set, codes):
    for seed in (7,) + C.SWEEP_SEEDS:
        a = ev.analyse(ev.make_dataset(seed=seed, **kwargs), ev.Settings(lag_set=lag_set))
        assert ev.verdict(a)[1] in codes, (seed, ev.verdict(a)[1])


def test_default_scene_verdict_is_sobi_wins_for_the_demo_seed():
    assert ev.verdict(ev.analyse(ev.make_dataset(), ev.Settings()))[1] == "sobi_wins"


def test_equal_spectra_verdict_at_the_preset_seed():
    a = ev.analyse(ev.make_dataset(spacing=0.0, seed=11), ev.Settings())
    assert ev.verdict(a)[1] == "both_fail"


def test_sweep_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED and len(C.SWEEP_SEEDS) >= 5


# --- Aussagen der App --------------------------------------------------------------------------------------------------------------------


def test_sobi_separates_the_gauss_pair_in_every_dataset_and_the_ica_only_by_chance():
    """Beleg für Preset-Hilfe und Erfolgsmeldung: SOBI Trennschärfe 0.82 (0.80-0.84), ICA im Mittel 0.41 (0.10-0.73), trennt in 2 von 5 festen Datensätzen zufällig."""
    a = _analyses()
    sobi = [x.methods["sobi"].bg_margin for x in a]
    ica = [x.methods["ica"].bg_margin for x in a]
    assert all(0.75 < v < 0.88 for v in sobi) and 0.78 < np.mean(sobi) < 0.86
    assert 0.3 < np.mean(ica) < 0.5 and min(ica) < 0.2 and max(ica) > 0.6 and sum(v >= ev.BG_SEPARATED for v in ica) in (1, 2, 3)
    assert np.mean(sobi) > np.mean(ica) + 0.3


def test_equal_spectra_put_sobi_at_the_chance_level():
    """Beleg für die Hilfe zum Spektren-Abstand und das Preset 'Gleiche Autokorrelation': SOBI 0.43 (0.30-0.50), ICA 0.27 (0.04-0.72, auf einer CI schon 0.58 statt 0.72 als Maximum beobachtet - ein einzelner der 5 Sweep-Seeds ist bei Abstand 0 numerisch grenzwertig) - beide um das Zufallsniveau 0.44 oder darunter."""
    a = _analyses(spacing=0.0)
    sobi = [x.methods["sobi"].bg_margin for x in a]
    ica = [x.methods["ica"].bg_margin for x in a]
    assert 0.35 < np.mean(sobi) < 0.5 and max(sobi) < 0.55 and min(sobi) > 0.25
    assert 0.15 < np.mean(ica) < 0.4 and min(ica) < 0.15 and max(ica) > 0.5


def test_random_success_counts_for_equal_spectra_over_24_datasets():
    """Beleg für den Text 'bei Abstand 0 trennt die ICA in 7 von 24 Datensätzen zufällig, SOBI in 2'."""
    ica_ok = sobi_ok = 0
    for seed in range(1, 25):
        a = ev.analyse(ev.make_dataset(spacing=0.0, seed=seed), ev.Settings())
        ica_ok += a.methods["ica"].bg_margin >= ev.BG_SEPARATED
        sobi_ok += a.methods["sobi"].bg_margin >= ev.BG_SEPARATED
    assert 5 <= ica_ok <= 9 and sobi_ok <= 4 and ica_ok >= sobi_ok + 2


@pytest.mark.parametrize("spacing,expected", [(0.0, 0.43), (0.05, 0.72), (0.1, 0.83)])
def test_spacing_help_text(spacing, expected):
    """Beleg für die Hilfe zum Spektren-Abstand: 0 -> 0.43, ab 0.05 -> 0.72, ab 0.1 -> 0.83 (SOBI-Trennschärfe)."""
    assert abs(_mean(_analyses(spacing=spacing), "sobi", "bg_margin") - expected) < 0.07


@pytest.mark.parametrize("noise,ica,sobi", [(0.05, 0.92, 0.90), (0.1, 0.83, 0.73), (0.2, 0.72, 0.48), (0.4, 0.59, 0.33)])
def test_noise_help_text(noise, ica, sobi):
    """Beleg für die Hilfe zum Rauschen: Neuronen-Korrelation ICA / SOBI 0.92/0.90, 0.83/0.73, 0.72/0.48, 0.59/0.33."""
    a = _analyses(noise=noise)
    assert abs(_mean(a, "ica", "neuron_corr") - ica) < 0.04 and abs(_mean(a, "sobi", "neuron_corr") - sobi) < 0.04


def test_missing_electrodes_hit_sobi_harder_than_the_ica():
    """Beleg für die Hilfe zu den Elektroden: bei 5 Elektroden (6 Quellen) ICA 0.82, SOBI 0.55."""
    a = _analyses(n=5)
    assert abs(_mean(a, "ica", "neuron_corr") - 0.82) < 0.04 and abs(_mean(a, "sobi", "neuron_corr") - 0.55) < 0.05
    for n in (3, 4):
        b = _analyses(n=n)
        assert _mean(b, "ica", "neuron_corr") > _mean(b, "sobi", "neuron_corr") + 0.1


@pytest.mark.parametrize("lag_set,expected", [("short", 0.63), ("medium", 0.82), ("long", 0.80), ("geometric", 0.68)])
def test_lag_set_help_text_for_the_gauss_pair(lag_set, expected):
    """Beleg für die Hilfe zu den Verzögerungen: Trennschärfe bei zwei Gauß-AR-Quellen kurz 0.63, mittel 0.82, lang 0.80, geometrisch 0.68."""
    assert abs(_mean(_analyses(settings=ev.Settings(lag_set=lag_set)), "sobi", "bg_margin") - expected) < 0.06


def test_rhythms_need_long_lags_and_the_ica_separates_them_anyway():
    """Beleg für Preset-Hilfen und Hilfe zu den Verzögerungen: Rhythmen mittel 0.38, geometrisch 0.94, ICA 0.76; Neuronen bei geometrisch 0.87 gegen 0.90."""
    med = _analyses(kind="rhythm")
    geo = _analyses(kind="rhythm", settings=ev.Settings(lag_set="geometric"))
    assert abs(_mean(med, "sobi", "bg_margin") - 0.38) < 0.05 and abs(_mean(geo, "sobi", "bg_margin") - 0.94) < 0.04 and abs(_mean(med, "ica", "bg_margin") - 0.76) < 0.05
    assert abs(_mean(med, "sobi", "neuron_corr") - 0.90) < 0.02 and abs(_mean(geo, "sobi", "neuron_corr") - 0.87) < 0.02


def test_close_rhythms_are_separated_by_the_ica_but_not_by_sobi():
    """Beleg für die Grenzen-Tabelle: Rhythmen mit fast gleicher Frequenz (10 und 10.7 Hz): ICA trennt (0.76), SOBI nicht (unter 0.55, auch geometrisch)."""
    a = _analyses(kind="rhythm", spacing=0.05, settings=ev.Settings(lag_set="geometric"))
    assert _mean(a, "ica", "bg_margin") > 0.7 and _mean(a, "sobi", "bg_margin") < 0.55


def test_sobi_is_close_to_the_ica_on_spikes_and_better_than_amuse():
    """Beleg für die Preset-Hilfe 'Nur Neuronen': SOBI 0.978 gegen ICA 0.981; AMUSE mit einem τ 0.93."""
    a = _analyses(g=0)
    ica, sobi, amuse = (_mean(a, name, "neuron_corr") for name in ("ica", "sobi", "amuse"))
    assert abs(ica - 0.981) < 0.01 and abs(sobi - 0.978) < 0.01 and abs(amuse - 0.93) < 0.03 and amuse < sobi - 0.02
    assert all(x.methods["sobi"].f1 > 0.98 for x in a)


def test_amuse_with_a_single_lag_is_worse_than_sobi_on_the_default_scene():
    a = _analyses()
    assert _mean(a, "sobi", "neuron_corr") > _mean(a, "amuse", "neuron_corr") + 0.15 and _mean(a, "sobi", "bg_margin") > _mean(a, "amuse", "bg_margin") + 0.3


def test_pca_alone_does_not_separate_the_background():
    a = _analyses()
    assert _mean(a, "pca", "bg_margin") < 0.3 and _mean(a, "pca", "neuron_corr") < 0.7
