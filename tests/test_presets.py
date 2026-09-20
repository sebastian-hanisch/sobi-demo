"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import sobi_constants as C
from sobi_evaluation import Settings, analyse, make_dataset, verdict


def _measure(p):
    ds = make_dataset(p["m"], p["n"], p["g"], p["kind"], p["spacing"], p["noise"], p["n_samples"], p["seed"])
    a = analyse(ds, Settings(contrast=p["contrast"], lag_set=p["lag_set"], init_start=p["init_start"]))
    code = verdict(a)[1]
    out = {"verdict": code}
    for name in ("ica", "sobi", "amuse", "pca"):
        out[f"{name}_neuron"] = a.methods[name].neuron_corr
        out[f"{name}_bg"] = a.methods[name].bg_margin
    return out


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_NEURONS_MIN <= p["m"] <= C.N_NEURONS_MAX and C.N_ELECTRODES_MIN <= p["n"] <= C.N_ELECTRODES_MAX and C.N_BACKGROUND_MIN <= p["g"] <= C.N_BACKGROUND_MAX
        assert p["kind"] in C.BACKGROUND_KINDS and C.SPACING_MIN <= p["spacing"] <= C.SPACING_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX
        assert C.N_SAMPLES_MIN <= p["n_samples"] <= C.N_SAMPLES_MAX and p["n_samples"] % 1000 == 0 and p["lag_set"] in C.LAG_SETS and p["contrast"] in C.CONTRASTS and p["init_start"] in C.INIT_STARTS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
