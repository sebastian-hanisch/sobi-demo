"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Zusatz-Experimente, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sobi_constants as C
from sobi_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
SUCCESS_PRESETS = ("Zwei Gauß-Quellen, verschiedene Autokorrelation",)
INFO_PRESETS = ("Zwei Rhythmen, geometrische Verzögerungen", "Nur Neuronen")


def _run(setup=None, timeout=300):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def test_default_renders_without_exception():
    at = _run()
    assert any("SOBI in Aktion" in m.value for m in at.markdown)
    assert not at.error and not at.warning and len(at.success) == 1


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert len(at.success) == (name in SUCCESS_PRESETS)
    assert bool(at.warning) == (name not in SUCCESS_PRESETS + INFO_PRESETS)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_neurons_slider"] = C.N_NEURONS_MIN
        at.session_state["n_electrodes_slider"] = C.N_ELECTRODES_MIN
        at.session_state["n_samples_slider"] = C.N_SAMPLES_MIN
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["lag_select"] = "geometric"
    _run(small)

    def no_background(at):
        at.session_state["n_background_slider"] = 0
        at.session_state["n_electrodes_slider"] = 2
        at.session_state["n_neurons_slider"] = 2
    _run(no_background)

    def large(at):
        at.session_state["n_neurons_slider"] = C.N_NEURONS_MAX
        at.session_state["n_electrodes_slider"] = C.N_ELECTRODES_MAX
        at.session_state["n_background_slider"] = C.N_BACKGROUND_MAX
        at.session_state["kind_select"] = "rhythm"
        at.session_state["spacing_slider"] = C.SPACING_MIN
        at.session_state["n_samples_slider"] = C.N_SAMPLES_MAX
        at.session_state["lag_select"] = "long"
        at.session_state["contrast_select"] = "exp"
        at.session_state["init_start_select"] = 5
    _run(large)


def test_kind_and_spacing_are_hidden_when_not_applicable_and_their_values_are_kept():
    at = _run(lambda a: (a.session_state.__setitem__("kind_select", "rhythm"), a.session_state.__setitem__("spacing_slider", 0.2)))
    assert any(s.key == "kind_select" for s in at.selectbox) and any(s.key == "spacing_slider" for s in at.slider)
    at.session_state["n_background_slider"] = 1
    at.run()
    assert not at.exception and any(s.key == "kind_select" for s in at.selectbox) and not any(s.key == "spacing_slider" for s in at.slider)
    at.session_state["n_background_slider"] = 0
    at.run()
    assert not at.exception and not any(s.key == "kind_select" for s in at.selectbox)
    at.session_state["n_background_slider"] = 2
    at.run()
    assert not at.exception and [s for s in at.selectbox if s.key == "kind_select"][0].value == "rhythm" and [s for s in at.slider if s.key == "spacing_slider"][0].value == pytest.approx(0.2)


def test_spacing_sweep_option_disappears_without_a_pair():
    at = _run()
    sel = [s for s in at.selectbox if s.key == "sweep_select"][0]
    assert "Spektren-Abstand" in sel.options
    sel.select("spacing")
    at.run()
    at.session_state["n_background_slider"] = 1
    at.run()
    assert not at.exception and [s for s in at.selectbox if s.key == "sweep_select"][0].value != "spacing"


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["sobi_step"] = 4
    at.run()
    assert not at.exception and at.session_state["sobi_step"] == 4
    at.session_state["noise_slider"] = 0.2
    at.run()
    assert not at.exception and at.session_state["sobi_step"] == 1


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_every_step_renders(step):
    def setup(at):
        at.session_state["sobi_step"] = step
    _run(setup)


def test_window_start_is_clamped_when_the_recording_gets_shorter():
    at = _run(lambda a: a.session_state.__setitem__("window_start", 1500))
    at.session_state["n_samples_slider"] = C.N_SAMPLES_MIN
    at.run()
    assert not at.exception and at.session_state["window_start"] == 0


def test_every_sweep_parameter_and_the_scene_experiment_run_on_demand():
    at = _run()
    for parameter in ("noise", "n_electrodes", "lag_set", "spacing"):
        [s for s in at.selectbox if s.key == "sweep_select"][0].select(parameter)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    [b for b in at.button if b.key == "scenes_start"][0].click()
    at.run()
    assert not at.exception and at.session_state["scenes_on"]


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "sobi_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M)) == 11
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_every_plotly_chart_has_an_explicit_key():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    calls = re.findall(r"plotly_chart\(", source)
    keyed = re.findall(r"plotly_chart\(.*?key=\"[a-z_]+\"\)", source)
    assert len(calls) == len(keyed) >= 15
