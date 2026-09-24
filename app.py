"""SOBI - Quellentrennung über die zeitliche Struktur, an Mehrelektroden-Signalen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - SOBI - und lässt stattdessen das Beispiel wachsen.
Zweites Stück der Quellentrennung-Linie der "Konzepte"-Reihe: ein Nachfolger der ICA-Demo, der die Schwäche "zwei Gauß'sche Quellen sind nicht trennbar" behebt - über die Autokorrelation statt
über die Verteilungsform. Was das bringt und wo es endet, wird hier gemessen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import sobi_constants as C
from sobi_algorithm import autocorrelation, power_spectrum
from sobi_evaluation import (
    CHANCE_PAIR_MARGIN, Settings, SWEEP_LABELS, analyse, correlation_matrix, diagonalized, lag_table, lags_of, make_dataset, scene_table, sobi_curve, sweep, verdict,
)
from sobi_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from sobi_visualization import (
    build_autocorrelation,
    build_corr_heatmap,
    build_covariances,
    build_diagonalization,
    build_lag_table,
    build_layout,
    build_method_bars,
    build_scenes,
    build_spectrum,
    build_sweep,
    build_traces,
    source_color,
    source_labels,
)

st.set_page_config(page_title="SOBI – Sebastian Hanisch", layout="wide")

STEP_LABELS = {1: "1 · Quellen", 2: "2 · Mischung", 3: "3 · Kovarianzen", 4: "4 · Diagonalisieren", 5: "5 · Ergebnis"}
WINDOW_WIDTH_MS = 60
MAX_ACF_LAG = 400
SWEEP_OPTIONS = {"spacing": "Spektren-Abstand", "noise": "Rauschen", "n_electrodes": "Anzahl Elektroden", "lag_set": "Verzögerungs-Menge"}


@st.cache_data(show_spinner=False)
def _dataset(m, n, g, kind, spacing, noise, n_samples, seed):
    return make_dataset(m, n, g, kind, spacing, noise, n_samples, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _spectra(data_params):
    ds = make_dataset(*data_params)
    freqs, spec = power_spectrum(ds.S)
    return autocorrelation(ds.S, MAX_ACF_LAG), freqs, spec


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    m, n, g, kind, spacing, noise, n_samples = base
    return sweep(parameter, settings=settings, m=m, n=n, g=g, kind=kind, spacing=spacing, noise=noise, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _scenes(base, settings):
    m, n, _g, _kind, _spacing, noise, n_samples = base
    return scene_table(settings, m=m, n=n, noise=noise, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _lag_table(base):
    m, n, g, kind, spacing, noise, n_samples = base
    return lag_table(m=m, n=n, g=g, kind=kind, spacing=spacing, noise=noise, n_samples=n_samples)


st.title("🌊 SOBI – Quellen trennen über die zeitliche Struktur")
st.markdown(
    """
Die **ICA** aus dem ersten Stück der Quellentrennung-Linie trennt Quellen über die **Form ihrer Verteilung** - und scheitert, wenn zwei Quellen Gauß'sch sind: Eine Drehung ändert eine Gauß-Verteilung nicht.
**SOBI** (Second-Order Blind Identification) nutzt etwas anderes: die **zeitliche Struktur**. Eine Quelle ist nicht nur eine Verteilung, sondern eine Zeitreihe - und zwei Zeitreihen mit **verschiedener Autokorrelation**
(die eine ändert sich langsam, die andere schnell) lassen sich trennen, selbst wenn beide perfekt Gauß'sch sind. Dafür genügen **Kovarianzen** bei mehreren Verzögerungen τ; sie werden nach dem Weißen
**gemeinsam diagonalisiert**. Die Demo stellt beide Verfahren auf demselben Elektrodenarray nebeneinander und misst, **wer wann trennt** - und wo SOBI scheitert: bei gleichem Spektrum, bei falsch gewählten Verzögerungen
und bei Rauschen. Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - zweites Stück der Quellentrennung-Linie der \"Konzepte\"-Reihe, Nachfolger der ICA-Demo - **ein** Verfahren "
    "an einem wachsenden Beispiel. Das Array ist dasselbe wie dort (ohne Laufzeitverzögerung); der Hintergrund ist ausgebaut."
)

with st.expander("So funktioniert SOBI", expanded=True):
    st.markdown(
        """
**Das Modell** ist dasselbe wie bei der ICA: $x(t) = A\\,s(t) + \\text{Rauschen}$ mit unbekannter Mischung $A$ und Quellen $s$. Neu ist, was von den Quellen verlangt wird: nicht Unabhängigkeit *und* Nicht-Gaußianität,
sondern **Unkorreliertheit bei allen Verzögerungen** - $E[s_i(t)\\,s_j(t+\\tau)] = 0$ für $i \\neq j$ - und **verschiedene Autokorrelation** $\\rho_i(\\tau) = E[s_i(t)\\,s_i(t+\\tau)]$.

1. **Weißen** (wie bei der ICA): danach ist die Mischung nur noch eine **Drehung** $Z = Q\\,s$.
2. **Verzögerte Kovarianzen.** Für jede Verzögerung $\\tau$ wird $R_\\tau = E[z(t)\\,z(t+\\tau)^\\top]$ berechnet (symmetrisiert). Für die wahren Quellen wäre $R_\\tau$ **diagonal** - mit den Autokorrelationen $\\rho_i(\\tau)$ auf der Diagonalen.
   Im weißen Raum gilt $R_\\tau = Q\\,\\text{diag}(\\rho_i(\\tau))\\,Q^\\top$: die **Eigenvektoren** von $R_\\tau$ sind die Quellenrichtungen - **wenn** die $\\rho_i(\\tau)$ verschieden sind.
3. **Ein τ genügt selten (AMUSE), viele τ genügen fast immer (SOBI).** Mit *einem* $\\tau$ hängt alles an einem Abstand zwischen zwei Eigenwerten; bei zufällig fast gleichen Werten ist die Richtung unbestimmt.
   SOBI sucht **eine** Drehung, die **alle** $R_\\tau$ gleichzeitig möglichst diagonal macht (Jacobi-Rotationen).
4. **Das τ muss zur Zeitskala der Quellen passen.** Spitzen haben nur über ihre Form (~ 30 Abtastwerte) Zeitstruktur, AR-Rauschen mit Koeffizient 0.95 hat eine Korrelationslänge um 20, ein 10-Hz-Rhythmus um 1000 (Abschnitt 🔧).

Was SOBI **nicht** verlangt: Nicht-Gaußianität. Was es **zusätzlich** braucht: verschiedene Spektren. Zwei Quellen mit **gleicher** Autokorrelation sind für SOBI ununterscheidbar - so wie zwei Gauß'sche Quellen für die ICA.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_neurons = st.slider(
        "Neuronen", *bounds("n_neurons_slider"), key="n_neurons_slider",
        help="Spitzenartige Quellen wie in der ICA-Demo (fester Ort, feste Spitzenform, Feuerrate 20-35 Hz). Die Quellenzahl wird den Verfahren als bekannt vorgegeben.",
    )
    n_electrodes = st.slider(
        "Elektroden", *bounds("n_electrodes_slider"), key="n_electrodes_slider",
        help="Aufnahmestellen auf einer Zeile. Weniger Elektroden als Quellen: die Mischung ist nicht umkehrbar. Bei 6 Quellen und 5 Elektroden fällt die Neuronen-Korrelation der ICA auf 0.82, die von SOBI auf 0.55 - SOBI leidet mehr.",
    )
    n_background = st.slider(
        "Hintergrundquellen", *bounds("n_background_slider"), key="n_background_slider",
        help="Zusätzliche flächige Quellen. Mit zwei Quellen desselben Typs lässt sich messen, ob ein Verfahren das Paar trennt. Sie zählen zur Quellenzahl.",
    )
    if n_background > 0:
        seed_widget("kind_select")
        kind = st.selectbox(
            "Art des Hintergrunds", C.BACKGROUND_KINDS, key="kind_select", format_func=lambda k: C.BACKGROUND_LABELS[k],
            help="Gauß-Rauschen: farbig (AR(1)), Gauß'sch - die ICA kann zwei davon nicht trennen, SOBI schon. Rhythmus: sinusförmig, unter-Gauß'sch - die ICA trennt sie, SOBI nur mit langen Verzögerungen.",
        )
        st.session_state["_kind_kept"] = kind
    else:
        kind = st.session_state.get("_kind_kept", C.DEFAULT_BACKGROUND_KIND)
    if n_background >= 2:
        seed_widget("spacing_slider")
        spacing = st.slider(
            "Spektren-Abstand", *bounds("spacing_slider"), key="spacing_slider", step=0.01,
            help="Wie verschieden die beiden Hintergrundquellen sind: AR-Koeffizient 0.95 gegen 0.95 minus Abstand (bei Rhythmen: 10 Hz gegen 10 Hz plus 29 Hz je 1.0). 0 = gleiches Spektrum: SOBI landet beim Zufall (Trennschärfe 0.43). "
                 "Ab 0.05 trennt SOBI (0.72), ab 0.1 sicher (0.83).",
        )
        st.session_state["_spacing_kept"] = spacing
    else:
        spacing = float(st.session_state.get("_spacing_kept", C.DEFAULT_SPACING))
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Sensorrauschen relativ zum Neuronen-Signal. SOBI ist rauschempfindlicher als die ICA: Neuronen-Korrelation 0.90 / 0.73 / 0.48 / 0.33 bei Rauschen 0.05 / 0.1 / 0.2 / 0.4, ICA 0.92 / 0.83 / 0.72 / 0.59.",
    )
    n_samples = st.slider(
        "Länge der Aufnahme", *bounds("n_samples_slider"), key="n_samples_slider", step=1000,
        help="Abtastwerte bei 10 kHz. Lange Verzögerungen (bis 256) brauchen genug Daten; unter 5000 Abtastwerte ist die Demo nicht sinnvoll.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**SOBI**")
    lag_name = st.selectbox(
        "Verzögerungen τ", list(C.LAG_SETS), key="lag_select", format_func=lambda k: C.LAG_SET_LABELS[k],
        help="Die Verzögerungen (Abtastwerte à 0.1 ms), deren Kovarianzen gemeinsam diagonalisiert werden. Bei zwei Gauß-AR-Quellen (Abstand 0.45) liegt die Trennschärfe bei kurz 0.63, mittel 0.82, lang 0.80, geometrisch 0.68; "
             "bei Sinusrhythmen braucht SOBI lange Verzögerungen (mittel 0.38, geometrisch 0.94). AMUSE nutzt das kleinste τ der Menge.",
    )
    st.markdown("**ICA (zum Vergleich)**")
    contrast = st.selectbox(
        "Kontrastfunktion", C.CONTRASTS, key="contrast_select", format_func=lambda c: C.CONTRAST_LABELS[c],
        help="Wie die ICA Nicht-Gaußianität misst (siehe ICA-Demo). Für dieses Stück nebensächlich.",
    )
    init_start = st.selectbox(
        "Start der ICA", C.INIT_STARTS, key="init_start_select", format_func=lambda i: f"Start {i}",
        help="Zufällige Anfangsrichtung der ICA. SOBI hat keinen Zufallsstart.",
    )

    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für Spikezeiten, Hintergrund und Rauschen.")

sync_query_params({
    "n_neurons_slider": int(n_neurons), "n_electrodes_slider": int(n_electrodes), "n_background_slider": int(n_background), "kind_select": kind, "spacing_slider": float(spacing), "noise_slider": noise,
    "n_samples_slider": int(n_samples), "lag_select": lag_name, "contrast_select": contrast, "init_start_select": int(init_start), "seed_input": int(seed),
})

data_params = (int(n_neurons), int(n_electrodes), int(n_background), kind, float(round(spacing, 2)), float(noise), int(n_samples), int(seed))
settings = Settings(contrast=contrast, lag_set=lag_name, init_start=int(init_start))
with st.spinner("Trenne die Quellen..."):
    ds = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
sobi_model = analysis.models["sobi"]
methods = analysis.methods
k = ds.S.shape[0]
m_n = ds.n_neurons
g = k - m_n
labels = source_labels(ds)
colors = [source_color(i) for i in range(k)]
level, code, vd = verdict(analysis)
lags = lags_of(lag_name)
data_key = data_params + (settings,)
ica_m, sobi_m, amuse_m, pca_m = methods["ica"], methods["sobi"], methods["amuse"], methods["pca"]

# --- SOBI in Aktion -------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 SOBI in Aktion")
if "sobi_step" not in st.session_state or st.session_state.get("sobi_step_owner") != data_key:
    st.session_state["sobi_step"] = 1
    st.session_state["sobi_step_owner"] = data_key
duration_ms = ds.S.shape[1] * 1000.0 / C.SAMPLE_RATE
max_start = int(duration_ms - WINDOW_WIDTH_MS)
if st.session_state.get("window_start", 0) > max_start:
    st.session_state["window_start"] = 0
step_col, play_col, win_col = st.columns([4, 2, 3])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="sobi_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
with win_col:
    window = st.slider(f"Zeitfenster ({WINDOW_WIDTH_MS} ms) ab [ms]", 0, max_start, key="window_start", step=10, help="Welchen Ausschnitt der Aufnahme die Signalspuren zeigen.")
peaks = ds.spike_times
view_slot = st.empty()
before_cov, after_cov = diagonalized(sobi_model)


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            acf, freqs, spec = _spectra(data_params)
            c1, c2 = st.columns(2)
            c1.markdown("**Die wahren Quellen** (in der Praxis unbekannt)")
            c1.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [peaks[i] if i < m_n else [] for i in range(k)]), width="stretch", key="step_sources")
            c2.markdown("**Autokorrelation je Quelle** (gepunktet: die gewählten τ)")
            c2.plotly_chart(build_autocorrelation(labels, acf, colors, lags), width="stretch", key="step_acf")
            c3, c4 = st.columns(2)
            c3.markdown("**Leistungsspektrum je Quelle**")
            c3.plotly_chart(build_spectrum(labels, freqs, spec, colors), width="stretch", key="step_spectrum")
            c4.markdown("**Ort von Neuronen und Elektroden**")
            c4.plotly_chart(build_layout(ds), width="stretch", key="step_layout")
        elif current_step == 2:
            st.markdown("**Was die Elektroden messen**: jede Spur mischt alle Quellen")
            st.plotly_chart(build_traces([f"E{j + 1}" for j in range(ds.n_electrodes)], list(ds.X), window, WINDOW_WIDTH_MS, normalise=False), width="stretch", key="step_electrodes")
        elif current_step == 3:
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"**Verzögerte Kovarianzen R_τ der weißen Daten** ({len(lags)} Verzögerungen: {lags[0]} ... {lags[-1]})")
            c1.plotly_chart(build_covariances(before_cov, None, lags), width="stretch", key="step_cov_before")
            c2.markdown("**Die weißen Komponenten** (PCA, ungedreht)")
            c2.plotly_chart(build_traces([f"PC{j + 1}" for j in range(pca_m.estimates.shape[0])], list(pca_m.estimates), window, WINDOW_WIDTH_MS), width="stretch", key="step_whitened")
        elif current_step == 4:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Vorher und nachher**: dieselben R_τ, gedreht mit der gefundenen Matrix")
            c1.plotly_chart(build_covariances(before_cov, after_cov, lags), width="stretch", key="step_cov_after")
            c2.markdown("**Jacobi-Verfahren**")
            c2.plotly_chart(build_diagonalization(sobi_model.history, sobi_curve(analysis)), width="stretch", key="step_diagonalization")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Wahre Quellen**")
            c1.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors), width="stretch", key="step_result_true")
            c2.markdown("**SOBI-Schätzungen** (Reihenfolge, Vorzeichen und Skala der wahren Quellen angepasst)")
            c2.plotly_chart(build_traces(labels, list(sobi_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="step_result_sobi")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption("Spitzen haben nur über ihre **Form** Zeitstruktur (kurze Autokorrelation, Neuronen mit verschieden breiten Spitzen unterscheiden sich leicht); das AR-Rauschen fällt langsam ab (Koeffizient 0.95: Korrelationslänge um 20 Abtastwerte), "
               "ein Rhythmus schwingt. Wo die Kurven **verschieden** sind, hat SOBI etwas zum Trennen; fallen zwei zusammen (Abstand 0), nicht. Rechts unten: dasselbe Array wie in der ICA-Demo.")
elif step == 2:
    st.caption(f"{ds.n_electrodes} Elektrode(n) messen jeweils eine gewichtete Summe aller {k} Quellen plus Rauschen ({noise:.2f} der Stärke des Neuronen-Signals). Die Mischung ist momentan - Laufzeitunterschiede kommen in dieser Demo nicht vor (sie brechen die "
               "Annahme; siehe ICA-Demo).")
elif step == 3:
    st.caption("Nach dem Weißen ist die Kovarianz bei Verzögerung 0 die Einheitsmatrix - aber bei τ > 0 stehen **außerhalb der Diagonalen** Werte: die weißen Komponenten sind Mischungen. Wären sie die wahren Quellen, wären alle Bilder diagonal. "
               "Die Farbskala ist für alle Verzögerungen gleich.")
elif step == 4:
    st.caption(f"Jacobi-Rotationen drehen jeweils zwei Komponenten so, dass die Summe der Nebendiagonal-Quadrate über **alle** R_τ sinkt. {'Konvergiert nach ' + str(len(sobi_model.angles)) + ' Sweeps.' if sobi_model.converged else 'Nicht konvergiert.'} "
               "Rechts: die Nicht-Diagonalität fällt auf fast null, die Korrelation der Neuronen mit den Schätzungen steigt (Sweep 0 = PCA).")
else:
    st.caption("Reihenfolge, Vorzeichen und Skala der SOBI-Ausgabe sind wie bei der ICA beliebig und hier nachträglich an die wahren Quellen angepasst.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wer trennt was: SOBI, ICA, AMUSE und PCA auf denselben Daten")
st.caption(
    "Neuronen-Korrelation: mittlere |Korrelation| der Neuronen mit der zugeordneten Schätzung. **Trennschärfe der Hintergrundquellen**: je Hintergrundquelle (c₁ − c₂)/(c₁ + c₂) aus der größten und zweitgrößten |Korrelation| über alle Schätzungen, "
    f"gemittelt - nahe 1 heißt, eine Schätzung gehört klar zu der Quelle; **{CHANCE_PAIR_MARGIN:.2f}** ist das Zufallsniveau eines gefundenen, aber beliebig gedrehten Paares."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Neuronen-Korrelation (SOBI)", f"{sobi_m.neuron_corr:.2f}", delta=f"{sobi_m.neuron_corr - ica_m.neuron_corr:+.2f} ggü. ICA", delta_color="normal", help="Mittlere |Korrelation| der Neuronen mit ihrer Schätzung. 1 = perfekt.")
if g >= 1:
    m2.metric("Trennschärfe Hintergrund (SOBI)", f"{sobi_m.bg_margin:.2f}", delta=f"{sobi_m.bg_margin - ica_m.bg_margin:+.2f} ggü. ICA", delta_color="normal",
              help=f"Klarheit, mit der die Hintergrundquellen je einer Schätzung zugeordnet sind. Zufallsniveau eines gefundenen Paares: {CHANCE_PAIR_MARGIN:.2f}.")
else:
    m2.metric("Trennschärfe Hintergrund (SOBI)", "-", help="Ohne Hintergrundquellen gibt es nichts zu messen.")
m3.metric("Spitzen-F1 (SOBI)", f"{sobi_m.f1:.2f}", delta=f"{sobi_m.f1 - ica_m.f1:+.2f} ggü. ICA", delta_color="normal", help="Wie viele der wahren Spitzen auf der Schätzung gefunden werden (Toleranz ±4 Abtastwerte).")
m4.metric("Jacobi-Sweeps", f"{len(sobi_model.angles)}", help="Anzahl der Durchläufe über alle Komponentenpaare, bis die gemeinsame Diagonalisierung stillsteht.")

if code == "underdetermined":
    st.warning(f"⚠️ Nur {ds.n_electrodes} Elektrode(n) für {k} Quellen: die Mischung ist nicht umkehrbar, beide Verfahren finden nur {min(ds.n_electrodes, k)} Komponenten. Neuronen-Korrelation ICA {vd['ica_neuron']:.2f}, SOBI {vd['sobi_neuron']:.2f} "
               "- SOBI leidet hier stärker (Sparse Component Analysis, nächstes Stück der Linie, ist für diesen Fall gebaut).")
elif code == "not_converged":
    st.warning("⚠️ Die gemeinsame Diagonalisierung ist nicht konvergiert - das Ergebnis ist unzuverlässig.")
elif code == "noise":
    st.warning(f"⚠️ Das Rauschen kostet viel: Neuronen-Korrelation ICA {vd['ica_neuron']:.2f} statt {vd['ref_clean']:.2f} ohne Rauschen, SOBI {vd['sobi_neuron']:.2f}. "
               "SOBI ist rauschempfindlicher: Kovarianzen bei Verzögerungen verlieren das Signal schneller im Rauschen als die Verteilungsform der Spitzen.")
elif code == "sobi_wins":
    st.success(f"✅ SOBI trennt die Hintergrundquellen (Trennschärfe {vd['sobi_bg']:.2f}), die ICA nicht (Trennschärfe {vd['ica_bg']:.2f}; Zufallsniveau {CHANCE_PAIR_MARGIN:.2f}). Beide Quellen sind Gauß'sch - die ICA hat nichts, woran sie "
               "sich orientieren kann, SOBI nutzt die verschiedene Autokorrelation. Die ICA trennt in einzelnen Datensätzen zufällig (im Test in 2 von 5 festen Datensätzen), SOBI in jedem.")
elif code == "both_work":
    st.info(f"ℹ️ Beide Verfahren trennen die Hintergrundquellen (Trennschärfe SOBI {vd['sobi_bg']:.2f}, ICA {vd['ica_bg']:.2f}).")
elif code == "lags_mismatch":
    st.warning(f"⚠️ SOBI trennt die Hintergrundquellen mit diesen Verzögerungen nicht (Trennschärfe {vd['sobi_bg']:.2f}), mit den geometrischen aber schon ({vd['geo_bg']:.2f}): die gewählten τ sind zu kurz oder zu lang für die Zeitskala der Quellen - "
               f"deren Autokorrelationen unterscheiden sich erst bei anderen Verzögerungen. Die ICA trennt {'sie ' if vd['ica_bg'] >= 0.65 else 'sie ebenfalls nicht '}({vd['ica_bg']:.2f}).")
elif code == "ica_wins_bg":
    st.warning(f"⚠️ SOBI trennt die Hintergrundquellen nicht (Trennschärfe {vd['sobi_bg']:.2f}, Zufallsniveau {CHANCE_PAIR_MARGIN:.2f}) - die Spektren sind zu ähnlich, auch mit anderen Verzögerungen. Die ICA trennt sie ({vd['ica_bg']:.2f}), "
               "weil sie unter-Gauß'sche Sinusquellen sind: die Verteilungsform unterscheidet sie von einer Mischung, auch wenn die Frequenzen fast gleich sind.")
elif code == "both_fail":
    st.warning(f"⚠️ Keines der Verfahren trennt die Hintergrundquellen (Trennschärfe SOBI {vd['sobi_bg']:.2f}, ICA {vd['ica_bg']:.2f}, Zufallsniveau {CHANCE_PAIR_MARGIN:.2f}): die Quellen sind Gauß'sch **und** haben (fast) dasselbe Spektrum. "
               "Ein Wert über dem Zufallsniveau ist Glück mit der Stichprobe - bei Abstand 0 trennt die ICA im Test in 7 von 24 Datensätzen zufällig, SOBI in 2.")
else:
    st.info(f"ℹ️ Neuronen-Korrelation: SOBI {vd['sobi_neuron']:.2f}, ICA {vd['ica_neuron']:.2f}, AMUSE {vd['amuse_neuron']:.2f}, PCA {vd['pca_neuron']:.2f}. Die Spitzen haben durch ihre verschieden breiten Wellenformen genug Zeitstruktur für SOBI; "
            "die ICA nutzt die Verteilung. Mit einer einzigen Verzögerung (AMUSE) ist das Ergebnis schlechter.")

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Wahre Quellen**")
    st.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [peaks[i] if i < m_n else [] for i in range(k)]), width="stretch", key="res_true")
with t2:
    st.markdown("**PCA (nur weißen)**")
    st.plotly_chart(build_traces(labels, list(pca_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="res_pca")
t3, t4 = st.columns(2)
with t3:
    st.markdown("**ICA**")
    st.plotly_chart(build_traces(labels, list(ica_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="res_ica")
with t4:
    st.markdown("**SOBI**")
    st.plotly_chart(build_traces(labels, list(sobi_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="res_sobi")

h1, h2, h3 = st.columns(3)
with h1:
    st.plotly_chart(build_corr_heatmap(correlation_matrix(ds.S, ica_m.estimates), labels, [f"IC{j + 1}" for j in range(ica_m.estimates.shape[0])], "ICA: |Korrelation|"), width="stretch", key="heat_ica")
with h2:
    st.plotly_chart(build_corr_heatmap(correlation_matrix(ds.S, sobi_m.estimates), labels, [f"SC{j + 1}" for j in range(sobi_m.estimates.shape[0])], "SOBI: |Korrelation|"), width="stretch", key="heat_sobi")
with h3:
    st.plotly_chart(build_method_bars(methods), width="stretch", key="method_bars")
st.caption("Links (Zeilen = wahre Quellen, Spalten = Komponenten): eine saubere Trennung hat in jeder Zeile und Spalte genau einen hellen Eintrag. Rechts: Korrelation der Neuronen, Spitzen-F1 und Trennschärfe der Hintergrundquellen "
           "(gestrichelt: Zufallsniveau).")

st.markdown("---")

# --- Verzögerungen ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔧 Welche Verzögerungen τ? Die Zeitskala entscheidet")
with st.spinner("Vergleiche die Verzögerungs-Mengen über 5 feste Datensätze..."):
    lag_rows = _lag_table(data_params[:7])
st.plotly_chart(build_lag_table(lag_rows), width="stretch", key="lag_table_chart")
st.table({
    "Menge": [C.LAG_SET_LABELS[r["lag_set"]] for r in lag_rows],
    "SOBI: Neuronen-Korrelation": [f"{r['neuron']:.3f}" for r in lag_rows],
    "SOBI: Trennschärfe Hintergrund": ["-" if np.isnan(r["bg"]) else f"{r['bg']:.2f}" for r in lag_rows],
    "AMUSE (kleinstes τ): Neuronen": [f"{r['amuse_neuron']:.3f}" for r in lag_rows],
    "AMUSE: Trennschärfe": ["-" if np.isnan(r["amuse_bg"]) else f"{r['amuse_bg']:.2f}" for r in lag_rows],
    "Jacobi-Sweeps": [f"{r['sweeps']:.0f}" for r in lag_rows],
})
st.caption("Mittel über 5 feste Datensätze mit den aktuellen Einstellungen. Es gibt **keine** Menge, die für alle Quellen passt: Spitzen brauchen kurze τ (~ 30 Abtastwerte Zeitstruktur), Rhythmen lange. "
           "Die geometrische Menge deckt beides ab, kostet aber Genauigkeit bei den Neuronen und dauert mehr Sweeps. AMUSE nutzt nur das kleinste τ der Menge und ist entsprechend anfällig.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Spektren-Abstand, Rauschen, Elektroden und Verzögerungen ab?")
sweep_options = [p for p in SWEEP_OPTIONS if p != "spacing" or g >= 2]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda p: SWEEP_OPTIONS[p], key="sweep_select")
current = {"spacing": float(spacing), "noise": float(noise), "n_electrodes": int(n_electrodes), "lag_set": None}[sweep_param]
with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
    rows = _sweep(sweep_param, data_params[:7], settings)
st.plotly_chart(build_sweep(rows, SWEEP_LABELS[sweep_param], current=current, categorical=(sweep_param == "lag_set")), width="stretch", key="sweep_chart")
st.caption("Mittel und Streuung über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Rechts: Trennschärfe der Hintergrundquellen; rot gestrichelt das Zufallsniveau eines Paares, "
           "grün gepunktet die Schwelle, ab der das Urteil ein Paar als getrennt wertet.")

st.markdown("---")

# --- Wer trennt was --------------------------------------------------------------------------------------------------------------------

st.subheader("🧩 Wer trennt was: die Annahmen im Vergleich")
if st.button("Fünf Szenen vergleichen (dauert einige Sekunden)", key="scenes_start"):
    st.session_state["scenes_on"] = True
if st.session_state.get("scenes_on"):
    with st.spinner("Vergleiche 5 Szenen × 5 Datensätze × 3 Verfahren..."):
        scene_rows = _scenes(data_params[:7], settings)
    st.plotly_chart(build_scenes(scene_rows), width="stretch", key="scenes_chart")
    st.table({
        "Szene": [r["scene"] for r in scene_rows],
        "Neuronen ICA": [f"{r['ica_neuron']:.3f}" for r in scene_rows],
        "Neuronen SOBI": [f"{r['sobi_neuron']:.3f}" for r in scene_rows],
        "Trennschärfe ICA (Spanne)": ["-" if np.isnan(r["ica_bg"]) else f"{r['ica_bg']:.2f} ({r['ica_bg_min']:.2f}-{r['ica_bg_max']:.2f})" for r in scene_rows],
        "Trennschärfe SOBI (Spanne)": ["-" if np.isnan(r["sobi_bg"]) else f"{r['sobi_bg']:.2f} ({r['sobi_bg_min']:.2f}-{r['sobi_bg_max']:.2f})" for r in scene_rows],
        "Trennschärfe SOBI geometrisch": ["-" if np.isnan(r["sobi_geometric_bg"]) else f"{r['sobi_geometric_bg']:.2f}" for r in scene_rows],
    })
    st.caption("Neuronen und Elektroden wie in der Seitenleiste, Rauschen und Länge ebenso; Hintergrund und Spektren-Abstand je Szene fest. 'SOBI' nutzt die gewählte Verzögerungs-Menge. Die Spanne zeigt, wie stark das Ergebnis "
               "von der Stichprobe abhängt: bei der ICA auf Gauß'schen Quellen groß (Zufall), bei SOBI klein.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden - und wer danach kommt")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **verschiedene Autokorrelation** | Gleiches Spektrum: SOBI landet beim Zufall (Preset "Gleiche Autokorrelation"). Nur ein **einziges** τ (AMUSE): Ergebnis hängt an einem Eigenwertabstand. | ICA, wenn die Quellen nicht Gauß'sch sind (Rhythmen mit fast gleicher Frequenz: ICA trennt, SOBI nicht) |
| **τ passt zur Zeitskala** | Zu kurze oder zu lange Verzögerungen sehen den Unterschied nicht (Rhythmen mit mittleren τ). | Geometrische Mengen (Abschnitt 🔧); die Wahl bleibt ein Kompromiss zwischen den Zeitskalen |
| **wenig Rauschen** | SOBI leidet mehr als die ICA (Preset "Starkes Rauschen"). | Rauschbereinigte Kovarianzen (τ = 0 weglassen ist hier schon der Fall); Mittelung über mehr Daten |
| **mindestens so viele Sensoren wie Quellen** | Nicht umkehrbar - SOBI fällt stärker ab als die ICA (Sidebar "Elektroden"). | **Sparse Component Analysis**: mehr Quellen als Sensoren, wenn die Quellen selten gleichzeitig aktiv sind |
| **Vorzeichen frei / Quellen beliebig** | Bei nur positiven Quellen (Leistung, Verbrauch) ist das Vorzeichen nicht beliebig. | **NMF**: Nichtnegativität statt Unabhängigkeit oder Autokorrelation |
| **momentane Mischung** | Laufzeitunterschiede zwischen den Elektroden (hier ausgeblendet, siehe ICA-Demo) brechen beide Verfahren. | **Spike-Sorting-Zweig** inkl. Verzögerungsgraph |
"""
)
st.caption("Die genannten Verfahren sind die nächsten Stücke der Quellentrennung-Linie; hier steht nur, welche Annahme sie jeweils lockern.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $x(t) = A\,s(t) + \varepsilon(t)$, $x \in \mathbb{R}^n$, $s \in \mathbb{R}^k$ mit **zeitlich strukturierten, paarweise unkorrelierten** Quellen: $C_s(\tau) = E[s(t)\,s(t+\tau)^\top] = \operatorname{diag}(\rho_1(\tau), \dots, \rho_k(\tau))$ für alle $\tau$.
Identifizierbar (bis auf Reihenfolge, Vorzeichen, Skala), wenn sich die Funktionen $\rho_i(\cdot)$ paarweise unterscheiden (Belouchrani et al.).

**Weißen.** $Z = K\,\tilde X$ mit $K = \Lambda_k^{-1/2} V_k^\top$; dann $C_z(0) = I$ und $Z = Q\,s$ mit orthogonalem $Q$.

**Verzögerte Kovarianzen.** $R_\tau = \tfrac12\left(\hat C_z(\tau) + \hat C_z(\tau)^\top\right)$ mit $\hat C_z(\tau) = \tfrac{1}{T-\tau}\sum_t z(t)\,z(t+\tau)^\top$. Im Modell gilt $R_\tau = Q\,\operatorname{diag}(\rho_i(\tau))\,Q^\top$.

**AMUSE.** Eigenzerlegung von $R_{\tau_0}$: $W$ = Zeilen der Eigenvektoren, sortiert nach absteigendem Eigenwert. Eindeutig nur, wenn die $\rho_i(\tau_0)$ verschieden sind; bei kleinen Abständen im Rauschen instabil.

**SOBI.** Minimiere $\sum_\tau \operatorname{off}(V^\top R_\tau V)$, $\operatorname{off}(M) = \sum_{i \neq j} M_{ij}^2$, über orthogonale $V$ (**gemeinsame näherungsweise Diagonalisierung**). Jacobi-Verfahren (Cardoso & Souloumiac):
für jedes Paar $(p, q)$ mit $g_\tau = (M_{pp} - M_{qq},\; M_{pq} + M_{qp})^\top$ und $G = \sum_\tau g_\tau g_\tau^\top$ ist der Rotationswinkel
$\theta = \tfrac12\operatorname{atan2}\!\big(2 G_{12},\; G_{11} - G_{22} + \sqrt{(G_{11}-G_{22})^2 + 4 G_{12}^2}\big)$; Sweeps über alle Paare bis $\max|\theta| < 10^{-8}$. Die Schätzquellen sind $\hat s = V^\top K \tilde X$.

**Kennzahlen.** Zuordnung und Regression wie in der ICA-Demo. **Trennschärfe der Hintergrundquellen** je Quelle $j$: $(c_{(1)} - c_{(2)})/(c_{(1)} + c_{(2)})$ mit der größten und zweitgrößten $|\text{Korrelation}|$ über alle Schätzungen;
für ein gefundenes, aber gleichverteilt gedrehtes Paar ist der Erwartungswert $\ln\sqrt{2}\,/\,(\pi/4) \approx 0.441$.

**Grenzen.** (1) Gleiche $\rho_i$: die Diagonalen der $R_\tau$ sind für zwei Quellen gleich - jede Drehung diagonalisiert sie. (2) Die Wahl der $\tau$ ist ein Kompromiss zwischen den Zeitskalen. (3) Rauschen: die Schätzfehler von $C_z(\tau)$
sind bei kleinen Quellen-Autokorrelationen groß gegen deren Unterschied. (4) Weniger Sensoren als Quellen: $A$ nicht umkehrbar. (5) Die **Quellenzahl** wird als bekannt angenommen. (6) SOBI nutzt nur Statistik zweiter Ordnung - die
Verteilungsform bleibt ungenutzt (bei Spitzen wäre sie die stärkere Information).

Implementiert in `sobi_algorithm.py` (verzögerte Kovarianzen, AMUSE, Jacobi-Diagonalisierung, Autokorrelation, Spektrum), `sobi_ica.py` (Weißen und FastICA, wortgleich aus ica-demo), `sobi_scenario.py` (Mehrelektroden-Generator),
`sobi_evaluation.py` (Zuordnung, Kennzahlen, Sweeps, Szenen, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
