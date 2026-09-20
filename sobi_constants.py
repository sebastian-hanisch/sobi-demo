"""Defaults, Slider-Grenzen und feste Szenario-Größen der SOBI-Demo (Presets folgen in sobi_presets.py-Nutzung nach den Messungen).
Das Szenario ist das der ica-demo (Neuronen, Wellenformen, Feuerraten, Mischung wortgleich); neu sind der ausgebaute Hintergrund und die SOBI-Regler."""

# --- Szenario (fest, wortgleich aus ica-demo) ---------------------------------------------------------------------------------
SAMPLE_RATE = 10_000                       # Hz
WAVEFORM_LENGTH = 30                       # Abtastwerte je Spike
PEAK_INDEX = 8                             # Lage der negativen Spitze in der Wellenform
OVERSHOOT = 0.4                            # Höhe des positiven Nachschlags
REFRACTORY = 20                            # Abtastwerte (2 ms)
NEURON_SIGMAS = (2.0, 3.0, 2.5, 4.0, 3.5)          # Breite der Spitze je Neuron
NEURON_RATES = (20.0, 28.0, 35.0, 24.0, 31.0)      # Feuerrate in Hz
NEURON_AMPLITUDES = (1.0, 0.8, 1.2, 0.7, 0.9)      # Spitzenamplitude an der nächsten Elektrode
NEURON_POSITIONS = ((0.10, 0.30), (0.35, 0.20), (0.60, 0.35), (0.85, 0.25), (0.50, 0.55))   # Elektroden liegen bei y = 0, x in [0, 1]
DISTANCE_EPS = 0.05
BACKGROUND_AMPLITUDE = 0.8
# --- Szenario (neu): Hintergrund mit einstellbarem Spektren-Abstand ------------------------------------------------------------------
GAUSS_AR_BASE = 0.95                       # AR(1)-Koeffizient der ersten Hintergrundquelle; die j-te hat 0.95 - j * Abstand (Abstand 0.45 = die beiden Quellen der ica-demo: 0.95 und 0.5)
RHYTHM_BASE_HZ = 10.0                      # Frequenz des ersten Rhythmus; der j-te hat 10 + j * Abstand * RHYTHM_HZ_PER_SPACING Hz (Abstand 0.45 = 10 und 23 Hz wie in der ica-demo)
RHYTHM_HZ_PER_SPACING = 13.0 / 0.45
BACKGROUND_KINDS = ("gauss", "rhythm")
BACKGROUND_LABELS = {"gauss": "Gauß-Rauschen (farbig, AR(1))", "rhythm": "Rhythmus (sinusförmig)"}

# --- Regler ---------------------------------------------------------------------------------------------------------------------
DEFAULT_N_NEURONS = 4
N_NEURONS_MIN, N_NEURONS_MAX = 2, 5
DEFAULT_N_ELECTRODES = 6
N_ELECTRODES_MIN, N_ELECTRODES_MAX = 1, 8
DEFAULT_N_BACKGROUND = 2
N_BACKGROUND_MIN, N_BACKGROUND_MAX = 0, 2
DEFAULT_BACKGROUND_KIND = "gauss"
DEFAULT_SPACING = 0.45
SPACING_MIN, SPACING_MAX = 0.0, 0.45
DEFAULT_NOISE = 0.05
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_N_SAMPLES = 20_000
N_SAMPLES_MIN, N_SAMPLES_MAX = 5_000, 40_000
# Verzögerungs-Mengen τ (Abtastwerte bei 10 kHz) für SOBI; die Zeitskalen der Quellen unterscheiden sich stark (Spitzen ~ 30 Abtastwerte, AR(0.95) ~ 20, Rhythmen ~ 1000)
LAG_SETS = {
    "short": tuple(range(1, 6)),
    "medium": tuple(range(2, 21, 2)),
    "long": tuple(range(10, 101, 10)),
    "geometric": tuple(2 ** j for j in range(9)),
}
LAG_SET_LABELS = {"short": "kurz: 1, 2, …, 5", "medium": "mittel: 2, 4, …, 20", "long": "lang: 10, 20, …, 100", "geometric": "geometrisch: 1, 2, 4, …, 256"}
DEFAULT_LAG_SET = "medium"
CONTRASTS = ("logcosh", "exp", "cube")
CONTRAST_LABELS = {"logcosh": "log cosh (robust)", "exp": "Gauß-Ableitung (sehr robust)", "cube": "Kurtosis (u³)"}
DEFAULT_CONTRAST = "logcosh"
METHODS = ("symmetric", "deflation")
DEFAULT_METHOD = "symmetric"
INIT_STARTS = (1, 2, 3, 4, 5)
DEFAULT_INIT_START = 1
DEFAULT_SEED = 7
MAX_ITER = 200                             # FastICA
TOL = 1e-6
JD_MAX_SWEEPS = 100                        # gemeinsame Diagonalisierung (Jacobi-Sweeps)
JD_TOL = 1e-8

# --- Auswertung -----------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = (100000, 100001, 100002, 100003, 100004)            # feste Datensätze der Sweeps, getrennt vom Demo-Seed


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(m=DEFAULT_N_NEURONS, n=DEFAULT_N_ELECTRODES, g=DEFAULT_N_BACKGROUND, kind=DEFAULT_BACKGROUND_KIND, spacing=DEFAULT_SPACING, noise=DEFAULT_NOISE, n_samples=DEFAULT_N_SAMPLES,
                lag_set=DEFAULT_LAG_SET, contrast=DEFAULT_CONTRAST, init_start=DEFAULT_INIT_START, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Zwei Gauß-Quellen, verschiedene Autokorrelation": _preset(),
    "Gleiche Autokorrelation": _preset(spacing=0.0, seed=11),
    "Zwei Rhythmen, Verzögerungen zu kurz": _preset(kind="rhythm"),
    "Zwei Rhythmen, geometrische Verzögerungen": _preset(kind="rhythm", lag_set="geometric"),
    "Starkes Rauschen": _preset(noise=0.4),
    "Nur Neuronen": _preset(g=0),
}
PRESET_HELP = {
    "Zwei Gauß-Quellen, verschiedene Autokorrelation": "Zwei Gauß'sche Hintergrundquellen mit AR-Koeffizient 0.95 und 0.5 - das Paar, an dem die ICA scheiterte. SOBI nutzt den Unterschied ihrer Autokorrelation und trennt sie: "
                                                        "Trennschärfe 0.82 (in jedem Datensatz 0.80-0.84), die ICA im Mittel nur 0.41 (0.10-0.73 je Datensatz, Zufall; Zufallsniveau 0.44).",
    "Gleiche Autokorrelation": "Beide Hintergrundquellen haben dasselbe Spektrum: SOBI findet keinen Unterschied und landet beim Zufall (Trennschärfe im Mittel 0.43, 0.30-0.50 je Datensatz), die ICA ebenfalls (0.27, "
                               "0.04-0.72). Wer hier zufällig trennt, hat Glück mit der Stichprobe, nicht mit dem Verfahren.",
    "Zwei Rhythmen, Verzögerungen zu kurz": "Zwei Sinusrhythmen (10 und 23 Hz). Die mittleren Verzögerungen (0.2-2 ms) sind winzig gegen die Periode (100 ms): die verzögerten Kovarianzen beider Rhythmen sind fast gleich, SOBI trennt sie nicht "
                                            "(Trennschärfe 0.38), die ICA schon (0.76, Sinus ist unter-Gauß'sch).",
    "Zwei Rhythmen, geometrische Verzögerungen": "Dieselben Rhythmen, Verzögerungen 1, 2, 4, ..., 256: die langen Verzögerungen unterscheiden die Frequenzen, SOBI trennt jetzt (Trennschärfe 0.94, ICA 0.76). "
                                                 "Der Preis: die Neuronen werden etwas schlechter getrennt (0.87 gegen 0.90 mit den mittleren Verzögerungen).",
    "Starkes Rauschen": "Rauschen von 40 % des Neuronen-Signals. SOBI leidet mehr als die ICA: Korrelation der Neuronen 0.33 gegen 0.59 - Kovarianzen bei Verzögerungen verlieren das Signal schneller im Rauschen "
                        "als die Verteilungsform der Spitzen.",
    "Nur Neuronen": "Kein Hintergrund. Überraschung: SOBI trennt die Spitzen fast so gut wie die ICA (Korrelation 0.978 gegen 0.981) - die Wellenformen der Neuronen sind verschieden breit, ihre Autokorrelationen "
                    "unterscheiden sich also; AMUSE mit einer einzigen Verzögerung kommt auf 0.93.",
}
# Bänder (Seed des Presets; Werte mit dem ausgelieferten Code kalibriert, bewusst weit): Neuronen-Korrelation (ica_neuron, sobi_neuron), Trennschärfe der Hintergrundquellen (ica_bg, sobi_bg), Urteil (verdict)
PRESET_EXPECTED_BANDS = {
    "Zwei Gauß-Quellen, verschiedene Autokorrelation": {"ica_neuron": (0.85, 0.97), "sobi_neuron": (0.82, 0.96), "ica_bg": (0.0, 0.6), "sobi_bg": (0.72, 0.95), "verdict": "sobi_wins"},
    "Gleiche Autokorrelation": {"sobi_bg": (0.0, 0.55), "ica_bg": (0.0, 0.55), "verdict": "both_fail"},
    "Zwei Rhythmen, Verzögerungen zu kurz": {"ica_bg": (0.68, 0.9), "sobi_bg": (0.2, 0.55), "verdict": "lags_mismatch"},
    "Zwei Rhythmen, geometrische Verzögerungen": {"ica_bg": (0.68, 0.9), "sobi_bg": (0.8, 1.0), "sobi_neuron": (0.8, 0.93), "verdict": "both_work"},
    "Starkes Rauschen": {"ica_neuron": (0.45, 0.7), "sobi_neuron": (0.15, 0.5), "verdict": "noise"},
    "Nur Neuronen": {"ica_neuron": (0.96, 1.0), "sobi_neuron": (0.95, 1.0), "amuse_neuron": (0.85, 0.98), "verdict": "neutral"},
}
