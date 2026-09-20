"""Mehrelektroden-Szenario der SOBI-Demo: dasselbe Array wie in ica-demo (Neuronen, Wellenformen, Feuern, Mischung, Rauschen - dort wortgleich übernommen), aber der Hintergrund ist ausgebaut:
bis zu zwei farbige Quellen, deren **Autokorrelation** (bzw. Rhythmusfrequenz) sich um den Spektren-Abstand `spacing` unterscheidet. Ohne Verzögerung (die Annahme der momentanen Mischung gilt hier immer).

Nur numpy. Jede Quelle hat einen eigenen Zufallsstrom (Seed, Nummer) - Neuron 0 feuert für einen Seed immer gleich, egal wie viele Neuronen, Elektroden oder Hintergrundquellen eingestellt sind."""

from dataclasses import dataclass

import numpy as np

import sobi_constants as C


@dataclass(frozen=True)
class Dataset:
    S: np.ndarray                 # (k, T) wahre Quellen, Zeilen 0..m-1 = Neuronen, danach Hintergrund; jede Zeile hat Varianz 1
    X: np.ndarray                 # (n, T) Elektrodensignale (mit Rauschen)
    X_clean: np.ndarray           # (n, T) ohne Rauschen
    A: np.ndarray                 # (n, k) Mischmatrix
    spike_times: tuple            # je Neuron: Zeitpunkte der negativen Spitze (Abtastwerte)
    kinds: tuple                  # je Quelle: "neuron", "gauss" oder "rhythm"
    phis: tuple                   # je Hintergrundquelle: AR-Koeffizient (gauss) bzw. Frequenz in Hz (rhythm)
    noise_sigma: float
    n_neurons: int
    n_electrodes: int
    seed: int


def spike_waveform(index):
    """Biphasische Wellenform: tiefe negative Spitze bei PEAK_INDEX, flacher positiver Nachschlag."""
    sigma = C.NEURON_SIGMAS[index]
    t = np.arange(C.WAVEFORM_LENGTH, dtype=float)
    return -np.exp(-((t - C.PEAK_INDEX) / sigma) ** 2) + C.OVERSHOOT * np.exp(-((t - (C.PEAK_INDEX + 2.5 * sigma)) / (1.6 * sigma)) ** 2)


def firing_times(index, n_samples, seed):
    """Poisson-artiges Feuern mit Refraktärzeit: Abstände = Refraktärzeit + Exponentialverteilung. Spike-Startzeitpunkte (Abtastwerte)."""
    rng = np.random.default_rng([seed, index])
    mean_isi = C.SAMPLE_RATE / C.NEURON_RATES[index]
    n_draw = int(n_samples / (mean_isi - C.REFRACTORY) * 1.5) + 20
    isi = C.REFRACTORY + rng.exponential(mean_isi - C.REFRACTORY, n_draw)
    times = np.cumsum(isi) - isi[0] + rng.uniform(0, mean_isi)
    return times[times < n_samples - C.WAVEFORM_LENGTH].astype(int)


def neuron_source(index, n_samples, seed):
    starts = firing_times(index, n_samples, seed)
    impulses = np.zeros(n_samples)
    impulses[starts] = 1.0
    s = np.convolve(impulses, spike_waveform(index))[:n_samples]
    return s, starts + C.PEAK_INDEX


def background_parameters(kind, n_background, spacing):
    """Je Hintergrundquelle der Parameter, der ihr Spektrum festlegt: AR(1)-Koeffizient (gauss, 0.95 - j*spacing) bzw. Frequenz in Hz (rhythm, 10 + j*spacing*RHYTHM_HZ_PER_SPACING).
    spacing = 0: alle Quellen haben dasselbe Spektrum."""
    if kind == "rhythm":
        return tuple(C.RHYTHM_BASE_HZ + j * spacing * C.RHYTHM_HZ_PER_SPACING for j in range(n_background))
    return tuple(C.GAUSS_AR_BASE - j * spacing for j in range(n_background))


def background_source(index, kind, parameter, n_samples, seed):
    rng = np.random.default_rng([seed, 2000 + index])
    if kind == "rhythm":
        t = np.arange(n_samples) / C.SAMPLE_RATE
        return np.sin(2 * np.pi * parameter * t + rng.uniform(0, 2 * np.pi))
    e = rng.standard_normal(n_samples)
    s = np.empty(n_samples)
    s[0] = e[0]
    for t_ in range(1, n_samples):                                    # AR(1); n_samples <= 40000, einmalig je Datensatz
        s[t_] = parameter * s[t_ - 1] + e[t_]
    return s


def electrode_positions(n):
    return np.array([[0.5, 0.0]]) if n == 1 else np.column_stack([np.linspace(0.0, 1.0, n), np.zeros(n)])


def mixing_matrix(m, n, g):
    """(n, m+g): Neuronen mit 1/(d^2+eps)-Abfall (Spitzenamplitude je Neuron), Hintergrund als flächiges Feld (gleich stark, Rampe)."""
    pos = electrode_positions(n)
    cols = []
    for i in range(m):
        d = np.linalg.norm(pos - np.array(C.NEURON_POSITIONS[i]), axis=1)
        a = 1.0 / (d ** 2 + C.DISTANCE_EPS)
        cols.append(C.NEURON_AMPLITUDES[i] * a / a.max())
    ramp = np.linspace(-1.0, 1.0, n) if n > 1 else np.ones(1)
    patterns = (np.ones(n), ramp)
    for j in range(g):
        cols.append(C.BACKGROUND_AMPLITUDE * patterns[j])
    return np.column_stack(cols)


def make_dataset(n_neurons, n_electrodes, n_background, background_kind, spacing, noise, n_samples, seed):
    m, n, g = n_neurons, n_electrodes, n_background
    sources, spikes, kinds = [], [], []
    for i in range(m):
        s, peaks = neuron_source(i, n_samples, seed)
        sources.append(s), spikes.append(peaks), kinds.append("neuron")
    phis = background_parameters(background_kind, g, spacing)
    for j in range(g):
        sources.append(background_source(j, background_kind, phis[j], n_samples, seed)), kinds.append(background_kind)
    S = np.array(sources)
    S = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    A = mixing_matrix(m, n, g)
    X_clean = A @ S
    sigma = noise * (A[:, :m] @ S[:m]).std()                            # Rauschen relativ zum Neuronen-Signal: unabhängig von der Zahl der Hintergrundquellen
    X = X_clean + sigma * np.random.default_rng([seed, 1000]).standard_normal(X_clean.shape)
    return Dataset(S, X, X_clean, A, tuple(spikes), tuple(kinds), phis, float(sigma), m, n, seed)
