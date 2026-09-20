"""Auswertung der SOBI-Demo: Zuordnung, Kennzahlen (Korrelation, SIR, Amari-Index, Spike-Treffer) - aus ica-demo übernommen - und die neuen Größen für den Hintergrund
(Amari-Index des Hintergrund-Blocks mit bekanntem Zufalls-Niveau), Sweeps, Verzögerungs-Mengen, Vergleichstabellen und das Urteil."""

from dataclasses import dataclass

import numpy as np

import sobi_algorithm as alg
import sobi_constants as C
import sobi_ica as ica
import sobi_scenario as sc


@dataclass(frozen=True)
class Settings:
    contrast: str = C.DEFAULT_CONTRAST
    lag_set: str = C.DEFAULT_LAG_SET
    init_start: int = C.DEFAULT_INIT_START


def make_dataset(m=C.DEFAULT_N_NEURONS, n=C.DEFAULT_N_ELECTRODES, g=C.DEFAULT_N_BACKGROUND, kind=C.DEFAULT_BACKGROUND_KIND, spacing=C.DEFAULT_SPACING, noise=C.DEFAULT_NOISE,
                 n_samples=C.DEFAULT_N_SAMPLES, seed=C.DEFAULT_SEED):
    return sc.make_dataset(m, n, g, kind, spacing, noise, n_samples, seed)


def n_components(ds):
    """Die Zahl der Quellen wird als bekannt angenommen (Standardannahme von FastICA und SOBI); bei weniger Elektroden als Quellen bleibt nur n."""
    return min(ds.n_electrodes, ds.S.shape[0])


def correlation_matrix(S, estimates):
    """|Korrelation| (k, nc) zwischen wahren Quellen und Schätzungen."""
    a = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    b = estimates - estimates.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    b = b / np.where(sd > 0, sd, 1.0)
    return np.abs(a @ b.T) / S.shape[1]


def assign(C_abs):
    """Optimale Zuordnung Quelle -> Schätzung (jede Schätzung höchstens einer Quelle), Summe der |Korrelationen| maximal. Exakt per Bitmasken-DP.
    Rückgabe: Liste je Quelle mit dem Index der Schätzung oder -1 (nicht zugeordnet, nur wenn es weniger Schätzungen als Quellen gibt)."""
    k, nc = C_abs.shape
    best = {}

    def solve(row, used):
        if row == k:
            return 0.0, ()
        key = (row, used)
        if key in best:
            return best[key]
        result = (solve(row + 1, used)[0], (-1,) + solve(row + 1, used)[1])
        for j in range(nc):
            if not used >> j & 1:
                value, rest = solve(row + 1, used | 1 << j)
                if value + C_abs[row, j] > result[0] + 1e-12:
                    result = (value + C_abs[row, j], (j,) + rest)
        best[key] = result
        return result

    return list(solve(0, 0)[1])


def matched(S, estimates):
    """Zuordnung + Vorzeichen-/Skalenkorrektur per Regression: (Zuordnung, |Korrelation| je Quelle, Schätzquellen in Skala und Vorzeichen der wahren Quellen)."""
    cm = correlation_matrix(S, estimates)
    idx = assign(cm)
    corr = np.array([cm[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)])
    aligned = np.zeros_like(S)
    for i, j in enumerate(idx):
        if j >= 0:
            e = estimates[j] - estimates[j].mean()
            aligned[i] = (e @ (S[i] - S[i].mean()) / (e @ e)) * e + S[i].mean()
    return idx, corr, aligned


def sir_db(corr):
    """Signal-zu-Interferenz in dB aus der Korrelation: rho^2 / (1 - rho^2)."""
    r2 = np.clip(np.asarray(corr) ** 2, 1e-9, 1.0 - 1e-9)
    return 10.0 * np.log10(r2 / (1.0 - r2))


def amari_index(P):
    """Amari-Index einer (nc, k)-Matrix P = Entmischung x Mischung; 0 = perfekt (nur Permutation und Skalierung), 1 = schlechtestmöglich; verallgemeinert auf nicht quadratische P."""
    P = np.abs(P)
    k = P.shape[1]
    rows = (P.sum(axis=1) / P.max(axis=1) - 1.0).sum()
    cols = (P.sum(axis=0) / P.max(axis=0) - 1.0).sum()
    d = max(k, P.shape[0])
    return float((rows + cols) / (2.0 * d * (d - 1))) if d > 1 else 0.0


# --- Spike-Erkennung auf den Schätzquellen ------------------------------------------------------------------------------------
DETECT_MIN_SEPARATION = 15         # Abtastwerte zwischen zwei erkannten Spitzen
DETECT_TOLERANCE = 4               # erlaubte Abweichung zur wahren Spitze
DETECT_SIGMA_FACTOR = 4.0          # Schwelle: 4 robuste Standardabweichungen (MAD) ...
DETECT_DEPTH_FRACTION = 0.3        # ... mindestens aber 30 % der typischen Spitzentiefe (sonst würde ein rauschfreies Signal jeden Rest melden)


def detect_spikes(x):
    """Negative Spitzen von x (Vorzeichen bereits wie beim Neuron): tiefste zuerst, Mindestabstand; Schwelle max(4 sigma_MAD, 0.3 x typische Tiefe)."""
    sigma = np.median(np.abs(x - np.median(x))) / 0.6745
    threshold = -DETECT_SIGMA_FACTOR * sigma
    taken = np.zeros(len(x), bool)
    peaks = []
    for t in np.argsort(x):
        if x[t] > threshold:
            break
        if taken[max(0, t - DETECT_MIN_SEPARATION): t + DETECT_MIN_SEPARATION + 1].any():
            continue
        taken[t] = True
        peaks.append(t)
        if len(peaks) == 10:                                             # typische Tiefe = Median der 10 tiefsten Spitzen
            threshold = min(threshold, DETECT_DEPTH_FRACTION * float(np.median(x[peaks])))
    return np.sort(np.array(peaks, dtype=int))


def spike_f1(detected, true):
    """F1 der erkannten gegen die wahren Spitzenzeiten (Zuordnung je wahrer Spitze zur nächsten, jede erkannte höchstens einmal)."""
    if len(detected) == 0 or len(true) == 0:
        return 0.0
    used = np.zeros(len(detected), bool)
    tp = 0
    for t in true:
        d = np.abs(detected - t)
        j = int(np.argmin(d))
        if d[j] <= DETECT_TOLERANCE and not used[j]:
            used[j] = True
            tp += 1
    if tp == 0:
        return 0.0
    precision, recall = tp / len(detected), tp / len(true)
    return 2 * precision * recall / (precision + recall)



def excess_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
    return (x ** 4).mean(axis=-1) - 3.0




# --- Trennschärfe der Hintergrundquellen ------------------------------------------------------------------------------------------

CHANCE_PAIR_MARGIN = float(np.log(np.sqrt(2.0)) / (np.pi / 4.0))       # zufällige Drehung eines gefundenen Paares: E[(max - min)/(max + min)] = ln(sqrt 2)/(pi/4) = 0.441
BG_SEPARATED = 0.65                # Trennschärfe ab der ein Verfahren die Hintergrundquellen als getrennt gilt (Messwerte: getrennt 0.72-0.97, nicht getrennt 0.06-0.5)
BG_FAILED = 0.55                   # darunter gilt sie als nicht getrennt


def bg_margin(cm, m):
    """Trennschärfe der Hintergrundquellen: je Hintergrundquelle (c1 - c2)/(c1 + c2) aus der größten und zweitgrößten |Korrelation| über ALLE Schätzungen, gemittelt.
    ~1 = eine Schätzung gehört klar zu der Quelle; 0 = zwei Schätzungen sind gleich gut (vermischt); 0.44 = zufällige Drehung eines gefundenen Gauß-Paares. NaN ohne Hintergrund."""
    if cm.shape[0] <= m:
        return float("nan")
    out = []
    for i in range(m, cm.shape[0]):
        r = np.sort(cm[i])[::-1]
        out.append((r[0] - r[1]) / max(r[0] + r[1], 1e-12) if len(r) > 1 else 1.0)
    return float(np.mean(out))


# --- Kennzahlen je Verfahren -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Method:
    name: str
    idx: list                     # je Quelle: Index der zugeordneten Schätzung (-1 = keine)
    corr: np.ndarray              # |Korrelation| je Quelle
    aligned: np.ndarray           # (k, T) zugeordnete Schätzungen in Vorzeichen und Skala der wahren Quellen
    neuron_corr: float
    neuron_sir: float
    f1: float
    bg_margin: float              # Trennschärfe der Hintergrundquellen (NaN ohne Hintergrund)
    amari: float                  # Amari-Index der gesamten Entmischung
    estimates: np.ndarray         # (nc, T) rohe Schätzungen


def evaluate_method(name, ds, estimates, unmixing):
    idx, corr, aligned = matched(ds.S, estimates)
    m = ds.n_neurons
    P = unmixing @ ds.A
    f1 = float(np.mean([spike_f1(detect_spikes(aligned[i]), ds.spike_times[i]) for i in range(m)]))
    return Method(name, idx, corr, aligned, float(corr[:m].mean()), float(sir_db(corr[:m]).mean()), f1, bg_margin(correlation_matrix(ds.S, estimates), m), amari_index(P), estimates)


def lags_of(name):
    return C.LAG_SETS[name]


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    models: dict                  # "ica", "sobi", "amuse"
    methods: dict                 # "pca", "ica", "amuse", "sobi" -> Method
    ref_geometric: float          # Trennschärfe von SOBI mit den geometrischen Verzögerungen (nur nötig, wenn die gewählte Menge eine andere ist); NaN sonst
    ref_clean: float              # Neuronen-Korrelation der ICA ohne Rauschen (nur bei Rauschen > 0)


def spacing_of(ds):
    """Spektren-Abstand aus den Parametern der Hintergrundquellen zurückrechnen (für Referenzläufe auf demselben Datensatz)."""
    if len(ds.phis) < 2:
        return C.DEFAULT_SPACING
    if ds.kinds[-1] == "rhythm":
        return (ds.phis[1] - ds.phis[0]) / C.RHYTHM_HZ_PER_SPACING
    return ds.phis[0] - ds.phis[1]


def analyse(ds, settings):
    nc = n_components(ds)
    lags = lags_of(settings.lag_set)
    ica_model = ica.fit_ica(ds.X, nc, settings.contrast, C.DEFAULT_METHOD, settings.init_start)
    sobi_model = alg.fit_sobi(ds.X, nc, lags)
    amuse_model = alg.fit_amuse(ds.X, nc, lags[0])
    wh = ica.whiten(ds.X, nc)
    methods = {
        "pca": evaluate_method("pca", ds, wh.Z, wh.K),
        "ica": evaluate_method("ica", ds, ica_model.sources, ica_model.unmixing),
        "amuse": evaluate_method("amuse", ds, amuse_model.sources, amuse_model.unmixing),
        "sobi": evaluate_method("sobi", ds, sobi_model.sources, sobi_model.unmixing),
    }
    g = ds.S.shape[0] - ds.n_neurons
    ref_geometric = float("nan")
    if g >= 1 and settings.lag_set != "geometric":
        geo = alg.fit_sobi(ds.X, nc, lags_of("geometric"))
        ref_geometric = evaluate_method("sobi", ds, geo.sources, geo.unmixing).bg_margin
    ref_clean = float("nan")
    if ds.noise_sigma > 0:
        clean = sc.make_dataset(ds.n_neurons, ds.n_electrodes, g, ds.kinds[-1] if g else C.DEFAULT_BACKGROUND_KIND, spacing_of(ds), 0.0, ds.S.shape[1], ds.seed)
        ref_clean = float(matched(clean.S, ica.fit_ica(clean.X, nc, settings.contrast, C.DEFAULT_METHOD, settings.init_start).sources)[1][:ds.n_neurons].mean())
    return Analysis(ds, settings, {"ica": ica_model, "sobi": sobi_model, "amuse": amuse_model}, methods, ref_geometric, ref_clean)


# --- Verläufe und Anschauungsdaten ----------------------------------------------------------------------------------------------------------


def sobi_curve(a):
    """Mittlere |Korrelation| der Neuronen nach jedem Jacobi-Sweep (Index 0 = vor der ersten Rotation, also PCA)."""
    model, ds = a.models["sobi"], a.ds
    Z = model.whitening.Z
    return [float(matched(ds.S, V.T @ Z)[1][:ds.n_neurons].mean()) for V in model.rotations]


def diagonalized(model):
    """Verzögerte Kovarianzen vor und nach der gefundenen Rotation: (vorher, nachher), je (Anzahl Verzögerungen, nc, nc)."""
    W = model.W
    return model.covariances, np.einsum("ij,kjl,ml->kim", W, model.covariances, W)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "spacing": (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45),
    "noise": (0.0, 0.02, 0.05, 0.1, 0.2, 0.4),
    "n_electrodes": (2, 3, 4, 5, 6, 8),
    "lag_set": tuple(C.LAG_SETS),
}
SWEEP_LABELS = {"spacing": "Spektren-Abstand (0 = gleiche Autokorrelation)", "noise": "Rauschen (relativ zum Neuronen-Signal)", "n_electrodes": "Anzahl Elektroden", "lag_set": "Verzögerungs-Menge"}
_SWEEP_KEYWORD = {"spacing": "spacing", "noise": "noise", "n_electrodes": "n"}
SWEEP_METHODS = ("ica", "sobi", "amuse")


def _summarise(x, per_seed):
    out = {"x": x}
    for method in SWEEP_METHODS:
        for key in ("neuron", "bg"):
            arr = np.array([r[method][key] for r in per_seed], dtype=float)
            out[f"{method}_{key}"] = float(np.nanmean(arr)) if not np.isnan(arr).all() else float("nan")
            out[f"{method}_{key}_std"] = float(np.nanstd(arr)) if not np.isnan(arr).all() else float("nan")
    return out


def _record(a):
    return {name: {"neuron": a.methods[name].neuron_corr, "bg": a.methods[name].bg_margin} for name in SWEEP_METHODS}


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel und Streuung (über die festen Sweep-Datensätze) der Neuronen-Korrelation und der Trennschärfe der Hintergrundquellen für ICA, SOBI und AMUSE in Abhängigkeit von einem Regler."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        per_seed = []
        for seed in C.SWEEP_SEEDS:
            if parameter == "lag_set":
                a = analyse(make_dataset(seed=seed, **base), Settings(settings.contrast, x, settings.init_start))
            else:
                kw = dict(base)
                kw[_SWEEP_KEYWORD[parameter]] = x
                a = analyse(make_dataset(seed=seed, **kw), settings)
            per_seed.append(_record(a))
        rows.append(_summarise(x, per_seed))
    return rows


SCENES = (
    ("Nur Neuronen", dict(g=0)),
    ("Zwei Gauß-Quellen, verschiedene Autokorrelation", dict(g=2, kind="gauss", spacing=0.45)),
    ("Zwei Gauß-Quellen, gleiche Autokorrelation", dict(g=2, kind="gauss", spacing=0.0)),
    ("Zwei Rhythmen (10 und 23 Hz)", dict(g=2, kind="rhythm", spacing=0.45)),
    ("Zwei nahe Rhythmen (10 und 10.7 Hz)", dict(g=2, kind="rhythm", spacing=0.05)),
)


def scene_table(settings=Settings(), **base):
    """Wer trennt was: je Szene ICA, SOBI (gewählte Verzögerungs-Menge) und SOBI geometrisch - Neuronen-Korrelation und Trennschärfe der Hintergrundquellen, Mittel über die Sweep-Datensätze."""
    rows = []
    for label, scene in SCENES:
        acc = {"ica": [], "sobi": [], "sobi_geometric": []}
        for seed in C.SWEEP_SEEDS:
            kw = dict(base)
            kw.update(scene)
            ds = make_dataset(seed=seed, **kw)
            a = analyse(ds, settings)
            nc = n_components(ds)
            geo = alg.fit_sobi(ds.X, nc, lags_of("geometric"))
            mg = evaluate_method("sobi_geometric", ds, geo.sources, geo.unmixing)
            acc["ica"].append(a.methods["ica"]), acc["sobi"].append(a.methods["sobi"]), acc["sobi_geometric"].append(mg)
        row = {"scene": label}
        for name, ms in acc.items():
            row[name + "_neuron"] = float(np.mean([m.neuron_corr for m in ms]))
            vals = [m.bg_margin for m in ms]
            row[name + "_bg"] = float(np.mean(vals)) if not np.isnan(vals).all() else float("nan")
            row[name + "_bg_min"] = float(np.min(vals)) if not np.isnan(vals).all() else float("nan")
            row[name + "_bg_max"] = float(np.max(vals)) if not np.isnan(vals).all() else float("nan")
        rows.append(row)
    return rows


def lag_table(**base):
    """Verzögerungs-Mengen im Vergleich (plus AMUSE mit dem kleinsten τ der Menge): Neuronen-Korrelation, Trennschärfe, Jacobi-Sweeps - Mittel über die Sweep-Datensätze."""
    rows = []
    for name, lags in C.LAG_SETS.items():
        acc = {"neuron": [], "bg": [], "sweeps": [], "amuse_neuron": [], "amuse_bg": []}
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **base)
            nc = n_components(ds)
            model = alg.fit_sobi(ds.X, nc, lags)
            r = evaluate_method("sobi", ds, model.sources, model.unmixing)
            am = alg.fit_amuse(ds.X, nc, lags[0])
            ra = evaluate_method("amuse", ds, am.sources, am.unmixing)
            acc["neuron"].append(r.neuron_corr), acc["bg"].append(r.bg_margin), acc["sweeps"].append(len(model.angles))
            acc["amuse_neuron"].append(ra.neuron_corr), acc["amuse_bg"].append(ra.bg_margin)
        rows.append({"lag_set": name, "lags": lags, **{k: float(np.nanmean(v)) if not np.isnan(v).all() else float("nan") for k, v in acc.items()}})
    return rows


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

VERDICT_NOISE_GAP = 0.20          # Verlust der Neuronen-Korrelation durch das Rauschen (ICA gegenüber demselben Datensatz ohne Rauschen)
VERDICT_MARGIN_GAP = 0.15         # Vorsprung der Trennschärfe, ab dem ein Verfahren als klar besser gilt


def verdict(a):
    """(Art, Code, Kennzahlen). Nur mit großer Marge - kein Urteil nahe an einer Schwelle."""
    ds = a.ds
    k, m = ds.S.shape[0], ds.n_neurons
    g = k - m
    ica_m, sobi_m, amuse_m = a.methods["ica"], a.methods["sobi"], a.methods["amuse"]
    data = {"ica_neuron": ica_m.neuron_corr, "sobi_neuron": sobi_m.neuron_corr, "amuse_neuron": amuse_m.neuron_corr, "pca_neuron": a.methods["pca"].neuron_corr,
            "ica_bg": ica_m.bg_margin, "sobi_bg": sobi_m.bg_margin, "amuse_bg": amuse_m.bg_margin, "pca_bg": a.methods["pca"].bg_margin, "geo_bg": a.ref_geometric,
            "sweeps": len(a.models["sobi"].angles), "ref_clean": a.ref_clean, "sobi_f1": sobi_m.f1, "ica_f1": ica_m.f1}
    if ds.n_electrodes < k:
        return "warning", "underdetermined", data
    if not a.models["sobi"].converged:
        return "warning", "not_converged", data
    if ds.noise_sigma > 0 and a.ref_clean - ica_m.neuron_corr > VERDICT_NOISE_GAP:
        return "warning", "noise", data
    if g >= 1:
        i, s = ica_m.bg_margin, sobi_m.bg_margin
        sobi_ok, ica_ok = s >= BG_SEPARATED, i >= BG_SEPARATED
        geo_ok = a.settings.lag_set != "geometric" and a.ref_geometric >= BG_SEPARATED
        if sobi_ok and ica_ok:
            return "info", "both_work", data
        if sobi_ok and i < s - VERDICT_MARGIN_GAP:
            return "success", "sobi_wins", data
        if not sobi_ok and geo_ok:
            return "warning", "lags_mismatch", data
        if not sobi_ok and ica_ok:
            return "warning", "ica_wins_bg", data
        if s < BG_FAILED and i < BG_FAILED:
            return "warning", "both_fail", data
    return "info", "neutral", data
