"""Plotly-Visualisierungen der SOBI-Demo: Elektrodenlayout, Signalspuren, Autokorrelation und Spektrum der Quellen, verzögerte Kovarianzen vor/nach der Diagonalisierung, Jacobi-Verlauf,
Zuordnungsmatrix, Kennzahlen-Balken, Sweeps, Szenen-Vergleich und Verzögerungs-Mengen. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sobi_constants as C
from sobi_evaluation import BG_SEPARATED, CHANCE_PAIR_MARGIN
from sobi_scenario import electrode_positions

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf"
NEURON_COLORS = ("#1f77b4", "#d68a2e", "#2ca02c", "#8e5fbf", "#c2185b")
BACKGROUND_COLORS = ("#7f7f7f", "#a0a0a0")
METHOD_COLORS = {"pca": GRAY, "ica": ORANGE, "amuse": PURPLE, "sobi": BLUE, "sobi_geometric": GREEN}
METHOD_NAMES = {"pca": "PCA (nur weißen)", "ica": "ICA", "amuse": "AMUSE (ein τ)", "sobi": "SOBI", "sobi_geometric": "SOBI geometrisch"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def source_color(i):
    return NEURON_COLORS[i] if i < len(NEURON_COLORS) else BACKGROUND_COLORS[(i - len(NEURON_COLORS)) % 2]


def source_labels(ds):
    labels = [f"Neuron {i + 1}" for i in range(ds.n_neurons)]
    for j, kind in enumerate(ds.kinds[ds.n_neurons:]):
        labels.append(f"{'Gauß-Hintergrund' if kind == 'gauss' else 'Rhythmus'} {j + 1}")
    return labels


def build_layout(ds):
    """Elektroden (Quadrate auf y = 0) und Neuronen (Kreise, Größe = Spitzenamplitude); der Hintergrund wirkt flächig auf alle Elektroden."""
    pos = electrode_positions(ds.n_electrodes)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[f"E{j + 1}" for j in range(ds.n_electrodes)], textposition="bottom center", name="Elektroden",
                             marker=dict(symbol="square", size=14, color=GRAY), hoverinfo="skip"))
    for i in range(ds.n_neurons):
        x, y = C.NEURON_POSITIONS[i]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"N{i + 1}"], textposition="top center", name=f"Neuron {i + 1}", hoverinfo="skip",
                                 marker=dict(size=10 + 14 * C.NEURON_AMPLITUDES[i], color=source_color(i), opacity=0.85)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-0.1, 1.1], title="Ort (willkürliche Einheit)", zeroline=False),
                      yaxis=dict(range=[-0.15, 0.7], title="Abstand", zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_traces(labels, arrays, t0, width, colors=None, spike_times=None, height=None, normalise=True):
    """Gestapelte Spuren eines Zeitfensters [t0, t0 + width) in ms (Abtastrate 10 kHz); optional Markierungen der wahren Spitzen je Zeile."""
    fs = C.SAMPLE_RATE
    lo, hi = int(t0 * fs / 1000), int((t0 + width) * fs / 1000)
    fig = go.Figure()
    n = len(arrays)
    scale_all = max(float(np.abs(a).max()) for a in arrays) if not normalise else None
    for r, (label, y) in enumerate(zip(labels, arrays)):
        seg = y[lo:hi]
        scale = float(np.abs(y).max()) if normalise else scale_all
        offset = (n - 1 - r) * 1.3
        color = colors[r] if colors else BLUE
        fig.add_trace(go.Scatter(x=np.arange(lo, hi) * 1000.0 / fs, y=offset + seg / max(scale, 1e-12), mode="lines", line=dict(color=color, width=1.2), name=label, hoverinfo="skip"))
        if spike_times is not None and r < len(spike_times):
            marks = [t for t in spike_times[r] if lo <= t < hi]
            if marks:
                fig.add_trace(go.Scatter(x=np.array(marks) * 1000.0 / fs, y=[offset + 0.75] * len(marks), mode="markers", marker=dict(symbol="triangle-down", size=7, color=color), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height or max(180, 42 * n + 60), margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis=dict(title="Zeit [ms]"), yaxis=dict(tickmode="array", tickvals=[(n - 1 - r) * 1.3 for r in range(n)], ticktext=list(labels), zeroline=False))
    return lock_axes(fig)


def build_corr_heatmap(cm, row_labels, col_labels, title):
    """|Korrelation| wahre Quelle (Zeile) gegen Schätzung (Spalte): ein sauberes Bild hat in jeder Zeile und Spalte genau einen hellen Eintrag."""
    fig = go.Figure(go.Heatmap(z=cm, x=col_labels, y=row_labels, zmin=0, zmax=1, colorscale="Blues", text=np.round(cm, 2), texttemplate="%{text}", showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=60 + 40 * len(row_labels), margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_autocorrelation(labels, acf, colors, lags=()):
    """Autokorrelationsfunktion je Quelle über die Verzögerung τ (Abtastwerte, logarithmisch); senkrechte Linien: die gewählten τ von SOBI. Quellen mit verschiedener Kurve lassen sich trennen."""
    fig = go.Figure()
    taus = np.arange(acf.shape[1])
    for label, y, color in zip(labels, acf, colors):
        fig.add_trace(go.Scatter(x=taus[1:], y=y[1:], mode="lines", line=dict(color=color, width=2), name=label, hoverinfo="skip"))
    for tau in lags:
        if 1 <= tau < acf.shape[1]:
            fig.add_vline(x=tau, line=dict(color=GRAY, dash="dot", width=1))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Verzögerung τ [Abtastwerte, 0.1 ms]", type="log"), yaxis=dict(title="Autokorrelation", range=[-0.5, 1.05]),
                      legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)


def build_spectrum(labels, freqs, spec, colors):
    """Leistungsspektrum je Quelle (normiert auf gleiche Gesamtleistung), logarithmische Achsen: farbige Quellen haben unterschiedliche Formen, Spitzen sind fast weiß."""
    fig = go.Figure()
    for label, y, color in zip(labels, spec, colors):
        fig.add_trace(go.Scatter(x=freqs[1:], y=y[1:] / max(y[1:].sum(), 1e-300), mode="lines", line=dict(color=color, width=2), name=label, hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Frequenz [Hz]", type="log"), yaxis=dict(title="Leistung (normiert)", type="log", range=[-7, -0.5]),
                      legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)


def build_covariances(before, after, lags):
    """Verzögerte Kovarianzen R_τ der weißen Daten für drei Verzögerungen (erste, mittlere, letzte): oben vor der gemeinsamen Rotation, unten (falls `after` gegeben) danach - fast diagonal."""
    picks = sorted({0, len(lags) // 2, len(lags) - 1})
    two = after is not None
    titles = [f"τ = {lags[j]}" for j in picks] * (2 if two else 1)
    fig = make_subplots(rows=2 if two else 1, cols=len(picks), subplot_titles=titles, vertical_spacing=0.14, horizontal_spacing=0.05)
    lim = float(max(np.abs(before[picks]).max(), np.abs(after[picks]).max() if two else 0.0, 1e-9))
    for c, j in enumerate(picks, start=1):
        for r, data in ((1, before[j]), (2, after[j] if two else None)):
            if data is None:
                continue
            fig.add_trace(go.Heatmap(z=data, zmin=-lim, zmax=lim, colorscale="RdBu", showscale=False, hoverinfo="skip", text=np.round(data, 2), texttemplate="%{text}" if data.shape[0] <= 6 else None), row=r, col=c)
            fig.update_yaxes(autorange="reversed", row=r, col=c, showticklabels=False)
            fig.update_xaxes(showticklabels=False, row=r, col=c)
    if two:
        fig.add_annotation(text="vor der Rotation", xref="paper", yref="paper", x=-0.06, y=0.8, textangle=-90, showarrow=False, font=dict(size=12))
        fig.add_annotation(text="nach der Rotation", xref="paper", yref="paper", x=-0.06, y=0.2, textangle=-90, showarrow=False, font=dict(size=12))
    fig.update_layout(height=520 if two else 300, margin=dict(l=60 if two else 10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_diagonalization(history, corr_curve):
    """Links: Anteil der Energie außerhalb der Diagonalen nach jedem Jacobi-Sweep (logarithmisch); rechts: Korrelation der Neuronen mit den Schätzungen nach jedem Sweep (0 = nur weißen)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Nicht-Diagonalität der R_τ", "Korrelation der Neuronen"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=np.arange(len(history)), y=np.maximum(history, 1e-16), mode="lines+markers", line=dict(color=BLUE, width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(corr_curve)), y=corr_curve, mode="lines+markers", line=dict(color=GREEN, width=2), hoverinfo="skip"), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    fig.update_xaxes(title="Jacobi-Sweep", row=1, col=1)
    fig.update_xaxes(title="Jacobi-Sweep", row=1, col=2)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return lock_axes(fig)


def build_method_bars(methods):
    """Neuronen-Korrelation, Spitzen-F1 und Trennschärfe der Hintergrundquellen für PCA, ICA, AMUSE und SOBI (gestrichelt: Zufalls-Niveau eines gefundenen Gauß-Paares)."""
    fig = go.Figure()
    has_bg = not np.isnan(methods["ica"].bg_margin)
    cats = ["Korrelation der Neuronen", "Spitzen-F1"] + (["Trennschärfe Hintergrund"] if has_bg else [])
    for name in ("pca", "ica", "amuse", "sobi"):
        m = methods[name]
        y = [m.neuron_corr, m.f1] + ([m.bg_margin] if has_bg else [])
        fig.add_trace(go.Bar(x=cats, y=y, name=METHOD_NAMES[name], marker_color=METHOD_COLORS[name], text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip"))
    if has_bg:
        fig.add_hline(y=CHANCE_PAIR_MARGIN, line=dict(color=RED, dash="dash", width=1), annotation_text="Zufall (Paar)", annotation_position="top left", annotation_font_size=10)
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.15))
    return lock_axes(fig)


def build_sweep(rows, xlabel, current=None, log=False, categorical=False):
    """Zwei Felder: Korrelation der Neuronen (links) und Trennschärfe der Hintergrundquellen (rechts, mit Zufalls-Niveau) für ICA, SOBI und AMUSE, Mittel und Streuung über die Sweep-Datensätze."""
    has_bg = any(not np.isnan(r["ica_bg"]) for r in rows)
    cols = 2 if has_bg else 1
    fig = make_subplots(rows=1, cols=cols, subplot_titles=("Korrelation der Neuronen",) + (("Trennschärfe der Hintergrundquellen",) if has_bg else ()), horizontal_spacing=0.1)
    xs = [r["x"] for r in rows]
    xs_plot = [str(x) for x in xs] if categorical else xs
    for col, key in enumerate(("neuron", "bg")[:cols], start=1):
        for method, name, color in (("ica", "ICA", ORANGE), ("sobi", "SOBI", BLUE), ("amuse", "AMUSE", PURPLE)):
            y = np.array([r[f"{method}_{key}"] for r in rows])
            sd = np.array([r[f"{method}_{key}_std"] for r in rows])
            if not categorical:
                fig.add_trace(go.Scatter(x=xs_plot + xs_plot[::-1], y=list(y + sd) + list(y - sd)[::-1], fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(x=xs_plot, y=y, mode="lines+markers", name=name, line=dict(color=color, width=2), showlegend=(col == 1), hoverinfo="skip",
                                     error_y=dict(type="data", array=sd, visible=True) if categorical else None), row=1, col=col)
        fig.update_xaxes(title=xlabel, type="log" if log else ("category" if categorical else "linear"), row=1, col=col)
        fig.update_yaxes(range=[0, 1.05], row=1, col=col)
    if has_bg:
        fig.add_hline(y=CHANCE_PAIR_MARGIN, line=dict(color=RED, dash="dash", width=1), row=1, col=2)
        fig.add_hline(y=BG_SEPARATED, line=dict(color=GREEN, dash="dot", width=1), row=1, col=2)
    if current is not None and not categorical:
        for col in range(1, cols + 1):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_scenes(rows):
    """Wer trennt was: Trennschärfe der Hintergrundquellen je Szene für ICA, SOBI (gewählte Verzögerungen) und SOBI geometrisch; Balken = Mittel, Fehlerbalken = Spanne über die Datensätze."""
    rows = [r for r in rows if not np.isnan(r["ica_bg"])]
    labels = [r["scene"].replace(", ", ",<br>").replace("Zwei ", "2 ") for r in rows]
    fig = go.Figure()
    for name in ("ica", "sobi", "sobi_geometric"):
        y = [r[f"{name}_bg"] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=METHOD_NAMES[name], marker_color=METHOD_COLORS[name], text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip",
                             error_y=dict(type="data", symmetric=False, array=[r[f"{name}_bg_max"] - r[f"{name}_bg"] for r in rows], arrayminus=[r[f"{name}_bg"] - r[f"{name}_bg_min"] for r in rows])))
    fig.add_hline(y=CHANCE_PAIR_MARGIN, line=dict(color=RED, dash="dash", width=1), annotation_text="Zufall (Paar)", annotation_position="top left", annotation_font_size=10)
    fig.update_layout(height=380, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Trennschärfe der Hintergrundquellen", range=[0, 1.2]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_lag_table(rows):
    """Verzögerungs-Mengen im Vergleich: SOBI-Neuronen-Korrelation und Trennschärfe der Hintergrundquellen (AMUSE mit dem kleinsten τ der Menge gestrichelt)."""
    labels = [C.LAG_SET_LABELS[r["lag_set"]] for r in rows]
    fig = go.Figure()
    for key, name, color in (("neuron", "SOBI: Korrelation der Neuronen", BLUE), ("bg", "SOBI: Trennschärfe der Hintergrundquellen", GREEN), ("amuse_bg", "AMUSE: Trennschärfe der Hintergrundquellen", PURPLE)):
        y = [r[key] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=name, marker_color=color, text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip"))
    fig.add_hline(y=CHANCE_PAIR_MARGIN, line=dict(color=RED, dash="dash", width=1), annotation_text="Zufall (Paar)", annotation_position="top left", annotation_font_size=10)
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.2]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)
