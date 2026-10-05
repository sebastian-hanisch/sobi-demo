# SOBI – Quellen trennen über die zeitliche Struktur – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-sobi-demo.streamlit.app/)**

Zweites Stück der **Quellentrennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **SOBI** (Second-Order Blind Identification) – an einem wachsenden Beispiel, mit der **ICA** und **AMUSE** als Vergleich.
Vehikel: dasselbe **Mehrelektroden-Array** wie in [ica-demo](../ica-demo) (Neuronen, Wellenformen, Feuerraten und Mischung sind dort übernommen, per Test gegen eingefrorene Werte geprüft), jetzt ohne Laufzeitverzögerung und mit **ausgebautem Hintergrund**:
zwei farbige Quellen, deren Autokorrelation (bzw. Rhythmusfrequenz) sich um einen einstellbaren **Spektren-Abstand** unterscheidet.

**Einordnung in die Reihe (die Kanten des Graphen):** SOBI behebt die Schwäche der ICA-Demo – zwei Gauß'sche Quellen sind für die ICA nicht trennbar –, indem es statt der Verteilungsform die **Autokorrelation** nutzt. Die Kante ist ein **Ast direkt nach der ICA**;
die Demo misst, dass die Schwäche wandert, statt zu verschwinden: gleiche Spektren, falsche Verzögerungen und Rauschen treffen SOBI.
```
ica-demo → sobi-demo          (zeitliche Struktur statt Nicht-Gaußianität)
         → NMF                (Nichtnegativität statt Unabhängigkeit: nmf-demo)
         → Sparse Component Analysis   (mehr Quellen als Sensoren: sca-demo)
         → Spike-Sorting-Zweig         (Standard-Pipeline, Vorlagenabgleich, Verzögerungsgraph: spike-sorting-demo, template-matching-demo, delay-graph-demo)
```
Gebaut sind inzwischen SCA, NMF und der Spike-Sorting-Zweig – die Linie ist vollständig. Die Demo selbst markiert nur, welche Annahme die Nachfolger jeweils lockern.

| Frage | Ergebnis (4 Neuronen, 6 Elektroden, Rauschen 0.05, 20000 Abtastwerte; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Zwei Gauß'sche Quellen, verschiedene Autokorrelation (0.95 / 0.5) | ✅ SOBI **Trennschärfe 0.82** (in jedem Datensatz 0.80–0.84), ICA im Mittel **0.41** (0.10–0.73, in 2 von 5 Datensätzen zufällig getrennt); Zufallsniveau eines Paares 0.44; PCA 0.15 |
| Spektren-Abstand (Gauß-AR) | ✅ 0 → 0.43 (Zufall), 0.02 → 0.22, 0.05 → 0.72, 0.1 → 0.83, 0.2/0.3 → 0.87, 0.45 → 0.82 |
| Gleiche Autokorrelation | ❌ SOBI 0.43 (0.30–0.50, Zufallsniveau), ICA 0.27 (0.04–0.72, je nach Rechner bis 0.58). Bei 24 weiteren Datensätzen trennt die ICA in 7 zufällig, SOBI in 2 |
| Zwei Rhythmen (10 und 23 Hz), mittlere Verzögerungen (2–20) | ❌ SOBI **0.38**, ICA 0.76 – die Verzögerungen (0.2–2 ms) sind winzig gegen die Periode (100 ms) |
| Dieselben Rhythmen, geometrische Verzögerungen (1…256) | ✅ SOBI **0.94**, ICA 0.76; Preis: Neuronen 0.87 statt 0.90 |
| Nahe Rhythmen (10 und 11.4 Hz) | ❌ SOBI 0.42 (auch geometrisch 0.49), ICA 0.76 – die Verteilungsform (unter-Gauß'sch) trennt, wo das Spektrum es nicht kann |
| Verzögerungs-Mengen (Gauß-Paar) | Trennschärfe kurz (1–5) 0.63, mittel 0.82, lang (10–100) 0.80, geometrisch 0.68; Neuronen 0.89 / 0.90 / 0.82 / 0.88; AMUSE (ein τ) 0.16 / 0.33 / 0.77 / 0.16 |
| Neuronen (nur Spitzen) | ✅ überraschend: SOBI **0.978** gegen ICA 0.981, AMUSE 0.931 – verschieden breite Spitzen geben verschiedene Autokorrelationen; Spitzen-F1 1.00 |
| Rauschen | ❌ Neuronen-Korrelation SOBI / ICA: 0.05 → 0.90 / 0.92, 0.1 → 0.73 / 0.83, 0.2 → 0.48 / 0.72, 0.4 → 0.33 / 0.59 – SOBI ist überall rauschempfindlicher |
| Zu wenige Elektroden (6 Quellen) | ❌ 5 Elektroden: ICA 0.82, SOBI 0.55; 4: 0.63 / 0.46; 3: 0.37 / 0.20 |
| Rechenzeit | ✅ 0.06 s für SOBI bei T = 40000, 8 Elektroden, 40 Verzögerungen; gesamte Analyse (vier Verfahren) ≈ 0.1–0.2 s |

## Was die Demo zeigt

1. **SOBI in Aktion** (Schritt-Slider + Abspielen, Zeitfenster-Regler): **Quellen** (Spuren, **Autokorrelation** und **Leistungsspektrum** je Quelle, Ort von Neuronen und Elektroden) → **Mischung** → **Kovarianzen** (verzögerte Kovarianzen R_τ der weißen Daten: nicht diagonal) →
   **Diagonalisieren** (dieselben R_τ vor und nach der gemeinsamen Rotation, Jacobi-Verlauf: Nicht-Diagonalität und Korrelation der Neuronen je Sweep) → **Ergebnis** (wahre Quellen gegen SOBI-Schätzungen).
2. **Wer trennt was: SOBI, ICA, AMUSE und PCA auf denselben Daten:** Neuronen-Korrelation, Spitzen-F1, **Trennschärfe der Hintergrundquellen** (mit Zufallsniveau), Jacobi-Sweeps; die Spurgruppen (wahr, PCA, ICA, SOBI); Zuordnungsmatrizen; Urteil
   (Codes: zu wenige Elektroden → nicht konvergiert → Rauschen → SOBI vorn → beide trennen → Verzögerungen passen nicht → ICA trennt, SOBI nicht → beide scheitern → neutral).
3. **🔧 Welche Verzögerungen τ?** Vier benannte Mengen (kurz, mittel, lang, geometrisch) im Vergleich, mit AMUSE (kleinstes τ) daneben.
4. **📐 Sweeps** über Spektren-Abstand, Rauschen, Elektrodenzahl und Verzögerungs-Menge (feste Datensätze ab 100000, Streuung, aktueller Wert markiert; Zufallsniveau und Trennungsschwelle eingezeichnet).
5. **🧩 Wer trennt was:** fünf Szenen (nur Neuronen, Gauß-Paar mit verschiedener / gleicher Autokorrelation, Rhythmen, nahe Rhythmen) für ICA, SOBI und SOBI geometrisch, mit Spanne über die Datensätze (Experiment auf Abruf).
6. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (ICA, geometrische Mengen, Sparse Component Analysis, NMF, Spike-Sorting-Zweig).

Regler: Neuronen (2–5), Elektroden (1–8), Hintergrundquellen (0–2) mit Art (Gauß-AR oder Rhythmus; bei 0 ausgeblendet) und **Spektren-Abstand** (nur bei zwei Quellen sichtbar; beide Werte bleiben beim Ausblenden erhalten), Rauschen, Länge, Verzögerungs-Menge, ICA-Kontrastfunktion und -Start.

## Messwerte der Presets (sie prüfen sich mit weiten Bändern selbst)

| Preset (Seed) | Neuronen ICA / SOBI | Trennschärfe ICA / SOBI | Urteil |
|---|---|---|---|
| Zwei Gauß-Quellen, verschiedene Autokorrelation (7) | 0.92 / 0.90 | 0.07 / 0.81 | SOBI vorn |
| Gleiche Autokorrelation (11) | 0.92 / 0.89 | 0.29 / 0.40 | beide scheitern |
| Zwei Rhythmen, Verzögerungen zu kurz (7) | 0.92 / 0.90 | 0.77 / 0.37 | Verzögerungen passen nicht (geometrisch: 0.94) |
| Zwei Rhythmen, geometrische Verzögerungen (7) | 0.92 / 0.86 | 0.77 / 0.94 | beide trennen |
| Starkes Rauschen 0.4 (7) | 0.60 / 0.33 | 0.14 / 0.45 | Rauschen |
| Nur Neuronen (7) | 0.981 / 0.978 | – | neutral (AMUSE 0.931) |

## Modell und Verfahren

- **Szenario** (`sobi_scenario.py`): wie in ica-demo (10 kHz, Neuronen mit biphasischer Wellenform und Poisson-artigem Feuern, Mischung ∝ 1/(d² + ε), Rauschen relativ zum Neuronen-Signal, eigener Zufallsstrom je Quelle), ohne Verzögerung. Hintergrund: AR(1) mit Koeffizient 0.95 − j·Abstand
  (Abstand 0.45 = die Quellen der ICA-Demo) oder Sinus mit 10 + j·Abstand·28.9 Hz; Abstand 0 = gleiches Spektrum. Der Hintergrund wirkt flächig auf alle Elektroden (gleich stark, Rampe).
- **SOBI** (`sobi_algorithm.py`, numpy von Grund auf): Weißen, symmetrisierte verzögerte Kovarianzen R_τ, **gemeinsame näherungsweise Diagonalisierung per Jacobi-Rotationen** (Cardoso & Souloumiac; Winkel aus der 2×2-Matrix G = Σ g gᵀ), Verlauf je Sweep; **AMUSE** als Eigenzerlegung von R_τ bei einem τ.
  Vergleichsverfahren **ICA** und Weißen wortgleich aus ica-demo (`sobi_ica.py`, gegen eingefrorene Werte geprüft).
- **Auswertung** (`sobi_evaluation.py`): Zuordnung (Bitmasken-DP), Neuronen-Korrelation, Spitzen-F1, Amari-Index (aus ica-demo); neu die **Trennschärfe der Hintergrundquellen** (c₁ − c₂)/(c₁ + c₂) über alle Schätzungen mit dem berechneten Zufallsniveau ln√2/(π/4) ≈ 0.441 für ein gefundenes,
  gleichverteilt gedrehtes Paar (per Simulation geprüft); Sweeps, Verzögerungs-Tabelle, Szenen-Tabelle; Urteil mit Referenzläufen (SOBI mit geometrischen Verzögerungen, ICA ohne Rauschen).

## Was nicht funktioniert hat / Grenzen

- **Die ICA trennt Gauß'sche Paare in etwa jedem dritten Datensatz zufällig.** Deshalb sagt die App nie "die ICA scheitert", sondern nennt Mittel **und** Spanne; die Presets verwenden einen Seed, an dem das typische Ergebnis auftritt (Seed 11 bei gleicher Autokorrelation:
  Seed 7 und 100000 trennen dort zufällig), und die Erfolgsmeldung nennt die Zahl der Zufallserfolge (2 von 5 bzw. 7 von 24). Die Tests prüfen Mittel und Spannen, nicht einzelne Datensätze.
- **Meine erste Kennzahl war falsch:** der Amari-Index des 2×2-Hintergrundblocks meldete die PCA mit 0.055 als "fast getrennt", weil die Hintergrundquellen die stärksten Hauptkomponenten belegen und der Block Verwechslungen mit den Neuronen ignoriert. Ersetzt durch die Trennschärfe über *alle* Schätzungen.
  Ebenso ist das Zufallsniveau des SIR eines Paares kein kleiner Wert (~ 10 dB im Mittel ohne Rauschen), deshalb wird das Paar nicht mit SIR bewertet.
- **Die Erwartung "SOBI ist auf Spitzen deutlich schwächer" stimmt nicht:** verschieden breite Wellenformen geben den Neuronen verschiedene Autokorrelationen, SOBI kommt auf 0.978 gegen 0.981 der ICA. Schwächer ist SOBI erst bei Rauschen und bei zu wenigen Elektroden.
- **Sehr kleine Spektren-Abstände sind schlechter als gar keiner** (0.02 → 0.22, 0 → 0.43): die R_τ-Diagonalen liegen so nah beieinander, dass das Rauschen die Reihenfolge der Eigenwerte dominiert; die Rotation wird dann systematisch falsch, nicht zufällig.
- **Es gibt keine Verzögerungs-Menge für alle Quellen:** kurze τ sehen die Rhythmen nicht, lange τ verlieren die Spitzen (Neuronen 0.82 bei 10–100). Die geometrische Menge deckt beides ab, trennt das Gauß-Paar aber schlechter (0.68 gegen 0.82) und braucht mehr Sweeps (im Mittel 24 gegen 10).
  Verzögerungen werden deshalb als vier benannte Mengen angeboten statt als zwei Regler (Anzahl, Abstand) – so bleibt jede Aussage im Text ein Test.
- **Auf zwei Hintergrundquellen begrenzt:** mit drei Quellen und sechs Elektroden wäre die Mischung nicht umkehrbar (7 Quellen), und für Triples gibt es kein sauberes Zufallsniveau. **Laufzeitverzögerung entfällt** (Annahme der momentanen Mischung; siehe ica-demo).
- **Synthetische Daten:** die Spitzen haben feste Form, die Mischung ist exakt linear, das Rauschen weiß und Gauß'sch. **Die Quellenzahl wird als bekannt angenommen.**
- **Grenzen des Verfahrens:** SOBI nutzt nur Statistik zweiter Ordnung; bei Spitzen bleibt die Verteilungsform ungenutzt. Es gibt keinen Zufallsstart, aber die Wahl der Verzögerungen ist ein Hyperparameter mit großem Einfluss.
- **Die Spitzenerkennung des Spitzen-F1 (Kopie aus ica-demo) wandte die Mindesttiefe erst ab 10 gefundenen Spitzen an:** bei kürzeren Spuren blieben Rauschspitzen über 4 σ_MAD als Falschtreffer stehen, obwohl dieselbe Spur mit mehr Spitzen sie verworfen hätte. Jetzt gilt die Regel (30 % der typischen Tiefe, typische Tiefe = Median der höchstens 10 tiefsten Spitzen) auch dort, dann mit dem Median der gefundenen Spitzen. Gemessen über die 5 festen Sweep-Datensätze mit den Standard-Einstellungen: alle Sweeps, Tabellen und Presets sowie alle in dieser Datei genannten Zahlen bleiben unverändert. Als Test hinterlegt (`tests/test_detect_spikes_depth.py`).

## Verifikation

- Verzögerte Kovarianzen (Symmetrie, Handrechnung); Nicht-Diagonalität (Handinstanzen); **Jacobi-Diagonalisierung**: exakt diagonalisierbarer Stapel wird bis auf Reihenfolge und Vorzeichen erholt (Nicht-Diagonalität < 1e-20), monoton fallend, orthogonal,
  **Kreuzprüfung gegen `numpy.linalg.eigh`** für eine einzelne Matrix; rauschfreie farbige Gauß-Mischung wird von SOBI mit Korrelation > 0.99 getrennt, von AMUSE bei weit auseinanderliegenden Spektren; bei gleichem Spektrum bleibt das Paar unaufgelöst, der Unterraum und die dritte Quelle sind gefunden.
- Autokorrelation und Spektrum gegen die Theorie (AR(1): φ^τ, Sinus: cos(2πfτ)); Szenario bit-genau gegen eingefrorene Werte aus ica-demo; ICA-Kopie gegen eingefrorenen Wert (Toleranz 2e-3); Zufallsniveau der Trennschärfe per Simulation.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Spektren-Abstand, Rauschen, Elektroden, Verzögerungs-Mengen, Rhythmen, nahe Rhythmen, Nur-Neuronen-Werte, Zufallserfolge über 24 Datensätze, Preset-Hilfen); Verdict-Codes über mehrere Datensätze;
  alle 6 Presets in weiten Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt, Randgrößen, ausgeblendete Regler behalten ihren Wert, Fenster-Klemmung, Sweep-Optionen, Szenen-Experiment), Achsensperre und explizite Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 🔧 Verzögerungen, 📐 Sweeps, 🧩 Szenen, 🚧 Grenzen, Mathe |
| `sobi_algorithm.py` | verzögerte Kovarianzen, AMUSE, SOBI (Jacobi-Diagonalisierung), Autokorrelation, Spektrum |
| `sobi_ica.py` | Weißen und FastICA (wortgleich aus ica-demo) |
| `sobi_scenario.py`, `sobi_constants.py` | Mehrelektroden-Generator mit einstellbarem Spektren-Abstand, Konstanten, Presets |
| `sobi_evaluation.py` | Zuordnung, Kennzahlen, Trennschärfe, Sweeps, Szenen, Urteil |
| `sobi_presets.py`, `sobi_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Algorithmus, Kreuzprüfungen, Szenario, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Quellentrennung: von ICA bis Verzögerungsgraph](https://sebastianhanisch.net/konzepte-quellentrennung.html).
