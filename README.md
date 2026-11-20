# MonterreyMove_sEMG
M6 Machine Learning and IoT apps. Project MONTERREY MOVE: tell a contraction of the semitendinosus from rest with one surface EMG (sEMG) signal. The signal drives a biofeedback rehabilitation game for people with a transtibial amputation and for post-operative patients.

Project rules:
- The physiotherapist prescribes. The system measures and records.
- The clinical condition never enters the game logic.
- No value is invented. Every assumption is marked **[DESIGN]**. Every value that is still missing is marked **[TO COMPLETE]**.

## The team — Team 4

| Member code | GitHub user | In charge of | AI tool |
|---|---|---|---|
| M01 | [TO COMPLETE] | [TO COMPLETE] | Claude Code |
| M02 | [TO COMPLETE] | [TO COMPLETE] | Claude Code |
| M03 | [TO COMPLETE] | [TO COMPLETE] | Claude Code |
| M04 | [TO COMPLETE] | [TO COMPLETE] | Claude Code |

Every member has at least one commit. See the commit list of this repository.

We measure one healthy team member. We do not say which one, for privacy. In the data that person has the code **MM-0001**, and the physiotherapist has the code **FT-01**.

## The six parts

### 1. Data acquisition

**11 recordings, 125 game trials, 33 calibration trials, 11 sessions, 1 person.**

- **Sensor:** one sEMG channel on the **right semitendinosus**.
  - Bipolar pre-gelled Ag/AgCl electrodes, 2 cm apart.
  - Placed at the midpoint of the line between the ischial tuberosity and the knee (SENIAM), aligned with the fibres.
  - Reference electrode on the ankle.
- **Analog front-end:**
  - AD8220 → OPA4197 → OPA197. Total gain 142.8.
  - Band-pass 20–477 Hz (Sallen-Key, 8th order) and a Twin-T notch at 60 Hz (8th order).
- **Sampling:** the internal 12-bit ADC of an ESP32 samples the signal at **2000 Hz**. We measured this from the data: 2,437,800 samples for 1,218 s. The hardware sheet says 1 kHz and must be corrected.
- **Resolution:** 5.6433 µV per count. Every stored value sits on this grid, which is 3.3 V / 4095 / 142.8, with the total gain of 142.8 set by Rg = 15.4 kΩ on the AD8220. Each session uses 396–687 distinct levels.
- **Cue:** there is no cue video. The person follows a timed calibration script and then a Unity biofeedback game.
- **Trial length:**
  - A calibration trial lasts 5 s.
  - A game trial is valid when the contraction is held for at least **1 s**. The median time between valid game trials is 3 s.

Evidence:
- `data/raw/`: 33 files, 3 per session.
  - `*_senal_cruda.npy`: raw signal, float32, µV.
  - `*_por_segundo.csv`: one row per second.
  - `*_resumen.json`: session summary.
  - The `.npy` and `.csv` files are cumulative: each one repeats all earlier sessions.
- `data/metadata.csv`: 11 rows, one per recording.
- `data/monterrey_move_ML_unido.csv`: 1,218 rows, only the new seconds of each session.
- `PROTOCOL.md`, sections 2–4.

### 2. Task

- **Calibration:** the person lies prone with the knee at 90° (`PRONO_RODILLA_90`) and contracts the hamstrings 3 times, following a script.
- **Game:** the person does a bilateral supine bridge (`BRIDGING`): hip extension with both feet on the mat. Knee angle: [TO COMPLETE]. This posture differs from the calibration posture.
- **Classes:** two, **contraction** and **rest**.

Evidence: `PROTOCOL.md`, section 6; the columns `exercise` and `reference_position` of `data/metadata.csv`; `figures/placement_S01.jpg` [TO COMPLETE].

### 3. Trial and cue timing

**Calibration script, 36 s:**

| Window | Seconds | What the person does | Label |
|---|---|---|---|
| Rest before | 0 to 6 | rest | REST |
| Trial 1 | 7 to 11 | contract | CONTRACTION |
| Rest between | 12 to 18 | rest | REST |
| Trial 2 | 19 to 23 | contract | CONTRACTION |
| Rest between | 24 to 30 | rest | REST |
| Trial 3 | 31 to 35 | contract | CONTRACTION |

**Game, 35.5–45.1 s (median 38.5 s):**
- 10–13 trials, one every 3 s (median).
- A trial is valid when it stays in the target zone (70–130 % of the calibration reference) for ≥ 1 s.
- The session ends after 10 valid trials.

One recording lasts 101 s (median; range 92–170 s). It includes the time before the script, the script and the game.

Evidence:
- `PROTOCOL.md`, section 6.
- `data/metadata.csv`, columns `calibration_s`, `game_duration_s_json`, `trials_total`, `hold_min_s` and `median_s_between_valid_trials`.
- `results/fig_promedio_por_sesion.png` (game phase shaded).
- `results/guion_calibracion.csv`.

Notes:
- **The start of the script is not logged.** Matching the signal against the script (`code/05_guion_calibracion.py`) places it at second 16–23 in 9 of 11 sessions, and at second 71 (S01) and 49 (S04) in the other 2. In S01 and S07 the match is ambiguous.
- `*_resumen.json` logs 10 valid trials per session, but the per-second CSV only reaches 9, because it loses the last second. We take the counts from the JSON.
- 12 of the 15 invalid trials are logged while the CSV column `fase` still says CALIBRACION. [TO COMPLETE: check in the game code whether `fase` changes late.]

### 4. Sessions

**1 person. 11 sessions on 1 day (2026-11-10), from 10:04:52 to 10:25:11. Electrodes replaced before every session.**

- The time in each file name is the save time, that is, the end of the session. It matches `fecha_hora` in the JSON.
- Before each session: skin cleaned with alcohol and dried, new electrodes, 2 min wait (team).
- Calibration repeatability: the CV of the 3 calibration trials is 0.6–21.7 %. The system flags S11 as low confidence (CV 21.7 % > 15 %).

Evidence: the columns `participant`, `session`, `date`, `start_time`, `file_saved`, `electrodes_replaced`, `calibration_cv_pct` and `calibration_low_confidence` of `data/metadata.csv`.

### 5. Cleaning

**Filters.** The full list, with types and orders, is in `PROTOCOL.md`, sections 3 and 5.
- **Analog:** band-pass 20–477 Hz (Sallen-Key, 8th order, Q 0.707) and a 60 Hz notch (the power line in Mexico; Twin-T, 8th order, ≈ −68.84 dB).
- **Digital, in the Python app** (causal, filter state carried between blocks):
  - Butterworth band-pass, order 4 (8 poles), 20–450 Hz.
  - Three 2nd-order IIR notch filters at 60, 120 and 180 Hz, Q = 30.
  - Moving RMS envelope over 150 ms (300 samples).
  - Filter delay 4.3 ms; whole system latency 130.7 ms.
- Both upper limits (477 and 450 Hz) are below half of our sampling rate (1000 Hz).
- In the raw signal, power at 59–61 Hz is 0.6–0.9 times the neighbouring band, as expected after the analog notch.

**Rejection rules [DESIGN].** They use neither the labels nor any model score. The unit is 1 s, the unit the model classifies.
- **R1 missing samples.** The raw signal has flat stretches: one value repeated 51 to 601 times, always in multiples of 50 samples (25 ms). They look like lost packets filled with the last value. A sample counts as missing if it is NaN/inf or belongs to a run of 10 or more identical samples. A second is rejected if more than 1 % of its samples are missing.
- **R2 saturation.** A second is rejected if it holds the extreme value of its session 3 or more times.
- **Start artefact.** The first 2 s of each session are rejected.

**Result:**
- Missing samples per session: 0.00–1.09 %. There are no NaN.
- R1/R2 rejects 22 of 1,225 raw seconds. 8 of them are in S01, seconds 163–170.
- The start artefact rejects 22 of 1,218 CSV seconds.
- 109 "ambiguous" seconds are left out of training.
- **1,070 s remain: 523 rest and 547 contraction.**
- The raw files are untouched. The scripts write new files.

**[DESIGN] Provisional labels:**
- Rest baseline = 20th percentile of `promedio_uv` of the session (43–61 µV).
- Contraction = 3× the baseline or more. Rest = 1.6× the baseline or less. In between = ambiguous.
- These labels depend on amplitude. The timed calibration script will give independent labels once its start is logged.

Evidence: `QUALITY-REPORT.md`, `code/02_calidad_senal.py`, `data/segundos_calidad.csv`, `results/calidad_senal.csv`, and the columns `raw_missing_pct` and `seconds_rejected_quality` of `data/metadata.csv`.

### 6. Fair split

- **Method:** leave-one-session-out. We train on 10 sessions and test on the remaining one, 11 times. Each test session has its own electrode placement. We never split rows at random, because neighbouring seconds are almost identical.
- **Inputs:** only `promedio_uv`, `maximo_uv`, `minimo_uv`, `desviacion_uv`, and ratios from the same 1-s window.
- **Forbidden inputs**, because they depend on the threshold or on the game result: `zona_dominante`, `pct_tiempo_*`, `promedio_pct_ref`, `repeticiones_*`, `puntaje_acumulado`, the thresholds, `REF_*`, `fase`, `segundo`, `id_sesion` and `base_estimada_uv`.

| Method | Sensitivity | Specificity | F1 (mean ± SD, 11 folds) |
|---|---|---|---|
| Logistic regression, 7 features | 1.000 | 1.000 | 1.000 ± 0.000 |
| Logistic regression, shape only (max/mean, min/mean, CV) | 0.817 | 0.877 | 0.848 ± 0.034 |
| Baseline: `promedio_uv` > k × calibration baseline (k = 2.40–2.45) | 1.000 | 1.000 | 1.000 |
| Baseline: fixed threshold (110–120 µV) | 1.000 | 1.000 | 1.000 |

**Limits:**
- F1 = 1.000 means that the model copies the amplitude rule that made the labels. It does not show that it detects real contraction.
- All 11 sessions come from 1 day. A test on other days is still missing.
- Everything is built and tested on 1 healthy person and 1 muscle. We do not claim that it works for other people or for people with an amputation.
- Over the day, the rest baseline rises from 43 to 61 µV and the calibration reference falls from 457 to 255 µV (`reference_uv`).
- The prototype uses the internal ESP32 ADC (≈ 9–10 effective bits), not the external MCP3208 of an earlier report.

Evidence: `code/03_clasificador.py`, `results/reporte_resultados.md`, `results/metricas_por_sesion.csv` (confusion matrix per session), `results/resumen_metodos.csv` and `results/predicciones_loso.csv`.

## What comes next

- Log the start of the calibration script, and label with it instead of amplitude.
- Record sessions on other days.
- Use 250 ms windows of the raw 2000 Hz signal, with these features: MAV, RMS, WL, ZC, SSC and MDF. `*_resumen.json` already logs the MDF per trial.

Our decisions about features, the model and the route are in DECISIONS.md [TO COMPLETE]. The rationale of each member is in `rationale/` [TO COMPLETE].

## Repository layout

```
data/raw/          33 original files (unchanged)
data/              monterrey_move_ML_unido.csv, segundos_calidad.csv, metadata.csv
code/              01_unir_csv → 02_calidad_senal → 03_clasificador → 04_metadata → 05_guion_calibracion
results/           figure, metrics, predictions, signal quality, calibration script fit
PROTOCOL.md        how we record
QUALITY-REPORT.md  cleaning rules and results
```

Run the scripts in order from the repository root. You need Python 3.12 with numpy, pandas, scikit-learn and matplotlib.

```
python code/01_unir_csv.py
python code/02_calidad_senal.py
python code/03_clasificador.py
python code/04_metadata.py
python code/05_guion_calibracion.py
```

## Where to search

- How to organise the files: Wilson and others, *Good enough practices in scientific computing*, 2017.
- What to report about a muscle recording: Merletti, *Standards for Reporting EMG Data*, 1999, and the newer checklist CEDE-Check.
- The steps from signal to model, and the fair split: Riascos Salas, Molinas and Lotte, *Machine learning methods for BCI*, 2024. It is about brain signals, and the same steps apply to muscle signals.
- Features and models: the scikit-learn user guide. Models for a microcontroller: emlearn.

## How we present it

Numbers, not adjectives. Seconds, counts and file names. Codes, never names: MM-0001, FT-01, M01–M04, S01–S11.

## How to repeat our recording

Read `PROTOCOL.md`. It is enough to record the same way without asking us. Items still open there:
- [TO COMPLETE] the knee angle in the bridge.
- [TO COMPLETE] logging the start of the calibration script.

## Versions

Two versions are marked on GitHub: **First version** and **Report 1**. Open "Releases" on the right of the main page to see them with their dates.
