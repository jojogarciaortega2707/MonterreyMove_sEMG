# PROTOCOL — MONTERREY MOVE sEMG recording

Read this file to record the same way without asking us. Values reported by the team are marked (team). Values measured from the files are marked (data). Missing values are marked [TO COMPLETE].

## 1. Participant

- 1 healthy adult, code **MM-0001**. Identity withheld for privacy (team).
- Physiotherapist code: **FT-01** (data, `*_resumen.json`).
- The clinical condition is never used by the game logic.

## 2. Electrodes and placement

| Item | Value |
|---|---|
| Muscle | semitendinosus, **right** leg (team) |
| Electrodes | bipolar, pre-gelled Ag/AgCl surface electrodes; new electrodes at each placement (team) |
| Location on the muscle | midpoint (50 %) of the line between the ischial tuberosity and the knee (medial epicondyle of the tibia), as in SENIAM, aligned with the muscle fibres (team) |
| Inter-electrode distance | **2 cm**, centre to centre (team) |
| Reference (ground) electrode | **ankle** (team) |
| Skin preparation | alcohol, then let the skin dry; wait 2 min after placing the electrodes before recording (team) |
| Channels | 1 |
| Placement between sessions | electrodes replaced before every session (team) |

## 3. Analog front-end (team)

Signal chain: electrodes → pre-amplifier → band-pass → notch → post-amplifier → clamp with DC offset → ESP32 ADC.

| Stage | Part | Detail |
|---|---|---|
| Pre-amplification | AD8220 instrumentation amplifier | gain 4.2 (Rg = 15.4 kΩ; G = 1 + 49.4 kΩ / Rg = 4.21) |
| Filtering | OPA4197 quad op-amp | gain 1 |
| Post-amplification | OPA197 op-amp | gain 34 |
| **Total gain** | | **142.8** (4.2 × 1 × 34), checked on the board through Rg (team) |
| High-pass | Sallen-Key, 8th order, Q 0.707 | cut-off **20 Hz** |
| Low-pass | Sallen-Key, 8th order, Q 0.707 | cut-off **477 Hz** |
| Notch | Twin-T, 8th order, Q 0.707 | centre **60 Hz**, max. attenuation ≈ −68.84 dB |
| Clamp / level shift | diode voltage clamp | DC offset **+1 V** |

Analog bandwidth: 20–477 Hz.

**Corrected:** the hardware sheet listed the 20 Hz cut-off as "low pass" and 477 Hz as "high pass". For a 20–477 Hz band, the 20 Hz filter must be the high-pass and the 477 Hz filter the low-pass, so the table above uses that order.

**Check:**
- The stored signal is bipolar (−2,208 to +3,848 µV), so it is not rectified after conversion. The offset is removed in software (section 4).
- The hardware offset is +1 V (team). The software does not use a fixed value: it estimates the offset from 1.5 s of rest (section 4).
- The raw `.npy` files show no 60 Hz peak (power at 59–61 Hz is 0.6–0.9 times the 50–70 Hz band; `QUALITY-REPORT.md`). This agrees with the analog notch.

## 4. Digitisation

| Item | Value |
|---|---|
| Microcontroller | ESP32 |
| Sampling rate | **2000 Hz** (data). The new samples of each session divided by its CSV rows give 1,986–2,011 per second. For example, there are 213,000 samples in the 107 s between saving S01 and saving S02 |
| ADC | internal ESP32 ADC, 12 bits (4096 levels), 0–3.3 V (team) |
| Conversion | µV = (counts − offset) × (3.3 / 4095) / gain × 10⁶, with gain = 142.8. The offset is estimated from 1.5 s of rest at connection (team) |
| Resolution | **5.6433 µV per count** at the electrodes. Checked in the data: every value of the 11 `.npy` files sits on a grid of exactly 5.6433 µV. This confirms the 12-bit count, the 3.3/4095 scale and the gain of 142.8 used in the software, which matches the gain on the board (section 3) |
| Range used | 396–687 distinct levels per session; the signal spans 536–1,067 counts out of 4,096. The effective resolution of the internal ESP32 ADC is ≈ 9–10 bits (team) |
| Stored unit | µV, float32 (`*_senal_cruda.npy`), before any digital filter |

The hardware sheet says "1 kHz sampling rate". The stored data are at 2000 Hz. The sheet must be corrected.

**Limitation:** an earlier report described an external MCP3208 ADC. This prototype uses the internal ESP32 ADC, which is noisier and less linear: ≈ 9–10 effective bits.

The Nyquist frequency at 2000 Hz is 1000 Hz, above the 477 Hz analog low-pass.

## 5. Digital processing (Python app, live)

Causal, real-time chain at 2000 Hz. The filter state is carried from block to block, and there is no zero-phase filtering because the loop must be causal (team):

| Step | Value |
|---|---|
| 1. Band-pass | Butterworth, order 4, 20–450 Hz. A band-pass doubles the order of the prototype, so it has 8 poles and 9 coefficients per side |
| 2. Notch | three 2nd-order IIR notch filters at 60, 120 and 180 Hz, Q = 30 (bandwidths 2, 4 and 6 Hz) |
| 3. Envelope | moving RMS over 150 ms (300 samples). This is envelope detection, not a filter |
| Output to Unity | the envelope, compared with the activation and release thresholds |
| Delay | filters 4.3 ms; whole system 130.7 ms (team) |

The app derives these thresholds for each session from calibration (data, `*_resumen.json`):

| Value | Field | Range over 11 sessions |
|---|---|---|
| Rest baseline | `base_uv` | 43.2–61.0 µV |
| Reference contraction | `ref_uv`, chosen from the 3 calibration trials | 255.0–588.8 µV |
| Target zone | `p_bajo_pct`–`p_alto_pct` of the reference | 70–130 % |
| Activation threshold | `umbral_activacion_uv` | 71.2–117.8 µV |
| Release threshold | `umbral_liberacion_uv` | 55.2–92.5 µV |
| Calibration quality | `cv_calibracion_pct` | 0.6–21.7 %. The app warns above 15 % |

## 6. Session procedure

1. Replace and place the electrodes (section 2).
2. Start the recording. The file is cumulative: it also contains all earlier sessions.
3. **Calibration script**, 36 s, prone with the knee at 90° (`PRONO_RODILLA_90`):

| Seconds | Instruction | Label |
|---|---|---|
| 0–6 | rest | REST |
| 7–11 | trial 1, contract | CONTRACTION |
| 12–18 | rest | REST |
| 19–23 | trial 2, contract | CONTRACTION |
| 24–30 | rest | REST |
| 31–35 | trial 3, contract | CONTRACTION |

4. **Game:** the bridge exercise (`BRIDGING`): bilateral supine bridge, with hip extension and both feet on the mat. This posture differs from the calibration posture (prone, knee at 90°). Knee angle during the bridge: [TO COMPLETE]. A trial is valid when the contraction stays in the target zone for at least **1.0 s** (`t_sosten_minimo_s`). The session ends after **10 valid trials**. Measured (data): 10–13 trials per session, a median of 3 s between valid trials, and 35.5–45.1 s of game.
5. Save. The app writes `<id>_senal_cruda.npy`, `<id>_por_segundo.csv` and `<id>_resumen.json`. The time in the file name is the save time, that is, the end of the session.

**Known gaps:**
- The start of the calibration script is not logged. `code/05_guion_calibracion.py` finds it by matching the signal: it starts at second 16–23 of the session in 9 of 11 sessions, and at second 71 (S01) and 49 (S04) in the other 2. In S01 and S07 the match is ambiguous.
- The calibration phase in the CSV lasts 58–135 s, longer than the 36 s script.
- [TO COMPLETE: log the start of the script and of each instruction in the CSV, so the script can give labels that do not depend on amplitude.]

## 7. Files produced per session

| File | Content |
|---|---|
| `*_senal_cruda.npy` | raw signal, float32, µV, 2000 Hz. Cumulative |
| `*_por_segundo.csv` | one row per second: amplitude statistics, phase, zone, trial counters, score. Cumulative; loses the last second |
| `*_resumen.json` | calibration, thresholds, trial counts, score, MDF per trial |
