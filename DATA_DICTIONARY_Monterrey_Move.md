# Data dictionary — MONTERREY MOVE (sEMG biofeedback)

Built from the real files of the project (11 sessions, patient MM-0001, 2026-11-10, exercise BRIDGING).
Language of column names is kept exactly as in the files (mixed Spanish/English).

## Files

| File | One row / element is… | Rows |
|---|---|---|
| `<id_sesion>_senal_cruda.npy` | one sample of the raw signal | ~2,000,000 (cumulative, see notes) |
| `<id_sesion>_por_segundo.csv` | one second of one session | 1 per second |
| `<id_sesion>_resumen.json` | one whole session | 1 |
| `monterrey_move_ML_unido.csv` | one second of one session (all sessions merged, new rows only) | 1,218 |
| `ventanas_250ms.csv` | one 250 ms window of the raw signal | thousands |
| `resultados_LOSO.csv` | one model evaluated on one held-out session | — |

`id_sesion` = `<patient>_<yyyymmdd>-<hhmmss>` (e.g. `MM-0001_20261110-100742`). It links every file.

## 1. Raw signal — `*_senal_cruda.npy`

| Element | Meaning | Unit / type |
|---|---|---|
| array of float32, 1-D | sEMG sample, already converted from ADC counts to electrode-referred microvolts with the offset removed: µV = (counts − offset) × (3.3/4095) / gain × 10⁶ | µV |
| position in the array | sample number; time = index / 2000 | fs = 2000 Hz |

Gain used: 142.8 (4.2 × 1 × 34), pending check against 108.8. Resolution ≈ 5.64 µV/count, 12-bit ESP32 ADC, 3.3 V.

## 2. Merged table — `monterrey_move_ML_unido.csv` (40 columns)

### 2.1 Session identity and participant (constant inside a session)

| Column | Meaning | Unit / allowed values | Source |
|---|---|---|---|
| id_sesion | session identifier | text | json `id_sesion` |
| date | day of the session | yyyy-mm-dd | json `fecha_hora` |
| start_time | start time | hh:mm:ss | json `fecha_hora` |
| pacient | participant code, never a name | `MM-0001` | json `id_paciente` |
| physiotherapist | therapist code | `FT-01` | json `id_fisioterapeuta` |
| gender | participant sex | — | **empty** (not recorded) |
| age | participant age | years | **empty** |
| hamstring_length_cm | hamstring length | cm | **empty** |
| hamstring_diameter_cm | hamstring diameter | cm | **empty** |
| side | limb side | left / right | **empty** |
| excercise | exercise (spelling as in the file) | `BRIDGING` | json `ejercicio` |
| Channel | acquisition channel | — | **empty** |
| electrode_type | electrode type | text | **empty** |
| electrode_spacing_cm | inter-electrode distance | cm | **empty** |
| ground_site | reference electrode site | text | **empty** |

### 2.2 Calibration and thresholds (constant inside a session)

| Column | Meaning | Unit | Source |
|---|---|---|---|
| REF_min_uV | lower edge of the **target band** (70 % of REF). Despite the name it is not REF itself | µV | json `objetivo_bajo_uv` |
| REF_max_uV | upper edge of the target band (130 % of REF) | µV | json `objetivo_alto_uv` |
| activation_threshold_uV | amplitude above which a repetition starts | µV | json `umbral_activacion_uv` |
| release_threshold_uV | amplitude below which a repetition ends (hysteresis) | µV | json `umbral_liberacion_uv` |
| base_estimada_uv | resting baseline estimated for this analysis | µV | json `base_uv` |

REF itself (median of valid calibration trials) is in the json as `ref_uv`; it is not a column of the merged table.

### 2.3 Session summary (constant inside a session)

| Column | Meaning | Unit | Source |
|---|---|---|---|
| session_average_uv | mean RMS amplitude of the session | µV | json `promedio_uv` |
| session_max_uV | maximum | µV | json `maximo_uv` |
| session_min_uV | minimum | µV | json `minimo_uv` |
| session_stdev_uV | standard deviation | µV | json `desviacion_uv` |
| sessionduration_sec | session duration | s | json `duracion_s` |

### 2.4 Per-second measurements (change every row)

| Column | Meaning | Unit / allowed values |
|---|---|---|
| segundo | second within the session, starting at 0 (renumbered after removing the repeated old rows) | s |
| fase | stage of the session | `CALIBRACION`, `JUEGO` |
| promedio_uv | mean of the 150 ms RMS envelope in that second | µV |
| maximo_uv | maximum in that second | µV |
| minimo_uv | minimum in that second | µV |
| desviacion_uv | standard deviation in that second | µV |
| promedio_pct_ref | promedio_uv as a percentage of REF | % of REF |
| zona_dominante | zone where the signal spent most of the second | `BAJA`, `TERAPEUTICA`, `ALTA` |
| pct_tiempo_objetivo | share of the second inside the target band | % |
| pct_tiempo_alto | share above the band | % |
| pct_tiempo_bajo | share below the band | % |
| repeticiones_validas | valid repetitions accumulated so far | count |
| repeticiones_invalidas | invalid repetitions accumulated so far | count |
| puntaje_acumulado | score accumulated so far | points |

### 2.5 Label added for ML

| Column | Meaning | Allowed values |
|---|---|---|
| etiqueta | state of the muscle, derived **from the signal** (not from cue timestamps, which were not logged) | `reposo`, `contraccion`, `ambiguo`, `artefacto_*` (artifact flag, check the exact text in the file) |

Caution: labels come from amplitude, so any amplitude-based classifier is partly circular. Cue timestamps are needed for independent labels.

## 3. Windowed features — `ventanas_250ms.csv`

Windows of 250 ms with a 125 ms step, computed from the raw `.npy`.

| Column | Meaning | Unit / values |
|---|---|---|
| id_sesion | session | text |
| t_s | window start time within the session | s |
| y | class: 0 = rest, 1 = contraction | 0 / 1 |
| base_uv | resting baseline of the session | µV |
| rms | root mean square of the window | µV |
| mav | mean absolute value | µV |
| wl | waveform length (sum of absolute differences) | µV·samples |
| zc | zero crossings | count |
| ssc | slope sign changes | count |
| mdf | median frequency | Hz |
| mnf | mean frequency | Hz |
| log_rms | log10 of rms | — |
| wl_rms | wl / rms (amplitude-free) | — |
| mav_rms | mav / rms (amplitude-free) | — |

Amplitude-free set: zc, ssc, mdf, mnf, wl_rms, mav_rms.

## 4. Model results — `resultados_LOSO.csv`

| Column | Meaning | Unit / values |
|---|---|---|
| conjunto | feature set used | `A_todas` (all features), others for amplitude-free set |
| modelo | classifier | e.g. LogReg |
| sesion_fuera | session left out for testing (leave-one-session-out) | id_sesion |
| n | windows in the held-out session | count |
| sensibilidad | true-positive rate for contraction | 0–1 |
| especificidad | true-negative rate for rest | 0–1 |
| f1 | F1 score | 0–1 |

Only one healthy participant: results show consistency across sessions, not across people.

## 5. Session summary json (extra fields not in the merged table)

`ref_uv` (REF, µV), `sd_base_uv`, `ensayos_uv` (calibration trials), `cv_calibracion_pct`, `calibracion_baja_confianza`, `calibracion_provisional`, `avisos_calibracion`, `p_bajo_pct` / `p_alto_pct` (70 / 130 %), `t_sosten_minimo_s`, `maximo_pct_ref`, `tiempo_objetivo_s` / `tiempo_alto_s` / `tiempo_bajo_s`, `repeticiones_totales`, `pct_repeticiones_validas`, `calidad_promedio_pct`, `puntaje_total`, `puntaje_maximo_posible`, `mdf_por_repeticion_hz`, `pendiente_mdf_hz_por_rep`, `posicion_referencia` (`PRONO_RODILLA_90`), `condicion_paciente`, `notas_clinicas`, `comentario_fisioterapeuta`.

## Known issues

1. **Cumulative files:** every `_por_segundo.csv` and `.npy` repeats all earlier sessions; the merge keeps only the new rows/samples of each session.
2. **Empty columns:** gender, age, hamstring length/diameter, side, Channel, electrode type/spacing, ground site were never recorded.
3. **Gain:** 142.8 vs 108.8 unresolved; absolute µV may be scaled, %REF is not affected.
4. **Name mismatch:** `REF_min_uV` / `REF_max_uV` are the target-band limits, not REF.
