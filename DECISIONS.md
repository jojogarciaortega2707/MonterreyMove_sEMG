# DECISIONS — MONTERREY MOVE

Each decision has a date. A decision is not edited after the date: a change is a new, dated entry.

## 2026-11-20 — Frozen model for the sealed test

Decided before recording any test session. Nothing below changes until `code/07_test.py` has run once.

**Training data:** the 11 sessions in `data/raw/` (MM-0001, 2026-11-10). 1,070 usable seconds: 523 rest and 547 contraction (`QUALITY-REPORT.md`).

**Primary method: `LogReg_todas`.** Standardised logistic regression, C = 1, on 7 features of the same 1-s window: log of `promedio_uv`, `maximo_uv`, `minimo_uv` and `desviacion_uv`; max/mean; min/mean; SD/mean. Only this method answers "does the model work?". The other three are comparisons:

| Method | Frozen value |
|---|---|
| `LogReg_sin_amplitud` | logistic regression on max/mean, min/mean and SD/mean only |
| `Umbral_k_x_base` | contraction if `promedio_uv` > **2.425** × base. The base is the P20 of `promedio_uv` in the CALIBRACION phase of the same session, from second 2 on. Training optimum: 1.75–3.10 |
| `Umbral_absoluto` | contraction if `promedio_uv` > **114.5 µV**. Training optimum: 95–134 µV |

modelo_congelado.pkl sha256: `6c844db8177e4741add2ece77a72387025bacae4c85b982359ec77aa8ebd7094`

The coefficients and versions are in `models/modelo_congelado.json`. `code/07_test.py` stops if the hash of the model file does not match the line above.

**Quality rules:** R1 and R2 exactly as in `code/02_calidad_senal.py` (runs of ≥ 10 identical samples count as missing; reject a second with > 1 % missing; reject a second with ≥ 3 samples at the session extreme). The first 2 s of each session are left out.

**Test labels [DESIGN]:** from the calibration-script instructions (`PROTOCOL.md`, section 6), not from amplitude. The first second of each block is left out (reaction time, and ±1 s error when the start is noted by hand):
- rest: script seconds 1–6, 13–18, 25–30 (18 s per session);
- contraction: script seconds 8–11, 20–23, 32–35 (12 s per session).

Game seconds are not evaluated: their only label would be the threshold again.

**Test set:** 6 new sessions of MM-0001 on 3 different days, 2 per day. The electrodes are re-placed each day as in `PROTOCOL.md`. The files go to `data/test/raw/` and the script start goes to `data/test/guion_inicio.csv`. Nobody opens or plots these files before the test runs.

**Metrics reported:** sensitivity, specificity and F1 per session, per day and over all test seconds, with the confusion matrix. All four methods are reported, whatever the result.

**Success criterion:** [TO COMPLETE before recording the first test session.]

## Other decisions

- Features, model and route: [TO COMPLETE]
- Rationale of each member: `rationale/` [TO COMPLETE]
