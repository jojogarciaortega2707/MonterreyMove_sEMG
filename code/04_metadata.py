"""
MONTERREY MOVE - Paso 4: data/metadata.csv, una fila por recording (= sesion).

Fuentes: data/monterrey_move_ML_unido.csv (paso 1), data/raw/*_resumen.json y
results/calidad_senal.csv (paso 3). Las columnas que no estan en ningun archivo
(lado, electrodos, colocacion, cambio de electrodos) quedan vacias para captura manual.

Uso:  python code/04_metadata.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

df = pd.read_csv(ROOT / "data" / "monterrey_move_ML_unido.csv")
cal_path = ROOT / "results" / "calidad_senal.csv"
cal = pd.read_csv(cal_path).set_index("id_sesion") if cal_path.exists() else None

filas = []
for n, (sid, s) in enumerate(df.groupby("id_sesion", sort=False), start=1):
    s = s.reset_index(drop=True)
    js = json.loads((RAW / f"{sid}_resumen.json").read_text(encoding="utf-8"))
    inc = (s.repeticiones_validas.diff().fillna(0) > 0) & (s.fase == "JUEGO")
    seg_val = s.loc[inc, "segundo"].to_numpy()
    fila = dict(
        recording=f"R{n:02d}",
        session=f"S{n:02d}",
        participant=js["id_paciente"],
        physiotherapist=js["id_fisioterapeuta"],
        date=s.date.iloc[0],
        start_time=s.start_time.iloc[0],
        file_saved=js["fecha_hora"],
        exercise=js["ejercicio"],
        reference_position=js["posicion_referencia"],
        muscle="semitendinosus",
        duration_s_csv=len(s),
        calibration_s=int((s.fase == "CALIBRACION").sum()),
        game_s=int((s.fase == "JUEGO").sum()),
        game_duration_s_json=js["duracion_s"],
        calibration_trials=len(js["ensayos_uv"]),
        calibration_trials_uv=";".join(f"{v:.1f}" for v in js["ensayos_uv"]),
        calibration_cv_pct=js["cv_calibracion_pct"],
        calibration_low_confidence=js["calibracion_baja_confianza"],
        rest_base_uv=js["base_uv"],
        rest_base_sd_uv=js["sd_base_uv"],
        reference_uv=js["ref_uv"],
        activation_threshold_uv=js["umbral_activacion_uv"],
        release_threshold_uv=js["umbral_liberacion_uv"],
        target_low_uv=js["objetivo_bajo_uv"],
        target_high_uv=js["objetivo_alto_uv"],
        hold_min_s=js["t_sosten_minimo_s"],
        trials_total=js["repeticiones_totales"],
        trials_valid=js["repeticiones_validas"],
        trials_invalid=js["repeticiones_invalidas"],
        trials_valid_in_csv=int(s.repeticiones_validas.iloc[-1]),
        median_s_between_valid_trials=float(np.median(np.diff(seg_val))) if len(seg_val) > 1 else np.nan,
        score=js["puntaje_total"],
        score_max=js["puntaje_maximo_posible"],
        mdf_slope_hz_per_trial=js["pendiente_mdf_hz_por_rep"],
        label_base_uv=float(s.base_estimada_uv.iloc[0]),
        seconds_rest=int((s.etiqueta == "reposo").sum()),
        seconds_contraction=int((s.etiqueta == "contraccion").sum()),
        seconds_ambiguous=int((s.etiqueta == "ambiguo").sum()),
        seconds_rejected_start=int((s.etiqueta == "artefacto_inicio").sum()),
    )
    if cal is not None and sid in cal.index:
        fila.update(raw_samples=int(cal.loc[sid, "muestras_nuevas"]),
                    raw_duration_s=float(cal.loc[sid, "duracion_s"]),
                    raw_missing_pct=float(cal.loc[sid, "faltantes_pct"]),
                    seconds_rejected_quality=int(cal.loc[sid, "segundos_rechazados"]))
    # Reportado por el equipo (ver PROTOCOL.md). electrodes_replaced = recolocados antes de esta sesion.
    fila.update(side="right", electrode_type="surface, gel", electrode_spacing_cm=2, ground_site="ankle",
                electrodes_replaced="yes" if n > 1 else "first placement", notes="")
    filas.append(fila)

meta = pd.DataFrame(filas)
meta.to_csv(ROOT / "data" / "metadata.csv", index=False)
pd.set_option("display.width", 250)
print(meta[["recording", "start_time", "duration_s_csv", "calibration_s", "game_s", "game_duration_s_json",
            "calibration_trials", "calibration_cv_pct", "trials_total", "trials_valid", "trials_invalid",
            "trials_valid_in_csv", "median_s_between_valid_trials", "raw_duration_s"]].to_string(index=False))
print("\nTOTAL trials:", meta.trials_total.sum(), "valid:", meta.trials_valid.sum(),
      "invalid:", meta.trials_invalid.sum(), "calibration trials:", meta.calibration_trials.sum())
