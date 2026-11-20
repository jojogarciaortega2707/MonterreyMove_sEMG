"""
MONTERREY MOVE - Paso 1: unir los CSV acumulativos por segundo en un solo archivo.

Los archivos MM-0001_<fecha>-<hora>_por_segundo.csv son ACUMULATIVOS: cada uno repite
todas las filas de las sesiones anteriores y reescribe id_sesion y fase con los de la
sesion mas reciente. Aqui se conservan solo las filas nuevas de cada archivo, se verifica
que el prefijo coincida con el archivo anterior y `segundo` reinicia en 0 por sesion.

Agrega:
  - columnas de sesion. Las que existen en <id_sesion>_resumen.json se copian de ahi
    (ver JSON_A_COLUMNA); las demas quedan vacias para captura manual,
  - base_estimada_uv   [DISENO] percentil 20 de promedio_uv de la sesion (sin artefacto_inicio),
  - etiqueta           [DISENO] provisional, regla de amplitud relativa a la base:
        segundo < 2                       -> artefacto_inicio
        promedio_uv >= 3.0 x base         -> contraccion
        promedio_uv <= 1.6 x base         -> reposo
        intermedio                        -> ambiguo

Uso:  python code/01_unir_csv.py      (lee data/raw/, escribe data/monterrey_move_ML_unido.csv)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Parametros [DISENO] ---------------------------------------------------
SEG_ARTEFACTO = 2          # primeros segundos de cada sesion descartados
PCT_BASE = 20              # percentil de promedio_uv usado como base de reposo
K_CONTRACCION = 3.0        # >= K x base -> contraccion
K_REPOSO = 1.6             # <= K x base -> reposo

# Columna del CSV unido <- clave de <id_sesion>_resumen.json.
# [DISENO] REF_min_uV <- base_uv y REF_max_uV <- ref_uv (base de reposo y referencia de
# calibracion). session_* vienen de las estadisticas del JSON, que cubren la fase JUEGO
# (duracion_s del JSON = 35-45 s), no la sesion completa.
# Valores reportados por el equipo (iguales en las 11 sesiones; ver PROTOCOL.md).
REPORTADO_EQUIPO = {
    "side": "right",
    "Channel": 1,
    "electrode_type": "surface, gel",
    "electrode_spacing_cm": 2,
    "ground_site": "ankle",
}

JSON_A_COLUMNA = {
    "physiotherapist": "id_fisioterapeuta",
    "excercise": "ejercicio",
    "REF_min_uV": "base_uv",
    "REF_max_uV": "ref_uv",
    "activation_threshold_uV": "umbral_activacion_uv",
    "release_threshold_uV": "umbral_liberacion_uv",
    "session_average_uv": "promedio_uv",
    "session_max_uV": "maximo_uv",
    "session_min_uV": "minimo_uv",
    "session_stdev_uV": "desviacion_uv",
}

COLUMNAS_ORDEN = [
    "id_sesion", "date", "start_time", "pacient", "physiotherapist", "gender", "age",
    "hamstring_length_cm", "hamstring_diameter_cm", "side", "excercise", "Channel",
    "electrode_type", "electrode_spacing_cm", "ground_site", "REF_min_uV", "REF_max_uV",
    "activation_threshold_uV", "release_threshold_uV", "session_average_uv",
    "session_max_uV", "session_min_uV", "session_stdev_uV", "sessionduration_sec",
    "segundo", "fase", "promedio_uv", "maximo_uv", "minimo_uv", "desviacion_uv",
    "promedio_pct_ref", "zona_dominante", "pct_tiempo_objetivo", "pct_tiempo_alto",
    "pct_tiempo_bajo", "repeticiones_validas", "repeticiones_invalidas", "puntaje_acumulado",
    "etiqueta", "base_estimada_uv",
]
# Se capturan a mano: se dejan vacias (no inventar valores).
COLUMNAS_MANUALES = [
    "physiotherapist", "gender", "age", "hamstring_length_cm", "hamstring_diameter_cm",
    "side", "excercise", "Channel", "electrode_type", "electrode_spacing_cm", "ground_site",
    "REF_min_uV", "REF_max_uV", "activation_threshold_uV", "release_threshold_uV",
    "session_average_uv", "session_max_uV", "session_min_uV", "session_stdev_uV",
]
# Columnas que no dependen de la sesion "actual" (sirven para verificar el prefijo).
COLS_VERIFICAR = ["segundo", "promedio_uv", "maximo_uv", "minimo_uv", "desviacion_uv"]


def etiquetar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validos = df["segundo"] >= SEG_ARTEFACTO
    base = np.percentile(df.loc[validos, "promedio_uv"], PCT_BASE)
    df["base_estimada_uv"] = round(float(base), 2)
    ratio = df["promedio_uv"] / base
    et = np.where(ratio >= K_CONTRACCION, "contraccion",
                  np.where(ratio <= K_REPOSO, "reposo", "ambiguo"))
    et = np.where(~validos, "artefacto_inicio", et)
    df["etiqueta"] = et
    return df


ROOT = Path(__file__).resolve().parents[1]


def main(carpeta: Path, salida: Path) -> None:
    archivos = sorted(carpeta.glob("MM-*_por_segundo.csv"))
    if not archivos:
        sys.exit(f"No se encontraron archivos *_por_segundo.csv en {carpeta}")

    previo = None
    sesiones = []
    for f in archivos:
        df = pd.read_csv(f)
        n_prev = 0 if previo is None else len(previo)
        if previo is not None:
            pref = df.iloc[:n_prev][COLS_VERIFICAR].reset_index(drop=True)
            ok = np.allclose(pref.values, previo[COLS_VERIFICAR].values, equal_nan=True)
            if not ok:
                sys.exit(f"ERROR: {f.name} no repite exactamente las filas de {archivos[len(sesiones)-1].name}")
        nuevas = df.iloc[n_prev:].copy()
        id_ses = f.name.replace("_por_segundo.csv", "")
        assert (nuevas["id_sesion"] == id_ses).all(), f"id_sesion inesperado en {f.name}"

        nuevas["segundo"] = np.arange(len(nuevas))
        paciente, stamp = id_ses.split("_")
        fecha, hora = stamp.split("-")
        nuevas["pacient"] = paciente
        nuevas["date"] = f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}"
        # La hora del nombre de archivo es la hora de GUARDADO (fin de la sesion): la diferencia
        # entre nombres consecutivos coincide (+/-1 s) con el numero de filas nuevas.
        # Por eso start_time = hora_guardado - duracion.
        fin = pd.Timestamp(f"{nuevas['date'].iloc[0]} {hora[:2]}:{hora[2:4]}:{hora[4:]}")
        nuevas["start_time"] = (fin - pd.Timedelta(seconds=len(nuevas))).strftime("%H:%M:%S")
        nuevas["sessionduration_sec"] = len(nuevas)
        for c in COLUMNAS_MANUALES:
            nuevas[c] = np.nan
        for c, v in REPORTADO_EQUIPO.items():
            nuevas[c] = v
        js =carpeta / f"{id_ses}_resumen.json"
        if js.exists():
            meta = json.loads(js.read_text(encoding="utf-8"))
            for col, clave in JSON_A_COLUMNA.items():
                if meta.get(clave) not in (None, ""):
                    nuevas[col] = meta[clave]
        nuevas = etiquetar(nuevas)
        sesiones.append(nuevas)
        print(f"{id_ses}: {n_prev:>5} filas previas, {len(nuevas):>4} nuevas  "
              f"(CAL={int((nuevas.fase=='CALIBRACION').sum())}, JUEGO={int((nuevas.fase=='JUEGO').sum())})  "
              f"base={nuevas.base_estimada_uv.iloc[0]:.1f} uV  "
              + " ".join(f"{k}={v}" for k, v in nuevas.etiqueta.value_counts().items()))
        previo = df

    unido = pd.concat(sesiones, ignore_index=True)[COLUMNAS_ORDEN]
    unido.to_csv(salida, index=False)
    print(f"\nTotal: {len(unido)} filas, {unido.id_sesion.nunique()} sesiones -> {salida.name}")
    print(unido.etiqueta.value_counts().to_string())


if __name__ == "__main__":
    main(ROOT / "data" / "raw", ROOT / "data" / "monterrey_move_ML_unido.csv")
