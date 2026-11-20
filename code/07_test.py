"""
MONTERREY MOVE - Paso 7: test final con las sesiones nuevas. Se corre UNA sola vez.

Entrada
  data/test/raw/              los archivos de las sesiones nuevas tal como los guarda la app
                              (<id>_senal_cruda.npy, <id>_por_segundo.csv, <id>_resumen.json).
                              Pueden ser de varios dias en la misma carpeta: si un archivo no
                              repite al anterior, se toma como el inicio de una nueva cadena.
  data/test/guion_inicio.csv  una fila por sesion, con id_sesion y UNA de estas columnas:
                                inicio_guion_s     segundo del CSV (0 = primera fila de la sesion)
                                                   en que empieza el segundo 0 del guion, o
                                hora_inicio_guion  HH:MM:SS del reloj de la computadora que graba.
  models/modelo_congelado.pkl (code/06_congelar_modelo.py)

Etiquetas: por INSTRUCCION del guion de calibracion, no por amplitud (PROTOCOL.md, seccion 6):
  reposo 0-6, 12-18, 24-30 | contraccion 7-11, 19-23, 31-35   (segundos del guion)
  [DISENO] se descarta el primer segundo de cada bloque (tiempo de reaccion y error de +/-1 s
  al anotar la hora): quedan reposo 1-6, 13-18, 25-30 y contraccion 8-11, 20-23, 32-35.
  Se descartan tambien los segundos < 2 de la sesion y los rechazados por R1/R2 (paso 2).
  Los segundos del JUEGO no se evaluan: su verdad seria otra vez el umbral.

Salida: results/test/ (metricas_test.csv, resumen_test.csv, predicciones_test.csv, reporte_test.md).
Si results/test/metricas_test.csv ya existe, el script se detiene: el test ya se corrio.

Uso:  python code/07_test.py
      python code/07_test.py <carpeta_test> <carpeta_salida>   (ensayo con otra carpeta)
"""
import hashlib
import importlib.util
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FS = 2000
BLOQUES = [(0, 6, 0), (7, 11, 1), (12, 18, 0), (19, 23, 1), (24, 30, 0), (31, 35, 1)]
MARGEN = 1  # [DISENO] segundos descartados al inicio de cada bloque


def cargar(nombre):
    spec = importlib.util.spec_from_file_location(nombre.replace(".py", ""), ROOT / "code" / nombre)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c2 = cargar("02_calidad_senal.py")
c3 = cargar("03_clasificador.py")


def es_continuacion(actual, previo, iguales) -> bool:
    return previo is not None and len(actual) >= len(previo) and iguales(actual[:len(previo)], previo)


def leer_sesiones(raw: Path) -> pd.DataFrame:
    """Une CSV y .npy acumulativos: solo las filas/muestras nuevas, con control de calidad por segundo."""
    cols = ["promedio_uv", "maximo_uv", "minimo_uv", "desviacion_uv"]
    filas, csv_prev, npy_prev = [], None, None
    for f in sorted(raw.glob("MM-*_por_segundo.csv")):
        sid = f.name.replace("_por_segundo.csv", "")
        df = pd.read_csv(f)
        a = np.load(raw / f"{sid}_senal_cruda.npy", allow_pickle=False)
        sigue_csv = es_continuacion(df[cols].values, None if csv_prev is None else csv_prev[cols].values,
                                    lambda x, y: np.allclose(x, y, equal_nan=True))
        sigue_npy = es_continuacion(a, npy_prev, lambda x, y: np.array_equal(x, y, equal_nan=True))
        if sigue_csv != sigue_npy:
            sys.exit(f"ERROR: en {sid} el CSV y el .npy no coinciden en si continuan la sesion anterior")
        n_csv = len(csv_prev) if sigue_csv else 0
        n_npy = len(npy_prev) if sigue_npy else 0
        csv_prev, npy_prev = df, a
        s = df.iloc[n_csv:].copy()
        x = a[n_npy:].astype(np.float64)
        s["id_sesion"] = sid
        s["segundo"] = np.arange(len(s))
        s["fecha"] = sid.split("_")[1][:8]
        hora = sid.split("-")[-1]
        fin = pd.Timestamp(f"{s.fecha.iloc[0]} {hora[:2]}:{hora[2:4]}:{hora[4:]}")
        s["inicio_sesion"] = fin - pd.Timedelta(seconds=len(s))   # hora de guardado - duracion
        s["nueva_cadena"] = not sigue_csv

        # Control de calidad R1/R2, mismas reglas y constantes que code/02_calidad_senal.py
        falt = ~np.isfinite(x) | c2.mascara_rachas(x, c2.RACHA_MIN)
        xf = np.where(np.isfinite(x), x, np.nan)
        extremo = (x == np.nanmax(xf)) | (x == np.nanmin(xf))
        rech = []
        for k in range(len(s)):
            sl = slice(k * FS, min((k + 1) * FS, len(x)))
            vacio = sl.start >= len(x)
            rech.append(vacio or falt[sl].mean() > c2.MAX_FALTANTES or extremo[sl].sum() >= c2.N_EXTREMO)
        s["rechazado_calidad"] = rech
        s["dif_duracion_s"] = round(len(x) / FS - len(s), 2)
        filas.append(s)
    if not filas:
        sys.exit(f"No hay archivos MM-*_por_segundo.csv en {raw}")
    return pd.concat(filas, ignore_index=True)


def etiquetar_por_guion(df: pd.DataFrame, guion: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = np.nan
    df["seg_guion"] = np.nan
    for sid, s in df.groupby("id_sesion", sort=False):
        g = guion[guion.id_sesion == sid]
        if g.empty:
            sys.exit(f"ERROR: falta {sid} en guion_inicio.csv")
        g = g.iloc[0]
        if "inicio_guion_s" in g and pd.notna(g.get("inicio_guion_s")):
            t0 = int(g.inicio_guion_s)
        else:
            h = pd.Timestamp(f"{s.fecha.iloc[0]} {g.hora_inicio_guion}")
            t0 = int(round((h - s.inicio_sesion.iloc[0]).total_seconds()))
        if t0 < 0 or t0 + 35 >= len(s):
            sys.exit(f"ERROR: en {sid} el guion (segundo {t0}) cae fuera de la sesion de {len(s)} s")
        for a, b, clase in BLOQUES:
            for j in range(a + MARGEN, b + 1):
                idx = s.index[s.segundo == t0 + j]
                df.loc[idx, "y"] = clase
                df.loc[idx, "seg_guion"] = j
    return df


def main():
    test = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "test"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "test"
    if (out / "metricas_test.csv").exists():
        sys.exit(f"El test ya se corrio ({out / 'metricas_test.csv'}). No se vuelve a correr.")

    pkl = ROOT / "models" / "modelo_congelado.pkl"
    hash_pkl = hashlib.sha256(pkl.read_bytes()).hexdigest()
    dec = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    m = re.search(r"modelo_congelado\.pkl sha256:\s*`?([0-9a-f]{64})", dec)
    if not m or m.group(1) != hash_pkl:
        sys.exit("ERROR: el hash de models/modelo_congelado.pkl no coincide con el de DECISIONS.md")
    mc = pickle.loads(pkl.read_bytes())

    df = leer_sesiones(test / "raw")
    df = etiquetar_por_guion(df, pd.read_csv(test / "guion_inicio.csv", dtype={"hora_inicio_guion": str}))

    d = df[df.y.notna() & ~df.rechazado_calidad & (df.segundo >= mc["SEG_ARTEFACTO"])].copy()
    d["y"] = d.y.astype(int)
    X = c3.caracteristicas(d)
    for nombre, cols in mc["columnas"].items():
        d[nombre] = mc["modelos"][nombre].predict(X[cols])
    base = {sid: c3.base_calibracion(s) for sid, s in df.groupby("id_sesion")}
    d["Umbral_k_x_base"] = (d.promedio_uv / d.id_sesion.map(base) > mc["k_x_base"]).astype(int)
    d["Umbral_absoluto"] = (d.promedio_uv > mc["umbral_absoluto_uv"]).astype(int)
    metodos = list(mc["columnas"]) + ["Umbral_k_x_base", "Umbral_absoluto"]

    filas = []
    for (sid, fecha), s in d.groupby(["id_sesion", "fecha"], sort=False):
        for mt in metodos:
            filas.append(dict(id_sesion=sid, fecha=fecha, metodo=mt, **c3.metricas(s.y, s[mt])))
    res = pd.DataFrame(filas)

    resumen = []
    for mt in metodos:
        r = res[res.metodo == mt]
        glob = c3.metricas(d.y, d[mt])
        por_dia = {f: round(c3.metricas(g.y, g[mt])["F1"], 3) for f, g in d.groupby("fecha")}
        resumen.append(dict(metodo=mt, sens_media=r.sensibilidad.mean(), sens_sd=r.sensibilidad.std(),
                            espec_media=r.especificidad.mean(), espec_sd=r.especificidad.std(),
                            F1_media=r.F1.mean(), F1_sd=r.F1.std(), F1_min=r.F1.min(),
                            F1_global=glob["F1"], F1_por_dia=por_dia,
                            **{k: glob[k] for k in ["TP", "FN", "FP", "TN"]}))
    resumen = pd.DataFrame(resumen)

    out.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(out / "resumen_test.csv", index=False, float_format="%.4f")
    d[["id_sesion", "fecha", "segundo", "seg_guion", "fase", "promedio_uv", "y"] + metodos].to_csv(
        out / "predicciones_test.csv", index=False)

    L = ["# Test result (sealed sessions, run once)", "",
         f"Model: `models/modelo_congelado.pkl`, frozen {mc['fecha']}, sha256 `{hash_pkl}`. "
         f"Primary method declared before recording: **{mc['metodo_principal']}**.", "",
         f"Sessions: {d.id_sesion.nunique()} on {d.fecha.nunique()} day(s) ({', '.join(sorted(d.fecha.unique()))}). "
         f"Seconds evaluated: {len(d)} ({int((d.y == 0).sum())} rest, {int(d.y.sum())} contraction). "
         f"Labels come from the calibration-script instructions, not from amplitude.", "",
         f"Seconds of the script left out by R1/R2: {int((df.y.notna() & df.rechazado_calidad).sum())}. "
         f"Raw duration minus CSV rows per session: {df.groupby('id_sesion').dif_duracion_s.first().min():+.2f} "
         f"to {df.groupby('id_sesion').dif_duracion_s.first().max():+.2f} s.", "",
         "| Method | Sensitivity | Specificity | F1 (mean ± SD per session) | F1 min | F1 global | F1 per day | TP | FN | FP | TN |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in resumen.iterrows():
        L.append(f"| {r.metodo} | {r.sens_media:.3f} ± {r.sens_sd:.3f} | {r.espec_media:.3f} ± {r.espec_sd:.3f} "
                 f"| {r.F1_media:.3f} ± {r.F1_sd:.3f} | {r.F1_min:.3f} | {r.F1_global:.3f} "
                 f"| {', '.join(f'{k}: {v}' for k, v in r.F1_por_dia.items())} | {r.TP} | {r.FN} | {r.FP} | {r.TN} |")
    L += ["", "## Per session", "", "| Session | Date | Method | TP | FN | FP | TN | Sens | Spec | F1 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in res.iterrows():
        L.append(f"| {x.id_sesion} | {x.fecha} | {x.metodo} | {x.TP} | {x.FN} | {x.FP} | {x.TN} "
                 f"| {x.sensibilidad:.3f} | {x.especificidad:.3f} | {x.F1:.3f} |")
    (out / "reporte_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    # metricas_test.csv se escribe al final: su existencia marca que el test ya se corrio.
    res.to_csv(out / "metricas_test.csv", index=False, float_format="%.4f")
    print("\n".join(L))


if __name__ == "__main__":
    main()
