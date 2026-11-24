"""
MONTERREY MOVE - realtime/rt_evaluar.py

Evalua la ruta EN TIEMPO REAL sin hardware: toma la senal cruda guardada,
la convierte en los mismos paquetes de 20 Hz que manda el servidor y se los
pasa uno por uno al ClasificadorEnVivo, como si llegaran en vivo.

Compara tres cosas por segundo:
  offline  el modelo congelado sobre los 2000 valores de envolvente del segundo
           (lo mismo que hace code/07_test.py)
  vivo_1s  el clasificador en vivo con los 20 paquetes de ese mismo segundo
  vivo_20Hz  todas las predicciones deslizantes (una cada 50 ms), contando solo
           las ventanas que caen enteras en segundos de la misma etiqueta

Etiquetas (elige una):
  --guion RUTA     guion_inicio.csv: etiquetas por INSTRUCCION del guion de
                   calibracion, con las mismas reglas que code/07_test.py
                   (se importa su funcion, no se copia). Es la que vale para
                   el test sellado.
  --amplitud RUTA  monterrey_move_ML_unido.csv, columna etiqueta (solo
                   entrenamiento; esas etiquetas salen de la amplitud).

Uso:
  python realtime/rt_evaluar.py <carpeta_raw> <carpeta_salida> --guion <guion_inicio.csv>
  python realtime/rt_evaluar.py dat1/raw results_rt/entrenamiento --amplitud dat1/monterrey_move_ML_unido.csv
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import envolvente as E  # noqa: E402
import rt_modelo as M   # noqa: E402

ROOT = M.ROOT
P = E.PAQUETES_POR_SEGUNDO


def cargar_07():
    spec = importlib.util.spec_from_file_location("t07", ROOT / "code" / "07_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metricas(y, p):
    y, p = np.asarray(y, int), np.asarray(p, int)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    n = tp + fn + fp + tn
    return dict(n=n, TP=tp, FN=fn, FP=fp, TN=tn,
                exactitud=(tp + tn) / n if n else np.nan,
                sensibilidad=tp / (tp + fn) if tp + fn else np.nan,
                especificidad=tn / (tn + fp) if tn + fp else np.nan,
                F1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("salida", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--guion", type=Path)
    g.add_argument("--amplitud", type=Path)
    a = ap.parse_args()

    t07 = cargar_07()
    df = t07.leer_sesiones(a.raw)              # mismas reglas de lectura y de calidad R1/R2
    if a.guion:
        df = t07.etiquetar_por_guion(df, pd.read_csv(a.guion, dtype={"hora_inicio_guion": str}))
        tipo = "guion de calibracion (instruccion, no amplitud)"
    else:
        lab = pd.read_csv(a.amplitud, usecols=["id_sesion", "segundo", "etiqueta"])
        df = df.merge(lab, on=["id_sesion", "segundo"], how="left")
        df["y"] = df.etiqueta.map({"reposo": 0, "contraccion": 1})
        tipo = "amplitud (columna etiqueta, solo entrenamiento)"

    # Desplazamiento de cada sesion dentro de su archivo acumulativo.
    off, acum = {}, 0
    for sid, s in df.groupby("id_sesion", sort=False):
        if bool(s.nueva_cadena.iloc[0]):
            acum = 0
        off[sid] = acum
        acum += len(s)

    modelo = M.ModeloCongelado("json")
    filas_s, filas_v, tiempos = [], [], []
    for sid, s in df.groupby("id_sesion", sort=False):
        env = E.envolvente(np.load(a.raw / f"{sid}_senal_cruda.npy").astype(np.float64))
        _, uv = E.paquetes(env)
        o = off[sid]
        n = len(s)
        uv = uv[o * P:(o + n) * P]
        env = env[o * E.FS:(o + n) * E.FS]

        # --- streaming: un paquete a la vez ---
        cl = M.ClasificadorEnVivo(modelo)
        pred = np.full(len(uv), -1)
        for j, v in enumerate(uv):
            t0 = time.perf_counter()
            r = cl.agregar(v)
            tiempos.append(time.perf_counter() - t0)
            if r is not None:
                pred[j] = r["y"]

        y = s.y.values
        valido = (~s.rechazado_calidad.values) & (s.segundo.values >= 2) & ~np.isnan(y.astype(float))
        for k in range(n):
            bloque = env[k * E.FS:(k + 1) * E.FS]
            if len(bloque) < E.FS:
                continue
            st = (bloque.mean(), bloque.max(), bloque.min(), bloque.std())
            filas_s.append(dict(id_sesion=sid, segundo=k, fase=s.fase.iloc[k], y=y[k], valido=valido[k],
                                offline=int(modelo.predecir(M.caracteristicas(*st))[0]),
                                vivo_1s=int(pred[(k + 1) * P - 1]) if (k + 1) * P - 1 < len(pred) else -1))
        # ventanas deslizantes: paquete j usa los paquetes j-19..j -> segundos que toca
        for j in range(P - 1, len(pred)):
            k0, k1 = (j - P + 1) // P, j // P
            if k1 >= n or not (valido[k0] and valido[k1]) or y[k0] != y[k1]:
                continue
            filas_v.append(dict(id_sesion=sid, paquete=j, y=int(y[k1]), vivo_20Hz=int(pred[j])))

    S = pd.DataFrame(filas_s)
    V = pd.DataFrame(filas_v)
    Sv = S[S.valido & (S.vivo_1s >= 0)].copy()
    Sv["y"] = Sv.y.astype(int)

    res = []
    for sid, g_ in Sv.groupby("id_sesion", sort=False):
        for met in ("offline", "vivo_1s"):
            res.append(dict(id_sesion=sid, metodo=met, **metricas(g_.y, g_[met])))
        gv = V[V.id_sesion == sid]
        res.append(dict(id_sesion=sid, metodo="vivo_20Hz", **metricas(gv.y, gv.vivo_20Hz)))
    res = pd.DataFrame(res)
    glob = pd.DataFrame([dict(metodo=m, **metricas(Sv.y, Sv[m])) for m in ("offline", "vivo_1s")] +
                        [dict(metodo="vivo_20Hz", **metricas(V.y, V.vivo_20Hz))])
    acuerdo = float((S.offline == S.vivo_1s)[S.vivo_1s >= 0].mean())
    t_us = np.array(tiempos) * 1e6

    a.salida.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.salida / "rt_metricas_por_sesion.csv", index=False, float_format="%.6f")
    glob.to_csv(a.salida / "rt_metricas_global.csv", index=False, float_format="%.6f")
    S.to_csv(a.salida / "rt_predicciones_por_segundo.csv", index=False)

    L = ["# Real-time path evaluation", "",
         f"Raw folder: `{a.raw}`. Sessions: {S.id_sesion.nunique()}. Labels: {tipo}.",
         f"Model: frozen `{M.METODO}` (models/modelo_congelado.json, pkl sha256 `{modelo.sha256[:16]}…`).", "",
         "The server packets (one envelope value every 50 ms) are rebuilt from the raw signal and fed one by one "
         "to `ClasificadorEnVivo`, as they would arrive live.", "",
         f"Agreement live 1-s window vs offline (all {int((S.vivo_1s >= 0).sum())} seconds): **{acuerdo:.3f}**", "",
         f"Classifier compute time per packet: median {np.median(t_us):.0f} µs, p99 {np.percentile(t_us, 99):.0f} µs "
         "(on top of this, the 1-s window means a decision reflects the last second of signal).", "",
         "| Path | n | Accuracy | Sensitivity | Specificity | F1 | TP | FN | FP | TN |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in glob.iterrows():
        L.append(f"| {r.metodo} | {r.n} | {r.exactitud:.3f} | {r.sensibilidad:.3f} | {r.especificidad:.3f} "
                 f"| {r.F1:.3f} | {r.TP} | {r.FN} | {r.FP} | {r.TN} |")
    L += ["", "## Per session", "", "| Session | Path | n | Accuracy | F1 |", "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r.id_sesion} | {r.metodo} | {r.n} | {r.exactitud:.3f} | {r.F1:.3f} |")
    (a.salida / "rt_reporte.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:14 + len(glob)]))


if __name__ == "__main__":
    main()
