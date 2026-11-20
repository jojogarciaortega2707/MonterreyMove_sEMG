"""
MONTERREY MOVE - Paso 5: ubicar el guion de calibracion dentro de la fase CALIBRACION.

Guion (PROTOCOL.md), 36 s:
  0-6 reposo | 7-11 ensayo 1 | 12-18 reposo | 19-23 ensayo 2 | 24-30 reposo | 31-35 ensayo 3

El inicio del guion NO se registra en los archivos. Este script busca, en cada sesion, el
desplazamiento t0 que mejor correlaciona log(promedio_uv) con la plantilla del guion, e informa
la segunda mejor opcion para mostrar cuando la ubicacion es ambigua. Es exploratorio: usa la
amplitud para alinear y por eso NO sirve como etiqueta independiente; para eso hay que registrar
el inicio del guion en el software.

Escribe results/guion_calibracion.csv.
Uso:  python code/05_guion_calibracion.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GUION = [(7, 11), (19, 23), (31, 35)]
DUR = 36

plantilla = np.zeros(DUR)
for a, b in GUION:
    plantilla[a:b + 1] = 1

df = pd.read_csv(ROOT / "data" / "monterrey_move_ML_unido.csv")
filas = []
for sid, s in df.groupby("id_sesion", sort=False):
    y = np.log(s.promedio_uv.clip(lower=0.1).to_numpy())
    inicio_juego = int((s.fase == "CALIBRACION").sum())
    cand = sorted(((np.corrcoef(y[t:t + DUR], plantilla)[0, 1], t)
                   for t in range(2, inicio_juego - DUR + 1)), reverse=True)
    r1, t1 = cand[0]
    # Segunda opcion = mejor desplazamiento a mas de 2 s del primero (los vecinos +/-1 s no cuentan).
    r2, t2 = next((r, t) for r, t in cand[1:] if abs(t - t1) > 2)
    filas.append(dict(id_sesion=sid, calibracion_s=inicio_juego, t0_mejor=t1, r_mejor=round(r1, 2),
                      t0_segundo=t2, r_segundo=round(r2, 2), ambiguo=bool(r1 - r2 < 0.05),
                      fin_guion=t1 + DUR, s_entre_guion_y_juego=inicio_juego - (t1 + DUR),
                      s_antes_del_guion=t1))
res = pd.DataFrame(filas)
res.to_csv(ROOT / "results" / "guion_calibracion.csv", index=False)
print(res.to_string(index=False))
