"""
MONTERREY MOVE - Azure ML / Foundry managed online endpoint (real-time scoring).

Same frozen model and same features as the local live classifier
(realtime/rt_modelo.py is imported, not copied). Only numpy is needed.

Request (any of the three):
  {"uv": [20 values]}                         one 1-s window of server packets (20 Hz)
  {"ventanas": [[20 values], [20 values]]}    several windows at once
  {"estadisticos": [{"promedio_uv":..,"maximo_uv":..,"minimo_uv":..,"desviacion_uv":..}]}

Response:
  {"modelo": "LogReg_todas", "sha256": "...", "predicciones": [{"clase","y","prob_contraccion"}]}
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1]))          # realtime/
import rt_modelo as M  # noqa: E402

MODELO = None


def init():
    global MODELO
    base = Path(os.environ.get("AZUREML_MODEL_DIR", M.ROOT))
    candidatos = list(base.rglob("modelo_congelado.json"))
    if not candidatos:
        raise FileNotFoundError(f"modelo_congelado.json no esta en {base}")
    MODELO = M.ModeloCongelado("json", ruta_json=candidatos[0])


def _stats(req):
    if "estadisticos" in req:
        e = req["estadisticos"]
        e = e if isinstance(e, list) else [e]
        return [(d["promedio_uv"], d["maximo_uv"], d["minimo_uv"], d["desviacion_uv"]) for d in e]
    ventanas = req.get("ventanas") or [req["uv"]]
    out = []
    for v in ventanas:
        if len(v) < 2:
            raise ValueError("cada ventana necesita al menos 2 valores (lo normal: 20 = 1 s a 20 Hz)")
        out.append(M.estadisticos(v))
    return out


def run(raw):
    try:
        req = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        st = np.array(_stats(req), float)
        p = MODELO.predecir_proba(M.caracteristicas(st[:, 0], st[:, 1], st[:, 2], st[:, 3]))
        return {"modelo": M.METODO, "sha256": MODELO.sha256,
                "predicciones": [{"clase": "CONTRACCION" if q >= 0.5 else "REPOSO", "y": int(q >= 0.5),
                                  "prob_contraccion": round(float(q), 4)} for q in p]}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
