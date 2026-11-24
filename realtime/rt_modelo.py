"""
MONTERREY MOVE - realtime/rt_modelo.py

El clasificador en tiempo real. NO es un modelo nuevo: usa el modelo
congelado el 2026-11-20 (models/modelo_congelado.*, metodo principal
LogReg_todas) tal cual, para que lo que corre en vivo sea exactamente lo
que se va a evaluar en el test sellado.

Dos formas de cargarlo, con el MISMO resultado:
  - "pkl":  models/modelo_congelado.pkl (scikit-learn). Antes de usarlo se
            comprueba su SHA-256 contra la linea de DECISIONS.md, igual que
            code/07_test.py.
  - "json": models/modelo_congelado.json, solo numpy. Es la regresion
            logistica escrita a mano (escalado + coeficientes). Sirve para el
            endpoint de Azure y para portar el modelo al ESP32 sin
            scikit-learn. realtime/pruebas.py verifica que coincide con el pkl.

Caracteristicas: las 7 de code/03_clasificador.py, sobre UNA ventana de 1 s.
En vivo esa ventana son los ultimos 20 paquetes del servidor (20 Hz).
"""
import hashlib
import json
import pickle
import re
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "models" / "modelo_congelado.pkl"
JSON = ROOT / "models" / "modelo_congelado.json"
DECISIONS = ROOT / "DECISIONS.md"

METODO = "LogReg_todas"
COLUMNAS = ["log_promedio", "log_maximo", "log_minimo", "log_desviacion",
            "max_sobre_prom", "min_sobre_prom", "cv"]
EPS = 0.1                      # igual que code/03_clasificador.py
VENTANA_PAQUETES = 20          # 1 s a 20 Hz


def caracteristicas(promedio, maximo, minimo, desviacion):
    """Mismas formulas que code/03_clasificador.caracteristicas. Acepta escalares o arreglos."""
    m = np.maximum(np.asarray(promedio, float), EPS)
    mx = np.asarray(maximo, float)
    mn = np.asarray(minimo, float)
    sd = np.asarray(desviacion, float)
    return np.stack([np.log(m), np.log(np.maximum(mx, EPS)), np.log(np.maximum(mn, EPS)),
                     np.log(np.maximum(sd, EPS)), mx / m, mn / m, sd / m], axis=-1)


def estadisticos(ventana):
    """promedio, maximo, minimo y desviacion (poblacional, como np.std del registro)."""
    v = np.asarray(ventana, float)
    return float(v.mean()), float(v.max()), float(v.min()), float(v.std())


def verificar_hash():
    h = hashlib.sha256(PKL.read_bytes()).hexdigest()
    m = re.search(r"modelo_congelado\.pkl sha256:\s*`?([0-9a-f]{64})", DECISIONS.read_text(encoding="utf-8"))
    if not m or m.group(1) != h:
        raise RuntimeError("El hash de models/modelo_congelado.pkl no coincide con DECISIONS.md")
    return h


class ModeloCongelado:
    """predecir_proba(X) -> probabilidad de CONTRACCION por fila (X con las 7 columnas)."""

    def __init__(self, fuente="json", ruta_json=JSON):
        self.fuente = fuente
        if fuente == "pkl":
            self.sha256 = verificar_hash()
            self._m = pickle.loads(PKL.read_bytes())["modelos"][METODO]
        elif fuente == "json":
            p = json.loads(Path(ruta_json).read_text(encoding="utf-8"))
            self.sha256 = p.get("sha256_pkl", "")
            d = p["modelos"][METODO]
            assert d["columnas"] == COLUMNAS, "columnas distintas a las del modelo congelado"
            self._mu = np.array(d["escalado_media"])
            self._sd = np.array(d["escalado_desv"])
            self._w = np.array(d["coeficientes"])
            self._b = float(d["intercepto"])
        else:
            raise ValueError("fuente debe ser 'json' o 'pkl'")

    def predecir_proba(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.fuente == "pkl":
            import pandas as pd
            return self._m.predict_proba(pd.DataFrame(X, columns=COLUMNAS))[:, 1]
        z = ((X - self._mu) / self._sd) @ self._w + self._b
        return 1.0 / (1.0 + np.exp(-z))

    def predecir(self, X):
        return (self.predecir_proba(X) >= 0.5).astype(int)


class ClasificadorEnVivo:
    """Recibe un valor de envolvente (uV) por paquete y responde en cada uno.

    Hasta juntar 20 paquetes (1 s) no hay prediccion. Despues, cada paquete
    nuevo desliza la ventana 50 ms y da una prediccion: 20 por segundo.
    """

    def __init__(self, modelo=None, ventana=VENTANA_PAQUETES):
        self.modelo = modelo or ModeloCongelado("json")
        self.buf = deque(maxlen=ventana)

    def reiniciar(self):
        self.buf.clear()

    def agregar(self, uv):
        self.buf.append(float(uv))
        if len(self.buf) < self.buf.maxlen:
            return None
        st = estadisticos(self.buf)
        p = float(self.modelo.predecir_proba(caracteristicas(*st))[0])
        return {"clase": "CONTRACCION" if p >= 0.5 else "REPOSO", "y": int(p >= 0.5),
                "prob_contraccion": round(p, 4), "promedio_uv": round(st[0], 1),
                "maximo_uv": round(st[1], 1), "minimo_uv": round(st[2], 1),
                "desviacion_uv": round(st[3], 1)}
