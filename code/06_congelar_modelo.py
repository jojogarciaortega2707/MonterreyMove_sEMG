"""
MONTERREY MOVE - Paso 6: congelar el modelo ANTES de grabar las sesiones de prueba.

Entrena los 4 metodos de code/03_clasificador.py con TODOS los segundos utiles de las 11
sesiones (data/raw/) y los guarda en models/. Despues de esto no se cambia nada: ni
caracteristicas, ni C, ni reglas de calidad, ni k, ni T.

  models/modelo_congelado.pkl    modelos y umbrales (lo lee code/07_test.py)
  models/modelo_congelado.json   los mismos valores en texto, para leerlos y citarlos

Al final imprime el SHA-256 del .pkl. Copialo en DECISIONS.md: 07_test.py se niega a correr
si el hash del modelo no coincide con el de DECISIONS.md.

No sobrescribe un modelo ya congelado (usa --forzar solo si aun no se ha grabado el test).
Uso:  python code/06_congelar_modelo.py
"""
import hashlib
import importlib.util
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
MODELOS = ROOT / "models"


def cargar(nombre):
    spec = importlib.util.spec_from_file_location(nombre.replace(".py", ""), ROOT / "code" / nombre)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main():
    pkl = MODELOS / "modelo_congelado.pkl"
    if pkl.exists() and "--forzar" not in sys.argv:
        sys.exit(f"Ya existe {pkl.relative_to(ROOT)}. El modelo esta congelado; no se sobrescribe.")

    c3 = cargar("03_clasificador.py")
    calidad = ROOT / "data" / "segundos_calidad.csv"
    df = pd.read_csv(c3.CSV).merge(pd.read_csv(calidad)[["id_sesion", "segundo", "rechazado_calidad"]],
                                   on=["id_sesion", "segundo"], how="left")
    df["rechazado_calidad"] = df.rechazado_calidad.fillna(False).astype(bool)
    d = df[df.etiqueta.isin(["reposo", "contraccion"]) & ~df.rechazado_calidad].copy()
    d["y"] = (d.etiqueta == "contraccion").astype(int)
    X = c3.caracteristicas(d)

    modelos = {n: c3.nuevo_modelo().fit(X[cols], d.y) for n, cols in c3.GRUPOS.items()}

    # Mismo procedimiento que en cada rotacion LOSO, ahora con las 11 sesiones.
    ratio = pd.concat([d.loc[d.id_sesion == s, "promedio_uv"] / c3.base_calibracion(df[df.id_sesion == s])
                       for s in df.id_sesion.unique()])
    k, kmin, kmax, f1k = c3.elegir_umbral(ratio.values, d.y.loc[ratio.index].values,
                                          np.round(np.arange(1.1, 8.0001, 0.05), 2))
    T, tmin, tmax, f1t = c3.elegir_umbral(d.promedio_uv.values, d.y.values, np.arange(40, 400.1, 1.0))

    congelado = dict(
        fecha=date.today().isoformat(),
        metodo_principal="LogReg_todas",
        modelos=modelos,
        columnas=c3.GRUPOS,
        k_x_base=k, umbral_absoluto_uv=T,
        SEG_ARTEFACTO=c3.SEG_ARTEFACTO, PCT_BASE=c3.PCT_BASE,
    )
    MODELOS.mkdir(exist_ok=True)
    pkl.write_bytes(pickle.dumps(congelado))

    legible = dict(
        fecha=congelado["fecha"],
        metodo_principal="LogReg_todas",
        entrenamiento=dict(sesiones=int(df.id_sesion.nunique()), segundos=len(d),
                           reposo=int((d.y == 0).sum()), contraccion=int(d.y.sum()),
                           sha256_csv_unido=sha256(c3.CSV), sha256_segundos_calidad=sha256(calidad)),
        modelos={n: dict(columnas=cols, escalado_media=m[0].mean_.round(6).tolist(),
                         escalado_desv=m[0].scale_.round(6).tolist(),
                         coeficientes=m[-1].coef_[0].round(6).tolist(),
                         intercepto=round(float(m[-1].intercept_[0]), 6), C=m[-1].C)
                 for (n, cols), m in zip(c3.GRUPOS.items(), modelos.values())},
        Umbral_k_x_base=dict(k=k, rango_optimo=[kmin, kmax], F1_entrenamiento=round(f1k, 4),
                             base=f"P{c3.PCT_BASE} de promedio_uv en CALIBRACION, segundo >= {c3.SEG_ARTEFACTO}"),
        Umbral_absoluto=dict(T_uv=T, rango_optimo=[tmin, tmax], F1_entrenamiento=round(f1t, 4)),
        versiones=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                       scikit_learn=sklearn.__version__),
        sha256_pkl=sha256(pkl),
    )
    (MODELOS / "modelo_congelado.json").write_text(json.dumps(legible, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    print(json.dumps({k_: v for k_, v in legible.items() if k_ != "modelos"}, indent=2, ensure_ascii=False))
    print(f"\nCopia en DECISIONS.md:\nmodelo_congelado.pkl sha256: {legible['sha256_pkl']}")


if __name__ == "__main__":
    main()
