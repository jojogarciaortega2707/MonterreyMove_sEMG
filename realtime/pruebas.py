"""
MONTERREY MOVE - realtime/pruebas.py

Comprueba que la ruta en tiempo real es fiel a la ruta offline. Solo usa
datos de ENTRENAMIENTO (data/raw o dat1/raw). Nunca toca data/test.

  1. envolvente.py reproduce los *_por_segundo.csv de la app (error <= 0.06 uV).
  2. rt_modelo.caracteristicas == code/03_clasificador.caracteristicas.
  3. El modelo leido del JSON (solo numpy) == el pkl congelado (scikit-learn).

Uso:  python realtime/pruebas.py
"""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import envolvente as E          # noqa: E402
import rt_modelo as M           # noqa: E402

ROOT = M.ROOT


def carpeta_datos():
    for d in ("data", "dat1"):
        if (ROOT / d / "raw").is_dir():
            return ROOT / d
    sys.exit("No se encontro data/raw ni dat1/raw")


def cargar_c3():
    spec = importlib.util.spec_from_file_location("c3", ROOT / "code" / "03_clasificador.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    datos = carpeta_datos()
    ok = True

    # 1. Los archivos son acumulativos: el ultimo contiene todas las sesiones.
    ultimo = sorted((datos / "raw").glob("*_por_segundo.csv"))[-1]
    sid = ultimo.name.replace("_por_segundo.csv", "")
    df = pd.read_csv(ultimo)
    env = E.envolvente(np.load(datos / "raw" / f"{sid}_senal_cruda.npy"))
    seg = np.stack([env[k * E.FS:(k + 1) * E.FS] for k in range(len(df))])
    err = {c: float(np.abs(f(seg, axis=1) - df[c]).max()) for c, f in
           [("promedio_uv", np.mean), ("maximo_uv", np.max), ("minimo_uv", np.min), ("desviacion_uv", np.std)]}
    p1 = max(err.values()) <= 0.06
    ok &= p1
    print(f"[{'OK' if p1 else 'FALLA'}] 1. envolvente vs CSV de la app, {len(df)} s: error max (uV) {err}")

    # 2. Mismas caracteristicas que el clasificador del equipo.
    c3 = cargar_c3()
    X_ref = c3.caracteristicas(df)[M.COLUMNAS].values
    X_rt = M.caracteristicas(df.promedio_uv, df.maximo_uv, df.minimo_uv, df.desviacion_uv)
    p2 = np.allclose(X_ref, X_rt)
    ok &= p2
    print(f"[{'OK' if p2 else 'FALLA'}] 2. caracteristicas identicas a code/03_clasificador.py")

    # 3. JSON (numpy) == pkl (sklearn)
    pj = M.ModeloCongelado("json").predecir_proba(X_rt)
    pp = M.ModeloCongelado("pkl").predecir_proba(X_rt)
    d = float(np.abs(pj - pp).max())
    p3 = d < 1e-4 and np.array_equal(pj >= 0.5, pp >= 0.5)
    ok &= p3
    print(f"[{'OK' if p3 else 'FALLA'}] 3. modelo JSON vs pkl: diferencia max de probabilidad {d:.2e}, "
          f"mismas clases en {len(pj)} s")

    print("\nTODO OK" if ok else "\nHAY FALLAS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
