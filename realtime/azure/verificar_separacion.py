"""
MONTERREY MOVE - realtime/azure/verificar_separacion.py

Primer paso del job del test sellado. Detiene el job (codigo 1) ANTES de
evaluar si los datos de test no estan separados del entrenamiento:

  - ninguna sesion de test puede estar en los datos de entrenamiento;
  - ninguna sesion de test puede ser anterior a la fecha en que se congelo
    el modelo (models/modelo_congelado.json -> fecha);
  - el hash del modelo tiene que coincidir con DECISIONS.md;
  - guion_inicio.csv tiene que traer una fila por cada sesion de test.

Uso:  python realtime/azure/verificar_separacion.py <carpeta_test>
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "realtime"))
sys.dont_write_bytecode = True
import rt_modelo as M  # noqa: E402


def main():
    test = Path(sys.argv[1])
    fallas = []

    h = M.verificar_hash()
    congelado = json.loads((ROOT / "models" / "modelo_congelado.json").read_text(encoding="utf-8"))
    fecha = congelado["fecha"].replace("-", "")

    ids = sorted(p.name.replace("_por_segundo.csv", "") for p in (test / "raw").glob("*_por_segundo.csv"))
    if not ids:
        fallas.append(f"no hay sesiones en {test / 'raw'}")

    entren = set()
    for d in ("data", "dat1"):
        f = ROOT / d / "monterrey_move_ML_unido.csv"
        if f.exists():
            entren = set(pd.read_csv(f, usecols=["id_sesion"]).id_sesion)
    if not entren:
        fallas.append("no se encontro la lista de sesiones de entrenamiento (monterrey_move_ML_unido.csv)")
    if set(ids) & entren:
        fallas.append(f"sesiones de test que tambien son de entrenamiento: {sorted(set(ids) & entren)}")
    antes = [s for s in ids if s.split("_")[1][:8] < fecha]
    if antes:
        fallas.append(f"sesiones grabadas antes de congelar el modelo ({congelado['fecha']}): {antes}")

    g = test / "guion_inicio.csv"
    if not g.exists():
        fallas.append("falta guion_inicio.csv")
    else:
        faltan = sorted(set(ids) - set(pd.read_csv(g).id_sesion))
        if faltan:
            fallas.append(f"sesiones sin fila en guion_inicio.csv: {faltan}")

    print(f"Modelo congelado {congelado['fecha']}, sha256 {h[:16]}… coincide con DECISIONS.md")
    print(f"Sesiones de test: {len(ids)}  |  sesiones de entrenamiento conocidas: {len(entren)}")
    if fallas:
        print("SEPARACION NO VALIDA:\n  - " + "\n  - ".join(fallas))
        sys.exit(1)
    print("Separacion entrenamiento/test: OK")


if __name__ == "__main__":
    main()
