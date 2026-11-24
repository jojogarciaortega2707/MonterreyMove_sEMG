"""
MONTERREY MOVE - realtime/azure/verificar_criterios.py

Corre en Azure al final del job del test sellado. No calcula nada nuevo:
lee lo que escribieron code/07_test.py y realtime/rt_evaluar.py y dice si se
cumplen los criterios del proyecto.

  C1  separacion: ninguna sesion de test aparece en los datos de entrenamiento,
      y todas se grabaron DESPUES de congelar el modelo (fecha del JSON).
  C2  al menos --min-sesiones sesiones nuevas (colocaciones nuevas de electrodos).
  C3  metodo principal LogReg_todas: exactitud y F1 globales >= --umbral.
  C4  la ruta en tiempo real (vivo_1s y vivo_20Hz): exactitud y F1 >= --umbral.

Escribe criterios.json y criterios.md en la carpeta de resultados. Sale con
codigo 0 aunque algun criterio falle: el resultado se reporta, no se esconde.

Uso:  python realtime/azure/verificar_criterios.py <resultados> --test <carpeta_test>
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def ids_entrenamiento():
    for d in ("data", "dat1"):
        f = ROOT / d / "monterrey_move_ML_unido.csv"
        if f.exists():
            return set(pd.read_csv(f, usecols=["id_sesion"]).id_sesion)
    return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resultados", type=Path)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--min-sesiones", type=int, default=4)
    ap.add_argument("--umbral", type=float, default=0.75)
    a = ap.parse_args()

    congelado = json.loads((ROOT / "models" / "modelo_congelado.json").read_text(encoding="utf-8"))
    fecha_congelado = congelado["fecha"].replace("-", "")
    test_ids = sorted(p.name.replace("_por_segundo.csv", "") for p in (a.test / "raw").glob("*_por_segundo.csv"))
    entren = ids_entrenamiento()
    repetidas = sorted(set(test_ids) & entren)
    anteriores = [s for s in test_ids if s.split("_")[1][:8] < fecha_congelado]

    res = pd.read_csv(a.resultados / "test" / "resumen_test.csv")
    met = pd.read_csv(a.resultados / "test" / "metricas_test.csv")
    p = res[res.metodo == "LogReg_todas"].iloc[0]
    n = p.TP + p.FN + p.FP + p.TN
    acc = (p.TP + p.TN) / n
    rt = pd.read_csv(a.resultados / "realtime" / "rt_metricas_global.csv").set_index("metodo")
    n_ses = met.id_sesion.nunique()

    c = {
        "C1_separacion": {"ok": not repetidas and not anteriores and bool(entren),
                          "sesiones_test": test_ids, "repetidas_en_entrenamiento": repetidas,
                          "grabadas_antes_de_congelar": anteriores, "modelo_congelado": congelado["fecha"]},
        "C2_sesiones_nuevas": {"ok": n_ses >= a.min_sesiones, "n": int(n_ses), "minimo": a.min_sesiones},
        "C3_modelo_principal": {"ok": bool(acc >= a.umbral and p.F1_global >= a.umbral),
                                "exactitud": round(float(acc), 4), "F1": round(float(p.F1_global), 4),
                                "segundos": int(n)},
        "C4_tiempo_real": {"ok": all(rt.loc[m, "exactitud"] >= a.umbral and rt.loc[m, "F1"] >= a.umbral
                                     for m in ("vivo_1s", "vivo_20Hz")),
                           **{m: {"exactitud": round(float(rt.loc[m, "exactitud"]), 4),
                                  "F1": round(float(rt.loc[m, "F1"]), 4), "n": int(rt.loc[m, "n"])}
                              for m in ("vivo_1s", "vivo_20Hz")}},
    }
    c["todo_ok"] = all(v["ok"] for v in c.values())
    (a.resultados / "criterios.json").write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Project criteria (sealed test, Azure)", "",
         f"Threshold: {a.umbral:.0%}. Minimum new sessions: {a.min_sesiones}.", "",
         "| Criterion | Result | Detail |", "|---|---|---|",
         f"| C1 test never trained the model | {'PASS' if c['C1_separacion']['ok'] else 'FAIL'} "
         f"| {len(test_ids)} test sessions, {len(repetidas)} also in training, {len(anteriores)} recorded "
         f"before the freeze date {congelado['fecha']} |",
         f"| C2 new electrode placements | {'PASS' if c['C2_sesiones_nuevas']['ok'] else 'FAIL'} | {n_ses} sessions |",
         f"| C3 LogReg_todas ≥ {a.umbral:.0%} | {'PASS' if c['C3_modelo_principal']['ok'] else 'FAIL'} "
         f"| accuracy {acc:.3f}, F1 {p.F1_global:.3f}, {int(n)} s |",
         f"| C4 real-time path ≥ {a.umbral:.0%} | {'PASS' if c['C4_tiempo_real']['ok'] else 'FAIL'} "
         f"| 1-s: acc {rt.loc['vivo_1s', 'exactitud']:.3f} F1 {rt.loc['vivo_1s', 'F1']:.3f}; "
         f"20 Hz: acc {rt.loc['vivo_20Hz', 'exactitud']:.3f} F1 {rt.loc['vivo_20Hz', 'F1']:.3f} |",
         "", f"**Overall: {'PASS' if c['todo_ok'] else 'FAIL'}**"]
    (a.resultados / "criterios.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    sys.exit(0)


if __name__ == "__main__":
    main()
