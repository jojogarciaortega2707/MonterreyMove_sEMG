"""
MONTERREY MOVE - Paso 3: clasificador reposo vs contraccion sobre el resumen de 1 Hz.
Excluye los segundos rechazados por el control de calidad (data/segundos_calidad.csv, paso 2).

  1. Grafica por sesion de promedio_uv (escala log), puntos por etiqueta, fase JUEGO sombreada.
  2. Clasificador binario reposo (0) vs contraccion (1); se descartan ambiguo y artefacto_inicio.
  3. Validacion leave-one-session-out (11 rotaciones): sensibilidad, especificidad, F1 y
     matriz de confusion por sesion + promedio.
  4. Baselines de umbral: promedio_uv > k x base de reposo (base de la fase CALIBRACION de la
     sesion de prueba) y umbral absoluto fijo.

Reglas anti-fuga: solo se usan promedio_uv, maximo_uv, minimo_uv, desviacion_uv y derivadas
de la MISMA ventana. Nada de zona_dominante, pct_tiempo_*, promedio_pct_ref, repeticiones_*,
puntaje_acumulado, umbrales, REF_*, fase, segundo, id_sesion ni base_estimada_uv.

Uso:  python code/03_clasificador.py   (lee data/monterrey_move_ML_unido.csv, escribe results/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "monterrey_move_ML_unido.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

SEG_ARTEFACTO = 2   # [DISENO] igual que en 01_unir_csv.py
PCT_BASE = 20       # [DISENO] percentil para la base de reposo del baseline

PERMITIDAS = ["promedio_uv", "maximo_uv", "minimo_uv", "desviacion_uv"]
PROHIBIDAS = ["zona_dominante", "pct_tiempo_objetivo", "pct_tiempo_alto", "pct_tiempo_bajo",
              "promedio_pct_ref", "repeticiones_validas", "repeticiones_invalidas",
              "puntaje_acumulado", "activation_threshold_uV", "release_threshold_uV",
              "REF_min_uV", "REF_max_uV", "fase", "segundo", "id_sesion", "base_estimada_uv"]

# Paleta (dataviz, slots categoricos 1-2 + gris neutro)
C_REPOSO, C_CONTR, C_AMBIG, C_ARTEF = "#2a78d6", "#eb6834", "#898781", "#52514e"
C_JUEGO, C_GRID, C_INK = "#f0efec", "#e1e0d9", "#0b0b0b"


# --------------------------------------------------------------------------- características
def caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Solo derivadas de la misma ventana de 1 s. eps evita log(0) en minimo_uv."""
    eps = 0.1
    m = df["promedio_uv"].clip(lower=eps)
    X = pd.DataFrame(index=df.index)
    X["log_promedio"] = np.log(m)
    X["log_maximo"] = np.log(df["maximo_uv"].clip(lower=eps))
    X["log_minimo"] = np.log(df["minimo_uv"].clip(lower=eps))
    X["log_desviacion"] = np.log(df["desviacion_uv"].clip(lower=eps))
    X["max_sobre_prom"] = df["maximo_uv"] / m
    X["min_sobre_prom"] = df["minimo_uv"] / m
    X["cv"] = df["desviacion_uv"] / m
    return X


GRUPOS = {
    "LogReg_todas": ["log_promedio", "log_maximo", "log_minimo", "log_desviacion",
                     "max_sobre_prom", "min_sobre_prom", "cv"],
    # Sin amplitud absoluta: solo forma de la ventana (invariantes a escala).
    "LogReg_sin_amplitud": ["max_sobre_prom", "min_sobre_prom", "cv"],
}


def nuevo_modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


# --------------------------------------------------------------------------- métricas
def metricas(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, sensibilidad=sens, especificidad=spec, F1=f1)


def base_calibracion(ses: pd.DataFrame) -> float:
    """Base de reposo conocida ANTES del juego: P20 de promedio_uv en CALIBRACION (sin artefacto)."""
    cal = ses[(ses.fase == "CALIBRACION") & (ses.segundo >= SEG_ARTEFACTO)]
    return float(np.percentile(cal.promedio_uv, PCT_BASE))


def elegir_umbral(valores, y, grid):
    """Devuelve el punto medio del intervalo de umbrales con F1 maximo en entrenamiento."""
    f1s = np.array([metricas(y, (valores > g).astype(int))["F1"] for g in grid])
    mejores = grid[np.isclose(f1s, f1s.max())]
    return float(np.median(mejores)), float(mejores.min()), float(mejores.max()), float(f1s.max())


# --------------------------------------------------------------------------- gráfica
def graficar(df: pd.DataFrame, ruta: Path):
    sesiones = list(df.id_sesion.unique())
    ncol = 3
    nrow = int(np.ceil(len(sesiones) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow), sharey=True)
    axes = axes.ravel()
    estilos = [
        ("reposo", dict(color=C_REPOSO, marker="o", s=14, label="reposo")),
        ("contraccion", dict(color=C_CONTR, marker="o", s=14, label="contracción")),
        ("ambiguo", dict(color=C_AMBIG, marker="D", s=14, label="ambiguo")),
        ("artefacto_inicio", dict(color=C_ARTEF, marker="x", s=30, linewidths=1.5,
                                  label="artefacto_inicio")),
    ]
    for ax, sid in zip(axes, sesiones):
        s = df[df.id_sesion == sid]
        juego = s[s.fase == "JUEGO"].segundo
        if len(juego):
            ax.axvspan(juego.min() - 0.5, juego.max() + 0.5, color=C_JUEGO, zorder=0,
                       label="fase JUEGO")
        ax.plot(s.segundo, s.promedio_uv, color=C_GRID, lw=1, zorder=1)
        base = s.base_estimada_uv.iloc[0]
        for k, ls in [(1.6, ":"), (3.0, "--")]:
            ax.axhline(k * base, color=C_AMBIG, lw=0.8, ls=ls, zorder=1)
        for et, kw in estilos:
            sel = s[s.etiqueta == et]
            ax.scatter(sel.segundo, sel.promedio_uv, zorder=3, edgecolors="none"
                       if kw["marker"] != "x" else None, **kw)
        ax.set_yscale("log")
        ax.set_title(f"{sid.split('_')[1]}  (n={len(s)} s, base={base:.0f} µV)",
                     fontsize=9, color=C_INK, loc="left")
        ax.grid(True, axis="y", which="major", color=C_GRID, lw=0.6)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    for ax in axes[len(sesiones):]:
        ax.axis("off")
    for i in range(0, len(axes), ncol):
        axes[i].set_ylabel("promedio_uv (µV, log)", fontsize=8)
    for ax in axes[-ncol:]:
        ax.set_xlabel("segundo", fontsize=8)
    # Leyenda en el panel vacío
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color=C_AMBIG, ls=":", lw=1), Line2D([], [], color=C_AMBIG, ls="--", lw=1)]
    l += ["1.6 × base (límite reposo)", "3 × base (límite contracción)"]
    axes[-1].legend(h, l, loc="center", frameon=False, fontsize=10)
    fig.suptitle("MONTERREY MOVE · MM-0001 · promedio_uv por segundo y etiqueta provisional [DISEÑO]",
                 fontsize=12, x=0.01, ha="left", color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ruta, dpi=130, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    df = pd.read_csv(CSV)
    qc_path = ROOT / "data" / "segundos_calidad.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)[["id_sesion", "segundo", "rechazado_calidad"]]
        df = df.merge(qc, on=["id_sesion", "segundo"], how="left")
        df["rechazado_calidad"] = df.rechazado_calidad.fillna(False).astype(bool)
    else:
        print("AVISO: falta data/segundos_calidad.csv; corre antes code/02_calidad_senal.py")
        df["rechazado_calidad"] = False
    assert not set(PROHIBIDAS) & set(GRUPOS["LogReg_todas"]), "fuga de variables"

    # Documentar columnas constantes / vacias
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    graficar(df, OUT / "fig_promedio_por_sesion.png")

    d = df[df.etiqueta.isin(["reposo", "contraccion"]) & ~df.rechazado_calidad].copy()
    d["y"] = (d.etiqueta == "contraccion").astype(int)
    X = caracteristicas(d)
    sesiones = list(df.id_sesion.unique())

    filas, preds = [], []
    grid_k = np.round(np.arange(1.1, 8.0001, 0.05), 2)
    grid_abs = np.arange(40, 400.1, 1.0)

    for sid in sesiones:
        tr, te = d.id_sesion != sid, d.id_sesion == sid
        ses_te = df[df.id_sesion == sid]
        pr = pd.DataFrame({"id_sesion": sid, "segundo": d.loc[te, "segundo"],
                           "y": d.loc[te, "y"]})

        # --- modelos ---
        for nombre, cols in GRUPOS.items():
            m = nuevo_modelo().fit(X.loc[tr, cols], d.loc[tr, "y"])
            p = m.predict(X.loc[te, cols])
            pr[nombre] = p
            filas.append(dict(id_sesion=sid, metodo=nombre, **metricas(d.loc[te, "y"], p)))

        # --- baseline k x base (base de calibración de cada sesión, k elegido en entrenamiento) ---
        ratio_tr = []
        for s2 in sesiones:
            if s2 == sid:
                continue
            b2 = base_calibracion(df[df.id_sesion == s2])
            ratio_tr.append(d.loc[d.id_sesion == s2, "promedio_uv"] / b2)
        ratio_tr = pd.concat(ratio_tr)
        k, kmin, kmax, _ = elegir_umbral(ratio_tr.values, d.loc[tr, "y"].loc[ratio_tr.index].values, grid_k)
        b = base_calibracion(ses_te)
        p = (d.loc[te, "promedio_uv"] / b > k).astype(int)
        pr["Umbral_k_x_base"] = p.values
        filas.append(dict(id_sesion=sid, metodo="Umbral_k_x_base", k=k, k_rango=f"{kmin:.2f}-{kmax:.2f}",
                          base_uv=round(b, 1), **metricas(d.loc[te, "y"], p)))

        # --- baseline umbral absoluto (sin base por sesión) ---
        T, tmin, tmax, _ = elegir_umbral(d.loc[tr, "promedio_uv"].values, d.loc[tr, "y"].values, grid_abs)
        p = (d.loc[te, "promedio_uv"] > T).astype(int)
        pr["Umbral_absoluto"] = p.values
        filas.append(dict(id_sesion=sid, metodo="Umbral_absoluto", k=T, k_rango=f"{tmin:.0f}-{tmax:.0f} uV",
                          **metricas(d.loc[te, "y"], p)))
        preds.append(pr)

    res = pd.DataFrame(filas)
    res.to_csv(OUT / "metricas_por_sesion.csv", index=False, float_format="%.4f")
    pd.concat(preds).to_csv(OUT / "predicciones_loso.csv", index=False)

    metodos = list(GRUPOS) + ["Umbral_k_x_base", "Umbral_absoluto"]
    resumen = []
    for mt in metodos:
        r = res[res.metodo == mt]
        tot = r[["TP", "FN", "FP", "TN"]].sum()
        agg = metricas(np.r_[np.ones(tot.TP + tot.FN), np.zeros(tot.FP + tot.TN)],
                       np.r_[np.ones(tot.TP), np.zeros(tot.FN), np.ones(tot.FP), np.zeros(tot.TN)])
        resumen.append(dict(
            metodo=mt,
            sens_media=r.sensibilidad.mean(), sens_sd=r.sensibilidad.std(),
            espec_media=r.especificidad.mean(), espec_sd=r.especificidad.std(),
            F1_media=r.F1.mean(), F1_sd=r.F1.std(), F1_min=r.F1.min(),
            F1_global=agg["F1"], TP=tot.TP, FN=tot.FN, FP=tot.FP, TN=tot.TN))
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(OUT / "resumen_metodos.csv", index=False, float_format="%.4f")

    # Coeficientes del modelo completo (interpretación, ajustado con las 11 sesiones)
    coefs = {}
    for nombre, cols in GRUPOS.items():
        m = nuevo_modelo().fit(X[cols], d.y)
        coefs[nombre] = pd.Series(m[-1].coef_[0], index=cols)

    # ¿Qué predice el modelo en los segundos 'ambiguo'? (sin verdad de referencia)
    amb = df[df.etiqueta == "ambiguo"]
    m_all = nuevo_modelo().fit(X[GRUPOS["LogReg_todas"]], d.y)
    frac_amb = m_all.predict(caracteristicas(amb)[GRUPOS["LogReg_todas"]]).mean()

    # ------------------------------------------------------------------ reporte markdown
    L = []
    L.append("# Resultados — clasificador reposo vs contracción (resumen 1 Hz)\n")
    L.append(f"Datos: `{CSV.name}` · {len(df)} filas · {len(sesiones)} sesiones · sujeto {df.pacient.iloc[0]}\n")
    L.append("Etiquetas: " + ", ".join(f"{k}={v}" for k, v in df.etiqueta.value_counts().items())
             + f"; rechazados por calidad (R1/R2) entre los etiquetados: "
             f"{int((df.etiqueta.isin(['reposo', 'contraccion']) & df.rechazado_calidad).sum())}"
             + f" → entrenamiento con {len(d)} segundos (reposo={int((d.y==0).sum())}, contracción={int(d.y.sum())}).\n")
    L.append("## Resumen LOSO (11 rotaciones; media ± DE entre sesiones; F1 global = matriz sumada)\n")
    L.append("| Método | Sensibilidad | Especificidad | F1 | F1 mín | F1 global | TP | FN | FP | TN |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in resumen.iterrows():
        L.append(f"| {r.metodo} | {r.sens_media:.3f} ± {r.sens_sd:.3f} | {r.espec_media:.3f} ± {r.espec_sd:.3f} "
                 f"| {r.F1_media:.3f} ± {r.F1_sd:.3f} | {r.F1_min:.3f} | {r.F1_global:.3f} "
                 f"| {r.TP} | {r.FN} | {r.FP} | {r.TN} |")
    L.append("")
    for mt in metodos:
        r = res[res.metodo == mt]
        L.append(f"### {mt} — por sesión\n")
        extra = "| k / T | rango óptimo en entrenamiento " if "Umbral" in mt else ""
        L.append(f"| Sesión | TP | FN | FP | TN | Sens | Espec | F1 {extra}|")
        L.append("|---|---|---|---|---|---|---|---" + ("|---|---" if extra else "") + "|")
        for _, x in r.iterrows():
            e = f"| {x.k:.2f} | {x.k_rango} " if extra else ""
            L.append(f"| {x.id_sesion.split('_')[1]} | {x.TP} | {x.FN} | {x.FP} | {x.TN} "
                     f"| {x.sensibilidad:.3f} | {x.especificidad:.3f} | {x.F1:.3f} {e}|")
        L.append("")
    L.append("## Coeficientes (regresión logística estandarizada, ajustada con las 11 sesiones)\n")
    for nombre, c in coefs.items():
        L.append(f"- **{nombre}**: " + ", ".join(f"{i}={v:+.2f}" for i, v in c.items()))
    L.append("")
    L.append(f"Segundos `ambiguo` (n={len(amb)}, sin verdad de referencia): el modelo LogReg_todas "
             f"predice contracción en el {100*frac_amb:.0f} %.\n")
    L.append("## Variables no usadas como entrada\n")
    L.append("- Prohibidas (fuga): " + ", ".join(PROHIBIDAS))
    L.append("- Constantes o vacías en las 11 sesiones (documentadas, no usadas): " + ", ".join(const))
    L.append("\n## Limitaciones (declarar siempre)\n")
    L.append("- **Circularidad de la etiqueta.** La etiqueta es una regla de amplitud relativa a la base "
             "(≥3× contracción, ≤1.6× reposo, intermedio descartado). Cualquier clasificador o umbral que vea "
             "`promedio_uv` puede reproducirla; F1 = 1.000 mide que se imitó la regla, no que se detecte "
             "contracción real. El hueco entre clases existe por construcción.")
    L.append("- **Un solo sujeto sano (MM-0001), un solo día.** LOSO estima la estabilidad entre sesiones del "
             "mismo sujeto, no la generalización a otros pacientes ni a amputados transtibiales.")
    L.append("- **Resumen a 1 Hz.** Se pierde la información espectral y la forma de onda; los segundos de "
             "transición mezclan reposo y contracción. La versión definitiva debe usar ventanas de la señal "
             "cruda a 2000 Hz (MAV, RMS, WL, ZC, SSC, MDF).")
    L.append("- **Etiquetado independiente pendiente.** Validar con un guion de calibración (instrucciones con "
             "marcas de tiempo) o con revisión visual de un fisioterapeuta, sin ver el umbral.")
    L.append("- **[DISEÑO]**: base = P20 de promedio_uv; k = 1.6 / 3.0; descarte de los 2 primeros segundos.")
    (OUT / "reporte_resultados.md").write_text("\n".join(L), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(resumen.round(3).to_string(index=False))
    print()
    print(res[["id_sesion", "metodo", "TP", "FN", "FP", "TN", "F1", "k", "k_rango"]].round(3).to_string(index=False))
    for n, c in coefs.items():
        print(n, c.round(2).to_dict())
    print(f"ambiguo -> contraccion: {frac_amb:.2f}")
    print("constantes:", const)


if __name__ == "__main__":
    main()
