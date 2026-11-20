"""
MONTERREY MOVE - Paso 2: control de calidad de la senal cruda (data/raw/*_senal_cruda.npy).

Los .npy son ACUMULATIVOS, igual que los CSV: cada archivo repite las muestras de los
anteriores. Se verifica el prefijo y se analizan solo las muestras nuevas de cada sesion.

Hallazgo: hay tramos planos (la misma muestra repetida) de 51, 101, 151, 301... muestras,
multiplos de 50 muestras = 25 ms. Son paquetes perdidos rellenados con el ultimo valor;
se cuentan como muestras FALTANTES.

Reglas [DISENO], sin usar etiquetas ni resultados del modelo. Unidad = 1 segundo, que es la
unidad que clasifica el modelo:
  R1  faltantes: una muestra falta si es NaN/inf o si pertenece a una racha de >= 10 muestras
      identicas seguidas (5 ms). Se rechaza el segundo si falta mas del 1 % de sus muestras.
  R2  saturacion: se rechaza el segundo si contiene el valor extremo exacto de la sesion
      repetido >= 3 veces (meseta en el limite).
  R3  duracion (por sesion, informativo): |muestras/2000 - filas del CSV| < 1 s.
Alineacion [DISENO]: el segundo k del CSV = muestras [2000k, 2000(k+1)) de la sesion.

Escribe results/calidad_senal.csv, data/segundos_calidad.csv y QUALITY-REPORT.md.
Los archivos crudos no se modifican.
Uso:  python code/02_calidad_senal.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FS = 2000
RACHA_MIN = 10          # muestras identicas seguidas que cuentan como faltantes
MAX_FALTANTES = 0.01    # por segundo
N_EXTREMO = 3


def mascara_rachas(x: np.ndarray, minimo: int) -> np.ndarray:
    """True en las muestras que pertenecen a una racha de >= minimo valores identicos."""
    cambio = np.flatnonzero(np.diff(x) != 0)
    ini = np.r_[0, cambio + 1]
    fin = np.r_[cambio, len(x) - 1]
    m = np.zeros(len(x), bool)
    for i, f in zip(ini, fin):
        if f - i + 1 >= minimo:
            m[i:f + 1] = True
    return m


def potencia_60(x: np.ndarray) -> float:
    """Potencia media en 59-61 Hz / mediana en 50-70 Hz (sin 58-62), segmentos de 1 s."""
    n = (len(x) // FS) * FS
    seg = x[:n].reshape(-1, FS)
    seg = seg - seg.mean(axis=1, keepdims=True)
    P = (np.abs(np.fft.rfft(seg * np.hanning(FS), axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(FS, 1 / FS)
    pico = P[(f >= 59) & (f <= 61)].mean()
    vecinos = P[(f >= 50) & (f <= 70) & ((f < 58) | (f > 62))]
    return float(pico / np.median(vecinos))


def main():
    ses_filas, seg_filas, previo, acum_csv = [], [], None, 0
    for f in sorted(RAW.glob("MM-*_senal_cruda.npy")):
        a = np.load(f, allow_pickle=False)
        n_prev = 0 if previo is None else len(previo)
        prefijo = True if previo is None else bool(np.array_equal(a[:n_prev], previo))
        previo = a
        x = a[n_prev:].astype(np.float64)
        sid = f.name.replace("_senal_cruda.npy", "")
        filas_csv = sum(1 for _ in open(RAW / f"{sid}_por_segundo.csv")) - 1
        filas_nuevas, acum_csv = filas_csv - acum_csv, filas_csv

        falt = ~np.isfinite(x)
        xf = np.where(falt, np.nan, x)
        falt |= mascara_rachas(x, RACHA_MIN)
        vmax, vmin = np.nanmax(xf), np.nanmin(xf)
        extremo = (x == vmax) | (x == vmin)

        n_seg = int(np.ceil(len(x) / FS))
        seg_ses = []
        for k in range(n_seg):
            sl = slice(k * FS, min((k + 1) * FS, len(x)))
            fr = float(falt[sl].mean())
            ext = int(extremo[sl].sum())
            r1, r2 = fr > MAX_FALTANTES, ext >= N_EXTREMO
            seg_ses.append(dict(id_sesion=sid, segundo=k, muestras=sl.stop - sl.start,
                                faltantes_pct=round(100 * fr, 2), muestras_extremo=ext,
                                R1_faltantes=r1, R2_saturacion=r2, rechazado_calidad=bool(r1 or r2)))
        seg_filas += seg_ses
        n_rech = sum(s["rechazado_calidad"] for s in seg_ses)
        dur = len(x) / FS
        # Cuantizacion: paso minimo entre valores distintos (= 1 cuenta del ADC en uV)
        niveles = np.unique(xf[np.isfinite(xf)])
        paso = float(np.diff(niveles).min()) if len(niveles) > 1 else np.nan
        en_rejilla = float(np.median(np.abs((niveles - niveles[0]) / paso - np.round((niveles - niveles[0]) / paso))))
        ses_filas.append(dict(
            id_sesion=sid, dtype=str(a.dtype), prefijo_igual=prefijo,
            muestras_nuevas=len(x), duracion_s=round(dur, 2), filas_csv_nuevas=filas_nuevas,
            dif_duracion_s=round(dur - filas_nuevas, 2), R3_duracion_ok=abs(dur - filas_nuevas) < 1,
            faltantes_pct=round(100 * falt.mean(), 2),
            segundos=n_seg, segundos_rechazados=int(n_rech),
            min_uv=round(float(vmin), 1), max_uv=round(float(vmax), 1),
            media_dc_uv=round(float(np.nanmean(xf)), 1),
            potencia_60hz_rel=round(potencia_60(np.nan_to_num(xf)), 1),
            paso_cuantizacion_uv=round(paso, 4), niveles_distintos=len(niveles),
            cuentas_usadas=int(round((vmax - vmin) / paso)) + 1, residuo_rejilla=round(en_rejilla, 4),
        ))

    q = pd.DataFrame(ses_filas)
    s = pd.DataFrame(seg_filas)
    (ROOT / "results").mkdir(exist_ok=True)
    q.to_csv(ROOT / "results" / "calidad_senal.csv", index=False)
    s.to_csv(ROOT / "data" / "segundos_calidad.csv", index=False)

    rech = s[s.rechazado_calidad]
    L = ["# Quality report",
         "",
         "Generated by `code/02_calidad_senal.py`. The raw files in `data/raw/` are not modified.",
         "",
         "## Raw signal (`data/raw/*_senal_cruda.npy`)",
         "",
         f"- {len(q)} files, dtype {q.dtype.iloc[0]}, cumulative. Each file repeats the previous one exactly: "
         f"{'yes, in all ' + str(len(q)) if q.prefijo_igual.all() else 'NO, see prefijo_igual'}.",
         f"- New samples per session: {q.muestras_nuevas.min():,} to {q.muestras_nuevas.max():,}. "
         f"Total: {q.muestras_nuevas.sum():,} samples = {q.duracion_s.sum():.1f} s at {FS} Hz.",
         f"- Sampling-rate check: raw duration minus rows of the per-second CSV = "
         f"{q.dif_duracion_s.min():+.2f} to {q.dif_duracion_s.max():+.2f} s per session (R3: "
         f"{int(q.R3_duracion_ok.sum())} of {len(q)} within ±1 s).",
         f"- Quantization: every value sits on a grid of {q.paso_cuantizacion_uv.min():.4f}–"
         f"{q.paso_cuantizacion_uv.max():.4f} µV (median residual from the grid {q.residuo_rejilla.max():.4f} steps). "
         f"That equals 3.3 V / 4095 / 142.8 = {3.3 / 4095 / 142.8 * 1e6:.4f} µV: a 12-bit count with the gain "
         "used in the software. "
         f"Distinct levels per session: {q.niveles_distintos.min()}–{q.niveles_distintos.max()}; counts spanned: "
         f"{q.cuentas_usadas.min():,}–{q.cuentas_usadas.max():,} of 4,096.",
         "- Flat stretches: the same value repeated 51, 101, 151, 301, 551 or 601 times, that is, multiples of "
         "50 samples (25 ms). They look like lost packets filled with the last value, so they count as missing samples.",
         "",
         "## Rules [DESIGN]",
         "",
         "The rules were written after a first model run. They use neither the labels nor any model score. "
         "The unit is one second, the unit the model classifies.",
         "",
         f"- **R1 missing samples.** A sample is missing if it is NaN/inf, or if it belongs to a run of {RACHA_MIN} or "
         f"more identical samples ({1000 * RACHA_MIN / FS:.0f} ms). A second is rejected if more than "
         f"{100 * MAX_FALTANTES:.0f} % of its samples are missing.",
         f"- **R2 saturation.** A second is rejected if it holds the exact extreme value of its session {N_EXTREMO} times "
         "or more. The converter limit in µV is not documented, so this is a proxy.",
         "- **R3 duration (per session).** |raw samples / 2000 − CSV rows| < 1 s.",
         "- **Alignment [DESIGN].** Second k of the CSV = samples [2000·k, 2000·(k+1)) of the session.",
         "",
         f"**Result: {len(rech)} of {len(s):,} seconds rejected** "
         f"(R1: {int(s.R1_faltantes.sum())}, R2: {int(s.R2_saturacion.sum())}). No whole session is rejected.",
         "",
         "| Session | Samples | s | Δ vs CSV (s) | Missing % | Seconds rejected | Min µV | Max µV | DC µV | 60 Hz rel. |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in q.iterrows():
        L.append(f"| {r.id_sesion.split('_')[1]} | {r.muestras_nuevas:,} | {r.duracion_s:.2f} | {r.dif_duracion_s:+.2f} "
                 f"| {r.faltantes_pct:.2f} | {r.segundos_rechazados} | {r.min_uv} | {r.max_uv} | {r.media_dc_uv} "
                 f"| {r.potencia_60hz_rel} |")
    if len(rech):
        L += ["", "Rejected seconds: " + "; ".join(
            f"{sid.split('_')[1]}: " + ", ".join(str(v) for v in g.segundo)
            for sid, g in rech.groupby("id_sesion", sort=False)) + "."]
    L += ["",
          "60 Hz rel. = mean power at 59–61 Hz divided by the median power at 50–70 Hz (outside 58–62 Hz), "
          "from 1-s segments. A value near 1 means no visible power-line peak in the raw signal.",
          "",
          "## Per-second summary (`data/monterrey_move_ML_unido.csv`)",
          ""]
    uni = ROOT / "data" / "monterrey_move_ML_unido.csv"
    if uni.exists():
        u = pd.read_csv(uni).merge(s[["id_sesion", "segundo", "rechazado_calidad"]],
                                   on=["id_sesion", "segundo"], how="left")
        u["rechazado_calidad"] = u.rechazado_calidad.fillna(False).astype(bool)
        e = u.etiqueta.value_counts()
        usable = u[u.etiqueta.isin(["reposo", "contraccion"]) & ~u.rechazado_calidad]
        L += [f"- First 2 s of each session rejected as a start artefact: {e.get('artefacto_inicio', 0)} of {len(u):,} s.",
              f"- Seconds between 1.6× and 3× the rest baseline are marked `ambiguo` and left out of training: "
              f"{e.get('ambiguo', 0)} s.",
              f"- Labelled seconds rejected by R1/R2: "
              f"{int((u.etiqueta.isin(['reposo', 'contraccion']) & u.rechazado_calidad).sum())}.",
              f"- **{len(usable):,} s remain: {int((usable.etiqueta == 'reposo').sum())} rest and "
              f"{int((usable.etiqueta == 'contraccion').sum())} contraction.**"]
    (ROOT / "QUALITY-REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(q.to_string(index=False))
    print("\n" + "\n".join(L[-12:]))


if __name__ == "__main__":
    main()
