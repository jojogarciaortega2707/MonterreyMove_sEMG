# Resultados — clasificador reposo vs contracción (resumen 1 Hz)

Datos: `monterrey_move_ML_unido.csv` · 1218 filas · 11 sesiones · sujeto MM-0001

Etiquetas: contraccion=556, reposo=531, ambiguo=109, artefacto_inicio=22; rechazados por calidad (R1/R2) entre los etiquetados: 17 → entrenamiento con 1070 segundos (reposo=523, contracción=547).

## Resumen LOSO (11 rotaciones; media ± DE entre sesiones; F1 global = matriz sumada)

| Método | Sensibilidad | Especificidad | F1 | F1 mín | F1 global | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|
| LogReg_todas | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 1.000 | 547 | 0 | 0 | 523 |
| LogReg_sin_amplitud | 0.817 ± 0.055 | 0.877 ± 0.050 | 0.848 ± 0.034 | 0.795 | 0.851 | 449 | 98 | 59 | 464 |
| Umbral_k_x_base | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 1.000 | 547 | 0 | 0 | 523 |
| Umbral_absoluto | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 1.000 | 547 | 0 | 0 | 523 |

### LogReg_todas — por sesión

| Sesión | TP | FN | FP | TN | Sens | Espec | F1 |
|---|---|---|---|---|---|---|---|
| 20261110-100742 | 50 | 0 | 0 | 102 | 1.000 | 1.000 | 1.000 |
| 20261110-100929 | 56 | 0 | 0 | 43 | 1.000 | 1.000 | 1.000 |
| 20261110-101109 | 53 | 0 | 0 | 39 | 1.000 | 1.000 | 1.000 |
| 20261110-101321 | 60 | 0 | 0 | 64 | 1.000 | 1.000 | 1.000 |
| 20261110-101453 | 48 | 0 | 0 | 31 | 1.000 | 1.000 | 1.000 |
| 20261110-101640 | 57 | 0 | 0 | 35 | 1.000 | 1.000 | 1.000 |
| 20261110-101815 | 48 | 0 | 0 | 34 | 1.000 | 1.000 | 1.000 |
| 20261110-101952 | 44 | 0 | 0 | 37 | 1.000 | 1.000 | 1.000 |
| 20261110-102132 | 47 | 0 | 0 | 38 | 1.000 | 1.000 | 1.000 |
| 20261110-102314 | 49 | 0 | 0 | 39 | 1.000 | 1.000 | 1.000 |
| 20261110-102511 | 35 | 0 | 0 | 61 | 1.000 | 1.000 | 1.000 |

### LogReg_sin_amplitud — por sesión

| Sesión | TP | FN | FP | TN | Sens | Espec | F1 |
|---|---|---|---|---|---|---|---|
| 20261110-100742 | 46 | 4 | 7 | 95 | 0.920 | 0.931 | 0.893 |
| 20261110-100929 | 50 | 6 | 6 | 37 | 0.893 | 0.860 | 0.893 |
| 20261110-101109 | 43 | 10 | 3 | 36 | 0.811 | 0.923 | 0.869 |
| 20261110-101321 | 48 | 12 | 7 | 57 | 0.800 | 0.891 | 0.835 |
| 20261110-101453 | 37 | 11 | 3 | 28 | 0.771 | 0.903 | 0.841 |
| 20261110-101640 | 49 | 8 | 4 | 31 | 0.860 | 0.886 | 0.891 |
| 20261110-101815 | 39 | 9 | 8 | 26 | 0.812 | 0.765 | 0.821 |
| 20261110-101952 | 33 | 11 | 6 | 31 | 0.750 | 0.838 | 0.795 |
| 20261110-102132 | 36 | 11 | 5 | 33 | 0.766 | 0.868 | 0.818 |
| 20261110-102314 | 41 | 8 | 6 | 33 | 0.837 | 0.846 | 0.854 |
| 20261110-102511 | 27 | 8 | 4 | 57 | 0.771 | 0.934 | 0.818 |

### Umbral_k_x_base — por sesión

| Sesión | TP | FN | FP | TN | Sens | Espec | F1 | k / T | rango óptimo en entrenamiento |
|---|---|---|---|---|---|---|---|---|---|
| 20261110-100742 | 50 | 0 | 0 | 102 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-100929 | 56 | 0 | 0 | 43 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-101109 | 53 | 0 | 0 | 39 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-101321 | 60 | 0 | 0 | 64 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-101453 | 48 | 0 | 0 | 31 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-101640 | 57 | 0 | 0 | 35 | 1.000 | 1.000 | 1.000 | 2.40 | 1.70-3.10 |
| 20261110-101815 | 48 | 0 | 0 | 34 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-101952 | 44 | 0 | 0 | 37 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-102132 | 47 | 0 | 0 | 38 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-102314 | 49 | 0 | 0 | 39 | 1.000 | 1.000 | 1.000 | 2.42 | 1.75-3.10 |
| 20261110-102511 | 35 | 0 | 0 | 61 | 1.000 | 1.000 | 1.000 | 2.45 | 1.75-3.15 |

### Umbral_absoluto — por sesión

| Sesión | TP | FN | FP | TN | Sens | Espec | F1 | k / T | rango óptimo en entrenamiento |
|---|---|---|---|---|---|---|---|---|---|
| 20261110-100742 | 50 | 0 | 0 | 102 | 1.000 | 1.000 | 1.000 | 120.50 | 95-146 uV |
| 20261110-100929 | 56 | 0 | 0 | 43 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-101109 | 53 | 0 | 0 | 39 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-101321 | 60 | 0 | 0 | 64 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-101453 | 48 | 0 | 0 | 31 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-101640 | 57 | 0 | 0 | 35 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-101815 | 48 | 0 | 0 | 34 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-101952 | 44 | 0 | 0 | 37 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-102132 | 47 | 0 | 0 | 38 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-102314 | 49 | 0 | 0 | 39 | 1.000 | 1.000 | 1.000 | 114.50 | 95-134 uV |
| 20261110-102511 | 35 | 0 | 0 | 61 | 1.000 | 1.000 | 1.000 | 110.50 | 87-134 uV |

## Coeficientes (regresión logística estandarizada, ajustada con las 11 sesiones)

- **LogReg_todas**: log_promedio=+2.53, log_maximo=+2.02, log_minimo=+1.86, log_desviacion=+1.34, max_sobre_prom=-0.70, min_sobre_prom=-1.39, cv=+0.26
- **LogReg_sin_amplitud**: max_sobre_prom=-0.21, min_sobre_prom=-5.80, cv=-2.09

Segundos `ambiguo` (n=109, sin verdad de referencia): el modelo LogReg_todas predice contracción en el 53 %.

## Variables no usadas como entrada

- Prohibidas (fuga): zona_dominante, pct_tiempo_objetivo, pct_tiempo_alto, pct_tiempo_bajo, promedio_pct_ref, repeticiones_validas, repeticiones_invalidas, puntaje_acumulado, activation_threshold_uV, release_threshold_uV, REF_min_uV, REF_max_uV, fase, segundo, id_sesion, base_estimada_uv
- Constantes o vacías en las 11 sesiones (documentadas, no usadas): date, pacient, physiotherapist, gender, age, hamstring_length_cm, hamstring_diameter_cm, side, excercise, Channel, electrode_type, electrode_spacing_cm, ground_site

## Limitaciones (declarar siempre)

- **Circularidad de la etiqueta.** La etiqueta es una regla de amplitud relativa a la base (≥3× contracción, ≤1.6× reposo, intermedio descartado). Cualquier clasificador o umbral que vea `promedio_uv` puede reproducirla; F1 = 1.000 mide que se imitó la regla, no que se detecte contracción real. El hueco entre clases existe por construcción.
- **Un solo sujeto sano (MM-0001), un solo día.** LOSO estima la estabilidad entre sesiones del mismo sujeto, no la generalización a otros pacientes ni a amputados transtibiales.
- **Resumen a 1 Hz.** Se pierde la información espectral y la forma de onda; los segundos de transición mezclan reposo y contracción. La versión definitiva debe usar ventanas de la señal cruda a 2000 Hz (MAV, RMS, WL, ZC, SSC, MDF).
- **Etiquetado independiente pendiente.** Validar con un guion de calibración (instrucciones con marcas de tiempo) o con revisión visual de un fisioterapeuta, sin ver el umbral.
- **[DISEÑO]**: base = P20 de promedio_uv; k = 1.6 / 3.0; descarte de los 2 primeros segundos.