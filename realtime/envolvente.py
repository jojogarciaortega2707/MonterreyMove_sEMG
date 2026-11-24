"""
MONTERREY MOVE - realtime/envolvente.py

Copia exacta, sin dependencias del resto de la app, de la cadena de
ProcesadorEMG (monterrey_move_core.py):

    pasa-banda Butterworth orden 4, 20-450 Hz
    -> notch IIR 60, 120 y 180 Hz, Q = 30
    -> RMS movil de 150 ms (300 muestras)
    -> correccion de ganancia medida con ruido blanco (semilla 0)

Se verifico contra los *_por_segundo.csv de dat1/raw/: promedio, maximo,
minimo y desviacion por segundo coinciden con un error maximo de 0.05 uV
(el redondeo del CSV). Ver realtime/pruebas.py.

Tambien emula lo que el servidor del juego manda por UDP: un paquete cada
bloque de 100 muestras (50 ms, 20 Hz) con el ULTIMO valor de la envolvente
de ese bloque (servidor: self.activacion = float(act[-1])).

No modifica ni importa ningun archivo de la app.
"""
import numpy as np
from scipy.signal import butter, iirnotch, lfilter

FS = 2000
BANDA_HZ = (20.0, 450.0)
ORDEN = 4
NOTCH_HZ = (60.0, 120.0, 180.0)
NOTCH_Q = 30.0
VENTANA_RMS_MS = 150
MUESTRAS_POR_BLOQUE = 100          # 50 ms -> 20 paquetes por segundo
PAQUETES_POR_SEGUNDO = FS // MUESTRAS_POR_BLOQUE


def _filtros(fs=FS):
    nyq = fs / 2.0
    f = [butter(ORDEN, [BANDA_HZ[0] / nyq, BANDA_HZ[1] / nyq], btype="bandpass")]
    f += [iirnotch(f0 / nyq, Q=NOTCH_Q) for f0 in NOTCH_HZ]
    return f


def _correccion(filtros, fs=FS):
    prueba = np.random.default_rng(0).standard_normal(fs * 5)
    for b, a in filtros:
        prueba = lfilter(b, a, prueba)
    factor = float(np.sqrt(np.mean(prueba[fs:] ** 2)))
    return factor if factor > 1e-9 else 1.0


def envolvente(cruda, fs=FS):
    """Envolvente RMS en uV de una senal cruda completa, desde la muestra 0.

    Es causal y arranca con estado cero, igual que el ProcesadorEMG de una
    sesion nueva, asi que da el mismo resultado que la app.
    """
    f = _filtros(fs)
    y = np.asarray(cruda, dtype=float)
    for b, a in f:
        y = lfilter(b, a, y)
    n = int(VENTANA_RMS_MS / 1000.0 * fs)
    acum = np.cumsum(y * y)
    suma = acum.copy()
    suma[n:] = acum[n:] - acum[:-n]
    return np.sqrt(np.maximum(suma / n, 0.0)) / _correccion(f, fs)


def paquetes(env):
    """Lo que el servidor manda por UDP: env[-1] de cada bloque de 100 muestras.

    Devuelve (t_s, uv). t_s es el instante del final de cada bloque.
    """
    n = len(env) // MUESTRAS_POR_BLOQUE
    idx = np.arange(1, n + 1) * MUESTRAS_POR_BLOQUE - 1
    return (idx + 1) / FS, env[idx]
