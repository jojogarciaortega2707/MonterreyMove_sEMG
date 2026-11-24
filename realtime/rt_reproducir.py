"""
MONTERREY MOVE - realtime/rt_reproducir.py

Reproduce una sesion GUARDADA como si fuera el servidor del juego en vivo:
manda por UDP un paquete cada 50 ms con la activacion (uV), con el mismo
formato basico que el servidor. Sirve para probar el clasificador en vivo
y el dashboard sin ESP32 ni electrodos.

  python realtime/rt_reproducir.py dat1/raw/MM-0001_20261110-100929        -> UDP 5007 (dashboard)
  python realtime/rt_reproducir.py dat1/raw/MM-0001_20261110-100929 --puerto 5008 --velocidad 4

Solo usar con sesiones de ENTRENAMIENTO. Las de test no se abren antes del test.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import envolvente as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sesion", help="ruta sin sufijo, p. ej. dat1/raw/MM-0001_20261110-100929")
    ap.add_argument("--puerto", type=int, default=5007)
    ap.add_argument("--velocidad", type=float, default=1.0, help="1 = tiempo real")
    ap.add_argument("--solo-nuevo", action="store_true",
                    help="los archivos son acumulativos: reproducir solo los segundos de ESTA sesion")
    a = ap.parse_args()

    base = Path(a.sesion)
    env = E.envolvente(np.load(f"{base}_senal_cruda.npy").astype(np.float64))
    t, uv = E.paquetes(env)
    fases = pd.read_csv(f"{base}_por_segundo.csv").fase.values
    i0 = 0
    if a.solo_nuevo:
        previos = sorted(base.parent.glob("*_por_segundo.csv"))
        previos = [p for p in previos if p.name < base.name + "_por_segundo.csv"]
        if previos:
            i0 = len(pd.read_csv(previos[-1])) * E.PAQUETES_POR_SEGUNDO

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    periodo = E.MUESTRAS_POR_BLOQUE / E.FS / a.velocidad
    print(f"Reproduciendo {base.name}: {(len(uv) - i0) / 20:.0f} s -> UDP 127.0.0.1:{a.puerto} "
          f"a x{a.velocidad:g}. Ctrl+C para parar.")
    sig = time.perf_counter()
    try:
        for j in range(i0, len(uv)):
            k = min(j // E.PAQUETES_POR_SEGUNDO, len(fases) - 1)
            p = {"t": round(float(t[j] - t[i0] + 0.05), 3), "uv": round(float(uv[j]), 1),
                 "phase": "REPRODUCCION", "msg": f"reproduccion {base.name} ({fases[k]})",
                 "zone": "BAJA", "state": "READY", "ref_uv": 0.0, "patient": base.name.split("_")[0]}
            s.sendto(json.dumps(p).encode("utf-8"), ("127.0.0.1", a.puerto))
            sig += periodo
            time.sleep(max(0.0, sig - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    print("Fin.")


if __name__ == "__main__":
    main()
