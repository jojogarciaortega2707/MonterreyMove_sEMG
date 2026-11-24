"""
MONTERREY MOVE - realtime/azure/cliente_endpoint.py

Manda ventanas de 1 s de una sesion de ENTRENAMIENTO al endpoint de Azure y
mide el tiempo de ida y vuelta. Compara la respuesta con el modelo local.

  set MM_ENDPOINT_URL=https://<endpoint>.<region>.inference.ml.azure.com/score
  set MM_ENDPOINT_KEY=<clave>        (az ml online-endpoint get-credentials -n mm-semg-tiempo-real)
  python realtime/azure/cliente_endpoint.py dat1/raw/MM-0001_20261110-100929 --n 50
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.dont_write_bytecode = True
import envolvente as E  # noqa: E402
import rt_modelo as M   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sesion")
    ap.add_argument("--n", type=int, default=50, help="ventanas a mandar, una por peticion")
    a = ap.parse_args()
    url, key = os.environ.get("MM_ENDPOINT_URL"), os.environ.get("MM_ENDPOINT_KEY")
    if not url or not key:
        sys.exit("Define MM_ENDPOINT_URL y MM_ENDPOINT_KEY")

    _, uv = E.paquetes(E.envolvente(np.load(f"{a.sesion}_senal_cruda.npy").astype(np.float64)))
    local = M.ModeloCongelado("json")
    paso = max(1, (len(uv) // 20) // a.n)
    rtt, iguales = [], 0
    for k in range(0, len(uv) // 20, paso)[:a.n]:
        v = [round(float(x), 1) for x in uv[k * 20:(k + 1) * 20]]
        req = urllib.request.Request(url, data=json.dumps({"uv": v}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
        rtt.append((time.perf_counter() - t0) * 1000)
        resp = json.loads(resp) if isinstance(resp, str) else resp
        y_az = resp["predicciones"][0]["y"]
        y_lo = int(local.predecir(M.caracteristicas(*M.estadisticos(v)))[0])
        iguales += y_az == y_lo
    rtt = np.array(rtt)
    print(f"{len(rtt)} peticiones. Ida y vuelta: mediana {np.median(rtt):.0f} ms, p95 {np.percentile(rtt, 95):.0f} ms")
    print(f"Misma clase que el modelo local: {iguales}/{len(rtt)}")


if __name__ == "__main__":
    main()
