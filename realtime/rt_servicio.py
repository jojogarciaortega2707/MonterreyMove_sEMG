"""
================================================================
 MONTERREY MOVE - CLASIFICADOR EN TIEMPO REAL  (complementario)
================================================================

Escucha la activacion que ya calcula el servidor del juego y, cada 50 ms,
dice si el musculo esta en CONTRACCION o en REPOSO con el modelo congelado
(LogReg_todas, models/modelo_congelado.*).

NO modifica nada de la app. Es un programa aparte que se conecta a la
SALIDA del dashboard. Dos formas:

  A) --fuente dashboard   (por defecto, cero configuracion)
        servidor --UDP 5007--> dashboard --HTTP /api/vivo--> ESTE PROGRAMA
     Pregunta a http://127.0.0.1:8080/api/vivo cada 100 ms y toma los
     paquetes nuevos. No hay que cambiar ni un parametro de la app.
     Retraso extra: hasta 100 ms.

  B) --fuente udp         (menor retraso)
        servidor --UDP 5008--> ESTE PROGRAMA --UDP 5007--> dashboard
     Recibe el paquete del servidor, lo clasifica y lo reenvia SIN TOCARLO
     al dashboard. Solo hay que arrancar el servidor con su puerto clinico
     en 5008 en lugar de 5007 (parametro de arranque, no cambio de codigo).

Salidas:
  http://127.0.0.1:8090          pantalla: senal en vivo coloreada por clase + CONTRACCION/REPOSO
  http://127.0.0.1:8090/api/estado   JSON (para Unity, Azure o lo que sea)
  UDP 127.0.0.1:5009             un JSON por prediccion (--udp-salida)
  realtime/registros/*.csv       una fila por prediccion

Uso:
  python realtime/rt_servicio.py
  python realtime/rt_servicio.py --fuente udp --udp-puerto 5008 --reenviar 127.0.0.1:5007

AVISO: prototipo academico, no es un dispositivo medico. El modelo se
entreno con 1 persona sana, 1 musculo, 1 dia.
================================================================
"""
import argparse
import collections
import csv
import json
import socket
import sys
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import rt_modelo as M  # noqa: E402

HISTORIA = 1200        # 60 s de predicciones a 20 Hz


class Estado:
    """Lo que se sabe ahora. Lo escriben las fuentes, lo leen la web y el registro."""

    def __init__(self, modelo, salida_udp=None, ruta_csv=None):
        self.lock = threading.Lock()
        self.cl = M.ClasificadorEnVivo(modelo)
        self.modelo = modelo
        self.t = collections.deque(maxlen=HISTORIA)
        self.uv = collections.deque(maxlen=HISTORIA)
        self.prob = collections.deque(maxlen=HISTORIA)
        self.y = collections.deque(maxlen=HISTORIA)
        self.n_contracciones = 0       # cambios REPOSO -> CONTRACCION desde que arranco
        self.n_contr_pred = 0          # predicciones en CONTRACCION
        self.t_cambio = None           # t del servidor del ultimo cambio de clase
        self.ultimo = {}
        self.fase = ""
        self.t_ult = None
        self.n = 0
        self.lat_ms = collections.deque(maxlen=400)
        self.visto = 0.0
        self.fuente_txt = ""
        self.error = ""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if salida_udp else None
        self.dest = salida_udp
        self.csv = None
        if ruta_csv:
            ruta_csv.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(ruta_csv, "w", newline="", encoding="utf-8")
            self.csv = csv.writer(self._f)
            self.csv.writerow(["hora", "t_servidor_s", "fase", "uv", "promedio_uv", "maximo_uv",
                               "minimo_uv", "desviacion_uv", "prob_contraccion", "clase"])

    def agregar(self, t, uv, fase="", llegada=None):
        """Un paquete del servidor. llegada = perf_counter() en que se recibio."""
        llegada = llegada or time.perf_counter()
        with self.lock:
            # Sesion nueva o reinicio del servidor: el tiempo retrocede o salta.
            if self.t_ult is not None and t <= self.t_ult - 0.5:
                # El servidor se reinicio (su reloj volvio a empezar): sesion nueva,
                # se borra la historia para no mezclar dos relojes en la grafica.
                self.cl.reiniciar()
                for d in (self.t, self.uv, self.prob, self.y):
                    d.clear()
                self.ultimo, self.t_cambio = {}, None
                self.n = self.n_contracciones = self.n_contr_pred = 0
            elif self.t_ult is not None and t > self.t_ult + 2.0:
                self.cl.reiniciar()                 # hueco largo: la ventana de 1 s ya no es continua
            elif self.t_ult is not None and t <= self.t_ult:
                return                              # paquete repetido o fuera de orden
            self.t_ult = t
            self.fase = fase or self.fase
            r = self.cl.agregar(uv)
            self.visto = time.time()
            if r is None:
                return
            self.n += 1
            lat = (time.perf_counter() - llegada) * 1000
            self.lat_ms.append(lat)
            r.update(t=round(t, 3), uv=round(uv, 1), fase=self.fase, n=self.n,
                     latencia_ms=round(lat, 3))
            if self.ultimo and r["y"] != self.ultimo.get("y"):
                self.t_cambio = t
                if r["y"] == 1:
                    self.n_contracciones += 1
            elif self.t_cambio is None:
                self.t_cambio = t
                self.n_contracciones += r["y"]
            self.n_contr_pred += r["y"]
            r["s_en_clase"] = round(t - self.t_cambio, 2)
            self.ultimo = r
            self.t.append(round(t, 2))
            self.uv.append(round(uv, 1))
            self.prob.append(r["prob_contraccion"])
            self.y.append(r["y"])
        if self.sock:
            try:
                self.sock.sendto(json.dumps(r).encode("utf-8"), self.dest)
            except OSError:
                pass
        if self.csv:
            self.csv.writerow([datetime.now().isoformat(timespec="milliseconds"), r["t"], r["fase"], r["uv"],
                               r["promedio_uv"], r["maximo_uv"], r["minimo_uv"], r["desviacion_uv"],
                               r["prob_contraccion"], r["clase"]])
            if self.n % 20 == 0:
                self._f.flush()

    def foto(self):
        with self.lock:
            vivo = bool(self.visto) and time.time() - self.visto < 2.0
            lat = sorted(self.lat_ms)
            return {"conectado": vivo, "fuente": self.fuente_txt, "error": self.error,
                    "modelo": M.METODO, "sha256": self.modelo.sha256[:16],
                    "ultimo": self.ultimo, "predicciones": self.n,
                    "latencia_ms_mediana": round(lat[len(lat) // 2], 3) if lat else None,
                    "t": list(self.t), "uv": list(self.uv), "prob": list(self.prob), "y": list(self.y),
                    "contracciones": self.n_contracciones,
                    "pct_contraccion": round(100 * self.n_contr_pred / self.n, 1) if self.n else None}


# ---------------------------------------------------------------- fuentes
def fuente_dashboard(est, url, periodo):
    """Lee la salida del dashboard: GET /api/vivo devuelve los ultimos 60 s de paquetes."""
    est.fuente_txt = f"dashboard {url}/api/vivo"
    ult = None
    while True:
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(url + "/api/vivo", timeout=2) as r:
                d = json.loads(r.read().decode("utf-8"))
            est.error = d.get("error", "") or ("" if d.get("conectado") else
                                               "el dashboard no recibe al servidor del juego")
            ts, uvs = d.get("t") or [], d.get("serie_uv") or []
            # La historia del dashboard no se borra cuando el servidor se reinicia: puede traer
            # dos relojes seguidos. Solo vale el tramo despues del ultimo retroceso del reloj.
            k = max((i for i in range(1, len(ts)) if ts[i] < ts[i - 1] - 0.5), default=0)
            ts, uvs = ts[k:], uvs[k:]
            if ts and ult is not None and ts[-1] < ult - 0.5:
                ult = None                                   # el servidor se reinicio
            nuevos = [(t, u) for t, u in zip(ts, uvs) if ult is None or t > ult + 1e-6]
            if ult is None:
                nuevos = nuevos[-M.VENTANA_PAQUETES:]        # al conectar, solo el ultimo segundo
            for t, u in nuevos:
                est.agregar(float(t), float(u), d.get("phase", ""), llegada=t0)
            if nuevos:
                ult = nuevos[-1][0]
        except Exception as e:  # noqa: BLE001
            est.error = f"no se pudo leer el dashboard en {url}: {e}"
        time.sleep(max(0.0, periodo - (time.perf_counter() - t0)))


def fuente_udp(est, puerto, reenviar):
    """Recibe el paquete del servidor, lo clasifica y lo reenvia intacto."""
    est.fuente_txt = f"UDP {puerto}" + (f" -> {reenviar[0]}:{reenviar[1]}" if reenviar else "")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", puerto))
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if reenviar else None
    while True:
        datos, _ = s.recvfrom(16384)
        llegada = time.perf_counter()
        if out:
            out.sendto(datos, reenviar)                      # primero el dashboard, sin cambios
        try:
            p = json.loads(datos.decode("utf-8"))
            est.agregar(float(p.get("t", 0.0)), float(p["uv"]), str(p.get("phase", "")), llegada)
        except (ValueError, KeyError, UnicodeDecodeError):
            continue


# ---------------------------------------------------------------- web
PAGINA = r"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clasificador en vivo · MONTERREY MOVE</title>
<style>
:root{--tinta:#0F141F;--panel:#171C29;--panel2:#242B3B;--borde:rgba(255,255,255,.13);--texto:#F2F5FA;
--suave:#B8C2D4;--tenue:#808CA1;--contr:#4DD973;--repo:#5CADFF;--alta:#FF6B61}
*{box-sizing:border-box}
body{margin:0;background:var(--tinta);color:var(--texto);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 22px 40px}
.top{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:15px;letter-spacing:.06em;margin:0}
.pill{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--suave);background:var(--panel);
 border:1px solid var(--borde);border-radius:20px;padding:4px 12px;margin-left:auto}
.dot{width:8px;height:8px;border-radius:50%;background:var(--alta)}.dot.on{background:var(--contr)}
.card{background:var(--panel);border:1px solid var(--borde);border-radius:12px;padding:16px 18px;margin-top:14px}
/* ---- clase en vivo ---- */
#banda{display:flex;align-items:center;gap:26px;flex-wrap:wrap;transition:background .15s,border-color .15s}
#banda.c{background:rgba(77,217,115,.13);border-color:rgba(77,217,115,.55)}
#banda.r{background:rgba(92,173,255,.10);border-color:rgba(92,173,255,.45)}
#icono{width:84px;height:84px;border-radius:50%;flex:0 0 84px;background:var(--panel2);
 display:grid;place-items:center;font-size:13px;font-weight:800;letter-spacing:.06em;color:#0b1118}
#banda.c #icono{background:var(--contr);box-shadow:0 0 0 8px rgba(77,217,115,.18)}
#banda.r #icono{background:var(--repo);box-shadow:0 0 0 8px rgba(92,173,255,.15)}
.et{font-size:10.5px;color:var(--tenue);text-transform:uppercase;letter-spacing:.08em}
#clase{font-size:58px;font-weight:800;line-height:1;letter-spacing:.01em}
#banda.c #clase{color:var(--contr)} #banda.r #clase{color:var(--repo)}
#desde{font-size:12.5px;color:var(--suave);margin-top:6px}
.kpis{display:flex;gap:26px;margin-left:auto;flex-wrap:wrap}
.k .v{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.15}
.k .u{font-size:12px;color:var(--suave);font-weight:400}
.meter{width:150px;height:8px;border-radius:5px;background:var(--panel2);margin-top:6px;position:relative;overflow:hidden}
.meter i{position:absolute;left:0;top:0;bottom:0;background:var(--contr)}
.meter b{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:rgba(255,255,255,.5)}
/* ---- señal ---- */
.hd{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:8px}
.hd h3{margin:0;font-size:14.5px}.hint{font-size:11.5px;color:var(--tenue)}
.tabs{margin-left:auto;display:flex;gap:4px;background:var(--tinta);padding:3px;border-radius:8px;border:1px solid var(--borde)}
.tabs button{background:transparent;border:0;color:var(--suave);font:inherit;font-size:12px;font-weight:600;
 padding:5px 12px;border-radius:6px;cursor:pointer}.tabs button.act{background:var(--panel2);color:var(--texto)}
canvas{width:100%;display:block}
.ley{display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:var(--suave);margin-top:8px}
.ley i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}
.st{background:var(--panel);border:1px solid var(--borde);border-radius:10px;padding:10px 12px}
.st .v{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.nota{font-size:11.5px;color:var(--tenue);margin-top:12px;line-height:1.55}
</style></head><body><main>
<div class="top"><h1>MONTERREY MOVE · CLASIFICADOR EN VIVO</h1>
 <div class="pill"><span class="dot" id="dot"></span><span id="sub">conectando…</span></div></div>

<div class="card" id="banda">
 <div id="icono">—</div>
 <div><div class="et">Clasificación en vivo</div><div id="clase">—</div><div id="desde">esperando señal…</div></div>
 <div class="kpis">
  <div class="k"><div class="et">P(contracción)</div><div class="v" id="prob">—</div>
   <div class="meter"><i id="barra" style="width:0"></i><b></b></div></div>
  <div class="k"><div class="et">Activación</div><div class="v" id="uv">—</div></div>
  <div class="k"><div class="et">Fase del juego</div><div class="v" id="fase" style="font-size:17px">—</div></div>
 </div>
</div>

<div class="card">
 <div class="hd"><h3>Señal en vivo</h3>
  <span class="hint">activación RMS (µV), coloreada por la clase que decidió el modelo</span>
  <div class="tabs" id="tabs"><button data-s="10" class="act">10 s</button><button data-s="30">30 s</button><button data-s="60">60 s</button></div>
 </div>
 <canvas id="cv"></canvas>
 <div class="ley"><span><i style="background:#4DD973"></i>contracción</span><span><i style="background:#5CADFF"></i>reposo</span>
  <span><i style="background:rgba(77,217,115,.16)"></i>tramo clasificado como contracción</span>
  <span>franja inferior: clase en cada ventana (cada 50 ms)</span></div>
</div>

<div class="stats">
 <div class="st"><div class="et">Contracciones detectadas</div><div class="v" id="ncon">—</div></div>
 <div class="st"><div class="et">Tiempo en contracción</div><div class="v" id="pcon">—</div></div>
 <div class="st"><div class="et">Predicciones</div><div class="v" id="npred">—</div></div>
 <div class="st"><div class="et">Latencia interna</div><div class="v" id="lat">—</div></div>
</div>
<div class="nota">Modelo congelado <b id="mod"></b>. Cada decisión usa el último segundo de señal (20 paquetes) y se
actualiza cada 50 ms. La señal es la envolvente RMS que el servidor manda al juego; la señal cruda de 2000 Hz no sale
del servidor. Prototipo académico, no es un dispositivo médico. Entrenado con 1 persona sana, 1 músculo y 1 día.</div>
</main><script>
"use strict";
const $=s=>document.querySelector(s);
const CC="#4DD973", CR="#5CADFF";
let VENT=10, D=null;
$("#tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{
  VENT=+b.dataset.s; $("#tabs").querySelectorAll("button").forEach(x=>x.classList.toggle("act",x===b)); dib();});

function paso(r){const e=Math.pow(10,Math.floor(Math.log10(r)));const f=r/e;return (f<=1?1:f<=2?2:f<=5?5:10)*e;}

function dib(){
  const c=$("#cv"),dpr=devicePixelRatio||1,W=c.clientWidth,H=340;
  c.style.height=H+"px";c.width=W*dpr;c.height=H*dpr;
  const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  const L=56,R=14,T=14,hS=18,B=26,gH=H-T-B-hS-10, yS=T+gH+10;
  x.fillStyle="rgba(0,0,0,.22)";x.fillRect(L,T,W-L-R,gH);
  if(!D||!D.t||D.t.length<2){x.fillStyle="#808CA1";x.font="13px system-ui";x.textAlign="center";
    x.fillText(D&&D.error?D.error:"Esperando señal del servidor…",W/2,T+gH/2);return;}
  const tN=D.t[D.t.length-1], t0=tN-VENT;
  let i0=D.t.findIndex(t=>t>=t0); if(i0<0)i0=0;
  const ts=D.t.slice(i0), uv=D.uv.slice(i0), y=D.y.slice(i0);
  const X=t=>L+(t-t0)/VENT*(W-L-R);
  const top=Math.max(60,...uv)*1.12, st=paso(top/4), yMax=Math.ceil(top/st)*st;
  const Y=v=>T+gH-Math.min(v,yMax)/yMax*gH;
  // bandas de contraccion
  x.fillStyle="rgba(77,217,115,.16)";
  for(let i=0;i<ts.length;i++){ if(y[i]!==1)continue; let j=i; while(j+1<ts.length&&y[j+1]===1)j++;
    const a=X(i?ts[i-1]:ts[i]), b=X(ts[j]); x.fillRect(a,T,Math.max(1,b-a),gH); i=j; }
  // rejilla y ejes
  x.font="10.5px system-ui";x.textAlign="right";x.textBaseline="middle";
  for(let v=0;v<=yMax+1e-9;v+=st){const yy=Math.round(Y(v))+.5;x.strokeStyle="rgba(255,255,255,.07)";x.lineWidth=1;
    x.beginPath();x.moveTo(L,yy);x.lineTo(W-R,yy);x.stroke();x.fillStyle="#808CA1";x.fillText(String(Math.round(v)),L-8,yy);}
  x.textAlign="left";x.fillStyle="#B8C2D4";x.fillText("µV",L+6,T+8);
  x.textAlign="center";x.textBaseline="top";const pS=VENT<=10?2:VENT<=30?5:10;
  for(let s=0;s<=VENT;s+=pS){const xx=X(t0+s);x.fillStyle="#808CA1";x.fillText(s===VENT?"ahora":"-"+(VENT-s)+" s",xx,yS+hS+6);}
  // señal coloreada por clase
  x.lineWidth=2;x.lineJoin="round";
  for(let i=1;i<ts.length;i++){x.strokeStyle=y[i]===1?CC:CR;x.beginPath();
    x.moveTo(X(ts[i-1]),Y(uv[i-1]));x.lineTo(X(ts[i]),Y(uv[i]));x.stroke();}
  // franja de clase
  for(let i=0;i<ts.length;i++){const a=X(i?ts[i-1]:ts[i]-.05),b=X(ts[i]);x.fillStyle=y[i]===1?CC:CR;
    x.fillRect(a,yS,Math.max(1,b-a+.5),hS);}
  x.strokeStyle="rgba(255,255,255,.15)";x.strokeRect(L+.5,yS+.5,W-L-R-1,hS-1);
  // punto actual
  const k=ts.length-1;x.fillStyle=y[k]===1?CC:CR;x.beginPath();x.arc(X(ts[k]),Y(uv[k]),4.5,0,7);x.fill();
}

async function tick(){
  try{ D=await (await fetch("/api/estado",{cache:"no-store"})).json(); }
  catch(e){ D=null; $("#dot").classList.remove("on"); $("#sub").textContent="sin conexión con el clasificador"; dib(); return; }
  $("#dot").classList.toggle("on",D.conectado);
  $("#sub").textContent=(D.conectado?"en vivo · ":"sin datos · ")+D.fuente+(D.error?" · "+D.error:"");
  $("#mod").textContent=D.modelo+" · sha256 "+D.sha256+"…";
  const u=D.ultimo||{};
  if(u.clase){
    const c=u.clase==="CONTRACCION";
    $("#banda").className="card "+(c?"c":"r");
    $("#clase").textContent=c?"CONTRACCIÓN":"REPOSO";
    $("#icono").textContent=c?"ACTIVO":"REPOSO";
    $("#desde").textContent=(D.conectado?"desde hace ":"última decisión, duró ")+(u.s_en_clase??0).toFixed(1)+" s";
    $("#prob").textContent=(u.prob_contraccion*100).toFixed(0)+" %";
    $("#barra").style.width=(u.prob_contraccion*100).toFixed(1)+"%";
    $("#barra").style.background=c?CC:CR;
    $("#uv").innerHTML=u.uv.toFixed(0)+' <span class="u">µV</span>';
    $("#fase").textContent=u.fase||"—";
  }
  $("#ncon").textContent=D.contracciones??"—";
  $("#pcon").textContent=D.pct_contraccion==null?"—":D.pct_contraccion.toFixed(0)+" %";
  $("#npred").textContent=D.predicciones;
  $("#lat").textContent=D.latencia_ms_mediana==null?"—":D.latencia_ms_mediana.toFixed(2)+" ms";
  dib();
}
addEventListener("resize",dib);
setInterval(tick,100);tick();
</script></body></html>"""


def servir_web(est, puerto):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _env(self, cuerpo, tipo):
            b = cuerpo.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/api/estado"):
                return self._env(json.dumps(est.foto(), ensure_ascii=False), "application/json; charset=utf-8")
            return self._env(PAGINA, "text/html; charset=utf-8")

    srv = ThreadingHTTPServer(("127.0.0.1", puerto), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _hostport(s):
    h, p = s.rsplit(":", 1)
    return h, int(p)


def main():
    ap = argparse.ArgumentParser(description="Clasificador en tiempo real (complementario) de MONTERREY MOVE")
    ap.add_argument("--fuente", choices=["dashboard", "udp"], default="dashboard")
    ap.add_argument("--dashboard", default="http://127.0.0.1:8080")
    ap.add_argument("--periodo", type=float, default=0.1, help="s entre lecturas del dashboard")
    ap.add_argument("--udp-puerto", type=int, default=5008)
    ap.add_argument("--reenviar", default="127.0.0.1:5007",
                    help="a donde reenviar el paquete intacto en modo udp; 'no' para no reenviar")
    ap.add_argument("--puerto-web", type=int, default=8090)
    ap.add_argument("--udp-salida", default="127.0.0.1:5009", help="'no' para desactivar")
    ap.add_argument("--modelo", choices=["json", "pkl"], default="json")
    ap.add_argument("--sin-registro", action="store_true")
    a = ap.parse_args()

    modelo = M.ModeloCongelado(a.modelo)
    ruta = None if a.sin_registro else AQUI / "registros" / f"rt_{datetime.now():%Y%m%d-%H%M%S}.csv"
    est = Estado(modelo, None if a.udp_salida == "no" else _hostport(a.udp_salida), ruta)
    servir_web(est, a.puerto_web)

    print("=" * 66)
    print(" MONTERREY MOVE - CLASIFICADOR EN TIEMPO REAL (complementario)")
    print("=" * 66)
    print(f"  Modelo        {M.METODO} ({a.modelo}), pkl sha256 {modelo.sha256[:16]}…")
    print(f"  Pantalla      http://127.0.0.1:{a.puerto_web}")
    print(f"  JSON          http://127.0.0.1:{a.puerto_web}/api/estado")
    if a.udp_salida != "no":
        print(f"  UDP salida    {a.udp_salida}")
    if ruta:
        print(f"  Registro      {ruta}")
    if a.fuente == "dashboard":
        print(f"  Fuente        {a.dashboard}/api/vivo (cada {a.periodo * 1000:.0f} ms)")
        hilo = threading.Thread(target=fuente_dashboard, args=(est, a.dashboard.rstrip("/"), a.periodo),
                                daemon=True)
    else:
        rv = None if a.reenviar == "no" else _hostport(a.reenviar)
        print(f"  Fuente        UDP {a.udp_puerto}" + (f", reenvia a {a.reenviar}" if rv else ""))
        print(f"  Arranca el servidor del juego con su puerto clinico en {a.udp_puerto}.")
        hilo = threading.Thread(target=fuente_udp, args=(est, a.udp_puerto, rv), daemon=True)
    hilo.start()
    print("  Ctrl+C para salir.\n")
    try:
        while True:
            time.sleep(1)
            u = est.ultimo
            if u:
                sys.stdout.write(f"\r  {u['clase']:<12} p={u['prob_contraccion']:.2f}  {u['uv']:>7.1f} uV  "
                                 f"{u.get('fase', ''):<12} n={est.n}   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n  Detenido.")


if __name__ == "__main__":
    main()
