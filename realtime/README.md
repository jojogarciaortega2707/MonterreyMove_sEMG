# Real-time classifier — contraction vs rest (complementary module)

This folder adds a **real-time** path and an **Azure** path to the project. It changes no existing file: the app, the dashboard, `code/`, `models/` and `DECISIONS.md` are untouched. The only file outside this folder is `.amlignore` at the root. Azure ML reads it to know what to leave out of the code it uploads.

**It is not a new model.** It runs the model frozen on 2026-11-20 (`LogReg_todas`, sha256 `6c844db8…`) without any change. The model that runs live is therefore the same model the sealed test evaluates.

## How it connects

```
ESP32 ))) server ──UDP 5007──> dashboard ──GET /api/vivo──> rt_servicio.py ──> http://127.0.0.1:8090
                                                    (mode A, default)      ├──> UDP 5009 (JSON per prediction)
                                                                           └──> realtime/registros/*.csv
ESP32 ))) server ──UDP 5008──> rt_servicio.py ──UDP 5007 (same bytes)──> dashboard
                    (mode B: lower delay; the server's clinical port is set to 5008 at launch)
```

- Every 50 ms the server sends one envelope value (µV). The classifier keeps the last **20 packets (1 s)** and computes the same 7 features as `code/03_clasificador.py`. It predicts every 50 ms, so 20 predictions per second.
- Compute time is 31–36 µs per packet (median). Delay added by the connection: mode A 3.8 ms (median; up to 100 ms because of polling), mode B 0.26 ms. In both modes the decision describes the **last 1 s** of signal.

## Files

| File | What it does |
|---|---|
| `envolvente.py` | Exact copy of the app's chain: band-pass 20–450 Hz, notches at 60/120/180 Hz, 150 ms RMS. It also rebuilds the server's 20 Hz packets. |
| `rt_modelo.py` | Loads the frozen model from the `.pkl` (hash checked against DECISIONS.md) or from the `.json` (numpy only, for Azure and the ESP32). Contains `ClasificadorEnVivo`. |
| `rt_servicio.py` | The live service: reads the dashboard (A) or relays UDP (B). Serves a web page and JSON, sends UDP and writes a CSV log. |
| `rt_reproducir.py` | Replays a saved training session as if it were the live server. For tests without hardware. |
| `rt_evaluar.py` | Feeds the saved raw signal packet by packet and reports metrics. Takes its labels from `code/07_test.py` (imported, not copied). |
| `pruebas.py` | Parity checks (below). |
| `azure/` | The sealed-test job, the separation and criteria checks, the online endpoint and its client. |
| `notebooks/colab_tiempo_real.ipynb` | The same checks in Google Colab, on training data only. |

## Checks on the training sessions (dat1/raw, 11 sessions)

`python realtime/pruebas.py`

| Check | Result |
|---|---|
| Envelope rebuilt from `.npy` vs the app's `*_por_segundo.csv` (1,218 s) | max error 0.05 µV (CSV rounding) |
| Features vs `code/03_clasificador.caracteristicas` | identical |
| JSON model (numpy) vs frozen pkl (scikit-learn) | max probability difference 8.2e-07, same class in all 1,218 s |

`python realtime/rt_evaluar.py …` Here *offline* means 2,000 envelope values per second (what `07_test.py` uses). *vivo_1s* means the 20 live packets of the same second. *vivo_20Hz* means every sliding window.

| Labels | Path | n | Accuracy | F1 |
|---|---|---|---|---|
| Amplitude (`etiqueta`) | offline / vivo_1s | 1,070 s | 1.000 / 1.000 | 1.000 / 1.000 |
| Amplitude (`etiqueta`) | vivo_20Hz | 17,334 windows | 0.993 | 0.993 |
| Calibration script, dry run | offline / vivo_1s | 323 s | 0.889 / 0.889 | 0.878 / 0.878 |
| Calibration script, dry run | vivo_20Hz | 5,149 windows | 0.931 | 0.917 |

The live 1-s path and the offline path agree on **99.6 %** of all 1,218 seconds.

How to read these numbers:
- **Amplitude labels are circular.** The labels came from an amplitude rule, so 1.000 shows only that the model copies that rule (as the main README already says).
- **The script dry run is not the test.** It uses training sessions. The script start was *estimated* by matching the signal (`results/guion_calibracion.csv`), and S01 and S07 are ambiguous. The errors are 35 false positives and 1 false negative. Most false positives are rest seconds right after a contraction, while the 150 ms RMS and the 1-s window are still coming down.
- **The game phase is a known limit.** Between game repetitions the signal rests for less than 1 s. A 1-s window almost always contains part of a contraction, so during the game the classifier reads continuous contraction (see the notebook figure). The sealed test does not evaluate game seconds. To separate game repetitions in real time, the next model needs shorter windows (250 ms, MAV/RMS/WL/ZC/SSC, as the main README plans). That would be a **new** model, frozen and tested again.

## Run it live

```
python monterrey_move_server.py --esp32 wifi        (as always)
python monterrey_move_dashboard.py                  (as always)
python realtime/rt_servicio.py                      (new, mode A)
```
Then open http://127.0.0.1:8090.

Mode B (lower delay): start the server with its clinical port set to 5008 instead of 5007. Then run `python realtime/rt_servicio.py --fuente udp`, which forwards every packet unchanged to the dashboard on 5007.

Without hardware: `python realtime/rt_reproducir.py dat1/raw/MM-0001_20261110-100929 --solo-nuevo`

## Azure (Microsoft Foundry / Azure Machine Learning)

Rule: **the test data lives only in Azure and never trains the model.** Three things enforce it:
1. The model was frozen (and its hash written in DECISIONS.md) before the test sessions existed. Nothing in Azure trains anything.
2. The test sessions are uploaded straight from the recording PC to an Azure data asset and mounted read-only. `.amlignore` keeps `data/test`, `dat1/test` and every `.npy` out of the code snapshot, and `realtime/.gitignore` keeps local logs out of GitHub.
3. `verificar_separacion.py` runs first and **stops the job** in any of these cases: a test session is also a training session, a test session is older than the freeze date, the model hash does not match, or `guion_inicio.csv` is missing a session.

You need an Azure Machine Learning workspace. In Foundry this is a hub-based project, which is an Azure ML workspace underneath. You also need the Azure CLI with the `ml` extension.

```bash
az extension add -n ml
az login
az configure --defaults group=<resource-group> workspace=<workspace>
az ml compute create -n cpu-mm --type amlcompute --size Standard_DS2_v2 --min-instances 0 --max-instances 1

# 1. Test data -> Azure only. Folder layout:  mm_test/raw/<id>_{senal_cruda.npy,por_segundo.csv,resumen.json}
#                                             mm_test/guion_inicio.csv   (id_sesion, inicio_guion_s or hora_inicio_guion)
az ml data create -n mm-test-sellado -v 1 --type uri_folder --path ./mm_test
#    then delete the local copy of mm_test

# 2. Sealed test, ONCE. The job history is the audit trail.
az ml job create -f realtime/azure/job_test_sellado.yml --stream
az ml job download -n <job-name> --output-name resultados --download-path results_azure
#    results_azure/.../criterios.md  -> C1 separation, C2 >= 4 new placements, C3 and C4 >= 75 %

# 3. Real-time endpoint (optional; billed hourly while it exists)
az ml model create -n mm-logreg-congelado -v 1 --path models --type custom_model
az ml online-endpoint create -f realtime/azure/endpoint/endpoint.yml
az ml online-deployment create -f realtime/azure/endpoint/deployment.yml --all-traffic
az ml online-endpoint invoke -n mm-semg-tiempo-real --request-file realtime/azure/endpoint/ejemplo.json
python realtime/azure/cliente_endpoint.py dat1/raw/MM-0001_20261110-100929   # round trip + agreement with local
az ml online-endpoint delete -n mm-semg-tiempo-real --yes
```

Which path to use for what:
- **Edge (laptop next to the patient)** is the real-time path: about 0.3 ms per decision, and it works without internet.
- **The Azure endpoint** shows the same model served from the cloud. Each request needs a network round trip, which `cliente_endpoint.py` measures.

The local dry run of the job (training data used as a fake test) behaved as expected:
- the separation check **stopped** it;
- when forced, `07_test.py` and `rt_evaluar.py` gave the same F1 (0.878);
- `07_test.py` refused a second run.

## Open decisions for the team

- **Test size.** DECISIONS.md declares 6 test sessions on 3 days. The current target is "4 new placements". Write the change in DECISIONS.md, with its date, before recording. The job enforces `--min-sesiones 4`.
- **Data folder.** The scripts default to `data/`, but the repository has `dat1/`. The Azure job passes its paths explicitly, so it is not affected.
- **Untested dashboard API.** `/api/vivo` was tested against the dashboard copy in the project files. If your local dashboard changed that route, use mode B.
- **Scope.** 1 healthy person, 1 muscle, 1 day. No claim for other people or for people with an amputation.
