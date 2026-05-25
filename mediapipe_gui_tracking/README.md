# mediapipe_gui_tracking

Nova, izolirana implementacija za analizo testa s pini/zatici. Obstojece skripte v `Robotski_Vid/` ostanejo nedotaknjene; ta mapa uporablja isto osnovno idejo kalibracije dveh 3x3 polj, vendar doda GUI-video izhod z grafi in loceno primerjavo s CoTracker3 sledilnikom iz vaje 9.

## Glavni zagon

Iz korena projekta:

```powershell
.\.venv\Scripts\python.exe .\mediapipe_gui_tracking\mediapipe_pin_gui_tracker.py `
  --input .\data\patient_001\patient_001camP_0_20241121_10_21_17.mp4 `
  --output-dir .\mediapipe_gui_tracking\outputs_test `
  --smooth-window 5
```

Ce `--input` ni podan, skripta poisce prvi `.mp4` pod `../data` ali `data`.

Glavni izhodi:

- `*_gui.mp4` ali fallback `*_gui.avi`
- `*.csv`
- `*_summary.json`
- `*_calibration.json`
- `*_calibration_debug.png`
- PNG grafi za hitrost, pospesek, razdaljo, zasedenost in trajektorijo

## Primerjava MediaPipe in vaja 9

```powershell
.\.venv\Scripts\python.exe .\mediapipe_gui_tracking\mediapipe_comparison_vaja9.py `
  --input .\data\patient_001\patient_001camP_0_20241121_10_21_17.mp4 `
  --output-dir .\mediapipe_gui_tracking\outputs_cotracker_test `
  --smooth-window 5
```

Primerjalna skripta najprej uporabi MediaPipe tocko kazalca za inicializacijo, nato pa to isto tocko poslje v CoTracker3 tracker iz `vaje/.../cotracker_studetns`. Shrani primerjalni video, CSV, summary JSON in PNG grafe. Za ta zagon mora biti na voljo okolje s `torch` in `cotracker`.

## Uporabni argumenti

- `--calibration-cache path\to\calibration.json` ponovno uporabi kalibracijo, ce se velikost frame-a ujema.
- `--max-frames 300` obdela samo prvih 300 frame-ov za hiter test.
- `--show` prikaze GUI med obdelavo.
- `--debug` obdrzi dodatne debug izhode.
- `--smooth-window 3..5` kratko glajenje polozaja, hitrosti in pospeska.

V tem projektu so osnovne MediaPipe odvisnosti ze namescene v `.venv`. Ce uporabljas drug Python, namesti pakete iz `requirements-local.txt`.
