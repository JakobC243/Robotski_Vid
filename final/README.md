# Final 9HPT pipeline

Ta mapa vsebuje aktualno kodo za obdelavo 9-hole peg test videov. Glavni vstop
je `run_hand_pipeline.py`; root `README.md` vsebuje celotna navodila za zagon.

## Hiter zagon

Iz korena projekta:

```powershell
.\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_001\patient_001camP_1_20241121_10_21_17.mp4 `
  --output .\outputs\p001_camP1_final.mp4 `
  --csv-output .\outputs\p001_camP1_final.csv `
  --calibration-output .\outputs\p001_camP1_calibration.json `
  --rotate-clockwise
```

Za odrezane videe brez LED zacetka:

```powershell
.\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_004\patient_004camP_1_20241010_14_47_20.mp4 `
  --output .\outputs\p004_camP1_fallback.mp4 `
  --csv-output .\outputs\p004_camP1_fallback.csv `
  --calibration-output .\outputs\p004_camP1_fallback_calibration.json `
  --rotate-clockwise `
  --hand-field-start-fallback on
```

## Moduli

- `run_hand_pipeline.py`: CLI, glavna frame zanka, timer, CSV, video output.
- `calibration.py`: kalibracija plosce iz 3x3 lukenj, rob plosce, fallback.
- `hand_tracking.py`: MediaPipe Hands, zaklep aktivne roke, soft ROI.
- `trial_timing.py`: LED start sekvenca.
- `peg_detection.py`: zaznava zaticov samo na ciljni 3x3 strani.
- `kinematics.py`: pot, hitrost, pospesek, razdalja palec-kazalec.
- `visualization.py`: overlay roke, lukenj, stoparice in grafov.
- `utils.py`: rotacija in video helperji.

## Trenutno vedenje

- Kalibracija pregleda vec zacetnih frame-ov in izbere najboljsi par 3x3 mrez.
- Grafi na videu so v metricnih enotah: pot v mm, hitrost v mm/s, pospesek v
  mm/s2 in razdalja palec-kazalec v mm. Pretvorba ni en sam faktor `px`,
  ampak homografija iz kalibrirane 3x3 mreze, kjer je razmik sosednjih lukenj
  32 mm.
- LED start je privzet in najbolj zaupanja vreden nacin za zacetek stoparice.
- `--hand-field-start-fallback on` je namenjen samo odrezanim posnetkom brez
  LED zacetka.
- Pri fallback startu detektor zaticov po prvem odhodu roke iz 3x3 polja
  primerja vseh 9 lukenj med sabo, da prvi ze odlozeni zatic ne postane prazna
  referenca.
- Timer se ustavi, ko je bilo dosezenih vsaj 8 zaticov in se ciljno polje nato
  stabilno izprazni.

Za vec primerov, parametre, CSV stolpce in seznam datotek za ciscenje glej
`README.md` v korenu projekta.
