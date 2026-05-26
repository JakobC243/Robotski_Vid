# Avtomatska analiza testa devetih zatičev z robotskim vidom

Projekt analizira videoposnetke testa devetih zatičev. Sistem iz enega pogleda
kamere izvede kalibracijo merilne plošče, zaznavo in sledenje roke z MediaPipe,
zaznavo začetka meritve z LED sekvenco ter izračun kinematičnih parametrov
roke, palca in kazalca.

GitHub repozitorij:
https://github.com/JakobC243/Robotski_Vid

## Delovanje

Najprej se na merilni plošči kalibrirata dve mreži lukenj velikosti 3 x 3. Iz
kalibracije se določijo območja zanimanja in homografija, s katero se slikovne
koordinate preslikajo v metrične koordinate. Roka se zazna z MediaPipe, nato se
iz izbranih točk roke, palca in kazalca računajo pot, hitrost in pospešek.

Za začetek merjenja časa se uporablja LED sekvenca na plošči. Zaznava zatičev v
ciljnem polju je vključena kot eksperimentalna funkcionalnost, saj je odvisna od
osvetlitve, zakritij roke in vidnosti posameznih lukenj.

## Struktura projekta

- `final/`: končna koda pipeline-a.
- `report/`: LaTeX poročilo za oddajo; mapa je preimenovana iz prejšnjega
  imena `overleaf_RV_9HPT/`.
- `report.zip`: arhiv poročila za uvoz v Overleaf.
- `outputs_example/`: primeri izhodov, če so priloženi.
- `Dockerfile`: Docker okolje za zagon.
- `final/requirements.txt`: Python odvisnosti.
- `.gitignore`: izključuje lokalne podatke, okolja in generirane izhode.

Mapa `data/` ni vključena v repozitorij oziroma oddajo, ker vsebuje velike
videoposnetke. Pri Docker zagonu se podatki pripnejo kot zunanji volume `/data`.

## Hiter Docker zagon

Zgradi sliko:

```bash
docker build -t jakobc_rv_izziv .
```

Zaženi kontejner:

```bash
docker run --name jakobc_rv_dev --shm-size=16g -it \
  -v /pot/do/projekta:/workspace \
  -v /pot/do/podatkov:/data \
  -w /workspace \
  jakobc_rv_izziv bash
```

Primer zagona programa v kontejnerju:

```bash
python final/run_hand_pipeline.py \
  --input /data/Data/patient_010/patient_010camP_1_20231130_12_54_11.mp4 \
  --output outputs/p010_camP1_final.mp4 \
  --csv-output outputs/p010_camP1_final.csv \
  --calibration-output outputs/p010_camP1_calibration.json \
  --rotate-clockwise
```

## Izhodi

Program ustvari:

- označen video z izrisom roke, kalibracije, stoparice in grafov,
- CSV z meritvami po okvirjih,
- JSON s kalibracijo merilne plošče.

Podrobnejša navodila za zagon in opis parametrov so v `final/README.md`.
