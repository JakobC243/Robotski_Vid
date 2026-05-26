# Final 9HPT pipeline

Ta mapa vsebuje aktualno kodo za obdelavo videoposnetkov testa devetih zatičev
(9-hole peg test). Glavni program je:

```bash
python final/run_hand_pipeline.py
```

Pipeline trenutno izvede:

- kalibracijo plošče iz dveh mrež lukenj velikosti 3 x 3,
- zaznavo začetka meritve iz LED sekvence,
- sledenje roke z MediaPipe,
- pretvorbo koordinat v mm s homografijo iz mreže 3 x 3, kjer je razmik
  sosednjih lukenj 32 mm,
- pot, hitrost in pospešek roke, palca in kazalca,
- eksperimentalno zaznavo zatičev v ciljnem 3 x 3 polju,
- anotiran video, CSV meritev in JSON kalibracijo.

## Docker na Linuxu

Iz korena projekta, kjer sta `Dockerfile` in mapa `final/`, najprej zgradi
Docker image:

```bash
docker build -t rv-9hpt .
```

Nato pripravi izhodno mapo, na primer:

```bash
mkdir -p /home/prof/rv_outputs
```

## Najbolj enostaven zagon

Podatki niso del repozitorija, zato je priporočeno ločeno priklopiti mapo z
vhodnimi videoposnetki in mapo za rezultate. V spodnjem ukazu popravi samo:

- `/ABS/POT/DO/VIDEO_MAPE`,
- `/ABS/POT/DO/IZHODNE_MAPE`,
- `IME_VIDEA.mp4`.

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -v "/ABS/POT/DO/VIDEO_MAPE":/input:ro \
  -v "/ABS/POT/DO/IZHODNE_MAPE":/output \
  rv-9hpt \
  python final/run_hand_pipeline.py \
    --input /input/IME_VIDEA.mp4 \
    --output /output/IME_VIDEA_annotated.mp4 \
    --csv-output /output/IME_VIDEA_measurements.csv \
    --calibration-output /output/IME_VIDEA_calibration.json \
    --rotate-clockwise
```

Primer:

```bash
mkdir -p /home/prof/rv_outputs

docker run --rm \
  -v "$(pwd)":/workspace \
  -v "/home/prof/videos":/input:ro \
  -v "/home/prof/rv_outputs":/output \
  rv-9hpt \
  python final/run_hand_pipeline.py \
    --input /input/patient_010camP_1_20231130_12_54_11.mp4 \
    --output /output/patient_010camP_1_annotated.mp4 \
    --csv-output /output/patient_010camP_1_measurements.csv \
    --calibration-output /output/patient_010camP_1_calibration.json \
    --rotate-clockwise
```

Če pot vsebuje presledke, naj ostane v narekovajih v `-v` delu ukaza.

## Če je video v projektu

Če so vhodni videoposnetki lokalno že v mapi projekta, lahko namesto ločenega
`/input` priklopa uporabiš tudi pot znotraj `/workspace`:

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  rv-9hpt \
  python final/run_hand_pipeline.py \
    --input /workspace/data/patient_010/patient_010camP_1_20231130_12_54_11.mp4 \
    --output /workspace/outputs/p010_camP1_annotated.mp4 \
    --csv-output /workspace/outputs/p010_camP1_measurements.csv \
    --calibration-output /workspace/outputs/p010_camP1_calibration.json \
    --rotate-clockwise
```

## Odrezan video brez LED začetka

Privzeto se čas začne iz LED sekvence. Če je posnetek odrezan in na začetku ni
vidnega LED zaporedja, dodaj:

```bash
--hand-field-start-fallback on
```

Celoten primer:

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -v "/ABS/POT/DO/VIDEO_MAPE":/input:ro \
  -v "/ABS/POT/DO/IZHODNE_MAPE":/output \
  rv-9hpt \
  python final/run_hand_pipeline.py \
    --input /input/IME_ODREZANEGA_VIDEA.mp4 \
    --output /output/IME_ODREZANEGA_VIDEA_annotated.mp4 \
    --csv-output /output/IME_ODREZANEGA_VIDEA_measurements.csv \
    --calibration-output /output/IME_ODREZANEGA_VIDEA_calibration.json \
    --rotate-clockwise \
    --hand-field-start-fallback on
```

## Ponovna uporaba kalibracije

Če želiš isti video ali isti pogled pognati še enkrat z že shranjeno
kalibracijo, uporabi `--calibration-input`.

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -v "/ABS/POT/DO/VIDEO_MAPE":/input:ro \
  -v "/ABS/POT/DO/IZHODNE_MAPE":/output \
  rv-9hpt \
  python final/run_hand_pipeline.py \
    --input /input/IME_VIDEA.mp4 \
    --output /output/IME_VIDEA_annotated_v2.mp4 \
    --csv-output /output/IME_VIDEA_measurements_v2.csv \
    --calibration-input /output/IME_VIDEA_calibration.json \
    --rotate-clockwise
```

## Izhodi

Vsak zagon naredi:

- anotiran video (`--output`),
- CSV z meritvami po okvirjih (`--csv-output`),
- JSON kalibracijo (`--calibration-output`, če je podan in se kalibracija
  izračuna v tem zagonu).

Na videu so prikazani:

- ime vhodnega videa,
- stoparica poskusa,
- kalibracija in ciljne luknje,
- MediaPipe roka,
- omejena sled roke,
- polni in prazni zatiči kot eksperimentalna zaznava,
- grafi v mm, mm/s in mm/s2,
- pozicija, pot, hitrost in pospešek palca ter kazalca.

Pomembni CSV stolpci:

- `measurement_time_s`, `measurement_start_source`, `measurement_target_side`,
- `trial_started`, `trial_side`, `trial_start_frame`,
- `hand_center_mm_x`, `hand_center_mm_y`,
- `path_length_mm_cumulative`,
- `speed_mm_s_smooth`, `acceleration_mm_s2_smooth`,
- `thumb_tip_mm_x_smooth`, `thumb_tip_mm_y_smooth`,
- `thumb_tip_path_mm_cumulative`, `thumb_tip_speed_mm_s_smooth`,
- `thumb_tip_acceleration_mm_s2_smooth`,
- `index_tip_mm_x_smooth`, `index_tip_mm_y_smooth`,
- `index_tip_path_mm_cumulative`, `index_tip_speed_mm_s_smooth`,
- `index_tip_acceleration_mm_s2_smooth`,
- `thumb_index_distance_mm`, `thumb_index_distance_mm_smooth`,
- `peg_target_count`, `peg_left_count`, `peg_right_count`,
- `peg_left_0` do `peg_left_8`, `peg_right_0` do `peg_right_8`.

## Glavni parametri

- `--rotate-clockwise`: zavrti video za 90 stopinj v smeri urinega kazalca.
- `--hole-spacing-mm 32`: fizični razmik med sosednjima luknjama v 3 x 3 polju.
- `--calibration-scan-frames 60`: koliko začetnih okvirjev pregleda za
  kalibracijo.
- `--calibration-scan-step 5`: korak med okvirji pri iskanju kalibracije.
- `--trail-length 125`: koliko zadnjih točk sledi roke ostane narisanih na
  videu.
- `--hand-field-start-fallback on`: uporabi samo za odrezane videe brez LED
  začetka.
- `--show-light-zones`: nariše LED diagnostične cone.
- `--show-field-zones`: nariše 3 x 3 polja za diagnostiko.
- `--show-trial-status`: prikaže debug tekst LED start detektorja.

## Lokalni zagon brez Dockerja

Če se program poganja brez Dockerja, je priporočen Python 3.11:

```bash
python3.11 -m venv .venv311
./.venv311/bin/python -m pip install --upgrade pip
./.venv311/bin/python -m pip install -r final/requirements.txt

./.venv311/bin/python final/run_hand_pipeline.py \
  --input data/patient_010/patient_010camP_1_20231130_12_54_11.mp4 \
  --output outputs/p010_camP1_annotated.mp4 \
  --csv-output outputs/p010_camP1_measurements.csv \
  --calibration-output outputs/p010_camP1_calibration.json \
  --rotate-clockwise
```

## Moduli

- `run_hand_pipeline.py`: CLI, glavna zanka po okvirjih, timer, CSV in video
  output.
- `calibration.py`: kalibracija plošče iz 3 x 3 lukenj, rob plošče in fallback.
- `hand_tracking.py`: MediaPipe Hands, zaklep aktivne roke in soft ROI.
- `trial_timing.py`: LED start sekvenca.
- `peg_detection.py`: eksperimentalna zaznava zatičev samo na ciljni 3 x 3
  strani.
- `kinematics.py`: pot, hitrost, pospešek, palec in kazalec.
- `visualization.py`: overlay roke, lukenj, stoparice, grafov in metrik.
- `utils.py`: rotacija in video helperji.
