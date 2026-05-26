# Robotski vid 9HPT

Aktualna koda za nalogo je v mapi `final/`. Namen pipeline-a je obdelava videov
9-hole peg testa: kalibracija plosce, sledenje roke, start casa iz LED sekvence,
zaznava zaticov v 3x3 polju in izvoz anotiranega videa ter CSV meritev.

## Kaj trenutno dela

`final/run_hand_pipeline.py` je glavni program.

Trenutno zaporedje:

1. Prebere video, po potrebi vsak frame zavrti z `--rotate-clockwise`.
2. Kalibrira plosco iz dveh 3x3 mrez lukenj.
3. Ce prvi frame ni dovolj dober, pregleda vec zacetnih frame-ov in izbere
   najboljsi frame za kalibracijo.
4. Pri kalibraciji uporablja svetle in temne luknje, rob plosce in po potrebi
   napove manjkajoco drugo 3x3 mrezo iz ene najdene mreze.
5. MediaPipe sledi aktivni roki. ROI okrog plosce je privzeto v `soft` nacinu:
   pocasen izhod roke iz ROI je dovoljen, hiter skok na drugo roko pa se zavrne.
6. LED start: pricakuje stanje obe 3x3 polji gorita, potem ugasneta, potem gori
   ciljna stran. Takrat se zacne stoparica.
7. Na videu je prikazana numericna stoparica. Debug napisi niso prikazani, razen
   ce jih vklopis s posebnimi parametri.
8. Zaznava zaticov tece samo na ciljni strani, ki jo doloci LED start.
9. Timer se ustavi, ko je bilo dosezenih vsaj 8 zaticov in se nato ciljno polje
   stabilno izprazni.

Posebno za odrezane videe brez pravilnega LED zacetka obstaja rocen fallback
`--hand-field-start-fallback on`. Ta ni privzet, ker je manj zanesljiv od LED
sekvence. Ce fallback zacne cas iz roke v 3x3 polju, detektor zaticov ne vzame
prvega prostega pogleda kot prazno referenco. Ko roka prvic zapusti ciljno 3x3
polje, primerja vseh 9 lukenj med sabo, iz vecine naredi referenco praznih
lukenj in lahko takoj oznaci prvi zatic, ki je bil vstavljen pred referenco.

## Namestitev okolja

Priporocen je Python 3.11, ker se uporablja `mediapipe==0.10.14`.

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r .\final\requirements.txt
```

Ce `.venv311` ze obstaja, je dovolj:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r .\final\requirements.txt
```

## Standardni zagon

Primer za normalen video, kjer je vidna LED start sekvenca:

```powershell
.\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_001\patient_001camP_1_20241121_10_21_17.mp4 `
  --output .\outputs\p001_camP1_final.mp4 `
  --csv-output .\outputs\p001_camP1_final.csv `
  --calibration-output .\outputs\p001_camP1_calibration.json `
  --rotate-clockwise
```

Ce video ni treba zavrteti, odstrani `--rotate-clockwise`.

Za ponovni zagon z ze shranjeno kalibracijo:

```powershell
.\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_001\patient_001camP_1_20241121_10_21_17.mp4 `
  --output .\outputs\p001_camP1_final.mp4 `
  --csv-output .\outputs\p001_camP1_final.csv `
  --calibration-input .\outputs\p001_camP1_calibration.json `
  --rotate-clockwise
```

## Odrezan video brez LED zacetka

Fallback z roko uporabi samo za posnetke, kjer zacetna LED sekvenca manjka:

```powershell
.\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_004\patient_004camP_1_20241010_14_47_20.mp4 `
  --output .\outputs\p004_camP1_fallback.mp4 `
  --csv-output .\outputs\p004_camP1_fallback.csv `
  --calibration-output .\outputs\p004_camP1_fallback_calibration.json `
  --rotate-clockwise `
  --hand-field-start-fallback on
```

V tem nacinu se v CSV zapise `measurement_start_source=hand_field`. Pri normalnem
LED startu je `measurement_start_source=light`.

## Uporabni parametri

- `--calibration-scan-frames 60`: koliko zacetnih frame-ov pregleda za boljso
  kalibracijo.
- `--calibration-scan-step 5`: korak med frame-i pri iskanju kalibracije.
- `--trial-light-start on|off`: vklop/izklop LED start detektorja.
- `--hand-field-start-fallback on|off`: fallback start iz roke v ciljnem 3x3
  polju, privzeto `off`.
- `--tracking-roi-mode soft|strict|off`: kako strogo naj ROI omejuje MediaPipe
  sled po zaklepu roke.
- `--peg-change-threshold 0.32`: prag spremembe lokalnega patcha za zatic.
- `--peg-stable-frames 5`: koliko zaporednih frame-ov mora biti stanje luknje
  stabilno.
- `--trial-end-peg-threshold 8`: test se lahko konca, ko je bilo dosezenih vsaj
  toliko zaticov in se nato ciljno polje izprazni.
- `--hole-spacing-mm 32`: fizicni razmik med sosednjima luknjama v 3x3 polju.
  Iz tega in homografije 3x3 mrez se racunajo metricni `mm` stolpci in grafi.
- `--show-light-zones`: narise LED diagnosticne cone.
- `--show-field-zones`: narise 3x3 polja za diagnostiko roke v polju.
- `--show-trial-status`: prikaze debug tekst za LED state.

## Outputi

Vsak zagon naredi:

- anotiran video (`--output`),
- frame-level CSV (`--csv-output`),
- JSON kalibracijo, ce je podan `--calibration-output`.

Pomembni CSV stolpci:

- `measurement_started`, `measurement_running`, `measurement_completed`
- `measurement_time_s`, `measurement_start_source`, `measurement_target_side`
- `trial_started`, `trial_side`, `trial_start_frame`
- `hand_detected`, `hand_center_x`, `hand_center_y`
- `thumb_tip_x`, `thumb_tip_y`, `index_tip_x`, `index_tip_y`
- `hand_center_mm_x`, `hand_center_mm_y`
- `thumb_index_distance_mm`, `thumb_index_distance_mm_smooth`
- `path_length_mm_cumulative`
- `speed_mm_s_smooth`, `acceleration_mm_s2_smooth`
- `metric_homography_source`
- `thumb_index_distance_px`, `speed_px_s_smooth`, `acceleration_px_s2_smooth`
  ostanejo v CSV kot debug stolpci v slikovnih enotah
- `peg_target_count`, `peg_left_count`, `peg_right_count`
- `peg_left_0` do `peg_left_8`, `peg_right_0` do `peg_right_8`
- `peg_target_phase`, uporaben za debug reference zaticov

## Batch primer

Primer za 15 videov iz razlicnih map:

```powershell
$out = ".\outputs_batch_check"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$videos = Get-ChildItem .\data -Recurse -Filter *.mp4 | Select-Object -First 15
foreach ($v in $videos) {
  $name = [IO.Path]::GetFileNameWithoutExtension($v.Name)
  .\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
    --input $v.FullName `
    --output "$out\$name.mp4" `
    --csv-output "$out\$name.csv" `
    --calibration-output "$out\$name`_calibration.json" `
    --rotate-clockwise
}
```

Za odrezane videe dodaj `--hand-field-start-fallback on`.

## Struktura projekta

Ohrani:

- `final/`: aktualna koda pipeline-a.
- `final/requirements.txt`: paketi za Python okolje.
- `data/`: lokalni primeri videov, ce jih zelis imeti v repozitoriju/delovni mapi.
- `Dockerfile`: pusti, ce bos se poganjal v containerju.
- `.gitignore`: ignorira okolja, outpute in video izhode.
- `README.md`: ta operativni opis.

Lahko odstranis oziroma ne prenasas naprej:

- `.venv/`, `.venv311/`, `venv/`, `env/`: lokalna Python okolja.
- `outputs/`, `outputs_*`: generirani videi, CSV-ji in debug slike.
- Stare root Python skripte iz prejsnjih iteracij, ce so se kje ostale:
  `hand_ai_tracking.py`, `hand_ai_tracking_calibrated.py`,
  `track_calibrated_hand_and_pins.py`, `track_calibrated_hand_and_pins_v2.py`,
  `track_hand_motion_v2.py`, `track_hand_motion_improved.py`,
  `hole_calibration.py`, `preview_video.py`,
  `batch_test_calibrated_hand_and_pins.py`, `dominant_hand_ai.py`,
  `test_detect_side_hole_grids.py`.
- `mediapipe_gui_tracking/`: star GUI/prototip, ni del aktualnega `final`
  pipeline-a.
- `run_local.ps1`: trenutno klice stare skripte, zato ga odstrani ali prepisuj
  samo, ce ga bos res uporabljal za nov `final/run_hand_pipeline.py`.
- `requirements-local.txt` in `test_env.py`: stara Docker/lokalna diagnostika,
  nista potrebna za aktualni final pipeline.
- `poganjanja.txt`: osebni zapiski, ni potreben za delovanje.

## Glavne datoteke v final

- `run_hand_pipeline.py`: CLI, zanka cez frame-e, timer, output video/CSV.
- `calibration.py`: iskanje 3x3 lukenj, rob plosce, homografija, JSON.
- `hand_tracking.py`: MediaPipe Hands, zaklep aktivne roke, soft ROI.
- `trial_timing.py`: LED start detektor.
- `peg_detection.py`: stabilna zaznava zaticov v ciljnem 3x3 polju.
- `kinematics.py`: pot, hitrost, pospesek, razdalja palec-kazalec.
- `visualization.py`: izris roke, kalibracije, stoparice, grafov.
- `utils.py`: pomocne funkcije za video/rotacijo.
- `final_quick_check.py`: starejsi helper; glavni tok je zdaj
  `run_hand_pipeline.py`.

## Trenutna omejitev

Zaznava zaticov trenutno temelji na spremembi slike okrog kalibriranih lukenj,
ne na semantiki prijema. Naslednji robustnejsi korak je povezati dogodek z
landmarkoma palca in kazalca: zatic naj se potrdi sele, ko palec/kazalec prideta
nad luknjo, nato roka zapusti 3x3 polje in se stanje luknje stabilno spremeni.
