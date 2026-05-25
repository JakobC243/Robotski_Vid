# Final 9HPT hand pipeline

Ta mapa vsebuje prvo cisto osnovo za RV izziv 9HPT oziroma test devetih zaticev.
Pipeline trenutno dela samo osnovni del:

- rotacija vhodnega videa po potrebi,
- kalibracija testne plosce,
- zaznava roke z MediaPipe Hands,
- sledenje centra roke skozi cas,
- glajenje koordinat in signalov,
- izracun poti, hitrosti, pospeska in razdalje palec-kazalec,
- izris roke, trajektorije, kalibracije in grafov na izhodni video,
- CSV z meritvami po frame-ih.

Zaznava zaticev, prijema/odlaganja in validacija s klinicnimi CSV podatki pridejo kasneje.

## Zagon

Priporoceno okolje je Python 3.11, ker `mediapipe==0.10.14` se uporablja prek
starega `mp.solutions` API-ja. Na novejsem Pythonu 3.14 PyPI namesti novejsi
Tasks-only paket, ki za roke potrebuje dodaten `.task` model.

```powershell
.\.venv311\Scripts\python.exe -m pip install -r .\final\requirements.txt
```

Iz korena repozitorija `Robotski_Vid`:

```powershell
.\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_008\patient_008camP_0_20241212_11_27_00.mp4 `
  --output .\outputs\patient_008camP_0_final.mp4 `
  --csv-output .\outputs\patient_008camP_0_final.csv `
  --calibration-output .\outputs\patient_008camP_0_calibration.json `
  --rotate-clockwise `
  --max-frames 300
```

Ce video ni treba rotirati, odstrani `--rotate-clockwise`.
Ce zelis sprotni prikaz, dodaj `--show`.

Privzeto skripta ne zaklene aktivne roke takoj na prvem frame-u. Najprej caka,
da ena roka pride v razsirjeno sredinsko cono kalibrirane plosce, nato sele zacne
tracking. To zmanjsa moznost, da se pipeline na zacetku zaklene na napacno roko.
Pri izbiri aktivne roke ne zaupamo samo oznaki `left/right` iz MediaPipe. Skripta
uporablja lokalno gibanje roke, blizino zadnje zaklenjene pozicije in recovery
pravila. Ko je roka enkrat zaklenjena, kandidat na drugi strani slike ne sme
kar prevzeti sledi; ce prava roka za nekaj frame-ov izgine, video raje pokaze
`NO HAND DETECTED`.
Po zacetnem zaklepu je privzeti ROI nacin `--tracking-roi-mode soft`. To pomeni,
da kalibrirana plosca z majhnim paddingom (`--tracking-roi-padding`) ni trda
meja, ampak varovalo pred skokom na napacno roko. Ce ista MediaPipe sled
postopno zapusti ROI, jo pipeline se vedno sprejme za omejeno stevilo frame-ov
(`--tracking-roi-outside-frames`). Ce kandidat nenadoma skoci iz ROI za vec kot
`--tracking-roi-exit-jump-px`, se zavrne. Staro strogo vedenje vklopis z
`--tracking-roi-mode strict`; ROI lahko izklopis z `--tracking-roi-mode off`.
Kratek izpad roke se ponovno ujame z `--reacquire-hits`, ce je kandidat blizu
zadnje pozicije. Daljsi izpad (`--max-missing-frames`) zaklep pocisti in
pipeline spet caka na roko v start coni.
Zacetni okvir se najprej izracuna iz zaznanih 3x3 mrez lukenj. Ce sta obe
mrezi najdeni, je start cona rotiran poligon okrog obeh polj za zatice z
zmernim paddingom; zato ni vezana na skoraj celo plosco ali celo sliko. Na videu
so izrisane tudi luknje, iz katerih je bila narejena kalibracija.
Ce zelis to izklopiti:

```powershell
.\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_008\patient_008camP_0_20241212_11_27_00.mp4 `
  --output .\outputs\patient_008camP_0_final.mp4 `
  --csv-output .\outputs\patient_008camP_0_final.csv `
  --calibration-output .\outputs\patient_008camP_0_calibration.json `
  --rotate-clockwise `
  --start-gate none
```

Pipeline privzeto meri tudi zacetek poskusa iz LED polj na plosci. Najprej mora
biti stabilno zaznano, da svetita obe 3x3 polji, nato morata obe strani za nekaj
frame-ov ugasniti, sele potem pa mora nekaj zaporednih frame-ov svetiti samo ena
ciljna stran. Kratki utripi se zato ne uporabijo kot zacetek poskusa. Privzeto
se kinematicna stoparica in grafi zacnejo sele, ko je ta start potrjen
(`--measure-from trial-start`). Ce zelis staro vedenje od zacetka videa, uporabi
`--measure-from immediate`. Diagnostika `CAS TECE`/`WAIT LIGHT START` se na video
ne izrise vec, razen ce dodas `--show-trial-status`. V CSV so dodani diagnosticni
stolpci:

- `trial_started`
- `trial_side`
- `trial_start_frame`
- `trial_time_s`
- `measurement_started`
- `measurement_time_s`
- `trial_light_state`
- `light_both_seen`, `light_off_seen`
- `light_left_score`, `light_right_score`
- `light_left_blob_count`, `light_right_blob_count`

Ce ima kalibracija zaznani obe 3x3 mrezi, detekcija LED ne uporablja vec
sirokega levega/desnega dela plosce. Namesto tega vzame 9 kalibriranih centrov
lukenj na vsaki strani in za vsako luknjo primerja majhen notranji krog z
lokalnim obrocem okrog luknje. Zato ni vezana samo na absolutno svetlost slike:
ko LED ne gori, je luknja lokalno temna; ko gori, ima luknja kontrast glede na
svojo neposredno okolico. Ce 3x3 mrezi nista najdeni, skripta pade nazaj na
stari siroki levi/desni ROI.

Za prvi test zaznave polja za zatice se iz vsake zaznane 3x3 mreze lukenj
naredi tesen stirikotnik. Ce katerikoli MediaPipe landmark roke pade v levo ali
desno polje, CSV dobi diagnosticne stolpce:

- `hand_field_zone`
- `hand_field_left_count`
- `hand_field_right_count`

Velikost stirikotnika nastavis z `--field-roi-padding`; vrednost je podana v
enotah razmika med luknjami. Izris teh polj in napis `ROKA LEVO/DESNO` vklopis z
`--show-field-zones`.

Uporabni parametri za tezje videe z dvema rokama na mizi:

- `--hand-lock-radius-px`: kako dalec od prejsnje pozicije sme biti kandidat pri normalnem sledenju,
- `--hand-reacquire-radius-px`: najvecji radij za ponovno zaklepanje po manjkajocih frame-ih,
- `--min-reacquire-motion-score`: koliko lokalnega gibanja mora imeti kandidat po izgubi sledi,
- `--reacquire-hits`: koliko zaporednih frame-ov mora biti kandidat stabilen pred ponovnim sprejemom.
- `--tracking-roi-mode`: `soft` za postopni izhod iz ROI, `strict` za staro trdo mejo, `off` brez ROI omejitve.

Za uporabo ze shranjene kalibracije:

```powershell
.\.venv311\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_008\patient_008camP_0_20241212_11_27_00.mp4 `
  --output .\outputs\patient_008camP_0_final.mp4 `
  --csv-output .\outputs\patient_008camP_0_final.csv `
  --calibration-input .\outputs\patient_008camP_0_calibration.json
```

## Outputi

- izhodni video z anotacijo roke, kalibracije, trajektorije in grafov,
- CSV z meritvami po frame-ih,
- JSON s kalibracijo, ce je podan `--calibration-output`.

## Glavni CSV stolpci

- `frame_idx`, `time_s`: frame in cas v sekundah,
- `hand_detected`: 1, ce je roka zaznana,
- `active_hand_label`, `hand_score`: MediaPipe oznaka roke in confidence; oznaka ni uporabljena kot glavna identiteta aktivne roke,
- `hand_motion_score`, `hand_missing_frames`: lokalno gibanje izbranega kandidata in stevilo zaporednih manjkajocih frame-ov,
- `tracking_roi_inside`, `tracking_roi_outside_frames`: ali je izbrana roka v tracking ROI in koliko zaporednih frame-ov je bila zunaj v `soft` nacinu,
- `hand_center_x_raw`, `hand_center_y_raw`: surov center roke,
- `hand_center_x`, `hand_center_y`: glajen center roke,
- `wrist_x`, `wrist_y`, `thumb_tip_x`, `thumb_tip_y`, `index_tip_x`, `index_tip_y`: pomembni landmarki,
- `thumb_index_distance_px`: razdalja palec-kazalec,
- `path_length_px_cumulative`: kumulativna dolzina poti centra roke,
- `speed_px_s`, `speed_px_s_smooth`: hitrost iz glajenih koordinat in dodatno glajena hitrost,
- `acceleration_px_s2`, `acceleration_px_s2_smooth`: pospesek in dodatno glajen pospesek,
- `calibrated`: 1, ce je kalibracija na voljo,
- `hand_center_board_x`, `hand_center_board_y`: center roke v koordinatnem sistemu plosce, ce obstaja homografija.

## Kalibracija

Skripta najprej poskusi najti dve 3x3 mrezi lukenj na plosci. Iz teh lukenj
izpelje obmocje plosce, shrani luknje v JSON in jih narise na video. Ce lukenj
ne najde dovolj stabilno, poskusi se staro avtomatsko iskanje roba plosce s
kombinacijo sivinske slike, CLAHE, Gaussian blur, Canny robov, kontur in
geometrijskih pogojev. Ce tudi to odpove, odpre rocni fallback, kjer kliknes 4
vogale plosce.

Pomembno: ce uporabljas `--rotate-clockwise`, se kalibracija, MediaPipe, CSV in
izris vsi izvajajo v koordinatah zarotiranega videa.
