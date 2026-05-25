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

Iz korena repozitorija `Robotski_Vid`:

```powershell
.\.venv\Scripts\python.exe .\final\run_hand_pipeline.py `
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
Zacetni pravokotnik se najprej izracuna iz zaznanih 3x3 mrez lukenj. Ce sta
obe mrezi najdeni, je start cona omejena na prostor med njima z zmernim
paddingom; zato ne sme vec pokriti skoraj cele slike. Na videu so izrisane tudi
luknje, iz katerih je bila narejena kalibracija.
Ce zelis to izklopiti:

```powershell
.\.venv\Scripts\python.exe .\final\run_hand_pipeline.py `
  --input .\data\patient_008\patient_008camP_0_20241212_11_27_00.mp4 `
  --output .\outputs\patient_008camP_0_final.mp4 `
  --csv-output .\outputs\patient_008camP_0_final.csv `
  --calibration-output .\outputs\patient_008camP_0_calibration.json `
  --rotate-clockwise `
  --start-gate none
```

Pipeline privzeto meri tudi zacetek poskusa iz LED polj na plosci. Najprej mora
biti stabilno zaznano, da svetita obe 3x3 polji, nato mora nekaj zaporednih
frame-ov svetiti samo ena stran. Kratki utripi se zato ne uporabijo kot zacetek
poskusa. Na videu je prikazan cas poskusa `TRIAL ...s`, v CSV pa so dodani
diagnosticni stolpci:

- `trial_started`
- `trial_side`
- `trial_start_frame`
- `trial_time_s`
- `trial_light_state`
- `light_left_score`, `light_right_score`
- `light_left_blob_count`, `light_right_blob_count`

Detekcija uporablja isti levi/desni del plosce kot kalibracija. Ni nujno, da je
v vsakem trenutku vidna cela 3x3 mreza; dovolj so stabilni delno vidni LED blobi
v ustrezni coni.

Uporabni parametri za tezje videe z dvema rokama na mizi:

- `--hand-lock-radius-px`: kako dalec od prejsnje pozicije sme biti kandidat pri normalnem sledenju,
- `--hand-reacquire-radius-px`: najvecji radij za ponovno zaklepanje po manjkajocih frame-ih,
- `--min-reacquire-motion-score`: koliko lokalnega gibanja mora imeti kandidat po izgubi sledi,
- `--reacquire-hits`: koliko zaporednih frame-ov mora biti kandidat stabilen pred ponovnim sprejemom.

Za uporabo ze shranjene kalibracije:

```powershell
.\.venv\Scripts\python.exe .\final\run_hand_pipeline.py `
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
