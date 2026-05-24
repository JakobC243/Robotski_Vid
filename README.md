# Robotski_Vid

Python projekt za robotski vid: sledenje roke, kalibracija plosce, analiza zaticnih lukenj in priprava meritev za 9HPT izziv.

## Koncni zagon za hiter check testa

Za koncni pregled uporabi skripto v mapi `final`. Ta najprej pozene zadnjo `pins-v2` kalibrirano sledenje, potem pa naredi se dodatni video, kjer je v kotu narisan zivi graf hitrosti in opozorila za osnovne napake testa.

Primer za lokalni zagon iz mape `Robotski_Vid`:

```powershell
.\.venv\Scripts\python.exe .\final\final_quick_check.py `
  --video ..\data\patient_001\patient_001camP_0_20241121_10_21_17.mp4 `
  --output-root .\outputs_final_quick `
  --expected-hand right `
  --target-side right
```

Ce roka ali stran odlaganja nista znani, pusti `auto`:

```powershell
.\.venv\Scripts\python.exe .\final\final_quick_check.py `
  --video ..\data\patient_003\patient_003camP_0_20231005_14_13_43.mp4 `
  --output-root .\outputs_final_quick_patient003 `
  --expected-hand auto `
  --target-side auto
```

Glavni outputi:

- `final_quick_check_live_check.avi` - video z oznaceno roko, luknjami, pini, z grafom hitrosti v kotu in opozorili v casu testa.
- `final_quick_check.csv` - frame-level meritve iz osnovnega trackerja.
- `final_quick_check_check.csv` - dodatni check stolpci: zglajena hitrost, opozorila, ciljna stran, aktivnost na napacni strani.
- `final_quick_check_check_summary.json` - povzetek napak in statistike.
- `final_quick_check_check_report.txt` - kratek berljiv report.

Uporabni parametri:

- `--expected-hand left|right|auto` pove, katero roko pricakujemo. Ce je nastavljeno `left` ali `right`, skripta opozori, ko MediaPipe zazna drugo roko.
- `--target-side left|right|auto` pove, na katero stran pacient odlaga pine. Ce je `auto`, skripta stran izbere glede na vec aktivnosti pinov/interakcij.
- `--velocity-window-frames`, `--ma-window`, `--speed-ema-alpha` gladijo hitrost, da kratki skoki pri spremembi smeri manj pokvarijo graf.
- `--brightness-jump-threshold` opozarja na nenadne spremembe svetlobe, npr. ko se prizgejo luci.
- `--missing-hand-seconds` opozori, ce roka izgine iz zaznave predolgo.
- `--skip-tracker` uporabi ze obstojeci CSV/video v isti output mapi in naredi samo finalni check video.

Primer Docker zagona, ce projekt in podatke mountas v container:

```powershell
docker build -t robotski-vid .
docker run --rm -it `
  -v C:\Users\Jakob\Desktop\RV_izziv\Robotski_Vid:/workspace `
  -v C:\Users\Jakob\Desktop\RV_izziv\data:/data `
  robotski-vid `
  python final/final_quick_check.py `
    --video /data/patient_001/patient_001camP_0_20241121_10_21_17.mp4 `
    --output-root /workspace/outputs_final_quick `
    --expected-hand right `
    --target-side right
```

## TODO za finalni video-check

- Graf na videu je dodan kot post-process overlay; za pravo kamero v zivo je naslednji korak isti overlay prestaviti direktno v tracker loop.
- Pin check trenutno uporablja pravilo "odlaganje samo na eno stran": ciljna stran steje pine, aktivnost na drugi strani je opozorilo. To je treba validirati na vec pacientih in po potrebi nastaviti threshold-e.
- Zaznavo luci je treba izboljsati iz globalne svetlosti na ROI plosce, da premiki roke ne sprozijo laznih alarmov.
- Za MS paciente dodati bolj jasna pravila napak testa: napacna roka, zacetek pred signalom/lucjo, odlaganje na napacno stran, predolga izguba roke, nenaraven skok hitrosti.
- Hitrosti so dodatno zglajene, ampak je treba se primerjati `velocity-window-frames`, `ma-window` in `speed-ema-alpha` na vec posnetkih.
- Povezati `dominant_hand_ai.py` ali metapodatke pacienta, da `--expected-hand` ni treba rocno nastavljati.

## Batch tracking za paciente

Velike CSV-je za vec pacientov zgenerira batch runner. Za enoten pogled kamere:

```powershell
.\run_local.ps1 -Mode batch-hand -InputRoot ..\data -Pattern "*camP_1*.mp4" -OutputDir .\outputs_batch_hand
```

Za vse kamere uporabi `-Pattern "*.mp4"`. Vsak video dobi svoj output folder in frame-level CSV.
Za vecji streznik lahko neposredno uporabis tudi Python batch runner z omejitvijo pacientov:

```powershell
python batch_test_calibrated_hand_and_pins.py --input-root D:\data --pattern "*camP_1*.mp4" --output-root D:\outputs_batch_hand --mode hand --max-patients 500
```

## AI za dominantno roko

Najprej iz tracking CSV-jev zgradi feature tabelo:

```powershell
python dominant_hand_ai.py build-features --data-root ..\data --tracking-root .\outputs_batch_hand --output .\outputs_dominant_hand\features.csv --camera 1
```

Nato treniraj model za dominantno stran pacienta (`right`/`left`):

```powershell
python dominant_hand_ai.py train --features .\outputs_dominant_hand\features.csv --task dominant_side --train-patients 300 --output-dir .\outputs_dominant_hand
```

Alternativno lahko treniras model za prepoznavo, ali je posamezen trial dominantna ali nedominantna roka:

```powershell
python dominant_hand_ai.py train --features .\outputs_dominant_hand\features.csv --task dominant_trial --train-patients 300 --output-dir .\outputs_dominant_hand
```
