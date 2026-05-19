# Robotski_Vid

Python projekt za robotski vid: sledenje roke, kalibracija plosce, analiza zaticnih lukenj in priprava meritev za 9HPT izziv.

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
