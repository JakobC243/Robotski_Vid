from pathlib import Path
import cv2

DATA_DIR = Path("/data/Data")
OUT_DIR = Path("/workspace/outputs")
OUT_DIR.mkdir(exist_ok=True)

# vzame prvi najdeni video
videos = sorted(DATA_DIR.glob("patient_*/*.mp4"))

if not videos:
    raise FileNotFoundError("Ni najdenih .mp4 videov v /data/Data")

video_path = videos[0]
out_path = OUT_DIR / "preview_first_video.mp4"

print("Input video:", video_path)
print("Output preview:", out_path)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Ne morem odpreti videa: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Shrani največ 10 sekund ali največ 300 frame-ov
max_frames = int(min(fps * 10, 300)) if fps > 0 else 300

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 25, (width, height))

count = 0

while count < max_frames:
    ok, frame = cap.read()
    if not ok:
        break

    # Dodaj tekst na video, da veš kateri file gledaš
    cv2.putText(
        frame,
        video_path.name,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    writer.write(frame)
    count += 1

cap.release()
writer.release()

print(f"Saved {count} frames to {out_path}")
