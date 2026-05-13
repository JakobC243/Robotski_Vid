import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


# ============================================================
# GLAVNE NASTAVITVE
# ============================================================
TRAIN_FRAMES_DEFAULT = 100      # na začetku "učenje"/kalibracija na 100 posnetkih
PREVIEW_FRAMES_DEFAULT = 3      # shrani samo 3 slike z označenim središčem in sklepi
MAX_HANDS = 1                   # za nalogo navadno slediš eni roki
SMOOTHING_ALPHA = 0.35          # 0 = zelo gladko, 1 = brez glajenja


# MediaPipe indeksi sklepov:
# 0 zapestje
# 1-4 palec
# 5-8 kazalec
# 9-12 sredinec
# 13-16 prstanec
# 17-20 mezinec
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]


def get_default_video_path(data_root: Path) -> Path:
    videos = sorted(data_root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"Ni .mp4 videov v mapi: {data_root}")
    return videos[0]


def create_output_paths(output_root: Path) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "video": output_root / "hand_ai_tracking_preview.avi",
        "csv": output_root / "hand_ai_tracking.csv",
        "preview_dir": output_root / "preview_frames",
        "pos_plot": output_root / "position_x_y.png",
        "speed_plot": output_root / "speed_over_time.png",
        "accel_plot": output_root / "acceleration_over_time.png",
        "traj_plot": output_root / "trajectory_xy.png",
    }


def landmarks_to_pixels(hand_landmarks, width: int, height: int) -> np.ndarray:
    """Pretvori normalizirane MediaPipe landmarke v pikselske koordinate."""
    points = []
    for lm in hand_landmarks.landmark:
        x = int(np.clip(lm.x * width, 0, width - 1))
        y = int(np.clip(lm.y * height, 0, height - 1))
        z = float(lm.z)
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def compute_hand_center(points: np.ndarray, mode: str = "palm") -> Tuple[int, int]:
    """
    Izračun središča roke.
    - palm: središče dlani iz sklepov 0, 1, 5, 9, 13, 17
    - bbox: središče pravokotnika vseh 21 sklepov
    - all: povprečje vseh 21 sklepov
    """
    if mode == "palm":
        xy = points[PALM_LANDMARKS, :2]
        center = np.mean(xy, axis=0)
    elif mode == "bbox":
        xy = points[:, :2]
        center = np.array([
            0.5 * (np.min(xy[:, 0]) + np.max(xy[:, 0])),
            0.5 * (np.min(xy[:, 1]) + np.max(xy[:, 1])),
        ])
    elif mode == "all":
        center = np.mean(points[:, :2], axis=0)
    else:
        raise ValueError("center_mode mora biti: palm, bbox ali all")

    return int(round(center[0])), int(round(center[1]))


def bbox_from_points(points: np.ndarray, margin: int, width: int, height: int) -> Tuple[int, int, int, int]:
    xy = points[:, :2]
    x1 = int(max(0, np.min(xy[:, 0]) - margin))
    y1 = int(max(0, np.min(xy[:, 1]) - margin))
    x2 = int(min(width - 1, np.max(xy[:, 0]) + margin))
    y2 = int(min(height - 1, np.max(xy[:, 1]) + margin))
    return x1, y1, x2 - x1, y2 - y1


def exponential_smoothing(
    current: Optional[Tuple[int, int]],
    previous: Optional[Tuple[float, float]],
    alpha: float,
) -> Optional[Tuple[float, float]]:
    if current is None:
        return previous
    if previous is None:
        return float(current[0]), float(current[1])
    return (
        alpha * current[0] + (1.0 - alpha) * previous[0],
        alpha * current[1] + (1.0 - alpha) * previous[1],
    )


def draw_annotations(
    frame_bgr: np.ndarray,
    hand_landmarks,
    points: Optional[np.ndarray],
    center: Optional[Tuple[int, int]],
    bbox: Optional[Tuple[int, int, int, int]],
    frame_idx: int,
    time_s: float,
    speed_px_s: float,
    acceleration_px_s2: float,
    confidence_text: str,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    if hand_landmarks is not None:
        mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

    if points is not None:
        # dodatno označi vseh 21 sklepov s številkami
        for i, (x, y, _) in enumerate(points):
            cv2.circle(annotated, (int(x), int(y)), 3, (0, 255, 255), -1)
            cv2.putText(
                annotated,
                str(i),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

    if center is not None:
        cv2.circle(annotated, center, 8, (0, 0, 255), -1)
        cv2.circle(annotated, center, 15, (255, 255, 255), 2)
        cv2.putText(
            annotated,
            "CENTER",
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    info = [
        f"frame={frame_idx}",
        f"t={time_s:.2f}s",
        confidence_text,
    ]

    if center is None:
        info.append("center=(nan,nan)")
    else:
        info.append(f"center=({center[0]},{center[1]})")

    info.append(f"v={speed_px_s:.1f}px/s" if not np.isnan(speed_px_s) else "v=nan")
    info.append(f"a={acceleration_px_s2:.1f}px/s2" if not np.isnan(acceleration_px_s2) else "a=nan")

    cv2.putText(
        annotated,
        "  ".join(info),
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if center is None:
        cv2.putText(
            annotated,
            "No hand detected",
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def plot_results(df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    valid = df.dropna(subset=["x_px", "y_px"]).copy()
    if valid.empty:
        print("Ni zaznanih točk, grafov ne morem narisati.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["frame"], df["x_px"], label="x", marker=".")
    plt.plot(df["frame"], df["y_px"], label="y", marker=".")
    plt.title("Položaj središča roke skozi čas")
    plt.xlabel("frame")
    plt.ylabel("položaj [px]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["pos_plot"])
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["frame"], df["speed_px_s"], marker=".")
    plt.title("Hitrost središča roke")
    plt.xlabel("frame")
    plt.ylabel("hitrost [px/s]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["speed_plot"])
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["frame"], df["acceleration_px_s2"], marker=".")
    plt.title("Pospešek središča roke")
    plt.xlabel("frame")
    plt.ylabel("pospešek [px/s²]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["accel_plot"])
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(valid["x_px"], valid["y_px"], marker=".", linestyle="-")
    plt.title("Trajektorija središča roke v slikovni ravnini")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["traj_plot"])
    plt.close()


def calibrate_on_first_frames(
    video_path: Path,
    train_frames: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    center_mode: str,
) -> Dict[str, float]:
    """
    To ni učenje nevronske mreže od začetka, ampak praktična kalibracija na začetnih N slikah:
    - oceni tipično velikost roke,
    - oceni povprečen položaj,
    - preveri delež uspešnih zaznav.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti videa: {video_path}")

    mp_hands = mp.solutions.hands
    centers = []
    bbox_sizes = []
    detected = 0
    processed = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:
        while processed < train_frames:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                points = landmarks_to_pixels(hand_landmarks, w, h)
                center = compute_hand_center(points, mode=center_mode)
                bbox = bbox_from_points(points, margin=0, width=w, height=h)
                centers.append(center)
                bbox_sizes.append(max(bbox[2], bbox[3]))
                detected += 1

            processed += 1

    cap.release()

    if centers:
        centers_np = np.asarray(centers, dtype=np.float32)
        mean_center = np.mean(centers_np, axis=0)
        mean_hand_size = float(np.mean(bbox_sizes))
    else:
        mean_center = np.array([np.nan, np.nan])
        mean_hand_size = np.nan

    detection_rate = detected / max(processed, 1)

    return {
        "processed_train_frames": processed,
        "detected_train_frames": detected,
        "detection_rate": detection_rate,
        "mean_center_x": float(mean_center[0]),
        "mean_center_y": float(mean_center[1]),
        "mean_hand_size_px": float(mean_hand_size),
    }


def process_video(
    video_path: Path,
    output_root: Path,
    train_frames: int,
    preview_frames: int,
    center_mode: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    max_frames: int,
    show: bool,
) -> None:
    paths = create_output_paths(output_root)
    paths["preview_dir"].mkdir(parents=True, exist_ok=True)

    print("Kalibracija / učenje na začetnih posnetkih ...")
    calib = calibrate_on_first_frames(
        video_path=video_path,
        train_frames=train_frames,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        center_mode=center_mode,
    )
    print("Rezultat kalibracije:")
    for k, v in calib.items():
        print(f"  {k}: {v}")

    # Če je roka redko zaznana, opozori, vendar nadaljuj.
    if calib["detection_rate"] < 0.3:
        print("Opozorilo: zaznava v učnih slikah je nizka. Poskusi z boljšo osvetlitvijo ali nižjim pragom confidence.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti videa: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print(f"Video: {video_path}")
    print(f"Velikost: {width}x{height}, fps={fps:.2f}, frames={frame_count}")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(paths["video"]), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Ne morem ustvariti izhodnega videa: {paths['video']}")

    mp_hands = mp.solutions.hands

    records: List[Dict[str, float]] = []
    path_points: List[Tuple[int, int]] = []

    prev_center_smoothed: Optional[Tuple[float, float]] = None
    prev_center_for_velocity: Optional[Tuple[float, float]] = None
    prev_time: Optional[float] = None
    prev_speed = np.nan

    saved_preview = 0
    detection_count = 0

    show_ok = show
    if show_ok:
        try:
            cv2.namedWindow("AI hand tracking", cv2.WINDOW_NORMAL)
        except cv2.error:
            show_ok = False
            print("GUI ni na voljo; nadaljujem brez prikaza.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_frames > 0 and frame_idx >= max_frames:
                break

            time_s = frame_idx / fps

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)

            hand_landmarks = None
            points = None
            raw_center = None
            center_smoothed_int = None
            bbox = None
            confidence_text = "conf=n/a"

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                points = landmarks_to_pixels(hand_landmarks, width, height)
                raw_center = compute_hand_center(points, mode=center_mode)

                # margin nastavimo glede na kalibrirano velikost roke
                if np.isfinite(calib["mean_hand_size_px"]):
                    margin = int(max(10, 0.12 * calib["mean_hand_size_px"]))
                else:
                    margin = 20

                bbox = bbox_from_points(points, margin=margin, width=width, height=height)
                detection_count += 1

            center_smoothed = exponential_smoothing(
                current=raw_center,
                previous=prev_center_smoothed,
                alpha=SMOOTHING_ALPHA,
            )
            prev_center_smoothed = center_smoothed

            if raw_center is not None and center_smoothed is not None:
                center_smoothed_int = (
                    int(round(center_smoothed[0])),
                    int(round(center_smoothed[1])),
                )
                path_points.append(center_smoothed_int)

            speed_px_s = np.nan
            acceleration_px_s2 = np.nan

            if center_smoothed is not None and raw_center is not None:
                if prev_center_for_velocity is not None and prev_time is not None:
                    dt = time_s - prev_time
                    if dt > 0:
                        dx = center_smoothed[0] - prev_center_for_velocity[0]
                        dy = center_smoothed[1] - prev_center_for_velocity[1]
                        speed_px_s = float(np.hypot(dx, dy) / dt)
                        if not np.isnan(prev_speed):
                            acceleration_px_s2 = float((speed_px_s - prev_speed) / dt)

                prev_center_for_velocity = center_smoothed
                prev_time = time_s
                prev_speed = speed_px_s
            else:
                prev_speed = np.nan

            annotated = draw_annotations(
                frame_bgr=frame,
                hand_landmarks=hand_landmarks,
                points=points,
                center=center_smoothed_int,
                bbox=bbox,
                frame_idx=frame_idx,
                time_s=time_s,
                speed_px_s=speed_px_s,
                acceleration_px_s2=acceleration_px_s2,
                confidence_text=confidence_text,
            )

            # nariši trajektorijo
            for a, b in zip(path_points[:-1], path_points[1:]):
                cv2.line(annotated, a, b, (0, 255, 255), 2)

            writer.write(annotated)

            # shrani samo prve 3 uspešno zaznane primere s sklepi in središčem
            if saved_preview < preview_frames and center_smoothed_int is not None:
                preview_path = paths["preview_dir"] / f"preview_{saved_preview + 1:02d}_frame_{frame_idx:06d}.png"
                cv2.imwrite(str(preview_path), annotated)
                saved_preview += 1

            if show_ok:
                cv2.imshow("AI hand tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            records.append(
                {
                    "frame": frame_idx,
                    "time_s": time_s,
                    "x_px": float(center_smoothed_int[0]) if center_smoothed_int is not None else np.nan,
                    "y_px": float(center_smoothed_int[1]) if center_smoothed_int is not None else np.nan,
                    "raw_x_px": float(raw_center[0]) if raw_center is not None else np.nan,
                    "raw_y_px": float(raw_center[1]) if raw_center is not None else np.nan,
                    "speed_px_s": float(speed_px_s) if not np.isnan(speed_px_s) else np.nan,
                    "acceleration_px_s2": float(acceleration_px_s2) if not np.isnan(acceleration_px_s2) else np.nan,
                    "detected": 1 if raw_center is not None else 0,
                    "train_detection_rate": calib["detection_rate"],
                    "mean_hand_size_px": calib["mean_hand_size_px"],
                }
            )

            frame_idx += 1

    cap.release()
    writer.release()

    if show_ok:
        cv2.destroyAllWindows()

    if not records:
        raise RuntimeError("Ni obdelanih slik.")

    df = pd.DataFrame(records)
    df.to_csv(paths["csv"], index=False)
    plot_results(df, paths)

    print("\nShranjeno:")
    print(f"  video: {paths['video']}")
    print(f"  csv: {paths['csv']}")
    print(f"  3 prikazne slike: {paths['preview_dir']}")
    print(f"  graf položaja: {paths['pos_plot']}")
    print(f"  graf hitrosti: {paths['speed_plot']}")
    print(f"  graf pospeška: {paths['accel_plot']}")
    print(f"  trajektorija: {paths['traj_plot']}")
    print(f"\nObdelanih slik: {len(records)}, zaznav roke: {detection_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI zaznava roke, sklepov in središča roke v videu."
    )

    parser.add_argument("--video", type=str, default=None, help="Pot do vhodnega .mp4 videa.")
    parser.add_argument("--data-root", type=str, default="/data/Data", help="Mapa, kjer iščem video, če --video ni podan.")
    parser.add_argument("--output-root", type=str, default="/workspace/outputs", help="Mapa za rezultate.")

    parser.add_argument("--train-frames", type=int, default=TRAIN_FRAMES_DEFAULT, help="Število začetnih slik za kalibracijo.")
    parser.add_argument("--preview-frames", type=int, default=PREVIEW_FRAMES_DEFAULT, help="Število shranjenih anotiranih primerov.")
    parser.add_argument("--max-frames", type=int, default=0, help="Največ slik za obdelavo. 0 = cel video.")

    parser.add_argument(
        "--center-mode",
        type=str,
        default="palm",
        choices=["palm", "bbox", "all"],
        help="Način izračuna središča roke.",
    )

    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.45)

    parser.add_argument("--show", action="store_true", help="Prikaži anotiran video med obdelavo.")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    if args.video is None:
        video_path = get_default_video_path(data_root)
    else:
        video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video ne obstaja: {video_path}")

    process_video(
        video_path=video_path,
        output_root=output_root,
        train_frames=args.train_frames,
        preview_frames=args.preview_frames,
        center_mode=args.center_mode,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        max_frames=args.max_frames,
        show=args.show,
    )


if __name__ == "__main__":
    main()