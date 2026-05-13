import argparse
import json
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
TRAIN_FRAMES_DEFAULT = 100
PREVIEW_FRAMES_DEFAULT = 3
MAX_HANDS = 1

# MediaPipe indeksi sklepov:
# 0 zapestje
# 1-4 palec
# 5-8 kazalec
# 9-12 sredinec
# 13-16 prstanec
# 17-20 mezinec
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]


class CenterKalman:
    """
    Kalmanov filter za središče roke.
    Stanje: [x, y, vx, vy]
    Meritev: [x, y]
    """

    def __init__(
        self,
        process_noise: float = 0.03,
        measurement_noise: float = 8.0,
    ):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False

    def update(self, measurement: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if measurement is not None:
            x, y = measurement

            if not self.initialized:
                self.kf.statePost = np.array(
                    [[x], [y], [0], [0]],
                    dtype=np.float32,
                )
                self.initialized = True
                return int(x), int(y)

            self.kf.predict()

            measured = np.array(
                [[np.float32(x)], [np.float32(y)]],
                dtype=np.float32,
            )

            corrected = self.kf.correct(measured)

            return int(corrected[0, 0]), int(corrected[1, 0])

        if not self.initialized:
            return None

        prediction = self.kf.predict()
        return int(prediction[0, 0]), int(prediction[1, 0])


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
        "summary_csv": output_root / "summary_metrics.csv",
        "summary_json": output_root / "summary_metrics.json",
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
        x = float(np.clip(lm.x * width, 0, width - 1))
        y = float(np.clip(lm.y * height, 0, height - 1))
        z = float(lm.z)
        points.append([x, y, z])

    return np.asarray(points, dtype=np.float32)


def compute_hand_center(points: np.ndarray, mode: str = "palm") -> Tuple[int, int]:
    """
    Izračun središča roke.

    Načini:
        palm: središče dlani iz stabilnejših sklepov [0, 1, 5, 9, 13, 17]
        bbox: središče pravokotnika okoli vseh 21 sklepov
        all: povprečje vseh 21 sklepov
    """

    if mode == "palm":
        xy = points[PALM_LANDMARKS, :2]
        center = np.mean(xy, axis=0)

    elif mode == "bbox":
        xy = points[:, :2]
        center = np.array(
            [
                0.5 * (np.min(xy[:, 0]) + np.max(xy[:, 0])),
                0.5 * (np.min(xy[:, 1]) + np.max(xy[:, 1])),
            ]
        )

    elif mode == "all":
        center = np.mean(points[:, :2], axis=0)

    else:
        raise ValueError("center_mode mora biti: palm, bbox ali all")

    return int(round(center[0])), int(round(center[1]))


def bbox_from_points(
    points: np.ndarray,
    margin: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    xy = points[:, :2]

    x1 = int(max(0, np.min(xy[:, 0]) - margin))
    y1 = int(max(0, np.min(xy[:, 1]) - margin))
    x2 = int(min(width - 1, np.max(xy[:, 0]) + margin))
    y2 = int(min(height - 1, np.max(xy[:, 1]) + margin))

    return x1, y1, x2 - x1, y2 - y1


def add_landmark_columns(record: Dict[str, float], points: Optional[np.ndarray]) -> None:
    """
    V CSV shrani vseh 21 MediaPipe točk.
    Če roka ni zaznana, vpiše NaN.
    """

    for i in range(21):
        if points is None:
            record[f"lm{i}_x"] = np.nan
            record[f"lm{i}_y"] = np.nan
            record[f"lm{i}_z"] = np.nan
        else:
            record[f"lm{i}_x"] = float(points[i, 0])
            record[f"lm{i}_y"] = float(points[i, 1])
            record[f"lm{i}_z"] = float(points[i, 2])


def draw_annotations(
    frame_bgr: np.ndarray,
    hand_landmarks,
    points: Optional[np.ndarray],
    raw_center: Optional[Tuple[int, int]],
    final_center: Optional[Tuple[int, int]],
    bbox: Optional[Tuple[int, int, int, int]],
    frame_idx: int,
    time_s: float,
    speed_px_s: float,
    acceleration_px_s2: float,
    cumulative_path_px: float,
    use_kalman: bool,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    h, _ = annotated.shape[:2]

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

    if raw_center is not None:
        cv2.circle(annotated, raw_center, 5, (255, 0, 255), -1)
        cv2.putText(
            annotated,
            "RAW",
            (raw_center[0] + 8, raw_center[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if final_center is not None:
        cv2.circle(annotated, final_center, 8, (0, 0, 255), -1)
        cv2.circle(annotated, final_center, 15, (255, 255, 255), 2)

        center_label = "CENTER KALMAN" if use_kalman else "CENTER"
        cv2.putText(
            annotated,
            center_label,
            (final_center[0] + 12, final_center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    method_text = "kalman=on" if use_kalman else "kalman=off"

    info = [
        f"frame={frame_idx}",
        f"t={time_s:.2f}s",
        method_text,
    ]

    if final_center is None:
        info.append("center=(nan,nan)")
    else:
        info.append(f"center=({final_center[0]},{final_center[1]})")

    info.append(f"path={cumulative_path_px:.1f}px")

    if not np.isnan(speed_px_s):
        info.append(f"v={speed_px_s:.1f}px/s")
    else:
        info.append("v=nan")

    if not np.isnan(acceleration_px_s2):
        info.append(f"a={acceleration_px_s2:.1f}px/s2")
    else:
        info.append("a=nan")

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

    if raw_center is None:
        if use_kalman:
            msg = "No hand detected - Kalman prediction used if initialized"
        else:
            msg = "No hand detected"

        cv2.putText(
            annotated,
            msg,
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def plot_results(df: pd.DataFrame, paths: Dict[str, Path], use_kalman: bool) -> None:
    valid = df.dropna(subset=["x_px", "y_px"]).copy()

    if valid.empty:
        print("Ni zaznanih točk, grafov ne morem narisati.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["frame"], df["raw_x_px"], label="raw x", marker=".", alpha=0.5)
    plt.plot(df["frame"], df["raw_y_px"], label="raw y", marker=".", alpha=0.5)

    if use_kalman:
        plt.plot(df["frame"], df["x_px"], label="kalman x", linewidth=2)
        plt.plot(df["frame"], df["y_px"], label="kalman y", linewidth=2)
    else:
        plt.plot(df["frame"], df["x_px"], label="center x", linewidth=2)
        plt.plot(df["frame"], df["y_px"], label="center y", linewidth=2)

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
    Kalibracija na prvih N slikah.
    To ni ponovno učenje MediaPipe mreže, ampak ocena detekcije in velikosti roke.
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


def compute_summary_metrics(
    df: pd.DataFrame,
    video_path: Path,
    fps: float,
    width: int,
    height: int,
    use_kalman: bool,
    center_mode: str,
    max_gap_frames_for_kinematics: int,
) -> Dict[str, float]:
    total_frames = int(len(df))
    detected_frames = int(df["detected"].sum())
    missing_frames = total_frames - detected_frames
    detection_rate = detected_frames / max(total_frames, 1)

    valid_speed = df["speed_px_s"].dropna()
    valid_accel = df["acceleration_px_s2"].dropna()

    total_path_px = float(df["cumulative_path_px"].dropna().iloc[-1]) if total_frames > 0 else 0.0

    summary = {
        "video": str(video_path),
        "frames_processed": total_frames,
        "detected_frames": detected_frames,
        "missing_frames": missing_frames,
        "detection_rate": float(detection_rate),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "use_kalman": bool(use_kalman),
        "center_mode": center_mode,
        "max_gap_frames_for_kinematics": int(max_gap_frames_for_kinematics),
        "total_path_px": total_path_px,
        "mean_speed_px_s": float(valid_speed.mean()) if not valid_speed.empty else np.nan,
        "median_speed_px_s": float(valid_speed.median()) if not valid_speed.empty else np.nan,
        "max_speed_px_s": float(valid_speed.max()) if not valid_speed.empty else np.nan,
        "mean_acceleration_px_s2": float(valid_accel.mean()) if not valid_accel.empty else np.nan,
        "median_acceleration_px_s2": float(valid_accel.median()) if not valid_accel.empty else np.nan,
        "max_abs_acceleration_px_s2": float(valid_accel.abs().max()) if not valid_accel.empty else np.nan,
    }

    return summary


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
    kalman_process_noise: float,
    kalman_measurement_noise: float,
    use_kalman: bool,
    max_gap_frames_for_kinematics: int,
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

    if calib["detection_rate"] < 0.3:
        print(
            "Opozorilo: zaznava v učnih slikah je nizka. "
            "Poskusi z drugo kamero, boljšo osvetlitvijo ali nižjim pragom confidence."
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti videa: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print(f"Video: {video_path}")
    print(f"Velikost: {width}x{height}, fps={fps:.2f}, frames={frame_count}")
    print(f"Kalman filter: {'ON' if use_kalman else 'OFF'}")
    print(f"Kinematika se računa samo čez vrzeli <= {max_gap_frames_for_kinematics} frame")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(paths["video"]), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Ne morem ustvariti izhodnega videa: {paths['video']}")

    mp_hands = mp.solutions.hands

    kalman = None
    if use_kalman:
        kalman = CenterKalman(
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
        )

    records: List[Dict[str, float]] = []
    path_points: List[Tuple[int, int]] = []

    prev_center_for_velocity: Optional[Tuple[int, int]] = None
    prev_time: Optional[float] = None
    prev_frame_for_velocity: Optional[int] = None
    prev_speed = np.nan
    cumulative_path_px = 0.0
    current_track_segment = 0

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
            final_center = None
            bbox = None

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                points = landmarks_to_pixels(hand_landmarks, width, height)
                raw_center = compute_hand_center(points, mode=center_mode)

                if np.isfinite(calib["mean_hand_size_px"]):
                    margin = int(max(10, 0.12 * calib["mean_hand_size_px"]))
                else:
                    margin = 20

                bbox = bbox_from_points(
                    points,
                    margin=margin,
                    width=width,
                    height=height,
                )

                detection_count += 1

            if use_kalman:
                final_center = kalman.update(raw_center)
            else:
                final_center = raw_center

            speed_px_s = np.nan
            acceleration_px_s2 = np.nan
            segment_distance_px = np.nan

            # Kinematiko računamo samo, kadar je roka dejansko zaznana.
            # Če roka izgine iz kadra, ne naredimo umetnega skoka od zadnje znane točke do nove.
            if final_center is not None and raw_center is not None:
                if (
                    prev_center_for_velocity is not None
                    and prev_time is not None
                    and prev_frame_for_velocity is not None
                ):
                    frame_gap = frame_idx - prev_frame_for_velocity
                    dt = time_s - prev_time

                    if 0 < frame_gap <= max_gap_frames_for_kinematics and dt > 0:
                        dx = final_center[0] - prev_center_for_velocity[0]
                        dy = final_center[1] - prev_center_for_velocity[1]
                        segment_distance_px = float(np.hypot(dx, dy))
                        cumulative_path_px += segment_distance_px

                        speed_px_s = float(segment_distance_px / dt)

                        if not np.isnan(prev_speed):
                            acceleration_px_s2 = float((speed_px_s - prev_speed) / dt)
                    else:
                        current_track_segment += 1
                        prev_speed = np.nan

                prev_center_for_velocity = final_center
                prev_time = time_s
                prev_frame_for_velocity = frame_idx
                prev_speed = speed_px_s

                path_points.append(final_center)

            else:
                # Roka ni zaznana. Prekinemo zvezni segment poti.
                prev_center_for_velocity = None
                prev_time = None
                prev_frame_for_velocity = None
                prev_speed = np.nan
                current_track_segment += 1

            annotated = draw_annotations(
                frame_bgr=frame,
                hand_landmarks=hand_landmarks,
                points=points,
                raw_center=raw_center,
                final_center=final_center,
                bbox=bbox,
                frame_idx=frame_idx,
                time_s=time_s,
                speed_px_s=speed_px_s,
                acceleration_px_s2=acceleration_px_s2,
                cumulative_path_px=cumulative_path_px,
                use_kalman=use_kalman,
            )

            for a, b in zip(path_points[:-1], path_points[1:]):
                cv2.line(annotated, a, b, (0, 255, 255), 2)

            writer.write(annotated)

            if saved_preview < preview_frames and raw_center is not None and final_center is not None:
                preview_path = (
                    paths["preview_dir"]
                    / f"preview_{saved_preview + 1:02d}_frame_{frame_idx:06d}.png"
                )
                cv2.imwrite(str(preview_path), annotated)
                saved_preview += 1

            if show_ok:
                cv2.imshow("AI hand tracking", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            record = {
                "frame": frame_idx,
                "time_s": time_s,

                # končno središče: Kalman ali raw, odvisno od nastavitve
                "x_px": float(final_center[0]) if final_center is not None else np.nan,
                "y_px": float(final_center[1]) if final_center is not None else np.nan,

                # surovo središče iz MediaPipe
                "raw_x_px": float(raw_center[0]) if raw_center is not None else np.nan,
                "raw_y_px": float(raw_center[1]) if raw_center is not None else np.nan,

                "segment_distance_px": (
                    float(segment_distance_px) if not np.isnan(segment_distance_px) else np.nan
                ),
                "cumulative_path_px": float(cumulative_path_px),

                "speed_px_s": float(speed_px_s) if not np.isnan(speed_px_s) else np.nan,
                "acceleration_px_s2": (
                    float(acceleration_px_s2)
                    if not np.isnan(acceleration_px_s2)
                    else np.nan
                ),

                "detected": 1 if raw_center is not None else 0,
                "track_segment": int(current_track_segment),
                "kalman_used": 1 if use_kalman else 0,
                "kalman_predicted": (
                    1 if use_kalman and raw_center is None and final_center is not None else 0
                ),

                "train_detection_rate": calib["detection_rate"],
                "mean_hand_size_px": calib["mean_hand_size_px"],
            }

            add_landmark_columns(record, points)
            records.append(record)

            frame_idx += 1

    cap.release()
    writer.release()

    if show_ok:
        cv2.destroyAllWindows()

    if not records:
        raise RuntimeError("Ni obdelanih slik.")

    df = pd.DataFrame(records)
    df.to_csv(paths["csv"], index=False)

    summary = compute_summary_metrics(
        df=df,
        video_path=video_path,
        fps=fps,
        width=width,
        height=height,
        use_kalman=use_kalman,
        center_mode=center_mode,
        max_gap_frames_for_kinematics=max_gap_frames_for_kinematics,
    )

    # dodamo še kalibracijske rezultate
    for k, v in calib.items():
        summary[f"calib_{k}"] = v

    pd.DataFrame([summary]).to_csv(paths["summary_csv"], index=False)

    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_results(df, paths, use_kalman=use_kalman)

    print("\nShranjeno:")
    print(f"  video: {paths['video']}")
    print(f"  csv po slikah: {paths['csv']}")
    print(f"  summary csv: {paths['summary_csv']}")
    print(f"  summary json: {paths['summary_json']}")
    print(f"  prikazne slike: {paths['preview_dir']}")
    print(f"  graf položaja: {paths['pos_plot']}")
    print(f"  graf hitrosti: {paths['speed_plot']}")
    print(f"  graf pospeška: {paths['accel_plot']}")
    print(f"  trajektorija: {paths['traj_plot']}")

    print("\nGlavne metrike:")
    print(f"  obdelanih slik: {summary['frames_processed']}")
    print(f"  zaznav roke: {summary['detected_frames']}")
    print(f"  delež zaznav: {summary['detection_rate']:.3f}")
    print(f"  dolžina poti: {summary['total_path_px']:.2f} px")
    print(f"  povprečna hitrost: {summary['mean_speed_px_s']:.2f} px/s")
    print(f"  največja hitrost: {summary['max_speed_px_s']:.2f} px/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI zaznava roke, sklepov in kinematičnih parametrov roke v videu."
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Pot do vhodnega .mp4 videa.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/Data",
        help="Mapa, kjer iščem video, če --video ni podan.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="/workspace/outputs",
        help="Mapa za rezultate.",
    )

    parser.add_argument(
        "--train-frames",
        type=int,
        default=TRAIN_FRAMES_DEFAULT,
        help="Število začetnih slik za kalibracijo.",
    )

    parser.add_argument(
        "--preview-frames",
        type=int,
        default=PREVIEW_FRAMES_DEFAULT,
        help="Število shranjenih anotiranih primerov.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Največ slik za obdelavo. 0 = cel video.",
    )

    parser.add_argument(
        "--center-mode",
        type=str,
        default="palm",
        choices=["palm", "bbox", "all"],
        help="Način izračuna središča roke.",
    )

    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.45,
        help="Minimalna zanesljivost detekcije roke.",
    )

    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.45,
        help="Minimalna zanesljivost sledenja roke.",
    )

    parser.add_argument(
        "--kalman-process-noise",
        type=float,
        default=0.03,
        help="Šum procesa v Kalmanovem filtru.",
    )

    parser.add_argument(
        "--kalman-measurement-noise",
        type=float,
        default=8.0,
        help="Šum meritve v Kalmanovem filtru.",
    )

    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="Izklopi Kalmanov filter in uporabi surovo MediaPipe središče.",
    )

    parser.add_argument(
        "--max-gap-frames-for-kinematics",
        type=int,
        default=1,
        help=(
            "Največja dovoljena vrzel med zaznavami za izračun poti/hitrosti. "
            "1 pomeni samo zaporedne zaznane slike."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Prikaži anotiran video med obdelavo.",
    )

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
        kalman_process_noise=args.kalman_process_noise,
        kalman_measurement_noise=args.kalman_measurement_noise,
        use_kalman=not args.no_kalman,
        max_gap_frames_for_kinematics=args.max_gap_frames_for_kinematics,
    )


if __name__ == "__main__":
    main()