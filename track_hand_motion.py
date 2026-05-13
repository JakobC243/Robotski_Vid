import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def get_default_video_path(data_root: Path) -> Path:
    videos = sorted(data_root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 videos found under {data_root}")
    return videos[0]


def create_output_paths(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "video": output_root / "hand_tracking_preview.avi",
        "csv": output_root / "hand_tracking.csv",
        "pos_plot": output_root / "position_x_y.png",
        "speed_plot": output_root / "speed_over_time.png",
        "accel_plot": output_root / "acceleration_over_time.png",
        "traj_plot": output_root / "trajectory_xy.png",
    }


def find_skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(frame_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 20, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_motion_mask(frame_bgr: np.ndarray, prev_gray: Optional[np.ndarray]) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        motion_mask = np.zeros_like(gray)
    else:
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    return motion_mask, gray


def find_hand_contour(
    mask: np.ndarray,
    previous_center: Optional[Tuple[int, int]] = None,
    min_area: float = 1500.0,
) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        distance = 0.0
        if previous_center is not None:
            dx = cx - previous_center[0]
            dy = cy - previous_center[1]
            distance = np.hypot(dx, dy)
        score = area - 0.15 * distance
        candidates.append((score, area, (cx, cy), contour))
    if not candidates:
        return None, None, None, 0.0
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, area, center, contour = candidates[0]
    bbox = cv2.boundingRect(contour)
    return center, contour, bbox, area


def safe_show(window_name: str, image: np.ndarray) -> bool:
    try:
        cv2.imshow(window_name, image)
        return True
    except cv2.error:
        return False


def plot_results(df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["frame"], df["x_px"], label="x", marker=".")
    plt.plot(df["frame"], df["y_px"], label="y", marker=".")
    plt.title("Hand position over frames")
    plt.xlabel("frame")
    plt.ylabel("position (px)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["pos_plot"])
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["frame"], df["speed_px_s"], color="tab:blue", marker=".")
    plt.title("Hand speed over time")
    plt.xlabel("frame")
    plt.ylabel("speed (px/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["speed_plot"])
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["frame"], df["acceleration_px_s2"], color="tab:red", marker=".")
    plt.title("Hand acceleration over time")
    plt.xlabel("frame")
    plt.ylabel("acceleration (px/s^2)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["accel_plot"])
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(df["x_px"], df["y_px"], marker=".", linestyle="-", color="tab:green")
    plt.title("Hand trajectory in image plane")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["traj_plot"])
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Track moving hand center in a video.")
    parser.add_argument("--video", type=str, default=None, help="Path to input .mp4 video.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated video while processing (press q to quit).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process. 0 means process full video.",
    )
    args = parser.parse_args()

    data_root = Path("/data/Data")
    output_root = Path("/workspace/outputs")
    paths = create_output_paths(output_root)

    if args.video is None:
        video_path = get_default_video_path(data_root)
    else:
        video_path = Path(args.video)
    print(f"Input video: {video_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Video size: {width}x{height}, fps={fps:.2f}, frames={frame_count}")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(paths["video"]), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open output video writer: {paths['video']}")

    records: List[Dict[str, float]] = []
    path_points: List[Tuple[int, int]] = []
    prev_gray = None
    prev_center = None
    prev_time = None
    prev_speed = np.nan
    detection_count = 0

    show_ok = args.show
    if show_ok:
        try:
            cv2.namedWindow("Hand tracking", cv2.WINDOW_NORMAL)
        except cv2.error:
            show_ok = False
            print("Warning: GUI display not available; continuing without --show.")

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        time_s = frame_idx / fps
        motion_mask, prev_gray = find_motion_mask(frame, prev_gray)
        skin_mask = find_skin_mask(frame)
        combined_mask = cv2.bitwise_and(skin_mask, motion_mask)
        if frame_idx == 0 or np.count_nonzero(combined_mask) < 100:
            combined_mask = skin_mask

        center, contour, bbox, area = find_hand_contour(combined_mask, previous_center=prev_center)
        speed_px_s = np.nan
        acceleration_px_s2 = np.nan

        if center is not None:
            detection_count += 1
            if prev_center is not None and prev_time is not None:
                dt = time_s - prev_time
                if dt > 0:
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]
                    speed_px_s = float(np.hypot(dx, dy) / dt)
                    if not np.isnan(prev_speed):
                        acceleration_px_s2 = float((speed_px_s - prev_speed) / dt)
            prev_center = center
            prev_time = time_s
            prev_speed = speed_px_s
            path_points.append(center)
        else:
            prev_speed = np.nan

        annotated = frame.copy()
        if contour is not None:
            cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if center is not None:
            cv2.circle(annotated, center, 8, (0, 0, 255), -1)
            cv2.circle(annotated, center, 14, (255, 255, 255), 2)

        for a, b in zip(path_points[:-1], path_points[1:]):
            cv2.line(annotated, a, b, (0, 255, 255), 2)

        info_lines = [
            f"frame={frame_idx}",
            f"t={time_s:.2f}s",
            f"area={area:.0f}",
        ]
        if center is not None:
            info_lines.append(f"pos=({center[0]},{center[1]})")
        else:
            info_lines.append("pos=(nan,nan)")
        if not np.isnan(speed_px_s):
            info_lines.append(f"speed={speed_px_s:.1f}px/s")
        else:
            info_lines.append("speed=nan")
        if not np.isnan(acceleration_px_s2):
            info_lines.append(f"acc={acceleration_px_s2:.1f}px/s^2")
        else:
            info_lines.append("acc=nan")
        info_text = "  ".join(info_lines)
        cv2.putText(
            annotated,
            info_text,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if center is None:
            cv2.putText(
                annotated,
                "No hand detected",
                (16, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(annotated)

        if show_ok:
            if not safe_show("Hand tracking", annotated):
                show_ok = False
                print("Warning: GUI display failed during processing; continuing without --show.")
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        records.append(
            {
                "frame": frame_idx,
                "time_s": time_s,
                "x_px": float(center[0]) if center is not None else np.nan,
                "y_px": float(center[1]) if center is not None else np.nan,
                "area_px": float(area),
                "speed_px_s": float(speed_px_s) if not np.isnan(speed_px_s) else np.nan,
                "acceleration_px_s2": float(acceleration_px_s2) if not np.isnan(acceleration_px_s2) else np.nan,
            }
        )

        frame_idx += 1

    cap.release()
    writer.release()
    if show_ok:
        cv2.destroyAllWindows()

    if not records:
        raise RuntimeError("No frames were processed.")

    df = pd.DataFrame(records)
    df.to_csv(paths["csv"], index=False)
    plot_results(df, paths)

    print(f"Saved annotated video: {paths['video']}")
    print(f"Saved tracking CSV: {paths['csv']}")
    print(f"Saved plots: {paths['pos_plot']}, {paths['speed_plot']}, {paths['accel_plot']}, {paths['traj_plot']}")
    print(f"Total frames processed: {frame_idx}, detections: {detection_count}")


if __name__ == "__main__":
    main()
