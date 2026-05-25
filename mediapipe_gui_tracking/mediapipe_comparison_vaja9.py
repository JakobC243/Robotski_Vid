from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mediapipe_pin_gui_tracker import (
    ActiveFieldDetector,
    BoardCalibration,
    GuiVideoRenderer,
    MediaPipeHandTracker,
    MotionEstimator,
    calibration_to_json,
    draw_calibration_debug,
    finite_float,
    get_default_video_path,
    load_or_build_calibration,
    open_video_writer,
    transform_point,
)


Point = Tuple[float, float]


def default_cotracker_students_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "vaje"
        / "Vaja09_Sledenje_in_analiza_gibanja 2 (1)"
        / "Vaja09_Sledenje_in_analiza_gibanja"
        / "cotracker_studetns"
    )


def create_paths(output_dir: Path, stem: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "video": output_dir / f"{stem}_comparison.mp4",
        "video_avi": output_dir / f"{stem}_comparison.avi",
        "csv": output_dir / f"{stem}_comparison.csv",
        "summary_json": output_dir / f"{stem}_summary.json",
        "calibration_json": output_dir / f"{stem}_calibration.json",
        "calibration_debug": output_dir / f"{stem}_calibration_debug.png",
        "diff_plot": output_dir / f"{stem}_difference.png",
        "speed_plot": output_dir / f"{stem}_speed_comparison.png",
        "accel_plot": output_dir / f"{stem}_acceleration_comparison.png",
        "trajectory_plot": output_dir / f"{stem}_trajectory_comparison.png",
    }


def resolve_input(args: argparse.Namespace) -> Path:
    return Path(args.input).expanduser().resolve() if args.input else get_default_video_path(Path(args.data_root))


def parse_xy_point(text: Optional[str]) -> Optional[Point]:
    if not text:
        return None
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--query-point expects 'x,y'")
    return float(parts[0]), float(parts[1])


def load_video_rgb_tensor(video_path: Path, max_frames: int, torch_module):
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for CoTracker: {video_path}")
    try:
        while True:
            if max_frames and len(frames) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded for CoTracker: {video_path}")
    video_np = np.stack(frames, axis=0)
    video = torch_module.from_numpy(video_np).permute(0, 3, 1, 2).float()
    return video.unsqueeze(0)


def import_cotracker(args: argparse.Namespace):
    students_root = Path(args.cotracker_root).expanduser().resolve()
    candidates = [
        students_root,
        students_root / "co-tracker",
        students_root.parent / "co-tracker",
        Path("/opt/co-tracker"),
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        import torch
        from cotracker.predictor import CoTrackerOnlinePredictor
    except ImportError as exc:
        raise RuntimeError(
            "CoTracker is not available in this Python environment. "
            "This comparison now uses the CoTracker3 tracker from "
            f"{students_root}. Run it in the cotracker_studetns Docker/Python "
            "environment, or install facebookresearch/co-tracker together with torch."
        ) from exc

    return torch, CoTrackerOnlinePredictor


def run_cotracker(
    video_path: Path,
    query_frame: int,
    query_point: Point,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, str]:
    torch, CoTrackerOnlinePredictor = import_cotracker(args)
    device = args.cotracker_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = Path(args.cotracker_checkpoint).expanduser()
    if not checkpoint.exists():
        raise RuntimeError(
            f"CoTracker checkpoint not found: {checkpoint}. "
            "In the supplied cotracker_studetns Dockerfile this is "
            "/opt/co-tracker/checkpoints/scaled_online.pth."
        )

    video = load_video_rgb_tensor(video_path, args.max_frames, torch).to(device)
    _, total_frames, _, _, _ = video.shape
    if query_frame < 0 or query_frame >= total_frames:
        raise RuntimeError(f"CoTracker query frame {query_frame} is outside video length {total_frames}.")

    queries = torch.tensor(
        [[[float(query_frame), float(query_point[0]), float(query_point[1])]]],
        dtype=torch.float32,
        device=device,
    )

    model = CoTrackerOnlinePredictor(checkpoint=str(checkpoint)).to(device)
    model.eval()
    window = int(model.step * 2)

    pred_tracks = None
    pred_visibility = None
    with torch.no_grad():
        model(video_chunk=video[:, :window], is_first_step=True, queries=queries)
        for start in range(0, total_frames - model.step, model.step):
            chunk = video[:, start : start + window]
            pred_tracks, pred_visibility = model(
                video_chunk=chunk,
                is_first_step=False,
                queries=queries,
            )

    if pred_tracks is None or pred_visibility is None:
        raise RuntimeError(
            "CoTracker did not return tracks. The video may be shorter than the "
            "online model step/window."
        )

    tracks = pred_tracks.detach().cpu().numpy()
    visibility = pred_visibility.detach().cpu().numpy()
    if tracks.ndim == 4:
        tracks = tracks[0, :, 0, :]
    elif tracks.ndim == 3:
        tracks = tracks[:, 0, :]
    else:
        raise RuntimeError(f"Unexpected CoTracker tracks shape: {tracks.shape}")

    if visibility.ndim == 3:
        visibility = visibility[0, :, 0]
    elif visibility.ndim == 2:
        visibility = visibility[:, 0]
    else:
        raise RuntimeError(f"Unexpected CoTracker visibility shape: {visibility.shape}")

    return tracks.astype(np.float32), visibility.astype(np.float32), str(device)


def find_mediapipe_query(
    video_path: Path,
    args: argparse.Namespace,
) -> Tuple[int, Point]:
    override_point = parse_xy_point(args.query_point)
    if override_point is not None:
        if args.query_frame < 0:
            raise RuntimeError("--query-frame must be set when --query-point is used.")
        return int(args.query_frame), override_point

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    tracker = MediaPipeHandTracker(args)
    try:
        frame_idx = 0
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            obs = tracker.detect(frame)
            if obs.detected and obs.point_px is not None:
                return frame_idx, obs.point_px
            frame_idx += 1
    finally:
        tracker.close()
        cap.release()
    raise RuntimeError("MediaPipe did not find an initial point for CoTracker.")


def draw_comparison_overlay(
    frame: np.ndarray,
    calibration: BoardCalibration,
    mp_point: Optional[Point],
    ct_point: Optional[Point],
    mp_traj: Sequence[Tuple[int, int]],
    ct_traj: Sequence[Tuple[int, int]],
    active_field: str,
) -> np.ndarray:
    image = frame.copy()
    if calibration.board_quad is not None:
        cv2.polylines(image, [np.round(calibration.board_quad).astype(np.int32)], True, (170, 170, 170), 2)
    for hole in calibration.holes:
        color = (0, 255, 255) if active_field in [hole.side, "both"] else (0, 150, 0)
        cv2.circle(image, (int(round(hole.x_px)), int(round(hole.y_px))), int(round(max(5, hole.r_px))), color, 1)

    for a, b in zip(mp_traj[:-1], mp_traj[1:]):
        cv2.line(image, a, b, (0, 255, 255), 2, cv2.LINE_AA)
    for a, b in zip(ct_traj[:-1], ct_traj[1:]):
        cv2.line(image, a, b, (255, 80, 255), 2, cv2.LINE_AA)
    if mp_point is not None:
        cv2.circle(image, (int(round(mp_point[0])), int(round(mp_point[1]))), 8, (0, 255, 255), -1)
        cv2.putText(image, "MediaPipe", (int(mp_point[0]) + 10, int(mp_point[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    if ct_point is not None:
        cv2.circle(image, (int(round(ct_point[0])), int(round(ct_point[1]))), 8, (255, 80, 255), 2)
        cv2.putText(image, "CoTracker3", (int(ct_point[0]) + 10, int(ct_point[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 80, 255), 1, cv2.LINE_AA)
    return image


def save_comparison_plots(df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["diff_px"], label="px")
    plt.plot(df["time_s"], df["diff_mm"], label="mm")
    plt.xlabel("time [s]")
    plt.ylabel("difference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["diff_plot"], dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["mp_speed_mm_s"], label="MediaPipe")
    plt.plot(df["time_s"], df["ct_speed_mm_s"], label="CoTracker3")
    plt.xlabel("time [s]")
    plt.ylabel("speed [mm/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["speed_plot"], dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["mp_acceleration_mm_s2"], label="MediaPipe")
    plt.plot(df["time_s"], df["ct_acceleration_mm_s2"], label="CoTracker3")
    plt.xlabel("time [s]")
    plt.ylabel("acceleration [mm/s^2]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["accel_plot"], dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    valid_mp = df.dropna(subset=["mp_x_mm", "mp_y_mm"])
    valid_ct = df.dropna(subset=["ct_x_mm", "ct_y_mm"])
    if not valid_mp.empty:
        plt.plot(valid_mp["mp_x_mm"], valid_mp["mp_y_mm"], label="MediaPipe")
    if not valid_ct.empty:
        plt.plot(valid_ct["ct_x_mm"], valid_ct["ct_y_mm"], label="CoTracker3")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["trajectory_plot"], dpi=150)
    plt.close()


def jitter_from_points(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    pts = df[[x_col, y_col]].dropna().to_numpy(dtype=np.float32)
    if pts.shape[0] < 3:
        return float("nan")
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.std(steps))


def cotracker_point_at(
    tracks: np.ndarray,
    visibility: np.ndarray,
    frame_idx: int,
    require_visibility: bool,
) -> Tuple[Optional[Point], str]:
    if frame_idx >= len(tracks):
        return None, "missing_track"
    visible = bool(visibility[frame_idx] > 0.5) if frame_idx < len(visibility) else True
    if require_visibility and not visible:
        return None, "not_visible"
    point = tracks[frame_idx]
    if point.shape[0] < 2 or not np.all(np.isfinite(point[:2])):
        return None, "invalid"
    return (float(point[0]), float(point[1])), "cotracker_visible" if visible else "cotracker_low_visibility"


def process_comparison(args: argparse.Namespace) -> Dict:
    video_path = resolve_input(args)
    paths = create_paths(Path(args.output_dir), args.output_stem)
    calibration = load_or_build_calibration(video_path, args, paths)
    draw_calibration_debug(calibration, paths["calibration_debug"])
    with open(paths["calibration_json"], "w", encoding="utf-8") as f:
        json.dump(calibration_to_json(calibration), f, indent=2)
    if not calibration.calibrated or len(calibration.holes) != 18:
        raise RuntimeError(f"Calibration failed. Debug image saved to {paths['calibration_debug']}")

    query_frame, query_point = find_mediapipe_query(video_path, args)
    ct_tracks, ct_visibility, ct_device = run_cotracker(video_path, query_frame, query_point, args)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or calibration.width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or calibration.height)
    renderer = GuiVideoRenderer(width, height, graph_width=args.graph_width)
    writer, actual_video_path = open_video_writer(paths["video"], paths["video_avi"], fps, (renderer.width, renderer.height))

    hand_tracker = MediaPipeHandTracker(args)
    mp_motion = MotionEstimator(args.smooth_window, speed_alpha=args.speed_ema_alpha, accel_alpha=args.acceleration_ema_alpha)
    ct_motion = MotionEstimator(args.smooth_window, speed_alpha=args.speed_ema_alpha, accel_alpha=args.acceleration_ema_alpha)
    active_detector = ActiveFieldDetector(calibration, args.active_value_threshold, args.active_diff_threshold)

    rows = []
    diff_history: List[float] = []
    mp_speed_history: List[float] = []
    ct_speed_history: List[float] = []
    mp_traj: Deque[Tuple[int, int]] = deque(maxlen=args.max_trajectory_points)
    ct_traj: Deque[Tuple[int, int]] = deque(maxlen=args.max_trajectory_points)
    show_ok = bool(args.show)
    mp_detected = 0
    ct_detected = 0

    try:
        frame_idx = 0
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            time_s = frame_idx / fps
            obs = hand_tracker.detect(frame)
            mp_point = obs.point_px if obs.detected else None
            mp_detected += int(mp_point is not None)
            ct_point, ct_status = cotracker_point_at(ct_tracks, ct_visibility, frame_idx, args.require_cotracker_visibility)
            ct_detected += int(ct_point is not None)
            active_field, _ = active_detector.detect(frame)

            if mp_point is not None:
                mp_traj.append((int(round(mp_point[0])), int(round(mp_point[1]))))
            if ct_point is not None:
                ct_traj.append((int(round(ct_point[0])), int(round(ct_point[1]))))

            mp_mm = transform_point(mp_point, calibration.image_to_mm_h, calibration.mm_per_px) if mp_point is not None else None
            ct_mm = transform_point(ct_point, calibration.image_to_mm_h, calibration.mm_per_px) if ct_point is not None else None
            mp_speed, mp_accel, mp_distance = mp_motion.update(time_s, mp_mm)
            ct_speed, ct_accel, ct_distance = ct_motion.update(time_s, ct_mm)

            if mp_point is not None and ct_point is not None:
                diff_px = float(np.hypot(mp_point[0] - ct_point[0], mp_point[1] - ct_point[1]))
            else:
                diff_px = np.nan
            if mp_mm is not None and ct_mm is not None:
                diff_mm = float(np.hypot(mp_mm[0] - ct_mm[0], mp_mm[1] - ct_mm[1]))
            else:
                diff_mm = np.nan
            diff_history.append(float(diff_px) if np.isfinite(diff_px) else np.nan)
            mp_speed_history.append(float(mp_speed) if np.isfinite(mp_speed) else np.nan)
            ct_speed_history.append(float(ct_speed) if np.isfinite(ct_speed) else np.nan)

            rows.append(
                {
                    "frame": int(frame_idx),
                    "time_s": float(time_s),
                    "mp_x_px": float(mp_point[0]) if mp_point is not None else np.nan,
                    "mp_y_px": float(mp_point[1]) if mp_point is not None else np.nan,
                    "ct_x_px": float(ct_point[0]) if ct_point is not None else np.nan,
                    "ct_y_px": float(ct_point[1]) if ct_point is not None else np.nan,
                    "ct_visibility": float(ct_visibility[frame_idx]) if frame_idx < len(ct_visibility) else np.nan,
                    "mp_x_mm": float(mp_mm[0]) if mp_mm is not None else np.nan,
                    "mp_y_mm": float(mp_mm[1]) if mp_mm is not None else np.nan,
                    "ct_x_mm": float(ct_mm[0]) if ct_mm is not None else np.nan,
                    "ct_y_mm": float(ct_mm[1]) if ct_mm is not None else np.nan,
                    "diff_px": diff_px,
                    "diff_mm": diff_mm,
                    "mp_speed_mm_s": float(mp_speed) if np.isfinite(mp_speed) else np.nan,
                    "mp_acceleration_mm_s2": float(mp_accel) if np.isfinite(mp_accel) else np.nan,
                    "mp_total_distance_mm": float(mp_distance),
                    "ct_speed_mm_s": float(ct_speed) if np.isfinite(ct_speed) else np.nan,
                    "ct_acceleration_mm_s2": float(ct_accel) if np.isfinite(ct_accel) else np.nan,
                    "ct_total_distance_mm": float(ct_distance),
                    "mp_detected": int(mp_point is not None),
                    "ct_detected": int(ct_point is not None),
                    "ct_status": ct_status,
                    "active_field": active_field,
                }
            )

            overlay = draw_comparison_overlay(frame, calibration, mp_point, ct_point, list(mp_traj), list(ct_traj), active_field)
            gui = renderer.compose(
                overlay,
                {"speed": diff_history, "acceleration": mp_speed_history, "distance": ct_speed_history},
                {
                    "frame": frame_idx,
                    "time_s": time_s,
                    "hand_detected": bool(mp_point is not None),
                    "active_field": active_field,
                    "occupied_count": 0,
                    "speed_mm_s": finite_float(diff_px, 0.0),
                    "acceleration_mm_s2": finite_float(mp_speed, 0.0),
                    "total_distance_mm": finite_float(ct_speed, 0.0),
                },
            )
            cv2.putText(gui, "Left graphs: diff_px | MediaPipe speed | CoTracker3 speed", (14, renderer.status_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1, cv2.LINE_AA)
            writer.write(gui)
            if show_ok:
                try:
                    cv2.imshow("MediaPipe vs CoTracker3", gui)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show_ok = False
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        hand_tracker.close()
        if show_ok:
            cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("No frames were processed.")

    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"], index=False)
    save_comparison_plots(df, paths)

    valid_diff_px = df["diff_px"].dropna()
    valid_diff_mm = df["diff_mm"].dropna()
    summary = {
        "video": str(video_path),
        "comparison_video": str(actual_video_path),
        "tracker": "CoTracker3 online via cotracker_studetns",
        "cotracker_students_root": str(Path(args.cotracker_root).expanduser().resolve()),
        "cotracker_checkpoint": str(Path(args.cotracker_checkpoint).expanduser()),
        "cotracker_device": ct_device,
        "query_frame": int(query_frame),
        "query_point_px": [float(query_point[0]), float(query_point[1])],
        "frames_processed": int(len(df)),
        "fps": float(fps),
        "mediapipe_detection_rate": float(mp_detected / max(1, len(df))),
        "cotracker_detection_rate": float(ct_detected / max(1, len(df))),
        "mediapipe_lost_frames": int(len(df) - mp_detected),
        "cotracker_lost_frames": int(len(df) - ct_detected),
        "mean_difference_px": float(valid_diff_px.mean()) if not valid_diff_px.empty else np.nan,
        "median_difference_px": float(valid_diff_px.median()) if not valid_diff_px.empty else np.nan,
        "mean_difference_mm": float(valid_diff_mm.mean()) if not valid_diff_mm.empty else np.nan,
        "median_difference_mm": float(valid_diff_mm.median()) if not valid_diff_mm.empty else np.nan,
        "mediapipe_jitter_px": jitter_from_points(df, "mp_x_px", "mp_y_px"),
        "cotracker_jitter_px": jitter_from_points(df, "ct_x_px", "ct_y_px"),
        "mediapipe_mean_speed_mm_s": float(df["mp_speed_mm_s"].dropna().mean()) if not df["mp_speed_mm_s"].dropna().empty else np.nan,
        "cotracker_mean_speed_mm_s": float(df["ct_speed_mm_s"].dropna().mean()) if not df["ct_speed_mm_s"].dropna().empty else np.nan,
        "outputs": {k: str(v) for k, v in paths.items() if k != "video_avi"},
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    for key in ["video", "csv", "summary_json", "diff_plot", "speed_plot", "accel_plot", "trajectory_plot"]:
        path = actual_video_path if key == "video" else paths[key]
        print(f"  {key}: {path}")
    print(
        "Summary: frames={} mp_rate={:.3f} cotracker_rate={:.3f} mean_diff_px={:.2f}".format(
            summary["frames_processed"],
            summary["mediapipe_detection_rate"],
            summary["cotracker_detection_rate"],
            finite_float(summary["mean_difference_px"], 0.0),
        )
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MediaPipe hand landmark tracking with the CoTracker3 tracker from Vaja 9/cotracker_studetns."
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--output-dir", type=str, default="mediapipe_gui_tracking/outputs")
    parser.add_argument("--output-stem", type=str, default="mediapipe_comparison_vaja9")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--calibration-cache", type=str, default=None)
    parser.add_argument("--smooth-window", type=int, default=5)

    parser.add_argument("--frame-samples", type=int, default=120)
    parser.add_argument("--hole-spacing-mm", type=float, default=32.0)
    parser.add_argument("--min-hole-radius", type=float, default=2.8)
    parser.add_argument("--max-hole-radius", type=float, default=9.0)
    parser.add_argument("--side-gray-threshold", type=int, default=170)
    parser.add_argument("--side-value-threshold", type=int, default=180)
    parser.add_argument("--side-min-area", type=float, default=15.0)
    parser.add_argument("--side-max-area", type=float, default=130.0)
    parser.add_argument("--side-min-circularity", type=float, default=0.25)
    parser.add_argument("--side-cluster-link-px", type=float, default=45.0)
    parser.add_argument("--side-min-grid-separation-px", type=float, default=90.0)
    parser.add_argument("--min-spacing-px", type=float, default=16.0)
    parser.add_argument("--max-spacing-px", type=float, default=44.0)
    parser.add_argument("--min-grid-matches", type=int, default=7)

    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.45)
    parser.add_argument("--speed-ema-alpha", type=float, default=0.40)
    parser.add_argument("--acceleration-ema-alpha", type=float, default=0.35)
    parser.add_argument("--active-value-threshold", type=float, default=145.0)
    parser.add_argument("--active-diff-threshold", type=float, default=18.0)
    parser.add_argument("--graph-width", type=int, default=420)
    parser.add_argument("--max-trajectory-points", type=int, default=1400)

    parser.add_argument("--cotracker-root", type=str, default=str(default_cotracker_students_root()))
    parser.add_argument("--cotracker-checkpoint", type=str, default="/opt/co-tracker/checkpoints/scaled_online.pth")
    parser.add_argument("--cotracker-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--require-cotracker-visibility", action="store_true")
    parser.add_argument("--query-frame", type=int, default=-1, help="Manual CoTracker query frame. Requires --query-point.")
    parser.add_argument("--query-point", type=str, default=None, help="Manual CoTracker query point as x,y. Defaults to first MediaPipe index tip.")
    return parser.parse_args(argv)


def main() -> None:
    process_comparison(parse_args())


if __name__ == "__main__":
    main()
