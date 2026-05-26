from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from calibration import BoardCalibration, ImageRegion, calibrate_frame, load_calibration, save_calibration
from hand_tracking import MediaPipeHandTracker
from kinematics import KinematicsTracker
from peg_detection import PegOccupancyDetector
from trial_timing import TrialLightStartDetector
from utils import ensure_parent, finite_point, open_video_writer, rotate_if_needed
from visualization import PANEL_WIDTH, compose_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean final 9HPT hand tracking pipeline: calibration, MediaPipe hand, kinematics, video graphs, CSV.")
    parser.add_argument("--input", required=True, help="Input video.")
    parser.add_argument("--output", required=True, help="Output video with overlays.")
    parser.add_argument("--csv-output", required=True, help="CSV output with frame-level measurements.")
    parser.add_argument("--calibration-output", default=None, help="Calibration JSON output.")
    parser.add_argument("--calibration-input", default=None, help="Optional existing calibration JSON.")
    parser.add_argument("--rotate-clockwise", action="store_true", help="Rotate each frame 90 degrees clockwise before all processing.")
    parser.add_argument("--show", action="store_true", help="Show OpenCV preview while processing.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for testing. 0 means full video.")
    parser.add_argument("--smooth-window", type=int, default=7, help="Moving average window for graph values.")
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="EMA alpha for live coordinate and metric smoothing.")
    parser.add_argument("--trail-length", type=int, default=200, help="Number of smoothed hand center points shown as trajectory.")
    parser.add_argument("--start-gate", choices=["center", "none"], default="center", help="Wait until a hand enters the center board zone before locking the active hand.")
    parser.add_argument("--start-gate-scale", type=float, default=0.75, help="Relative size of the expanded center start zone inside the board ROI.")
    parser.add_argument("--start-gate-padding", type=float, default=0.15, help="Relative padding added around the hole-based center start zone used for initial hand lock.")
    parser.add_argument("--start-gate-hits", type=int, default=1, help="Consecutive frames in the start zone required before tracking starts.")
    parser.add_argument("--tracking-roi-padding", type=float, default=0.08, help="Relative padding around the calibrated board ROI used to reject hand candidates after lock.")
    parser.add_argument("--tracking-roi-mode", choices=["soft", "strict", "off"], default="soft", help="How the tracking ROI is used after hand lock: soft allows gradual exits, strict rejects every outside candidate, off ignores it.")
    parser.add_argument("--tracking-roi-outside-frames", type=int, default=45, help="In soft ROI mode, maximum consecutive detected frames allowed outside the tracking ROI.")
    parser.add_argument("--tracking-roi-exit-jump-px", type=float, default=120.0, help="In soft ROI mode, reject a first outside-ROI candidate if it jumps farther than this from the previous center.")
    parser.add_argument("--hand-lock-radius-px", type=float, default=120.0, help="After the active hand is locked, reject hand candidates farther than this many pixels from the previous center.")
    parser.add_argument("--hand-reacquire-radius-px", type=float, default=280.0, help="Maximum radius used when reacquiring the locked hand after missed frames.")
    parser.add_argument("--max-missing-frames", type=int, default=10, help="After this many missed frames, clear the hand lock and wait for the start zone again.")
    parser.add_argument("--motion-weight", type=float, default=0.12, help="How strongly initial active-hand selection prefers the moving hand.")
    parser.add_argument("--min-start-motion-score", type=float, default=2.0, help="Minimum local motion score needed before a hand can start tracking.")
    parser.add_argument("--min-reacquire-motion-score", type=float, default=4.0, help="Minimum local motion score needed before a lost hand can be reacquired.")
    parser.add_argument("--reacquire-hits", type=int, default=2, help="Consecutive frames needed before a lost hand is accepted again.")
    parser.add_argument("--trial-light-start", choices=["on", "off"], default="on", help="Detect trial start from board lights: both fields on, then off, then one stable side.")
    parser.add_argument("--measure-from", choices=["trial-start", "immediate"], default="trial-start", help="Start kinematic stopwatch at detected trial start or immediately at video frame 0.")
    parser.add_argument("--show-trial-status", action="store_true", help="Show light-start text such as WAIT LIGHT START / CAS TECE on the output video.")
    parser.add_argument("--show-light-zones", action="store_true", help="Draw LED detector zones and point samples on the output video.")
    parser.add_argument("--show-field-zones", action="store_true", help="Draw 3x3 hand-in-field diagnostic zones and ROKA LEVO/DESNO text on the output video.")
    parser.add_argument("--field-roi-padding", type=float, default=0.75, help="Padding in hole-spacing units around each 3x3 pin field for the first hand-in-field diagnostic.")
    parser.add_argument("--peg-detection", choices=["on", "off"], default="on", help="Detect inserted pegs from stable local changes around calibrated 3x3 holes.")
    parser.add_argument("--peg-stable-frames", type=int, default=5, help="Stable frames needed before a peg hole changes occupied/empty state.")
    parser.add_argument("--peg-change-threshold", type=float, default=0.32, help="Local patch-change threshold for peg occupancy.")
    parser.add_argument("--trial-end-peg-threshold", type=int, default=8, help="Target peg count that must be reached before an empty target field can stop the trial timer.")
    parser.add_argument("--trial-end-empty-frames", type=int, default=5, help="Consecutive stable empty target-field frames needed to stop the trial timer.")
    parser.add_argument("--trial-end-min-visible", type=int, default=7, help="Minimum visible target holes needed before an empty field can stop the trial timer.")
    parser.add_argument("--light-delta-threshold", type=float, default=22.0, help="Brightness increase over baseline needed to mark a light field as on.")
    parser.add_argument("--light-side-gap-threshold", type=float, default=1.0, help="Minimum baseline-relative brightness gap between sides when deciding which single side is on.")
    parser.add_argument("--both-light-frames", type=int, default=3, help="Stable frames needed for the initial both-fields-on event.")
    parser.add_argument("--off-light-frames", type=int, default=2, help="Stable frames with both light fields off required before one side can start trial time.")
    parser.add_argument("--single-light-frames", type=int, default=5, help="Stable frames needed before the single-side light starts trial time.")
    return parser.parse_args()


def first_processed_frame(cap: cv2.VideoCapture, rotate_clockwise: bool) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from input video.")
    return rotate_if_needed(frame, rotate_clockwise)


def board_point(calibration: BoardCalibration, point: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    return calibration.image_to_board(point)


def activation_roi_from_calibration(calibration: BoardCalibration, padding_ratio: float) -> Optional[ImageRegion]:
    return calibration.start_zone_image_roi(padding_ratio=padding_ratio)


def padded_board_roi(calibration: BoardCalibration, padding_ratio: float) -> Optional[Tuple[int, int, int, int]]:
    if calibration.board_roi is None:
        return None
    x, y, w, h = calibration.board_roi
    pad_x = float(w) * float(max(0.0, padding_ratio))
    pad_y = float(h) * float(max(0.0, padding_ratio))
    x1 = int(max(0, np.floor(float(x) - pad_x)))
    y1 = int(max(0, np.floor(float(y) - pad_y)))
    x2 = int(min(calibration.frame_width - 1, np.ceil(float(x + w) + pad_x)))
    y2 = int(min(calibration.frame_height - 1, np.ceil(float(y + h) + pad_y)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def hand_field_contact(observation, field_regions: List[Dict]) -> Tuple[str, Dict[str, int]]:
    counts = {"left": 0, "right": 0}
    if not observation.detected or observation.landmarks_px is None:
        return "", counts
    points = np.asarray(observation.landmarks_px[:, :2], dtype=np.float32)
    for region in field_regions:
        side = str(region.get("side", ""))
        if side not in counts:
            continue
        polygon = np.asarray(region.get("polygon", []), dtype=np.float32).reshape(-1, 2)
        if polygon.shape[0] < 3:
            continue
        count = 0
        for point in points:
            if cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0.0:
                count += 1
        counts[side] = max(counts[side], count)
    if counts["left"] > 0 and counts["right"] > 0:
        if counts["left"] == counts["right"]:
            return "both", counts
        return ("left" if counts["left"] > counts["right"] else "right"), counts
    if counts["left"] > 0:
        return "left", counts
    if counts["right"] > 0:
        return "right", counts
    return "", counts


def covered_sides_from_hand_field(hand_field_zone: str) -> set:
    if hand_field_zone == "left":
        return {"left"}
    if hand_field_zone == "right":
        return {"right"}
    if hand_field_zone == "both":
        return {"left", "right", "both"}
    return set()


def peg_target_count(peg_info) -> int:
    target_side = str(getattr(peg_info, "target_side", ""))
    states = getattr(peg_info, "states", {}).get(target_side, []) if target_side else []
    return int(sum(int(value) for value in states))


def peg_target_visible_count(peg_info) -> int:
    target_side = str(getattr(peg_info, "target_side", ""))
    visible = getattr(peg_info, "visible", {}).get(target_side, []) if target_side else []
    return int(sum(int(value) for value in visible))


def peg_target_covered(peg_info) -> bool:
    target_side = str(getattr(peg_info, "target_side", ""))
    covered = getattr(peg_info, "covered", {})
    return bool(target_side and covered.get(target_side, False))


def empty_kinematics_row(path_length: float = 0.0) -> Dict[str, float]:
    return {
        "hand_center_x": float("nan"),
        "hand_center_y": float("nan"),
        "path_length_px_cumulative": float(path_length),
        "speed_px_s": float("nan"),
        "speed_px_s_smooth": float("nan"),
        "acceleration_px_s2": float("nan"),
        "acceleration_px_s2_smooth": float("nan"),
        "thumb_index_distance_px_smooth": float("nan"),
        "speed_px_s_raw": float("nan"),
        "acceleration_px_s2_raw": float("nan"),
    }


def build_row(
    frame_idx: int,
    time_s: float,
    measurement_started: bool,
    measurement_running: bool,
    measurement_completed: bool,
    measurement_end_frame: int,
    measurement_time_s: float,
    observation,
    kin: Dict[str, float],
    calibration: BoardCalibration,
    trial_info,
    hand_field_zone: str,
    hand_field_counts: Dict[str, int],
    peg_info,
) -> Dict[str, object]:
    raw_center = observation.center_raw if observation.detected else None
    wrist = observation.wrist if observation.detected else None
    thumb = observation.thumb_tip if observation.detected else None
    index = observation.index_tip if observation.detected else None
    smoothed_center = (
        float(kin["hand_center_x"]),
        float(kin["hand_center_y"]),
    ) if np.isfinite(kin["hand_center_x"]) and np.isfinite(kin["hand_center_y"]) else None

    center_board = board_point(calibration, smoothed_center)
    wrist_board = board_point(calibration, wrist)
    thumb_board = board_point(calibration, thumb)
    index_board = board_point(calibration, index)

    row = {
        "frame_idx": int(frame_idx),
        "time_s": float(time_s),
        "measurement_started": int(measurement_started),
        "measurement_running": int(measurement_running),
        "measurement_completed": int(measurement_completed),
        "measurement_end_frame": int(measurement_end_frame) if measurement_end_frame >= 0 else np.nan,
        "measurement_time_s": float(measurement_time_s) if np.isfinite(measurement_time_s) else np.nan,
        "hand_detected": int(observation.detected),
        "track_started": int(observation.track_started),
        "waiting_for_start": int(observation.waiting_for_start),
        "active_hand_label": observation.active_hand_label if observation.detected else "",
        "hand_score": float(observation.hand_score) if observation.detected else np.nan,
        "hand_motion_score": float(observation.motion_score) if observation.detected else np.nan,
        "hand_missing_frames": int(observation.missing_frames),
        "tracking_roi_inside": int(observation.tracking_roi_inside) if observation.detected else 0,
        "tracking_roi_outside_frames": int(observation.tracking_roi_outside_frames) if observation.detected else 0,
        "hand_center_x_raw": float(raw_center[0]) if finite_point(raw_center) else np.nan,
        "hand_center_y_raw": float(raw_center[1]) if finite_point(raw_center) else np.nan,
        "hand_center_x": kin["hand_center_x"],
        "hand_center_y": kin["hand_center_y"],
        "wrist_x": float(wrist[0]) if finite_point(wrist) else np.nan,
        "wrist_y": float(wrist[1]) if finite_point(wrist) else np.nan,
        "thumb_tip_x": float(thumb[0]) if finite_point(thumb) else np.nan,
        "thumb_tip_y": float(thumb[1]) if finite_point(thumb) else np.nan,
        "index_tip_x": float(index[0]) if finite_point(index) else np.nan,
        "index_tip_y": float(index[1]) if finite_point(index) else np.nan,
        "thumb_index_distance_px": float(observation.thumb_index_distance_px) if observation.detected else np.nan,
        "thumb_index_distance_px_smooth": kin["thumb_index_distance_px_smooth"],
        "path_length_px_cumulative": kin["path_length_px_cumulative"],
        "speed_px_s": kin["speed_px_s"],
        "speed_px_s_smooth": kin["speed_px_s_smooth"],
        "acceleration_px_s2": kin["acceleration_px_s2"],
        "acceleration_px_s2_smooth": kin["acceleration_px_s2_smooth"],
        "speed_px_s_raw": kin["speed_px_s_raw"],
        "acceleration_px_s2_raw": kin["acceleration_px_s2_raw"],
        "calibrated": int(calibration.calibrated),
        "hand_center_board_x": center_board[0],
        "hand_center_board_y": center_board[1],
        "wrist_board_x": wrist_board[0],
        "wrist_board_y": wrist_board[1],
        "thumb_tip_board_x": thumb_board[0],
        "thumb_tip_board_y": thumb_board[1],
        "index_tip_board_x": index_board[0],
        "index_tip_board_y": index_board[1],
        "hand_field_zone": hand_field_zone,
        "hand_field_left_count": int(hand_field_counts.get("left", 0)),
        "hand_field_right_count": int(hand_field_counts.get("right", 0)),
    }
    if trial_info is not None:
        row.update(trial_info.to_row())
    if peg_info is not None:
        row.update(peg_info.to_row())
    return row


def print_summary(
    args: argparse.Namespace,
    fps: float,
    total_frames: int,
    rows: List[Dict[str, object]],
    calibration: BoardCalibration,
) -> None:
    df = pd.DataFrame(rows)
    processed = len(df)
    detected = int(df["hand_detected"].sum()) if processed else 0
    detection_rate = float(detected / processed) if processed else 0.0
    speed = pd.to_numeric(df.get("speed_px_s_smooth", pd.Series(dtype=float)), errors="coerce").dropna()
    accel = pd.to_numeric(df.get("acceleration_px_s2_smooth", pd.Series(dtype=float)), errors="coerce").dropna()
    pinch = pd.to_numeric(df.get("thumb_index_distance_px_smooth", pd.Series(dtype=float)), errors="coerce").dropna()
    total_path = float(pd.to_numeric(df.get("path_length_px_cumulative", pd.Series([0.0])), errors="coerce").dropna().max()) if processed else 0.0
    duration_s = float(processed / fps) if fps > 0 else 0.0
    trial_started = bool("trial_started" in df and pd.to_numeric(df["trial_started"], errors="coerce").fillna(0).max() > 0)
    trial_side = ""
    trial_start_frame = np.nan
    measurement_completed = bool("measurement_completed" in df and pd.to_numeric(df["measurement_completed"], errors="coerce").fillna(0).max() > 0)
    measurement_end_frame = np.nan
    measurement_elapsed_s = np.nan
    if trial_started:
        started_rows = df[pd.to_numeric(df["trial_started"], errors="coerce").fillna(0) > 0]
        if not started_rows.empty:
            trial_side = str(started_rows["trial_side"].iloc[-1]) if "trial_side" in started_rows else ""
            trial_start_frame = pd.to_numeric(started_rows["trial_start_frame"], errors="coerce").dropna().min()
    if measurement_completed:
        completed_rows = df[pd.to_numeric(df["measurement_completed"], errors="coerce").fillna(0) > 0]
        if not completed_rows.empty:
            measurement_end_frame = pd.to_numeric(completed_rows["measurement_end_frame"], errors="coerce").dropna().min()
            measurement_elapsed_s = pd.to_numeric(completed_rows["measurement_time_s"], errors="coerce").dropna().min()

    print("\nFinal hand pipeline summary")
    print(f"  input video: {args.input}")
    print(f"  output video: {args.output}")
    print(f"  csv output: {args.csv_output}")
    print(f"  fps: {fps:.3f}")
    print(f"  total frames: {total_frames}")
    print(f"  processed frames: {processed}")
    print(f"  duration_s: {duration_s:.3f}")
    print(f"  detected_frames: {detected}")
    print(f"  detection_rate: {detection_rate:.3f}")
    print(f"  calibration_status: {calibration.status} ({calibration.source})")
    print(f"  trial_started: {trial_started}")
    print(f"  trial_side: {trial_side}")
    print(f"  trial_start_frame: {trial_start_frame}")
    print(f"  measurement_completed: {measurement_completed}")
    print(f"  measurement_end_frame: {measurement_end_frame}")
    print(f"  measurement_elapsed_s: {measurement_elapsed_s}")
    print(f"  total_path_length_px: {total_path:.3f}")
    print(f"  mean_speed_px_s: {float(speed.mean()) if not speed.empty else np.nan:.3f}")
    print(f"  max_speed_px_s: {float(speed.max()) if not speed.empty else np.nan:.3f}")
    print(f"  mean_acceleration_px_s2: {float(accel.mean()) if not accel.empty else np.nan:.3f}")
    print(f"  max_acceleration_px_s2: {float(accel.max()) if not accel.empty else np.nan:.3f}")
    print(f"  mean_thumb_index_distance_px: {float(pinch.mean()) if not pinch.empty else np.nan:.3f}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    csv_output_path = Path(args.csv_output)
    calibration_output_path = Path(args.calibration_output) if args.calibration_output else None
    ensure_parent(output_path)
    ensure_parent(csv_output_path)
    if calibration_output_path is not None:
        ensure_parent(calibration_output_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    first_frame = first_processed_frame(cap, args.rotate_clockwise)
    frame_h, frame_w = first_frame.shape[:2]

    if args.calibration_input:
        calibration = load_calibration(Path(args.calibration_input))
        if calibration.frame_width != frame_w or calibration.frame_height != frame_h:
            print(
                "Warning: calibration dimensions do not match processed video "
                f"({calibration.frame_width}x{calibration.frame_height} vs {frame_w}x{frame_h})."
            )
    else:
        calibration = calibrate_frame(first_frame, video=str(input_path), allow_manual=True)
        if calibration_output_path is not None:
            save_calibration(calibration, calibration_output_path)
    start_activation_roi = activation_roi_from_calibration(calibration, args.start_gate_padding)
    tracking_roi = padded_board_roi(calibration, args.tracking_roi_padding)
    field_regions = calibration.hole_grid_regions(padding_scale=args.field_roi_padding)
    peg_detector = (
        PegOccupancyDetector(
            calibration=calibration,
            stable_frames=args.peg_stable_frames,
            change_threshold=args.peg_change_threshold,
        )
        if args.peg_detection == "on"
        else None
    )

    writer = open_video_writer(output_path, fps, (frame_w + PANEL_WIDTH, frame_h))
    trial_detector = (
        TrialLightStartDetector(
            calibration=calibration,
            frame_shape=first_frame.shape,
            fps=fps,
            light_delta_threshold=args.light_delta_threshold,
            side_gap_threshold=args.light_side_gap_threshold,
            both_stable_frames=args.both_light_frames,
            off_stable_frames=args.off_light_frames,
            side_stable_frames=args.single_light_frames,
        )
        if args.trial_light_start == "on"
        else None
    )
    tracker = MediaPipeHandTracker(
        start_gate_enabled=args.start_gate == "center",
        start_gate_scale=args.start_gate_scale,
        activation_consecutive_frames=args.start_gate_hits,
        max_missing_frames=args.max_missing_frames,
        max_lock_jump_px=args.hand_lock_radius_px,
        max_reacquire_jump_px=args.hand_reacquire_radius_px,
        motion_weight=args.motion_weight,
        min_start_motion_score=args.min_start_motion_score,
        min_reacquire_motion_score=args.min_reacquire_motion_score,
        reacquire_consecutive_frames=args.reacquire_hits,
        tracking_roi_mode=args.tracking_roi_mode,
        max_tracking_roi_outside_frames=args.tracking_roi_outside_frames,
        max_tracking_roi_exit_jump_px=args.tracking_roi_exit_jump_px,
    )
    kinematics = KinematicsTracker(args.smooth_alpha, args.smooth_window)
    trail: Deque[Tuple[int, int]] = deque(maxlen=max(1, int(args.trail_length)))
    history: Dict[str, Deque[float]] = {
        "path": deque(maxlen=360),
        "speed": deque(maxlen=360),
        "acceleration": deque(maxlen=360),
        "thumb_index": deque(maxlen=360),
    }
    rows: List[Dict[str, object]] = []
    measurement_started = bool(args.measure_from == "immediate" or trial_detector is None)
    measurement_completed = False
    measurement_start_frame = 0 if measurement_started else -1
    measurement_end_frame = -1
    max_target_peg_count = 0
    empty_target_frames = 0
    show_ok = bool(args.show)
    if show_ok:
        try:
            cv2.namedWindow("final hand pipeline", cv2.WINDOW_NORMAL)
        except cv2.error:
            show_ok = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    try:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = rotate_if_needed(frame, args.rotate_clockwise)
            time_s = frame_idx / fps
            trial_info = trial_detector.update(frame_idx, frame) if trial_detector is not None else None
            activation_roi = tracker.get_activation_roi(frame.shape, calibration.board_roi, start_activation_roi)
            observation = tracker.detect(frame, calibration.board_roi, start_activation_roi, tracking_roi)
            hand_field_zone, hand_field_counts = hand_field_contact(observation, field_regions)
            if not measurement_started and trial_info is not None and bool(getattr(trial_info, "trial_started", False)):
                measurement_started = True
                measurement_start_frame = int(frame_idx)
                measurement_end_frame = -1
                measurement_completed = False
                max_target_peg_count = 0
                empty_target_frames = 0
                kinematics.reset()
                trail.clear()
                for values in history.values():
                    values.clear()

            measurement_running = bool(measurement_started and not measurement_completed)
            measurement_time_s = (
                max(0.0, (float(measurement_end_frame if measurement_completed else frame_idx) - float(measurement_start_frame)) / fps)
                if measurement_started and measurement_start_frame >= 0
                else float("nan")
            )
            if measurement_running:
                kin = kinematics.update(measurement_time_s, observation.detected, observation.center_raw, observation.thumb_index_distance_px)
            else:
                kin = empty_kinematics_row()
            peg_info = None
            if peg_detector is not None:
                target_side = str(getattr(trial_info, "trial_side", "")) if trial_info is not None else ""
                peg_info = peg_detector.update(
                    frame,
                    covered_sides=covered_sides_from_hand_field(hand_field_zone),
                    target_side=target_side,
                    measurement_active=measurement_started,
                )
                if measurement_running and target_side:
                    target_count = peg_target_count(peg_info)
                    max_target_peg_count = max(max_target_peg_count, target_count)
                    end_candidate = bool(
                        max_target_peg_count >= int(args.trial_end_peg_threshold)
                        and target_count == 0
                        and not peg_target_covered(peg_info)
                        and peg_target_visible_count(peg_info) >= int(args.trial_end_min_visible)
                    )
                    if end_candidate:
                        empty_target_frames += 1
                    else:
                        empty_target_frames = 0
                    if empty_target_frames >= int(args.trial_end_empty_frames):
                        measurement_completed = True
                        measurement_end_frame = int(frame_idx)
                        measurement_running = False
                        measurement_time_s = max(0.0, (float(measurement_end_frame) - float(measurement_start_frame)) / fps)
            smoothed_center = None
            if measurement_running and np.isfinite(kin["hand_center_x"]) and np.isfinite(kin["hand_center_y"]):
                smoothed_center = (float(kin["hand_center_x"]), float(kin["hand_center_y"]))
                trail.append((int(round(smoothed_center[0])), int(round(smoothed_center[1]))))

            if measurement_running:
                history["path"].append(float(kin["path_length_px_cumulative"]))
                history["speed"].append(float(kin["speed_px_s_smooth"]))
                history["acceleration"].append(float(kin["acceleration_px_s2_smooth"]))
                history["thumb_index"].append(float(kin["thumb_index_distance_px_smooth"]))

            rows.append(
                build_row(
                    frame_idx,
                    time_s,
                    measurement_started,
                    measurement_running,
                    measurement_completed,
                    measurement_end_frame,
                    measurement_time_s,
                    observation,
                    kin,
                    calibration,
                    trial_info,
                    hand_field_zone,
                    hand_field_counts,
                    peg_info,
                )
            )
            display_frame = frame.copy()
            if args.show_light_zones and trial_detector is not None and trial_info is not None:
                trial_detector.draw_zones(display_frame, trial_info)
            composed = compose_frame(
                display_frame,
                calibration,
                observation,
                smoothed_center,
                list(trail),
                history,
                fps,
                args.smooth_window,
                activation_roi=activation_roi,
                trial_info=trial_info,
                measurement_started=measurement_started,
                measurement_completed=measurement_completed,
                measurement_time_s=measurement_time_s,
                show_trial_status=args.show_trial_status,
                field_regions=field_regions if args.show_field_zones else [],
                hand_field_zone=hand_field_zone if args.show_field_zones else "",
                peg_detector=peg_detector,
                peg_info=peg_info,
            )
            writer.write(composed)
            if show_ok:
                cv2.imshow("final hand pipeline", composed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        tracker.close()
        if show_ok:
            cv2.destroyAllWindows()

    pd.DataFrame(rows).to_csv(csv_output_path, index=False)
    print_summary(args, fps, total_frames, rows, calibration)


if __name__ == "__main__":
    main()
