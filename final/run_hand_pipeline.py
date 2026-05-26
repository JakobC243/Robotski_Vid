from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from calibration import BoardCalibration, ImageRegion, calibrate_best_frame, load_calibration, save_calibration
from hand_tracking import MediaPipeHandTracker
from kinematics import KinematicsTracker
from peg_detection import PegOccupancyDetector
from trial_timing import TrialLightStartDetector
from utils import distance, ensure_parent, finite_point, open_video_writer, rotate_if_needed
from visualization import PANEL_WIDTH, compose_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean final 9HPT hand tracking pipeline: calibration, MediaPipe hand, kinematics, video graphs, CSV.")
    parser.add_argument("--input", required=True, help="Input video.")
    parser.add_argument("--output", required=True, help="Output video with overlays.")
    parser.add_argument("--csv-output", required=True, help="CSV output with frame-level measurements.")
    parser.add_argument("--calibration-output", default=None, help="Calibration JSON output.")
    parser.add_argument("--calibration-input", default=None, help="Optional existing calibration JSON.")
    parser.add_argument("--calibration-scan-frames", type=int, default=60, help="Number of initial frames scanned for the best hole-grid calibration.")
    parser.add_argument("--calibration-scan-step", type=int, default=5, help="Frame step used during initial multi-frame calibration scan.")
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
    parser.add_argument("--hand-field-start-fallback", choices=["on", "off"], default="off", help="Experimental fallback: start the stopwatch from hand entry into a left/right 3x3 field if light start is not detected.")
    parser.add_argument("--hand-field-start-frames", type=int, default=3, help="Consecutive frames in the same left/right 3x3 field needed for fallback stopwatch start.")
    parser.add_argument("--show-trial-status", action="store_true", help="Show light-start debug text on the output video.")
    parser.add_argument("--show-light-zones", action="store_true", help="Draw LED detector zones and point samples on the output video.")
    parser.add_argument("--show-field-zones", action="store_true", help="Draw 3x3 hand-in-field diagnostic zones on the output video.")
    parser.add_argument("--field-roi-padding", type=float, default=0.75, help="Padding in hole-spacing units around each 3x3 pin field for the first hand-in-field diagnostic.")
    parser.add_argument("--peg-detection", choices=["on", "off"], default="on", help="Detect inserted pegs from stable local changes around calibrated 3x3 holes.")
    parser.add_argument("--peg-stable-frames", type=int, default=5, help="Stable frames needed before a peg hole changes occupied/empty state.")
    parser.add_argument("--peg-post-cover-clear-frames", type=int, default=3, help="Clear frames after a hand leaves a peg field before peg state can change.")
    parser.add_argument("--peg-change-threshold", type=float, default=0.32, help="Local patch-change threshold for peg occupancy.")
    parser.add_argument("--trial-end-peg-threshold", type=int, default=8, help="Target peg count that must be reached before an empty target field can stop the trial timer.")
    parser.add_argument("--trial-end-empty-frames", type=int, default=5, help="Consecutive stable empty target-field frames needed to stop the trial timer.")
    parser.add_argument("--trial-end-min-visible", type=int, default=7, help="Minimum visible target holes needed before an empty field can stop the trial timer.")
    parser.add_argument("--hole-spacing-mm", type=float, default=32.0, help="Physical spacing between neighboring holes in each calibrated 3x3 field.")
    parser.add_argument("--light-delta-threshold", type=float, default=22.0, help="Brightness increase over baseline needed to mark a light field as on.")
    parser.add_argument("--light-side-gap-threshold", type=float, default=1.0, help="Minimum baseline-relative brightness gap between sides when deciding which single side is on.")
    parser.add_argument("--both-light-frames", type=int, default=3, help="Stable frames needed for the initial both-fields-on event.")
    parser.add_argument("--off-light-frames", type=int, default=2, help="Stable frames with both light fields off required before one side can start trial time.")
    parser.add_argument("--single-light-frames", type=int, default=5, help="Stable frames needed before the single-side light starts trial time.")
    return parser.parse_args()


def first_processed_frames(cap: cv2.VideoCapture, rotate_clockwise: bool, count: int, step: int) -> List[Tuple[int, np.ndarray]]:
    frames: List[Tuple[int, np.ndarray]] = []
    scan_frames = int(max(1, count))
    scan_step = int(max(1, step))
    for frame_idx in range(0, scan_frames, scan_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append((int(frame_idx), rotate_if_needed(frame, rotate_clockwise)))
    if not frames:
        raise RuntimeError("Could not read frames from input video.")
    return frames


def board_point(calibration: BoardCalibration, point: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    return calibration.image_to_board(point)


def build_metric_homography(calibration: BoardCalibration, hole_spacing_mm: float) -> Tuple[Optional[np.ndarray], str]:
    candidates = []
    for grid in calibration.hole_grids:
        expected_points = np.asarray(grid.get("expected_points", []), dtype=np.float32).reshape(-1, 2)
        if expected_points.shape[0] < 9 or not np.all(np.isfinite(expected_points)):
            continue
        expected_metric_points = []
        for row in range(3):
            for col in range(3):
                expected_metric_points.append([float(col) * hole_spacing_mm, float(row) * hole_spacing_mm])
        expected_metric_points_np = np.asarray(expected_metric_points, dtype=np.float32)

        matched_points = np.asarray(grid.get("matched_points", []), dtype=np.float32).reshape(-1, 2)
        matched_local_ids = [int(v) for v in grid.get("matched_local_ids", [])]
        if matched_points.shape[0] >= 4 and len(matched_local_ids) == matched_points.shape[0]:
            valid_pairs = [
                (point, expected_metric_points_np[local_id])
                for point, local_id in zip(matched_points, matched_local_ids)
                if 0 <= local_id < 9 and np.all(np.isfinite(point))
            ]
            image_points = np.asarray([point for point, _ in valid_pairs], dtype=np.float32)
            metric_points_np = np.asarray([point for _, point in valid_pairs], dtype=np.float32)
        else:
            image_points = expected_points[:9]
            metric_points_np = expected_metric_points_np

        if image_points.shape[0] < 4:
            continue
        homography, _ = cv2.findHomography(image_points, metric_points_np, method=0)
        if homography is None or not np.all(np.isfinite(homography)):
            continue
        matches = int(grid.get("matches", 0))
        mean_error = float(grid.get("mean_error_px", 99.0))
        inferred_penalty = 1 if bool(grid.get("inferred", False)) else 0
        score = matches * 1000.0 - mean_error * 80.0 - inferred_penalty * 250.0
        candidates.append((score, homography.astype(np.float32), int(grid.get("grid_id", len(candidates)))))
    if not candidates:
        return None, ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, homography, grid_id = candidates[0]
    return homography, f"hole_grid_{grid_id}"


def image_to_metric(homography: Optional[np.ndarray], point: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if homography is None or not finite_point(point):
        return float("nan"), float("nan")
    src = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


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


def dominant_hand_field_side(hand_field_zone: str, hand_field_counts: Dict[str, int]) -> str:
    if hand_field_zone in ("left", "right"):
        return hand_field_zone
    left_count = int(hand_field_counts.get("left", 0))
    right_count = int(hand_field_counts.get("right", 0))
    if left_count > right_count and left_count > 0:
        return "left"
    if right_count > left_count and right_count > 0:
        return "right"
    return ""


def lit_single_side(trial_info) -> str:
    if trial_info is None:
        return ""
    left_on = bool(getattr(trial_info, "left_on", False))
    right_on = bool(getattr(trial_info, "right_on", False))
    if left_on and not right_on:
        return "left"
    if right_on and not left_on:
        return "right"
    return ""


def empty_kinematics_row(path_length: float = 0.0) -> Dict[str, float]:
    return {
        "hand_center_x": float("nan"),
        "hand_center_y": float("nan"),
        "hand_center_mm_x": float("nan"),
        "hand_center_mm_y": float("nan"),
        "path_length_px_cumulative": float(path_length),
        "path_length_mm_cumulative": float("nan"),
        "speed_px_s": float("nan"),
        "speed_px_s_smooth": float("nan"),
        "acceleration_px_s2": float("nan"),
        "acceleration_px_s2_smooth": float("nan"),
        "thumb_index_distance_px_smooth": float("nan"),
        "speed_px_s_raw": float("nan"),
        "acceleration_px_s2_raw": float("nan"),
        "speed_mm_s": float("nan"),
        "speed_mm_s_smooth": float("nan"),
        "acceleration_mm_s2": float("nan"),
        "acceleration_mm_s2_smooth": float("nan"),
        "thumb_index_distance_mm_smooth": float("nan"),
        "speed_mm_s_raw": float("nan"),
        "acceleration_mm_s2_raw": float("nan"),
    }


def landmark_kinematics_columns(prefix: str, kin: Dict[str, float]) -> Dict[str, float]:
    return {
        f"{prefix}_x_smooth": kin["hand_center_x"],
        f"{prefix}_y_smooth": kin["hand_center_y"],
        f"{prefix}_mm_x_smooth": kin["hand_center_mm_x"],
        f"{prefix}_mm_y_smooth": kin["hand_center_mm_y"],
        f"{prefix}_path_px_cumulative": kin["path_length_px_cumulative"],
        f"{prefix}_path_mm_cumulative": kin["path_length_mm_cumulative"],
        f"{prefix}_speed_px_s": kin["speed_px_s"],
        f"{prefix}_speed_px_s_smooth": kin["speed_px_s_smooth"],
        f"{prefix}_speed_px_s_raw": kin["speed_px_s_raw"],
        f"{prefix}_speed_mm_s": kin["speed_mm_s"],
        f"{prefix}_speed_mm_s_smooth": kin["speed_mm_s_smooth"],
        f"{prefix}_speed_mm_s_raw": kin["speed_mm_s_raw"],
        f"{prefix}_acceleration_px_s2": kin["acceleration_px_s2"],
        f"{prefix}_acceleration_px_s2_smooth": kin["acceleration_px_s2_smooth"],
        f"{prefix}_acceleration_px_s2_raw": kin["acceleration_px_s2_raw"],
        f"{prefix}_acceleration_mm_s2": kin["acceleration_mm_s2"],
        f"{prefix}_acceleration_mm_s2_smooth": kin["acceleration_mm_s2_smooth"],
        f"{prefix}_acceleration_mm_s2_raw": kin["acceleration_mm_s2_raw"],
    }


def landmark_overlay_metrics(thumb_kin: Dict[str, float], index_kin: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        "thumb": {
            "x_mm": thumb_kin["hand_center_mm_x"],
            "y_mm": thumb_kin["hand_center_mm_y"],
            "path_mm": thumb_kin["path_length_mm_cumulative"],
            "speed_mm_s": thumb_kin["speed_mm_s_smooth"],
            "accel_mm_s2": thumb_kin["acceleration_mm_s2_smooth"],
        },
        "index": {
            "x_mm": index_kin["hand_center_mm_x"],
            "y_mm": index_kin["hand_center_mm_y"],
            "path_mm": index_kin["path_length_mm_cumulative"],
            "speed_mm_s": index_kin["speed_mm_s_smooth"],
            "accel_mm_s2": index_kin["acceleration_mm_s2_smooth"],
        },
    }


def build_row(
    frame_idx: int,
    time_s: float,
    measurement_started: bool,
    measurement_running: bool,
    measurement_completed: bool,
    measurement_end_frame: int,
    measurement_start_source: str,
    measurement_target_side: str,
    fallback_start_frame: int,
    measurement_time_s: float,
    observation,
    kin: Dict[str, float],
    calibration: BoardCalibration,
    trial_info,
    hand_field_zone: str,
    hand_field_counts: Dict[str, int],
    peg_info,
    metric_homography: Optional[np.ndarray],
    metric_homography_source: str,
    thumb_kin: Dict[str, float],
    index_kin: Dict[str, float],
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
    wrist_metric = image_to_metric(metric_homography, wrist)
    thumb_metric = image_to_metric(metric_homography, thumb)
    index_metric = image_to_metric(metric_homography, index)
    thumb_index_metric_distance = distance(thumb_metric, index_metric) if finite_point(thumb_metric) and finite_point(index_metric) else float("nan")

    row = {
        "frame_idx": int(frame_idx),
        "time_s": float(time_s),
        "measurement_started": int(measurement_started),
        "measurement_running": int(measurement_running),
        "measurement_completed": int(measurement_completed),
        "measurement_end_frame": int(measurement_end_frame) if measurement_end_frame >= 0 else np.nan,
        "measurement_start_source": measurement_start_source,
        "measurement_target_side": measurement_target_side,
        "fallback_start_frame": int(fallback_start_frame) if fallback_start_frame >= 0 else np.nan,
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
        "hand_center_mm_x": kin["hand_center_mm_x"],
        "hand_center_mm_y": kin["hand_center_mm_y"],
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
        "metric_homography_source": metric_homography_source,
        "thumb_index_distance_mm": thumb_index_metric_distance,
        "thumb_index_distance_mm_smooth": kin["thumb_index_distance_mm_smooth"],
        "path_length_mm_cumulative": kin["path_length_mm_cumulative"],
        "speed_mm_s": kin["speed_mm_s"],
        "speed_mm_s_smooth": kin["speed_mm_s_smooth"],
        "acceleration_mm_s2": kin["acceleration_mm_s2"],
        "acceleration_mm_s2_smooth": kin["acceleration_mm_s2_smooth"],
        "speed_mm_s_raw": kin["speed_mm_s_raw"],
        "acceleration_mm_s2_raw": kin["acceleration_mm_s2_raw"],
        "wrist_mm_x": wrist_metric[0],
        "wrist_mm_y": wrist_metric[1],
        "thumb_tip_mm_x": thumb_metric[0],
        "thumb_tip_mm_y": thumb_metric[1],
        "index_tip_mm_x": index_metric[0],
        "index_tip_mm_y": index_metric[1],
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
    row.update(landmark_kinematics_columns("thumb_tip", thumb_kin))
    row.update(landmark_kinematics_columns("index_tip", index_kin))
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
    speed = pd.to_numeric(df.get("speed_mm_s_smooth", pd.Series(dtype=float)), errors="coerce").dropna()
    accel = pd.to_numeric(df.get("acceleration_mm_s2_smooth", pd.Series(dtype=float)), errors="coerce").dropna()
    pinch = pd.to_numeric(df.get("thumb_index_distance_mm_smooth", pd.Series(dtype=float)), errors="coerce").dropna()
    total_path = float(pd.to_numeric(df.get("path_length_mm_cumulative", pd.Series([0.0])), errors="coerce").dropna().max()) if processed else 0.0
    metric_source = ""
    duration_s = float(processed / fps) if fps > 0 else 0.0
    trial_started = bool("trial_started" in df and pd.to_numeric(df["trial_started"], errors="coerce").fillna(0).max() > 0)
    trial_side = ""
    trial_start_frame = np.nan
    measurement_completed = bool("measurement_completed" in df and pd.to_numeric(df["measurement_completed"], errors="coerce").fillna(0).max() > 0)
    measurement_end_frame = np.nan
    measurement_elapsed_s = np.nan
    measurement_start_source = ""
    measurement_target_side = ""
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
    if "measurement_start_source" in df and processed:
        source_values = df["measurement_start_source"].astype(str)
        non_empty = source_values[source_values != ""]
        measurement_start_source = str(non_empty.iloc[-1]) if not non_empty.empty else ""
    if "measurement_target_side" in df and processed:
        side_values = df["measurement_target_side"].astype(str)
        non_empty = side_values[side_values != ""]
        measurement_target_side = str(non_empty.iloc[-1]) if not non_empty.empty else ""
    if "metric_homography_source" in df and processed:
        metric_values = df["metric_homography_source"].astype(str)
        non_empty = metric_values[metric_values != ""]
        metric_source = str(non_empty.iloc[-1]) if not non_empty.empty else ""

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
    print(f"  metric_homography_source: {metric_source}")
    print(f"  trial_started: {trial_started}")
    print(f"  trial_side: {trial_side}")
    print(f"  trial_start_frame: {trial_start_frame}")
    print(f"  measurement_start_source: {measurement_start_source}")
    print(f"  measurement_target_side: {measurement_target_side}")
    print(f"  measurement_completed: {measurement_completed}")
    print(f"  measurement_end_frame: {measurement_end_frame}")
    print(f"  measurement_elapsed_s: {measurement_elapsed_s}")
    print(f"  total_path_length_mm: {total_path:.3f}")
    print(f"  mean_speed_mm_s: {float(speed.mean()) if not speed.empty else np.nan:.3f}")
    print(f"  max_speed_mm_s: {float(speed.max()) if not speed.empty else np.nan:.3f}")
    print(f"  mean_acceleration_mm_s2: {float(accel.mean()) if not accel.empty else np.nan:.3f}")
    print(f"  max_acceleration_mm_s2: {float(accel.max()) if not accel.empty else np.nan:.3f}")
    print(f"  mean_thumb_index_distance_mm: {float(pinch.mean()) if not pinch.empty else np.nan:.3f}")


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
    calibration_frames = first_processed_frames(
        cap,
        args.rotate_clockwise,
        args.calibration_scan_frames,
        args.calibration_scan_step,
    )
    first_frame = calibration_frames[0][1]
    frame_h, frame_w = first_frame.shape[:2]

    if args.calibration_input:
        calibration = load_calibration(Path(args.calibration_input))
        if calibration.frame_width != frame_w or calibration.frame_height != frame_h:
            print(
                "Warning: calibration dimensions do not match processed video "
                f"({calibration.frame_width}x{calibration.frame_height} vs {frame_w}x{frame_h})."
            )
    else:
        calibration = calibrate_best_frame(calibration_frames, video=str(input_path), allow_manual=True)
        print(
            "Calibration selected "
            f"source={calibration.source}, frame={calibration.calibration_frame_idx}, "
            f"holes={len(calibration.holes)}, candidates={calibration.hole_candidate_count}"
        )
        if calibration_output_path is not None:
            save_calibration(calibration, calibration_output_path)
    start_activation_roi = activation_roi_from_calibration(calibration, args.start_gate_padding)
    tracking_roi = padded_board_roi(calibration, args.tracking_roi_padding)
    field_regions = calibration.hole_grid_regions(padding_scale=args.field_roi_padding)
    metric_homography, metric_homography_source = build_metric_homography(calibration, args.hole_spacing_mm)
    peg_detector = (
        PegOccupancyDetector(
            calibration=calibration,
            stable_frames=args.peg_stable_frames,
            post_cover_clear_frames=args.peg_post_cover_clear_frames,
            change_threshold=args.peg_change_threshold,
        )
        if args.peg_detection == "on"
        else None
    )

    writer = open_video_writer(output_path, fps, (frame_w + 2 * PANEL_WIDTH, frame_h))
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
    thumb_kinematics = KinematicsTracker(args.smooth_alpha, args.smooth_window)
    index_kinematics = KinematicsTracker(args.smooth_alpha, args.smooth_window)
    trail: Deque[Tuple[int, int]] = deque(maxlen=max(1, int(args.trail_length)))
    history: Dict[str, Deque[float]] = {
        "path": deque(maxlen=360),
        "speed": deque(maxlen=360),
        "acceleration": deque(maxlen=360),
        "thumb_index": deque(maxlen=360),
    }
    finger_history: Dict[str, Deque[float]] = {
        "thumb_path": deque(maxlen=360),
        "thumb_speed": deque(maxlen=360),
        "thumb_acceleration": deque(maxlen=360),
        "index_path": deque(maxlen=360),
        "index_speed": deque(maxlen=360),
        "index_acceleration": deque(maxlen=360),
    }
    rows: List[Dict[str, object]] = []
    measurement_started = bool(args.measure_from == "immediate" or trial_detector is None)
    measurement_completed = False
    measurement_start_frame = 0 if measurement_started else -1
    measurement_end_frame = -1
    measurement_start_source = "immediate" if measurement_started else ""
    measurement_target_side = ""
    fallback_start_frame = -1
    fallback_side_candidate = ""
    fallback_side_count = 0
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
            light_target_side = str(getattr(trial_info, "trial_side", "")) if trial_info is not None else ""
            fallback_side = ""
            if not measurement_started and args.hand_field_start_fallback == "on":
                hand_side = dominant_hand_field_side(hand_field_zone, hand_field_counts)
                single_light_side = lit_single_side(trial_info)
                light_sequence_not_seen = bool(
                    trial_info is not None
                    and not bool(getattr(trial_info, "both_seen", False))
                    and not bool(getattr(trial_info, "off_seen", False))
                )
                if light_sequence_not_seen and single_light_side and int(hand_field_counts.get(single_light_side, 0)) > 0:
                    fallback_side = single_light_side
                elif trial_info is None:
                    fallback_side = hand_side
                if fallback_side:
                    if fallback_side != fallback_side_candidate:
                        fallback_side_candidate = fallback_side
                        fallback_side_count = 1
                    else:
                        fallback_side_count += 1
                else:
                    fallback_side_candidate = ""
                    fallback_side_count = 0

            should_start_from_light = bool(not measurement_started and trial_info is not None and bool(getattr(trial_info, "trial_started", False)))
            should_start_from_hand = bool(
                not measurement_started
                and args.hand_field_start_fallback == "on"
                and fallback_side_candidate in ("left", "right")
                and fallback_side_count >= max(1, int(args.hand_field_start_frames))
            )
            if should_start_from_light or should_start_from_hand:
                measurement_started = True
                detected_start_frame = int(getattr(trial_info, "trial_start_frame", -1)) if should_start_from_light else -1
                measurement_start_frame = detected_start_frame if detected_start_frame >= 0 else int(frame_idx)
                measurement_end_frame = -1
                measurement_completed = False
                measurement_start_source = "light" if should_start_from_light else "hand_field"
                measurement_target_side = light_target_side if should_start_from_light and light_target_side in ("left", "right") else fallback_side_candidate
                fallback_start_frame = int(frame_idx) if measurement_start_source == "hand_field" else -1
                max_target_peg_count = 0
                empty_target_frames = 0
                kinematics.reset()
                thumb_kinematics.reset()
                index_kinematics.reset()
                trail.clear()
                for values in history.values():
                    values.clear()
                for values in finger_history.values():
                    values.clear()

            measurement_running = bool(measurement_started and not measurement_completed)
            measurement_time_s = (
                max(0.0, (float(measurement_end_frame if measurement_completed else frame_idx) - float(measurement_start_frame)) / fps)
                if measurement_started and measurement_start_frame >= 0
                else float("nan")
            )
            metric_center = image_to_metric(metric_homography, observation.center_raw) if observation.detected else (float("nan"), float("nan"))
            metric_thumb = image_to_metric(metric_homography, observation.thumb_tip) if observation.detected else (float("nan"), float("nan"))
            metric_index = image_to_metric(metric_homography, observation.index_tip) if observation.detected else (float("nan"), float("nan"))
            thumb_index_distance_mm = distance(metric_thumb, metric_index) if finite_point(metric_thumb) and finite_point(metric_index) else float("nan")
            if measurement_running:
                kin = kinematics.update(
                    measurement_time_s,
                    observation.detected,
                    observation.center_raw,
                    observation.thumb_index_distance_px,
                    metric_center=metric_center,
                    thumb_index_distance_mm=thumb_index_distance_mm,
                )
                thumb_kin = thumb_kinematics.update(
                    measurement_time_s,
                    bool(observation.detected and finite_point(observation.thumb_tip)),
                    observation.thumb_tip,
                    float("nan"),
                    metric_center=metric_thumb,
                    thumb_index_distance_mm=float("nan"),
                )
                index_kin = index_kinematics.update(
                    measurement_time_s,
                    bool(observation.detected and finite_point(observation.index_tip)),
                    observation.index_tip,
                    float("nan"),
                    metric_center=metric_index,
                    thumb_index_distance_mm=float("nan"),
                )
            else:
                kin = empty_kinematics_row()
                thumb_kin = empty_kinematics_row()
                index_kin = empty_kinematics_row()
            peg_info = None
            if peg_detector is not None:
                target_side = measurement_target_side if measurement_target_side in ("left", "right") else light_target_side
                peg_info = peg_detector.update(
                    frame,
                    covered_sides=covered_sides_from_hand_field(hand_field_zone),
                    target_side=target_side,
                    measurement_active=measurement_started,
                    reference_bootstrap=bool(measurement_started and measurement_start_source != "light"),
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
                history["path"].append(float(kin["path_length_mm_cumulative"]))
                history["speed"].append(float(kin["speed_mm_s_smooth"]))
                history["acceleration"].append(float(kin["acceleration_mm_s2_smooth"]))
                history["thumb_index"].append(float(kin["thumb_index_distance_mm_smooth"]))
                finger_history["thumb_path"].append(float(thumb_kin["path_length_mm_cumulative"]))
                finger_history["thumb_speed"].append(float(thumb_kin["speed_mm_s_smooth"]))
                finger_history["thumb_acceleration"].append(float(thumb_kin["acceleration_mm_s2_smooth"]))
                finger_history["index_path"].append(float(index_kin["path_length_mm_cumulative"]))
                finger_history["index_speed"].append(float(index_kin["speed_mm_s_smooth"]))
                finger_history["index_acceleration"].append(float(index_kin["acceleration_mm_s2_smooth"]))

            rows.append(
                build_row(
                    frame_idx,
                    time_s,
                    measurement_started,
                    measurement_running,
                    measurement_completed,
                    measurement_end_frame,
                    measurement_start_source,
                    measurement_target_side,
                    fallback_start_frame,
                    measurement_time_s,
                    observation,
                    kin,
                    calibration,
                    trial_info,
                    hand_field_zone,
                    hand_field_counts,
                    peg_info,
                    metric_homography,
                    metric_homography_source,
                    thumb_kin,
                    index_kin,
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
                finger_history,
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
                video_name=input_path.name,
                landmark_metrics=landmark_overlay_metrics(thumb_kin, index_kin) if measurement_started else None,
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
