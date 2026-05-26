from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from calibration import BoardCalibration, draw_calibration
from hand_tracking import HAND_CONNECTIONS, HandObservation, INDEX_TIP, THUMB_TIP, WRIST
from utils import moving_average_nan


PANEL_WIDTH = 390
ActivationRegion = Union[Tuple[int, int, int, int], np.ndarray]


def draw_text(image: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: Tuple[int, int, int]) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hand(image: np.ndarray, observation: HandObservation, smoothed_center: Optional[Tuple[float, float]]) -> None:
    if not observation.detected or observation.landmarks_px is None:
        return

    pts = observation.landmarks_px
    for a, b in HAND_CONNECTIONS:
        pa = (int(round(pts[a, 0])), int(round(pts[a, 1])))
        pb = (int(round(pts[b, 0])), int(round(pts[b, 1])))
        cv2.line(image, pa, pb, (60, 220, 120), 1, cv2.LINE_AA)
    for idx, point in enumerate(pts):
        color = (255, 255, 255)
        radius = 2
        if idx == WRIST:
            color = (255, 180, 60)
            radius = 3
        elif idx == THUMB_TIP:
            color = (80, 220, 255)
            radius = 3
        elif idx == INDEX_TIP:
            color = (255, 100, 220)
            radius = 3
        cv2.circle(image, (int(round(point[0])), int(round(point[1]))), radius, color, -1, cv2.LINE_AA)

    if smoothed_center is not None and np.all(np.isfinite(smoothed_center)):
        cx, cy = int(round(smoothed_center[0])), int(round(smoothed_center[1]))
        cv2.drawMarker(image, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1, line_type=cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_activation_roi(image: np.ndarray, activation_roi: Optional[ActivationRegion]) -> None:
    if activation_roi is None:
        return
    if isinstance(activation_roi, np.ndarray):
        pts = np.round(np.asarray(activation_roi, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
        if pts.shape[0] < 3:
            return
        cv2.polylines(image, [pts], True, (255, 170, 60), 1, cv2.LINE_AA)
        x = int(np.min(pts[:, 0]))
        y = int(np.min(pts[:, 1]))
        cv2.putText(image, "START ZONE", (x + 6, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 170, 60), 1, cv2.LINE_AA)
        return
    x, y, w, h = activation_roi
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 170, 60), 1, cv2.LINE_AA)
    cv2.putText(image, "START ZONE", (x + 6, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 170, 60), 1, cv2.LINE_AA)


def draw_trial_status(image: np.ndarray, trial_info) -> None:
    if trial_info is None:
        return
    started = bool(getattr(trial_info, "trial_started", False))
    trial_time = float(getattr(trial_info, "trial_time_s", np.nan))
    state = str(getattr(trial_info, "state", ""))
    if started and np.isfinite(trial_time):
        text = "STARTED"
        color = (80, 255, 130)
    else:
        text = f"WAIT LIGHT START  {state}"
        color = (0, 220, 255)
    draw_text(image, text, (15, 88), 0.52, color)


def draw_measurement_timer(
    image: np.ndarray,
    measurement_started: bool,
    measurement_completed: bool,
    measurement_time_s: float,
    y: int = 88,
) -> None:
    if not measurement_started or not np.isfinite(measurement_time_s):
        return
    color = (90, 190, 255) if measurement_completed else (80, 255, 130)
    minutes = int(measurement_time_s // 60.0)
    seconds = measurement_time_s - 60.0 * float(minutes)
    draw_text(image, f"{minutes:02d}:{seconds:05.2f}", (15, y), 0.72, color)


def draw_video_name(image: np.ndarray, video_name: str) -> None:
    if not video_name:
        return
    max_chars = max(18, int(image.shape[1] / 13))
    text = video_name if len(video_name) <= max_chars else "..." + video_name[-(max_chars - 3):]
    draw_text(image, text, (15, 28), 0.52, (245, 245, 245))


def metric_value(value: float, digits: int = 1) -> str:
    return "nan" if not np.isfinite(value) else f"{float(value):.{digits}f}"


def draw_landmark_metrics(image: np.ndarray, landmark_metrics: Optional[Dict[str, Dict[str, float]]]) -> None:
    if not landmark_metrics:
        return
    specs = [
        ("PAL", "thumb", (80, 220, 255)),
        ("KAZ", "index", (255, 110, 220)),
    ]
    y = 128
    for label, key, color in specs:
        values = landmark_metrics.get(key, {})
        x_mm = metric_value(float(values.get("x_mm", float("nan"))))
        y_mm = metric_value(float(values.get("y_mm", float("nan"))))
        path_mm = metric_value(float(values.get("path_mm", float("nan"))))
        speed_mm_s = metric_value(float(values.get("speed_mm_s", float("nan"))))
        accel_mm_s2 = metric_value(float(values.get("accel_mm_s2", float("nan"))))
        draw_text(image, f"{label} pos {x_mm},{y_mm} mm", (15, y), 0.40, color)
        draw_text(image, f"{label} pot {path_mm} mm", (15, y + 18), 0.40, color)
        draw_text(image, f"{label} v {speed_mm_s} mm/s", (15, y + 36), 0.40, color)
        draw_text(image, f"{label} a {accel_mm_s2} mm/s2", (15, y + 54), 0.40, color)
        y += 80


def draw_multi_graph(
    panel: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    series: Sequence[Tuple[str, Sequence[float], Tuple[int, int, int]]],
    smooth_window: int,
) -> None:
    cv2.rectangle(panel, (x, y), (x + w, y + h), (52, 56, 62), 1)
    cv2.putText(panel, title, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    prepared = []
    finite_parts = []
    for label, values, color in series:
        arr = np.asarray(list(values)[-w:], dtype=np.float32)
        smooth = moving_average_nan(arr, smooth_window)
        prepared.append((label, smooth, color))
        finite = smooth[np.isfinite(smooth)]
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        return
    finite_all = np.concatenate(finite_parts)
    if finite_all.size < 2:
        return
    v_min = float(np.min(finite_all))
    v_max = float(np.max(finite_all))
    if abs(v_max - v_min) < 1e-6:
        v_max = v_min + 1.0
    for label, smooth, color in prepared:
        points = []
        for idx, value in enumerate(smooth):
            if not np.isfinite(value):
                points.append(None)
                continue
            px = x + int(round(idx * (w - 1) / max(1, len(smooth) - 1)))
            py = y + h - 4 - int(round((float(value) - v_min) / (v_max - v_min) * (h - 8)))
            points.append((px, py))
        for p1, p2 in zip(points[:-1], points[1:]):
            if p1 is not None and p2 is not None:
                cv2.line(panel, p1, p2, color, 1, cv2.LINE_AA)


def draw_field_regions(image: np.ndarray, field_regions: Sequence[Dict], hand_field_zone: str) -> None:
    if not field_regions:
        return
    for region in field_regions:
        side = str(region.get("side", ""))
        polygon = np.asarray(region.get("polygon", []), dtype=np.float32).reshape(-1, 2)
        if polygon.shape[0] < 3:
            continue
        pts = np.round(polygon).astype(np.int32)
        active = side == hand_field_zone or hand_field_zone == "both"
        color = (0, 255, 255) if active else (180, 180, 180)
        thickness = 2 if active else 1
        cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA)
        label = "LEVO" if side == "left" else "DESNO" if side == "right" else side.upper()
        anchor = tuple(np.round(np.mean(polygon, axis=0)).astype(int))
        cv2.putText(image, label, (anchor[0] - 18, anchor[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


def draw_trail(image: np.ndarray, trail: Sequence[Tuple[int, int]]) -> None:
    if len(trail) < 2:
        return
    for i, (a, b) in enumerate(zip(trail[:-1], trail[1:])):
        alpha = (i + 1) / max(1, len(trail) - 1)
        color = (int(40 + 120 * alpha), int(120 + 120 * alpha), 255)
        cv2.line(image, a, b, color, 1, cv2.LINE_AA)


def draw_single_graph(
    panel: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    values: Sequence[float],
    smooth_window: int,
    color: Tuple[int, int, int],
) -> None:
    cv2.rectangle(panel, (x, y), (x + w, y + h), (52, 56, 62), 1)
    arr = np.asarray(values[-w:], dtype=np.float32)
    smooth = moving_average_nan(arr, smooth_window)
    finite = smooth[np.isfinite(smooth)]
    current = float(finite[-1]) if finite.size else float("nan")
    label_value = "nan" if not np.isfinite(current) else f"{current:.1f}"
    cv2.putText(panel, f"{title}: {label_value}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    if finite.size < 2:
        return
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    if abs(v_max - v_min) < 1e-6:
        v_max = v_min + 1.0
    points = []
    for idx, value in enumerate(smooth):
        if not np.isfinite(value):
            points.append(None)
            continue
        px = x + int(round(idx * (w - 1) / max(1, len(smooth) - 1)))
        py = y + h - 4 - int(round((float(value) - v_min) / (v_max - v_min) * (h - 8)))
        points.append((px, py))
    for p1, p2 in zip(points[:-1], points[1:]):
        if p1 is not None and p2 is not None:
            cv2.line(panel, p1, p2, color, 1, cv2.LINE_AA)


def draw_timeseries_panel(panel: np.ndarray, history: Dict[str, Deque[float]], fps: float, smooth_window: int) -> None:
    panel[:] = (28, 31, 36)
    cv2.putText(panel, "9HPT hand kinematics", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    graph_w = panel.shape[1] - 32
    graph_h = max(54, (panel.shape[0] - 92) // 4)
    y0 = 64
    specs = [
        ("d(t) path mm", "path", (70, 210, 255)),
        ("v(t) mm/s", "speed", (80, 220, 160)),
        ("a(t) mm/s2", "acceleration", (255, 180, 80)),
        ("thumb-index mm", "thumb_index", (220, 130, 255)),
    ]
    for idx, (title, key, color) in enumerate(specs):
        y = y0 + idx * (graph_h + 28)
        if y + graph_h + 4 > panel.shape[0]:
            break
        draw_single_graph(panel, 16, y, graph_w, graph_h, title, list(history.get(key, [])), smooth_window, color)
    cv2.putText(panel, f"fps={fps:.1f}  smooth={smooth_window}", (16, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 185), 1, cv2.LINE_AA)


def latest(history: Dict[str, Deque[float]], key: str) -> float:
    values = history.get(key)
    if not values:
        return float("nan")
    for value in reversed(values):
        if np.isfinite(value):
            return float(value)
    return float("nan")


def draw_finger_panel(
    panel: np.ndarray,
    finger_history: Dict[str, Deque[float]],
    landmark_metrics: Optional[Dict[str, Dict[str, float]]],
    fps: float,
    smooth_window: int,
) -> None:
    panel[:] = (28, 31, 36)
    cv2.putText(panel, "thumb / index kinematics", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)

    thumb_color = (80, 220, 255)
    index_color = (255, 110, 220)
    cv2.line(panel, (18, 50), (54, 50), thumb_color, 2, cv2.LINE_AA)
    cv2.putText(panel, "PAL", (62, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, thumb_color, 1, cv2.LINE_AA)
    cv2.line(panel, (126, 50), (162, 50), index_color, 2, cv2.LINE_AA)
    cv2.putText(panel, "KAZ", (170, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, index_color, 1, cv2.LINE_AA)

    thumb_values = (landmark_metrics or {}).get("thumb", {})
    index_values = (landmark_metrics or {}).get("index", {})
    rows = [
        ("PAL d", float(thumb_values.get("path_mm", latest(finger_history, "thumb_path"))), "mm", thumb_color),
        ("PAL v", float(thumb_values.get("speed_mm_s", latest(finger_history, "thumb_speed"))), "mm/s", thumb_color),
        ("PAL a", float(thumb_values.get("accel_mm_s2", latest(finger_history, "thumb_acceleration"))), "mm/s2", thumb_color),
        ("KAZ d", float(index_values.get("path_mm", latest(finger_history, "index_path"))), "mm", index_color),
        ("KAZ v", float(index_values.get("speed_mm_s", latest(finger_history, "index_speed"))), "mm/s", index_color),
        ("KAZ a", float(index_values.get("accel_mm_s2", latest(finger_history, "index_acceleration"))), "mm/s2", index_color),
    ]
    y = 82
    for idx, (label, value, unit, color) in enumerate(rows):
        col_x = 16 if idx < 3 else 196
        row_y = y + 20 * (idx % 3)
        text = f"{label}: {metric_value(value)} {unit}"
        cv2.putText(panel, text, (col_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    graph_w = panel.shape[1] - 32
    graph_h = max(46, min(100, (panel.shape[0] - 188) // 3))
    graph_gap = 28
    y0 = 150
    specs = [
        (
            "d(t) path mm",
            "thumb_path",
            "index_path",
        ),
        (
            "v(t) mm/s",
            "thumb_speed",
            "index_speed",
        ),
        (
            "a(t) mm/s2",
            "thumb_acceleration",
            "index_acceleration",
        ),
    ]
    for idx, (title, thumb_key, index_key) in enumerate(specs):
        graph_y = y0 + idx * (graph_h + graph_gap)
        if graph_y + graph_h + 4 > panel.shape[0]:
            break
        draw_multi_graph(
            panel,
            16,
            graph_y,
            graph_w,
            graph_h,
            title,
            [
                ("PAL", list(finger_history.get(thumb_key, [])), thumb_color),
                ("KAZ", list(finger_history.get(index_key, [])), index_color),
            ],
            smooth_window,
        )
    cv2.putText(panel, f"fps={fps:.1f}  smooth={smooth_window}", (16, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 185), 1, cv2.LINE_AA)


def compose_frame(
    frame: np.ndarray,
    calibration: BoardCalibration,
    observation: HandObservation,
    smoothed_center: Optional[Tuple[float, float]],
    trail: Sequence[Tuple[int, int]],
    history: Dict[str, Deque[float]],
    finger_history: Dict[str, Deque[float]],
    fps: float,
    smooth_window: int,
    activation_roi: Optional[ActivationRegion] = None,
    trial_info=None,
    measurement_started: bool = False,
    measurement_completed: bool = False,
    measurement_time_s: float = float("nan"),
    show_trial_status: bool = False,
    field_regions: Optional[List[Dict]] = None,
    hand_field_zone: str = "",
    peg_detector=None,
    peg_info=None,
    video_name: str = "",
    landmark_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    annotated = frame.copy()
    draw_video_name(annotated, video_name)
    draw_calibration(annotated, calibration)
    draw_activation_roi(annotated, activation_roi)
    if show_trial_status:
        draw_trial_status(annotated, trial_info)
    draw_measurement_timer(annotated, measurement_started, measurement_completed, measurement_time_s, y=118 if show_trial_status else 88)
    draw_field_regions(annotated, field_regions or [], hand_field_zone)
    if peg_detector is not None and peg_info is not None and getattr(peg_info, "measurement_active", False):
        peg_detector.draw(annotated, peg_info)
    draw_trail(annotated, trail)
    draw_hand(annotated, observation, smoothed_center)
    left_panel = np.zeros((annotated.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
    right_panel = np.zeros((annotated.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
    draw_finger_panel(left_panel, finger_history, landmark_metrics, fps, smooth_window)
    draw_timeseries_panel(right_panel, history, fps, smooth_window)
    return cv2.hconcat([left_panel, annotated, right_panel])
