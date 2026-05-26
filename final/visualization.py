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
        label = "WAITING FOR HAND IN START ZONE" if observation.waiting_for_start else "NO HAND DETECTED"
        draw_text(image, label, (15, 58), 0.62, (0, 190, 255))
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

    for idx, label in [(WRIST, "0"), (THUMB_TIP, "4"), (INDEX_TIP, "8")]:
        point = pts[idx]
        cv2.putText(image, label, (int(point[0]) + 4, int(point[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (245, 245, 245), 1, cv2.LINE_AA)

    if smoothed_center is not None and np.all(np.isfinite(smoothed_center)):
        cx, cy = int(round(smoothed_center[0])), int(round(smoothed_center[1]))
        cv2.drawMarker(image, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1, line_type=cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 5, (0, 255, 255), 1, cv2.LINE_AA)

    draw_text(
        image,
        f"hand={observation.active_hand_label} score={observation.hand_score:.2f} motion={observation.motion_score:.1f}",
        (15, 58),
        0.48,
        (235, 235, 235),
    )


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
        ("d(t) path px", "path", (70, 210, 255)),
        ("v(t) px/s", "speed", (80, 220, 160)),
        ("a(t) px/s2", "acceleration", (255, 180, 80)),
        ("thumb-index px", "thumb_index", (220, 130, 255)),
    ]
    for idx, (title, key, color) in enumerate(specs):
        y = y0 + idx * (graph_h + 28)
        if y + graph_h + 4 > panel.shape[0]:
            break
        draw_single_graph(panel, 16, y, graph_w, graph_h, title, list(history.get(key, [])), smooth_window, color)
    cv2.putText(panel, f"fps={fps:.1f}  smooth={smooth_window}", (16, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 185), 1, cv2.LINE_AA)


def compose_frame(
    frame: np.ndarray,
    calibration: BoardCalibration,
    observation: HandObservation,
    smoothed_center: Optional[Tuple[float, float]],
    trail: Sequence[Tuple[int, int]],
    history: Dict[str, Deque[float]],
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
) -> np.ndarray:
    annotated = frame.copy()
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
    panel = np.zeros((annotated.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
    draw_timeseries_panel(panel, history, fps, smooth_window)
    return cv2.hconcat([annotated, panel])
