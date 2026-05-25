from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration import BoardCalibration, draw_calibration
from hand_tracking import HAND_CONNECTIONS, HandObservation, INDEX_TIP, THUMB_TIP, WRIST
from utils import moving_average_nan


PANEL_WIDTH = 390


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


def draw_activation_roi(image: np.ndarray, activation_roi: Optional[Tuple[int, int, int, int]]) -> None:
    if activation_roi is None:
        return
    x, y, w, h = activation_roi
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 170, 60), 1, cv2.LINE_AA)
    cv2.putText(image, "START ZONE", (x + 6, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 170, 60), 1, cv2.LINE_AA)


def draw_trial_status(image: np.ndarray, trial_info) -> None:
    if trial_info is None:
        return
    started = bool(getattr(trial_info, "trial_started", False))
    side = str(getattr(trial_info, "trial_side", ""))
    trial_time = float(getattr(trial_info, "trial_time_s", np.nan))
    state = str(getattr(trial_info, "state", ""))
    if started and np.isfinite(trial_time):
        text = f"TRIAL {trial_time:5.2f}s"
        if side:
            text += f"  side={side}"
        color = (80, 255, 130)
    else:
        text = f"WAIT LIGHT START  {state}"
        color = (0, 220, 255)
    draw_text(image, text, (15, 88), 0.52, color)


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
    activation_roi: Optional[Tuple[int, int, int, int]] = None,
    trial_info=None,
) -> np.ndarray:
    annotated = frame.copy()
    draw_calibration(annotated, calibration)
    draw_activation_roi(annotated, activation_roi)
    draw_trial_status(annotated, trial_info)
    draw_trail(annotated, trail)
    draw_hand(annotated, observation, smoothed_center)
    panel = np.zeros((annotated.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
    draw_timeseries_panel(panel, history, fps, smooth_window)
    return cv2.hconcat([annotated, panel])
