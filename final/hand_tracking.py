from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import mediapipe as mp
import numpy as np

from utils import Point, distance


WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
ActivationRegion = Union[Tuple[int, int, int, int], np.ndarray]
DEFAULT_HAND_LANDMARKER_MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class HandObservation:
    detected: bool
    active_hand_label: str
    hand_score: float
    landmarks_px: Optional[np.ndarray]
    center_raw: Optional[Point]
    wrist: Optional[Point]
    thumb_tip: Optional[Point]
    index_tip: Optional[Point]
    thumb_index_distance_px: float
    missing_frames: int
    track_started: bool
    waiting_for_start: bool
    motion_score: float
    tracking_roi_inside: bool
    tracking_roi_outside_frames: int


class _LandmarkListAdapter:
    def __init__(self, landmarks) -> None:
        self.landmark = landmarks


class _HandednessAdapter:
    def __init__(self, categories) -> None:
        self.classification = categories


class _HandResultAdapter:
    def __init__(self, task_result) -> None:
        self.multi_hand_landmarks = [_LandmarkListAdapter(item) for item in getattr(task_result, "hand_landmarks", [])]
        self.multi_handedness = [_HandednessAdapter(item) for item in getattr(task_result, "handedness", [])]


def empty_observation(missing_frames: int, track_started: bool = False, waiting_for_start: bool = False) -> HandObservation:
    return HandObservation(
        detected=False,
        active_hand_label="",
        hand_score=float("nan"),
        landmarks_px=None,
        center_raw=None,
        wrist=None,
        thumb_tip=None,
        index_tip=None,
        thumb_index_distance_px=float("nan"),
        missing_frames=missing_frames,
        track_started=track_started,
        waiting_for_start=waiting_for_start,
        motion_score=float("nan"),
        tracking_roi_inside=False,
        tracking_roi_outside_frames=0,
    )


def landmarks_to_pixels(hand_landmarks, width: int, height: int) -> np.ndarray:
    points = []
    for lm in hand_landmarks.landmark:
        z = float(getattr(lm, "z", 0.0))
        points.append(
            [
                float(np.clip(lm.x * width, 0, width - 1)),
                float(np.clip(lm.y * height, 0, height - 1)),
                z,
            ]
        )
    return np.asarray(points, dtype=np.float32)


def center_from_landmarks(points: np.ndarray) -> Point:
    xy = points[:, :2]
    return float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))


def reference_center(frame_shape: Tuple[int, int, int], board_roi: Optional[Tuple[int, int, int, int]]) -> Point:
    h, w = frame_shape[:2]
    if board_roi is not None:
        x, y, bw, bh = board_roi
        return float(x + bw / 2.0), float(y + bh / 2.0)
    return float(w / 2.0), float(h / 2.0)


def center_roi(frame_shape: Tuple[int, int, int], board_roi: Optional[Tuple[int, int, int, int]], scale: float) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    if board_roi is None:
        board_roi = (0, 0, w, h)
    x, y, bw, bh = board_roi
    scale = float(np.clip(scale, 0.1, 1.0))
    roi_w = max(1, int(round(bw * scale)))
    roi_h = max(1, int(round(bh * scale)))
    roi_x = int(round(x + 0.5 * (bw - roi_w)))
    roi_y = int(round(y + 0.5 * (bh - roi_h)))
    return roi_x, roi_y, roi_w, roi_h


def point_in_roi(point: Point, roi: ActivationRegion) -> bool:
    if isinstance(roi, np.ndarray):
        polygon = np.asarray(roi, dtype=np.float32).reshape(-1, 2)
        if polygon.shape[0] < 3:
            return False
        return bool(cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0.0)
    x, y, w, h = roi
    return bool(x <= point[0] <= x + w and y <= point[1] <= y + h)


def roi_center(roi: ActivationRegion) -> Point:
    if isinstance(roi, np.ndarray):
        polygon = np.asarray(roi, dtype=np.float32).reshape(-1, 2)
        if polygon.size == 0:
            return float("nan"), float("nan")
        center = np.mean(polygon, axis=0)
        return float(center[0]), float(center[1])
    x, y, w, h = roi
    return float(x + w / 2.0), float(y + h / 2.0)


def bbox_from_landmarks(points: np.ndarray, width: int, height: int, margin: int = 24) -> Tuple[int, int, int, int]:
    xy = points[:, :2]
    x1 = int(max(0, np.floor(np.min(xy[:, 0]) - margin)))
    y1 = int(max(0, np.floor(np.min(xy[:, 1]) - margin)))
    x2 = int(min(width - 1, np.ceil(np.max(xy[:, 0]) + margin)))
    y2 = int(min(height - 1, np.ceil(np.max(xy[:, 1]) + margin)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def motion_score_for_hand(points: np.ndarray, motion: Optional[np.ndarray], width: int, height: int) -> float:
    if motion is None:
        return 0.0
    x, y, w, h = bbox_from_landmarks(points, width, height)
    roi = motion[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    # High percentile is more useful than the mean: fingers can move while most
    # of the hand bbox is static table/background.
    return float(np.percentile(roi.astype(np.float32), 85))


def select_active_hand(
    result,
    previous_center: Optional[Point],
    frame_shape: Tuple[int, int, int],
    board_roi: Optional[Tuple[int, int, int, int]],
    activation_roi: Optional[ActivationRegion] = None,
    tracking_roi: Optional[ActivationRegion] = None,
    require_activation_roi: bool = False,
    require_tracking_roi: bool = False,
    tracking_roi_mode: str = "soft",
    tracking_roi_outside_frames: int = 0,
    max_tracking_roi_outside_frames: int = 45,
    max_tracking_roi_exit_jump_px: float = 120.0,
    max_lock_jump_px: float = 120.0,
    max_reacquire_jump_px: float = 280.0,
    lock_missing_frames: int = 0,
    motion: Optional[np.ndarray] = None,
    motion_weight: float = 0.12,
    min_start_motion_score: float = 0.0,
    min_reacquire_motion_score: float = 4.0,
) -> Optional[Tuple[object, np.ndarray, str, float, float, bool]]:
    if not result.multi_hand_landmarks:
        return None
    h, w = frame_shape[:2]
    if previous_center is not None:
        ref = previous_center
    elif activation_roi is not None:
        ref = roi_center(activation_roi)
    else:
        ref = reference_center(frame_shape, board_roi)
    handedness_list = result.multi_handedness or []
    roi_mode = str(tracking_roi_mode or "soft").lower()
    candidates: List[Tuple[float, float, object, np.ndarray, str, float, float, bool]] = []
    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
        points = landmarks_to_pixels(hand_landmarks, w, h)
        center = center_from_landmarks(points)
        if require_activation_roi and activation_roi is not None and not point_in_roi(center, activation_roi):
            continue
        inside_tracking_roi = bool(tracking_roi is None or point_in_roi(center, tracking_roi))
        label = "unknown"
        score = 0.0
        if idx < len(handedness_list) and handedness_list[idx].classification:
            cls = handedness_list[idx].classification[0]
            label_value = getattr(cls, "label", getattr(cls, "category_name", "unknown"))
            label = str(label_value).lower()
            score = float(getattr(cls, "score", 0.0))
        dist = distance(center, ref)
        if require_tracking_roi and tracking_roi is not None and not inside_tracking_roi:
            if roi_mode == "strict":
                continue
            if roi_mode == "soft" and previous_center is not None:
                if int(tracking_roi_outside_frames) >= int(max_tracking_roi_outside_frames):
                    continue
                previous_inside = point_in_roi(previous_center, tracking_roi)
                if previous_inside and np.isfinite(dist) and dist > float(max_tracking_roi_exit_jump_px):
                    continue
            elif roi_mode != "off":
                continue
        motion_score = motion_score_for_hand(points, motion, w, h)
        if require_activation_roi and motion_score < min_start_motion_score:
            continue
        if previous_center is not None and np.isfinite(dist):
            missing = max(0, int(lock_missing_frames))
            allowed_jump = max_lock_jump_px
            if missing > 0:
                allowed_jump = min(float(max_reacquire_jump_px), max_lock_jump_px + 18.0 * missing)
            if dist > allowed_jump:
                continue
            if missing > 0 and motion_score < min_reacquire_motion_score:
                continue
        # Once the active hand is locked, identity continuity is more important
        # than MediaPipe handedness confidence. This prevents jumps to the other
        # visible hand when it receives a higher score.
        if previous_center is not None:
            if lock_missing_frames > 0:
                selection_score = float(motion_score) - 0.035 * float(dist) + 0.25 * score
            else:
                selection_score = -float(dist)
        else:
            selection_score = score + motion_weight * motion_score - 0.01 * (dist if np.isfinite(dist) else 0.0)
        if roi_mode == "soft" and require_tracking_roi and tracking_roi is not None and not inside_tracking_roi:
            selection_score -= 0.08 * (dist if np.isfinite(dist) else 0.0)
        candidates.append((selection_score, float(dist) if np.isfinite(dist) else float("inf"), hand_landmarks, points, label, score, motion_score, inside_tracking_roi))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, _, hand_landmarks, points, label, score, motion_score, inside_tracking_roi = candidates[0]
    return hand_landmarks, points, label, score, motion_score, inside_tracking_roi


class MediaPipeHandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_missing_frames: int = 10,
        start_gate_enabled: bool = True,
        start_gate_scale: float = 0.75,
        activation_consecutive_frames: int = 1,
        max_lock_jump_px: float = 120.0,
        max_reacquire_jump_px: float = 280.0,
        motion_weight: float = 0.12,
        min_start_motion_score: float = 2.0,
        min_reacquire_motion_score: float = 4.0,
        reacquire_consecutive_frames: int = 2,
        tracking_roi_mode: str = "soft",
        max_tracking_roi_outside_frames: int = 45,
        max_tracking_roi_exit_jump_px: float = 120.0,
    ):
        if hasattr(mp, "solutions"):
            self.backend = "solutions"
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            if not DEFAULT_HAND_LANDMARKER_MODEL.exists():
                raise FileNotFoundError(
                    "MediaPipe Tasks requires a hand landmarker model. "
                    f"Expected: {DEFAULT_HAND_LANDMARKER_MODEL}"
                )
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            self.backend = "tasks"
            self.running_mode = vision.RunningMode.VIDEO
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(DEFAULT_HAND_LANDMARKER_MODEL)),
                running_mode=self.running_mode,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.hands = vision.HandLandmarker.create_from_options(options)
            self.timestamp_ms = 0
        self.previous_center: Optional[Point] = None
        self.missing_frames = 0
        self.max_missing_frames = int(max_missing_frames)
        self.start_gate_enabled = bool(start_gate_enabled)
        self.start_gate_scale = float(start_gate_scale)
        self.activation_consecutive_frames = int(max(1, activation_consecutive_frames))
        self.max_lock_jump_px = float(max(1.0, max_lock_jump_px))
        self.max_reacquire_jump_px = float(max(self.max_lock_jump_px, max_reacquire_jump_px))
        self.motion_weight = float(motion_weight)
        self.min_start_motion_score = float(max(0.0, min_start_motion_score))
        self.min_reacquire_motion_score = float(max(0.0, min_reacquire_motion_score))
        self.reacquire_consecutive_frames = int(max(1, reacquire_consecutive_frames))
        self.tracking_roi_mode = str(tracking_roi_mode or "soft").lower()
        if self.tracking_roi_mode not in {"strict", "soft", "off"}:
            self.tracking_roi_mode = "soft"
        self.max_tracking_roi_outside_frames = int(max(0, max_tracking_roi_outside_frames))
        self.max_tracking_roi_exit_jump_px = float(max(1.0, max_tracking_roi_exit_jump_px))
        self.tracking_roi_outside_frames = 0
        self.track_started = not self.start_gate_enabled
        self.start_gate_hits = 0
        self.reacquire_hits = 0
        self.reacquire_candidate_center: Optional[Point] = None
        self.prev_gray: Optional[np.ndarray] = None

    def reset_lock(self) -> None:
        self.previous_center = None
        self.missing_frames = 0
        self.start_gate_hits = 0
        self.reacquire_hits = 0
        self.reacquire_candidate_center = None
        self.tracking_roi_outside_frames = 0
        self.track_started = not self.start_gate_enabled

    def close(self) -> None:
        self.hands.close()

    def get_activation_roi(
        self,
        frame_shape: Tuple[int, int, int],
        board_roi: Optional[Tuple[int, int, int, int]],
        activation_roi_override: Optional[ActivationRegion] = None,
    ) -> Optional[ActivationRegion]:
        if not self.start_gate_enabled or self.track_started:
            return None
        if activation_roi_override is not None:
            return activation_roi_override
        return center_roi(frame_shape, board_roi, self.start_gate_scale)

    def detect(
        self,
        frame_bgr: np.ndarray,
        board_roi: Optional[Tuple[int, int, int, int]],
        activation_roi_override: Optional[ActivationRegion] = None,
        tracking_roi_override: Optional[ActivationRegion] = None,
    ) -> HandObservation:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        motion = None if self.prev_gray is None else cv2.absdiff(gray, self.prev_gray)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        if self.backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = _HandResultAdapter(self.hands.detect_for_video(mp_image, self.timestamp_ms))
            self.timestamp_ms += 1
        else:
            result = self.hands.process(rgb)

        activation_roi = self.get_activation_roi(frame_bgr.shape, board_roi, activation_roi_override)
        waiting_for_start = bool(self.start_gate_enabled and not self.track_started)
        selected = select_active_hand(
            result,
            self.previous_center if self.track_started else None,
            frame_bgr.shape,
            board_roi,
            activation_roi=activation_roi,
            tracking_roi=tracking_roi_override,
            require_activation_roi=waiting_for_start,
            require_tracking_roi=bool(self.track_started and tracking_roi_override is not None),
            tracking_roi_mode=self.tracking_roi_mode,
            tracking_roi_outside_frames=self.tracking_roi_outside_frames,
            max_tracking_roi_outside_frames=self.max_tracking_roi_outside_frames,
            max_tracking_roi_exit_jump_px=self.max_tracking_roi_exit_jump_px,
            max_lock_jump_px=self.max_lock_jump_px,
            max_reacquire_jump_px=self.max_reacquire_jump_px,
            lock_missing_frames=self.missing_frames if self.track_started else 0,
            motion=motion,
            motion_weight=self.motion_weight,
            min_start_motion_score=self.min_start_motion_score if waiting_for_start else 0.0,
            min_reacquire_motion_score=self.min_reacquire_motion_score,
        )
        self.prev_gray = gray
        if selected is None:
            if waiting_for_start:
                self.start_gate_hits = 0
                return empty_observation(0, track_started=False, waiting_for_start=True)
            self.missing_frames += 1
            self.reacquire_hits = 0
            self.reacquire_candidate_center = None
            if self.start_gate_enabled and self.missing_frames >= self.max_missing_frames:
                self.reset_lock()
                return empty_observation(0, track_started=False, waiting_for_start=True)
            return empty_observation(self.missing_frames, track_started=self.track_started, waiting_for_start=False)

        if waiting_for_start:
            self.start_gate_hits += 1
            if self.start_gate_hits < self.activation_consecutive_frames:
                return empty_observation(0, track_started=False, waiting_for_start=True)
            self.track_started = True

        _, points, label, score, motion_score, inside_tracking_roi = selected
        center = center_from_landmarks(points)
        if self.missing_frames > 0 and self.reacquire_consecutive_frames > 1:
            if self.reacquire_candidate_center is not None and distance(center, self.reacquire_candidate_center) <= self.max_lock_jump_px:
                self.reacquire_hits += 1
            else:
                self.reacquire_hits = 1
            self.reacquire_candidate_center = center
            if self.reacquire_hits < self.reacquire_consecutive_frames:
                return empty_observation(self.missing_frames, track_started=True, waiting_for_start=False)

        self.previous_center = center
        self.missing_frames = 0
        if self.track_started and tracking_roi_override is not None and not inside_tracking_roi:
            self.tracking_roi_outside_frames += 1
        else:
            self.tracking_roi_outside_frames = 0
        self.reacquire_hits = 0
        self.reacquire_candidate_center = None
        wrist = (float(points[WRIST, 0]), float(points[WRIST, 1]))
        thumb = (float(points[THUMB_TIP, 0]), float(points[THUMB_TIP, 1]))
        index = (float(points[INDEX_TIP, 0]), float(points[INDEX_TIP, 1]))
        return HandObservation(
            detected=True,
            active_hand_label=label,
            hand_score=score,
            landmarks_px=points,
            center_raw=center,
            wrist=wrist,
            thumb_tip=thumb,
            index_tip=index,
            thumb_index_distance_px=distance(thumb, index),
            missing_frames=0,
            track_started=self.track_started,
            waiting_for_start=False,
            motion_score=float(motion_score),
            tracking_roi_inside=bool(inside_tracking_roi),
            tracking_roi_outside_frames=int(self.tracking_roi_outside_frames),
        )
