from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from calibration import BoardCalibration


Polygon = np.ndarray


def shrink_rect(rect: Tuple[float, float, float, float], scale: float) -> Tuple[float, float, float, float]:
    x, y, w, h = rect
    scale = float(np.clip(scale, 0.2, 1.0))
    new_w = w * scale
    new_h = h * scale
    return x + 0.5 * (w - new_w), y + 0.5 * (h - new_h), new_w, new_h


def rect_to_polygon(rect: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = rect
    return np.asarray(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )


def board_rect_to_image(calibration: BoardCalibration, rect: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    if calibration.inverse_homography_matrix is None:
        return None
    board_poly = rect_to_polygon(rect).reshape(1, 4, 2)
    image_poly = cv2.perspectiveTransform(board_poly, calibration.inverse_homography_matrix)[0]
    return image_poly.astype(np.float32)


def fallback_side_polygon(frame_shape: Tuple[int, int, int], side: str) -> np.ndarray:
    h, w = frame_shape[:2]
    third = w / 3.0
    if side == "left":
        return rect_to_polygon((0.0, 0.0, third, float(h)))
    return rect_to_polygon((2.0 * third, 0.0, third, float(h)))


def make_mask(shape: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask


def percentile_value(frame_bgr: np.ndarray, mask: np.ndarray, percentile: float = 85.0) -> float:
    value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    vals = value[mask > 0]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals.astype(np.float32), percentile))


@dataclass
class LightSideFeatures:
    value: float
    baseline: float
    delta: float
    bright_fraction: float
    blob_count: int
    score: float
    is_on: bool


@dataclass
class TrialLightInfo:
    state: str
    left_value: float
    right_value: float
    left_baseline: float
    right_baseline: float
    left_delta: float
    right_delta: float
    left_bright_fraction: float
    right_bright_fraction: float
    left_blob_count: int
    right_blob_count: int
    left_score: float
    right_score: float
    left_on: bool
    right_on: bool
    both_seen: bool
    trial_started: bool
    trial_side: str
    trial_start_frame: int
    trial_time_s: float

    def to_row(self) -> Dict[str, object]:
        return {
            "trial_light_state": self.state,
            "trial_started": int(self.trial_started),
            "trial_side": self.trial_side,
            "trial_start_frame": self.trial_start_frame if self.trial_start_frame >= 0 else np.nan,
            "trial_time_s": self.trial_time_s,
            "light_left_value": self.left_value,
            "light_right_value": self.right_value,
            "light_left_baseline": self.left_baseline,
            "light_right_baseline": self.right_baseline,
            "light_left_delta": self.left_delta,
            "light_right_delta": self.right_delta,
            "light_left_bright_fraction": self.left_bright_fraction,
            "light_right_bright_fraction": self.right_bright_fraction,
            "light_left_blob_count": self.left_blob_count,
            "light_right_blob_count": self.right_blob_count,
            "light_left_score": self.left_score,
            "light_right_score": self.right_score,
            "light_left_on": int(self.left_on),
            "light_right_on": int(self.right_on),
            "light_both_seen": int(self.both_seen),
        }


class TrialLightStartDetector:
    def __init__(
        self,
        calibration: BoardCalibration,
        frame_shape: Tuple[int, int, int],
        fps: float,
        light_delta_threshold: float = 22.0,
        side_gap_threshold: float = 1.0,
        both_stable_frames: int = 3,
        side_stable_frames: int = 5,
        baseline_frames: int = 30,
        zone_inner_scale: float = 0.82,
        min_blob_area_px: float = 6.0,
    ):
        self.calibration = calibration
        self.fps = float(fps) if fps > 0 else 25.0
        self.light_delta_threshold = float(light_delta_threshold)
        self.side_gap_threshold = float(side_gap_threshold)
        self.both_stable_frames = int(max(1, both_stable_frames))
        self.side_stable_frames = int(max(1, side_stable_frames))
        self.min_blob_area_px = float(max(1.0, min_blob_area_px))
        self.left_history: Deque[float] = deque(maxlen=max(5, int(baseline_frames)))
        self.right_history: Deque[float] = deque(maxlen=max(5, int(baseline_frames)))

        self.left_polygon, self.right_polygon = self._build_polygons(frame_shape, zone_inner_scale)
        self.left_mask = make_mask(frame_shape[:2], self.left_polygon)
        self.right_mask = make_mask(frame_shape[:2], self.right_polygon)
        self.left_mask_pixels = max(1, int(np.count_nonzero(self.left_mask)))
        self.right_mask_pixels = max(1, int(np.count_nonzero(self.right_mask)))

        self.both_seen = False
        self.both_count = 0
        self.side_candidate = ""
        self.side_count = 0
        self.side_candidate_start_frame = -1
        self.trial_started = False
        self.trial_side = ""
        self.trial_start_frame = -1

    def _build_polygons(self, frame_shape: Tuple[int, int, int], zone_inner_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        zones = self.calibration.zones or {}
        left_rect = zones.get("left_target_zone")
        right_rect = zones.get("right_target_zone")
        if self.calibration.calibrated and left_rect is not None and right_rect is not None:
            left_poly = board_rect_to_image(self.calibration, shrink_rect(tuple(left_rect), zone_inner_scale))
            right_poly = board_rect_to_image(self.calibration, shrink_rect(tuple(right_rect), zone_inner_scale))
            if left_poly is not None and right_poly is not None:
                return left_poly, right_poly
        return fallback_side_polygon(frame_shape, "left"), fallback_side_polygon(frame_shape, "right")

    def _baseline(self, history: Deque[float], current_value: float) -> float:
        if not history:
            return float(current_value)
        vals = np.asarray([v for v in history if np.isfinite(v)], dtype=np.float32)
        if vals.size == 0:
            return float(current_value)
        return float(np.percentile(vals, 30))

    def _update_baseline_if_safe(self, left_value: float, right_value: float, left_on: bool, right_on: bool) -> None:
        if self.both_seen or self.trial_started:
            return
        if left_on or right_on:
            return
        if np.isfinite(left_value):
            self.left_history.append(float(left_value))
        if np.isfinite(right_value):
            self.right_history.append(float(right_value))

    def _side_features(self, frame_bgr: np.ndarray, mask: np.ndarray, mask_pixels: int, baseline: float) -> LightSideFeatures:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        vals = value[mask > 0].astype(np.float32)
        if vals.size == 0:
            return LightSideFeatures(float("nan"), float(baseline), float("nan"), 0.0, 0, 0.0, False)

        side_value = float(np.percentile(vals, 85))
        delta = side_value - float(baseline)

        # LED evidence is intentionally local and partial: even if only a few
        # LEDs are visible, bright blobs above the dark baseline can start the
        # state machine once they persist for enough frames.
        pixel_threshold = max(float(baseline) + 0.60 * self.light_delta_threshold, float(np.percentile(vals, 75)) + 3.0)
        bright = np.zeros_like(value, dtype=np.uint8)
        bright[(mask > 0) & (value.astype(np.float32) >= pixel_threshold)] = 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        components, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        blob_count = 0
        for idx in range(1, components):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            if area >= self.min_blob_area_px:
                blob_count += 1

        bright_fraction = float(np.count_nonzero(bright) / max(1, mask_pixels))
        score = 0.0
        score += max(0.0, delta) / max(1e-6, self.light_delta_threshold)
        score += min(1.0, bright_fraction / 0.010) * 0.35
        score += min(1.0, blob_count / 3.0) * 0.35
        is_on = bool(
            delta >= self.light_delta_threshold
            or (delta >= 0.65 * self.light_delta_threshold and blob_count >= 1 and bright_fraction >= 0.0015)
            or score >= 1.25
        )
        return LightSideFeatures(side_value, float(baseline), float(delta), bright_fraction, int(blob_count), float(score), is_on)

    def update(self, frame_idx: int, frame_bgr: np.ndarray) -> TrialLightInfo:
        left_value = percentile_value(frame_bgr, self.left_mask)
        right_value = percentile_value(frame_bgr, self.right_mask)

        if not self.left_history and np.isfinite(left_value):
            self.left_history.append(float(left_value))
        if not self.right_history and np.isfinite(right_value):
            self.right_history.append(float(right_value))

        left_baseline = self._baseline(self.left_history, left_value)
        right_baseline = self._baseline(self.right_history, right_value)
        left_features = self._side_features(frame_bgr, self.left_mask, self.left_mask_pixels, left_baseline)
        right_features = self._side_features(frame_bgr, self.right_mask, self.right_mask_pixels, right_baseline)
        left_on = left_features.is_on
        right_on = right_features.is_on

        self._update_baseline_if_safe(left_value, right_value, left_on, right_on)
        left_baseline = self._baseline(self.left_history, left_value)
        right_baseline = self._baseline(self.right_history, right_value)
        left_features = self._side_features(frame_bgr, self.left_mask, self.left_mask_pixels, left_baseline)
        right_features = self._side_features(frame_bgr, self.right_mask, self.right_mask_pixels, right_baseline)
        left_value = left_features.value
        right_value = right_features.value
        left_on = left_features.is_on
        right_on = right_features.is_on

        state = "waiting_both_lights"
        if self.trial_started:
            state = "trial_running"
        elif left_on and right_on:
            self.both_count += 1
            self.side_candidate = ""
            self.side_count = 0
            if self.both_count >= self.both_stable_frames:
                self.both_seen = True
                state = "both_lights_seen"
            else:
                state = "both_lights_candidate"
        elif not self.both_seen:
            self.both_count = 0
            state = "waiting_both_lights"
        else:
            raw_side = ""
            left_delta = left_features.delta
            right_delta = right_features.delta
            if left_on and not right_on and left_delta - right_delta >= self.side_gap_threshold:
                raw_side = "left"
            elif right_on and not left_on and right_delta - left_delta >= self.side_gap_threshold:
                raw_side = "right"

            if raw_side:
                if raw_side != self.side_candidate:
                    self.side_candidate = raw_side
                    self.side_count = 1
                    self.side_candidate_start_frame = int(frame_idx)
                else:
                    self.side_count += 1
                state = f"{raw_side}_light_candidate"
                if self.side_count >= self.side_stable_frames:
                    self.trial_started = True
                    self.trial_side = raw_side
                    self.trial_start_frame = self.side_candidate_start_frame
                    state = "trial_running"
            else:
                self.side_candidate = ""
                self.side_count = 0
                self.side_candidate_start_frame = -1
                state = "waiting_stable_one_side"

        trial_time = (
            max(0.0, (float(frame_idx) - float(self.trial_start_frame)) / self.fps)
            if self.trial_started and self.trial_start_frame >= 0
            else float("nan")
        )
        return TrialLightInfo(
            state=state,
            left_value=float(left_value),
            right_value=float(right_value),
            left_baseline=float(left_features.baseline),
            right_baseline=float(right_features.baseline),
            left_delta=float(left_features.delta),
            right_delta=float(right_features.delta),
            left_bright_fraction=float(left_features.bright_fraction),
            right_bright_fraction=float(right_features.bright_fraction),
            left_blob_count=int(left_features.blob_count),
            right_blob_count=int(right_features.blob_count),
            left_score=float(left_features.score),
            right_score=float(right_features.score),
            left_on=left_on,
            right_on=right_on,
            both_seen=bool(self.both_seen),
            trial_started=bool(self.trial_started),
            trial_side=self.trial_side,
            trial_start_frame=int(self.trial_start_frame),
            trial_time_s=float(trial_time),
        )

    def draw_zones(self, frame_bgr: np.ndarray, info: TrialLightInfo) -> None:
        left_color = (0, 255, 255) if info.left_on else (120, 120, 120)
        right_color = (0, 255, 255) if info.right_on else (120, 120, 120)
        cv2.polylines(frame_bgr, [np.round(self.left_polygon).astype(np.int32)], True, left_color, 1, cv2.LINE_AA)
        cv2.polylines(frame_bgr, [np.round(self.right_polygon).astype(np.int32)], True, right_color, 1, cv2.LINE_AA)
