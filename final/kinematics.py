from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from utils import Point, distance, finite_point, nanmean


class EmaPoint:
    def __init__(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.value: Optional[Point] = None

    def reset(self) -> None:
        self.value = None

    def update(self, point: Optional[Point]) -> Optional[Point]:
        if not finite_point(point):
            return None
        p = (float(point[0]), float(point[1]))
        if self.value is None:
            self.value = p
        else:
            self.value = (
                self.alpha * p[0] + (1.0 - self.alpha) * self.value[0],
                self.alpha * p[1] + (1.0 - self.alpha) * self.value[1],
            )
        return self.value


class EmaScalar:
    def __init__(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.value: Optional[float] = None

    def reset(self) -> None:
        self.value = None

    def update(self, value: float) -> float:
        if not np.isfinite(value):
            return float("nan") if self.value is None else float(self.value)
        value = float(value)
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value
        return float(self.value)


class KinematicsTracker:
    def __init__(self, smooth_alpha: float, smooth_window: int):
        self.center_filter = EmaPoint(smooth_alpha)
        self.speed_filter = EmaScalar(smooth_alpha)
        self.accel_filter = EmaScalar(smooth_alpha)
        self.thumb_index_filter = EmaScalar(smooth_alpha)
        self.window = max(1, int(smooth_window))
        self.prev_smoothed_center: Optional[Point] = None
        self.prev_raw_center: Optional[Point] = None
        self.prev_time: Optional[float] = None
        self.prev_speed: Optional[float] = None
        self.prev_raw_speed: Optional[float] = None
        self.last_was_detected = False
        self.path_length = 0.0
        self.recent_speeds: Deque[float] = deque(maxlen=self.window)
        self.recent_accels: Deque[float] = deque(maxlen=self.window)
        self.recent_thumb_index: Deque[float] = deque(maxlen=self.window)

    def reset_live_state(self) -> None:
        self.center_filter.reset()
        self.speed_filter.reset()
        self.accel_filter.reset()
        self.thumb_index_filter.reset()
        self.prev_smoothed_center = None
        self.prev_raw_center = None
        self.prev_time = None
        self.prev_speed = None
        self.prev_raw_speed = None
        self.last_was_detected = False
        self.recent_speeds.clear()
        self.recent_accels.clear()
        self.recent_thumb_index.clear()

    def reset(self) -> None:
        self.reset_live_state()
        self.path_length = 0.0

    def update(self, time_s: float, detected: bool, raw_center: Optional[Point], thumb_index_distance_px: float) -> Dict[str, float]:
        if not detected or not finite_point(raw_center):
            self.reset_live_state()
            return {
                "hand_center_x": float("nan"),
                "hand_center_y": float("nan"),
                "path_length_px_cumulative": float(self.path_length),
                "speed_px_s": float("nan"),
                "speed_px_s_smooth": float("nan"),
                "acceleration_px_s2": float("nan"),
                "acceleration_px_s2_smooth": float("nan"),
                "thumb_index_distance_px_smooth": float("nan"),
                "speed_px_s_raw": float("nan"),
                "acceleration_px_s2_raw": float("nan"),
            }

        smoothed = self.center_filter.update(raw_center)
        if smoothed is None:
            self.last_was_detected = False
            return {
                "hand_center_x": float("nan"),
                "hand_center_y": float("nan"),
                "path_length_px_cumulative": float(self.path_length),
                "speed_px_s": float("nan"),
                "speed_px_s_smooth": float("nan"),
                "acceleration_px_s2": float("nan"),
                "acceleration_px_s2_smooth": float("nan"),
                "thumb_index_distance_px_smooth": float("nan"),
                "speed_px_s_raw": float("nan"),
                "acceleration_px_s2_raw": float("nan"),
            }

        speed = float("nan")
        speed_raw = float("nan")
        accel = float("nan")
        accel_raw = float("nan")
        if self.last_was_detected and self.prev_time is not None and self.prev_smoothed_center is not None:
            dt = max(1e-6, float(time_s) - float(self.prev_time))
            step = distance(self.prev_smoothed_center, smoothed)
            if np.isfinite(step):
                self.path_length += step
                speed = step / dt
            raw_step = distance(self.prev_raw_center, raw_center)
            if np.isfinite(raw_step):
                speed_raw = raw_step / dt
            if self.prev_speed is not None and np.isfinite(speed):
                accel = (speed - self.prev_speed) / dt
            if self.prev_raw_speed is not None and np.isfinite(speed_raw):
                accel_raw = (speed_raw - self.prev_raw_speed) / dt

        speed_smooth = self.speed_filter.update(speed)
        accel_smooth = self.accel_filter.update(accel)
        thumb_smooth = self.thumb_index_filter.update(thumb_index_distance_px)
        self.recent_speeds.append(speed_smooth)
        self.recent_accels.append(accel_smooth)
        self.recent_thumb_index.append(thumb_smooth)

        self.prev_smoothed_center = smoothed
        self.prev_raw_center = raw_center
        self.prev_time = float(time_s)
        if np.isfinite(speed):
            self.prev_speed = speed
        if np.isfinite(speed_raw):
            self.prev_raw_speed = speed_raw
        self.last_was_detected = True

        return {
            "hand_center_x": float(smoothed[0]),
            "hand_center_y": float(smoothed[1]),
            "path_length_px_cumulative": float(self.path_length),
            "speed_px_s": float(speed),
            "speed_px_s_smooth": float(nanmean(self.recent_speeds)),
            "acceleration_px_s2": float(accel),
            "acceleration_px_s2_smooth": float(nanmean(self.recent_accels)),
            "thumb_index_distance_px_smooth": float(nanmean(self.recent_thumb_index)),
            "speed_px_s_raw": float(speed_raw),
            "acceleration_px_s2_raw": float(accel_raw),
        }
