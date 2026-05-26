from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def output_path_key(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def unique_output_path(path: Path, reserved: Optional[Set[str]] = None) -> Path:
    ensure_parent(path)
    reserved_keys = reserved if reserved is not None else set()
    candidate = path
    candidate_key = output_path_key(candidate)
    if not candidate.exists() and candidate_key not in reserved_keys:
        reserved_keys.add(candidate_key)
        return candidate

    for idx in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        candidate_key = output_path_key(candidate)
        if not candidate.exists() and candidate_key not in reserved_keys:
            reserved_keys.add(candidate_key)
            return candidate
    raise RuntimeError(f"Could not find a free output path for: {path}")


def rotate_if_needed(frame: np.ndarray, rotate_clockwise: bool) -> np.ndarray:
    if rotate_clockwise:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def finite_point(point: Optional[Point]) -> bool:
    if point is None:
        return False
    return bool(np.isfinite(point[0]) and np.isfinite(point[1]))


def distance(a: Optional[Point], b: Optional[Point]) -> float:
    if not finite_point(a) or not finite_point(b):
        return float("nan")
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def moving_average_nan(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    window = max(1, int(window))
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for i in range(arr.size):
        start = max(0, i - window + 1)
        vals = arr[start : i + 1]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = float(np.mean(vals))
    return out


def open_video_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    ensure_parent(path)
    suffix = path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix == ".mp4" else "MJPG"))
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def safe_float(value: object) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value_f if math.isfinite(value_f) else float("nan")


def point_to_list(point: Optional[Point]) -> list[float]:
    if not finite_point(point):
        return [float("nan"), float("nan")]
    return [float(point[0]), float(point[1])]

