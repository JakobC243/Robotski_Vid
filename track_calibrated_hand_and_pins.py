import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd

from test_detect_side_hole_grids import (
    cluster_candidates as cluster_side_hole_candidates,
    detect_bright_hole_candidates as detect_side_hole_candidates,
    select_two_grids as select_two_side_hole_grids,
)


Circle = Tuple[float, float, float]
Point = Tuple[float, float]

PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]

# Nastavljivi casovni parametri. Za mirnejso hitrost/pospesek povecaj okna,
# za bolj odzivno meritev jih zmanjsaj.
DEFAULT_HAND_CENTER_MA_WINDOW_FRAMES = 3
DEFAULT_VELOCITY_WINDOW_FRAMES = 3
DEFAULT_ACCELERATION_WINDOW_FRAMES = 5

# Nastavitve za zaklep zasedenosti luknje. Zasedenost se potrdi sele po tem,
# ko sta palec/kazalec obiskala luknjo, se umaknila, in je dokaz stabilen.
DEFAULT_OCCUPANCY_INTERACTION_MEMORY_FRAMES = 24
DEFAULT_OCCUPANCY_SETTLE_FRAMES = 3
DEFAULT_OCCUPANCY_STABLE_OCCUPIED_FRAMES = 3
DEFAULT_OCCUPANCY_STABLE_EMPTY_FRAMES = 3


@dataclass
class Hole:
    hole_id: int
    grid_id: int
    local_id: int
    row: int
    col: int
    x_px: float
    y_px: float
    r_px: float
    x_mm: float
    y_mm: float
    matched: bool


@dataclass
class BoardCalibration:
    calibrated: bool
    width: int
    height: int
    median_frame: np.ndarray
    board_quad: Optional[np.ndarray]
    board_mask: np.ndarray
    holes: List[Hole]
    mm_per_px: float
    px_per_mm: float
    image_to_mm_h: Optional[np.ndarray]
    mm_to_image_h: Optional[np.ndarray]
    grid_info: List[Dict]
    candidate_count: int


@dataclass
class HandCandidate:
    index: int
    landmarks: object
    handedness: str
    points: np.ndarray
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    score: float


@dataclass
class PinDetection:
    x_px: float
    y_px: float
    r_px: float
    area_px: float
    mean_value: float
    mean_diff: float
    mean_motion: float
    near_hole: bool
    x_mm: float
    y_mm: float


def get_default_video_path(data_root: Path) -> Path:
    videos = sorted(data_root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 videos found under {data_root}")
    return videos[0]


def create_output_paths(output_root: Path, output_stem: str = "calibrated_hand_pins") -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "video": output_root / f"{output_stem}_preview.avi",
        "csv": output_root / f"{output_stem}.csv",
        "summary_json": output_root / "calibrated_summary.json",
        "calibration_json": output_root / "board_calibration.json",
        "median_frame": output_root / "median_frame.png",
        "calibration_debug": output_root / "board_calibration_debug.png",
        "position_plot": output_root / "position_mm_over_time.png",
        "speed_plot": output_root / "speed_mm_s_over_time.png",
        "accel_plot": output_root / "acceleration_mm_s2_over_time.png",
        "trajectory_plot": output_root / "trajectory_mm.png",
        "occupancy_plot": output_root / "hole_occupancy_over_time.png",
    }


def read_median_frame(video_path: Path, frame_samples: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while len(frames) < frame_samples:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No frames could be read for calibration.")

    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def detect_board_quad(frame_bgr: np.ndarray, min_area_frac: float = 0.10) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 130)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_frac * frame_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            score = area
        else:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype(np.float32)
            rect_area = max(1.0, cv2.contourArea(quad))
            fill_ratio = area / rect_area
            if fill_ratio < 0.45:
                continue
            score = area * fill_ratio

        ordered = order_quad_points(quad)
        q_area = abs(cv2.contourArea(ordered))
        if q_area < min_area_frac * frame_area:
            continue

        if score > best_score:
            best_score = score
            best_quad = ordered

    return best_quad


def make_board_mask(shape: Tuple[int, int], quad: Optional[np.ndarray]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if quad is None:
        mask[:, :] = 255
    else:
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask


def point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    xi = int(round(x))
    yi = int(round(y))
    if yi < 0 or yi >= mask.shape[0] or xi < 0 or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi] > 0)


def merge_close_circles(circles: List[Circle], merge_dist_px: float = 8.0) -> List[Circle]:
    merged: List[Circle] = []
    for x, y, r in circles:
        found = False
        for i, (ox, oy, orad) in enumerate(merged):
            if float(np.hypot(x - ox, y - oy)) < merge_dist_px:
                merged[i] = (
                    0.55 * ox + 0.45 * x,
                    0.55 * oy + 0.45 * y,
                    0.55 * orad + 0.45 * r,
                )
                found = True
                break
        if not found:
            merged.append((float(x), float(y), float(r)))
    return merged


def detect_hough_circles(
    frame_bgr: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_dist: int,
    param1: float,
    param2: float,
) -> List[Circle]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in np.round(circles[0, :], 2)]


def circular_blobs_from_mask(
    mask: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_circularity: float,
) -> List[Circle]:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = math.pi * min_radius * min_radius
    max_area = math.pi * max_radius * max_radius
    circles: List[Circle] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue
        (x, y), r = cv2.minEnclosingCircle(contour)
        if min_radius <= r <= max_radius:
            circles.append((float(x), float(y), float(r)))
    return circles


def detect_hole_candidates(
    frame_bgr: np.ndarray,
    board_mask: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_dist: int,
    hough_param1: float,
    hough_param2: float,
    dark_threshold: int,
    bright_threshold: int,
) -> List[Circle]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    valid_pixels = gray[board_mask > 0]
    if valid_pixels.size:
        dark_t = min(dark_threshold, int(np.percentile(valid_pixels, 25)))
        bright_t = max(bright_threshold, int(np.percentile(valid_pixels, 92)))
    else:
        dark_t = dark_threshold
        bright_t = bright_threshold

    hough = detect_hough_circles(
        frame_bgr=frame_bgr,
        min_radius=min_radius,
        max_radius=max_radius,
        min_dist=min_dist,
        param1=hough_param1,
        param2=hough_param2,
    )

    dark_mask = np.where((gray < dark_t) & (board_mask > 0), 255, 0).astype(np.uint8)
    bright_mask = np.where(
        ((value > bright_t) | ((value > max(120, bright_threshold - 30)) & (saturation > 60)))
        & (board_mask > 0),
        255,
        0,
    ).astype(np.uint8)

    dark_blobs = circular_blobs_from_mask(dark_mask, min_radius, max_radius, 0.28)
    bright_blobs = circular_blobs_from_mask(bright_mask, min_radius, max_radius, 0.25)

    circles = hough + dark_blobs + bright_blobs
    circles = [c for c in circles if point_in_mask(board_mask, c[0], c[1])]
    return merge_close_circles(circles, merge_dist_px=max(5.0, 0.6 * min_dist))


def nearest_candidate(
    expected: np.ndarray,
    centers: np.ndarray,
    used_indices: set,
    tolerance_px: float,
) -> Tuple[Optional[int], float]:
    if centers.size == 0:
        return None, float("inf")

    dists = np.linalg.norm(centers - expected.reshape(1, 2), axis=1)
    for idx in np.argsort(dists):
        idx = int(idx)
        if idx in used_indices:
            continue
        dist = float(dists[idx])
        if dist <= tolerance_px:
            return idx, dist
        break
    return None, float("inf")


def score_grid_candidate(
    centers: np.ndarray,
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    tolerance_px: float,
    min_matches: int,
) -> Optional[Dict]:
    spacing_u = float(np.linalg.norm(u))
    spacing_v = float(np.linalg.norm(v))
    if spacing_u <= 1e-6 or spacing_v <= 1e-6:
        return None

    spacing_ratio = max(spacing_u, spacing_v) / max(1e-6, min(spacing_u, spacing_v))
    if spacing_ratio > 2.0:
        return None

    cos_angle = abs(float(np.dot(u, v) / (spacing_u * spacing_v)))
    if cos_angle > 0.70:
        return None

    expected_points = []
    local_matches = []
    used_indices = set()
    errors = []
    matched_indices = []
    matched_points = []

    for row in range(3):
        for col in range(3):
            local_id = row * 3 + col
            expected = p0 + col * u + row * v
            expected_points.append(expected)
            idx, err = nearest_candidate(expected, centers, used_indices, tolerance_px)
            if idx is None:
                continue
            used_indices.add(idx)
            matched_indices.append(idx)
            matched_points.append(centers[idx])
            errors.append(err)
            local_matches.append(
                {
                    "local_id": int(local_id),
                    "row": int(row),
                    "col": int(col),
                    "candidate_index": int(idx),
                    "error_px": float(err),
                }
            )

    num_matches = len(matched_indices)
    if num_matches < min_matches:
        return None

    mean_error = float(np.mean(errors)) if errors else float("inf")
    score = (
        num_matches * 1000.0
        - mean_error * 25.0
        - abs(spacing_u - spacing_v) * 3.0
        - cos_angle * 150.0
    )

    return {
        "score": float(score),
        "num_matches": int(num_matches),
        "mean_error_px": mean_error,
        "matched_indices": matched_indices,
        "matched_points": np.asarray(matched_points, dtype=np.float32),
        "expected_points": np.asarray(expected_points, dtype=np.float32),
        "local_matches": local_matches,
        "p0": p0.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "spacing_u_px": spacing_u,
        "spacing_v_px": spacing_v,
        "cos_angle": cos_angle,
    }


def limit_grid_candidates(
    circles: List[Circle],
    min_spacing_px: float,
    max_spacing_px: float,
    max_candidates: int,
) -> List[Circle]:
    if len(circles) <= max_candidates:
        return circles

    centers = np.array([[c[0], c[1]] for c in circles], dtype=np.float32)
    scores = []
    for i, center in enumerate(centers):
        dists = np.linalg.norm(centers - center.reshape(1, 2), axis=1)
        neighbor_count = int(np.sum((dists >= min_spacing_px) & (dists <= max_spacing_px)))
        scores.append((neighbor_count, circles[i][2], i))

    keep = [idx for _, _, idx in sorted(scores, reverse=True)[:max_candidates]]
    return [circles[i] for i in keep]


def find_best_3x3_grid(
    circles: List[Circle],
    min_spacing_px: float,
    max_spacing_px: float,
    tolerance_ratio: float,
    min_matches: int,
    max_candidates: int,
) -> Optional[Dict]:
    limited = limit_grid_candidates(circles, min_spacing_px, max_spacing_px, max_candidates)
    if len(limited) < min_matches:
        return None

    centers = np.array([[c[0], c[1]] for c in limited], dtype=np.float32)
    best = None

    for i, p0 in enumerate(centers):
        for j, p_u in enumerate(centers):
            if j == i:
                continue
            u = p_u - p0
            du = float(np.linalg.norm(u))
            if du < min_spacing_px or du > max_spacing_px:
                continue

            for k, p_v in enumerate(centers):
                if k == i or k == j:
                    continue
                v = p_v - p0
                dv = float(np.linalg.norm(v))
                if dv < min_spacing_px or dv > max_spacing_px:
                    continue

                tolerance_px = max(5.0, tolerance_ratio * 0.5 * (du + dv))
                candidate = score_grid_candidate(centers, p0, u, v, tolerance_px, min_matches)
                if candidate is None:
                    continue

                candidate["source_circles"] = limited
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

    return best


def grid_center(grid: Dict) -> np.ndarray:
    return np.mean(grid["expected_points"], axis=0).astype(np.float32)


def grid_spacing(grid: Dict) -> float:
    return float(0.5 * (grid["spacing_u_px"] + grid["spacing_v_px"]))


def grid_bbox(grid: Dict) -> Tuple[float, float, float, float]:
    pts = grid["expected_points"]
    return (
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    )


def grid_orientation_similarity(a: Dict, b: Dict) -> float:
    axes_a = [a["u"].astype(np.float32), a["v"].astype(np.float32)]
    axes_b = [b["u"].astype(np.float32), b["v"].astype(np.float32)]
    axes_a = [axis / max(1e-6, float(np.linalg.norm(axis))) for axis in axes_a]
    axes_b = [axis / max(1e-6, float(np.linalg.norm(axis))) for axis in axes_b]

    same = 0.5 * (abs(float(np.dot(axes_a[0], axes_b[0]))) + abs(float(np.dot(axes_a[1], axes_b[1]))))
    swapped = 0.5 * (abs(float(np.dot(axes_a[0], axes_b[1]))) + abs(float(np.dot(axes_a[1], axes_b[0]))))
    return max(same, swapped)


def grid_overlap_ratio(a: Dict, b: Dict) -> float:
    ax1, ay1, ax2, ay2 = grid_bbox(a)
    bx1, by1, bx2, by2 = grid_bbox(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / min(area_a, area_b))


def homography_from_grid(grid: Dict, hole_spacing_mm: float = 32.0) -> Optional[np.ndarray]:
    image_points = grid["expected_points"].astype(np.float32)
    world_points = []
    for row in range(3):
        for col in range(3):
            world_points.append([col * hole_spacing_mm, row * hole_spacing_mm])
    world = np.asarray(world_points, dtype=np.float32)
    h, _ = cv2.findHomography(image_points, world, 0)
    return h


def rectified_grid_metrics(reference: Dict, candidate: Dict) -> Optional[Tuple[float, float, float]]:
    h = homography_from_grid(reference)
    if h is None:
        return None

    pts = candidate["expected_points"].astype(np.float32).reshape(1, -1, 2)
    rectified = cv2.perspectiveTransform(pts, h).reshape(3, 3, 2)

    horizontal = []
    vertical = []
    cosines = []
    for row in range(3):
        for col in range(2):
            horizontal.append(float(np.linalg.norm(rectified[row, col + 1] - rectified[row, col])))
    for row in range(2):
        for col in range(3):
            vertical.append(float(np.linalg.norm(rectified[row + 1, col] - rectified[row, col])))
    for row in range(2):
        for col in range(2):
            u = rectified[row, col + 1] - rectified[row, col]
            v = rectified[row + 1, col] - rectified[row, col]
            nu = float(np.linalg.norm(u))
            nv = float(np.linalg.norm(v))
            if nu > 1e-6 and nv > 1e-6:
                cosines.append(abs(float(np.dot(u, v) / (nu * nv))))

    if not horizontal or not vertical:
        return None

    median_h = float(np.median(horizontal))
    median_v = float(np.median(vertical))
    spacing_ratio = max(median_h, median_v) / max(1e-6, min(median_h, median_v))
    median_spacing = 0.5 * (median_h + median_v)
    mean_cos = float(np.mean(cosines)) if cosines else 1.0
    return spacing_ratio, mean_cos, median_spacing


def dedupe_grid_candidates(candidates: List[Dict], center_tol_px: float = 14.0) -> List[Dict]:
    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected: List[Dict] = []
    for candidate in candidates:
        center = grid_center(candidate)
        duplicate = False
        for existing in selected:
            if float(np.linalg.norm(center - grid_center(existing))) < center_tol_px:
                duplicate = True
                break
            if grid_overlap_ratio(candidate, existing) > 0.45:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return selected


def enumerate_3x3_grids(
    circles: List[Circle],
    min_spacing_px: float,
    max_spacing_px: float,
    tolerance_ratio: float,
    min_matches: int,
    max_candidates: int,
    max_output: int = 120,
) -> List[Dict]:
    limited = limit_grid_candidates(circles, min_spacing_px, max_spacing_px, max_candidates)
    if len(limited) < min_matches:
        return []

    centers = np.array([[c[0], c[1]] for c in limited], dtype=np.float32)
    candidates: List[Dict] = []

    for i, p0 in enumerate(centers):
        for j, p_u in enumerate(centers):
            if j == i:
                continue
            u = p_u - p0
            du = float(np.linalg.norm(u))
            if du < min_spacing_px or du > max_spacing_px:
                continue

            for k, p_v in enumerate(centers):
                if k == i or k == j:
                    continue
                v = p_v - p0
                dv = float(np.linalg.norm(v))
                if dv < min_spacing_px or dv > max_spacing_px:
                    continue

                tolerance_px = max(5.0, tolerance_ratio * 0.5 * (du + dv))
                candidate = score_grid_candidate(centers, p0, u, v, tolerance_px, min_matches)
                if candidate is None:
                    continue
                candidate["source_circles"] = limited
                candidates.append(candidate)

    return dedupe_grid_candidates(candidates)[:max_output]


def select_side_grid_pair(
    candidates: List[Dict],
    width: int,
    height: int,
    min_pair_separation_frac: float = 0.34,
) -> List[Dict]:
    if len(candidates) < 2:
        return []

    min_sep = max(min_pair_separation_frac * width, 3.2 * np.median([grid_spacing(g) for g in candidates]))
    best_pair: List[Dict] = []
    best_score = -1e18

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if a["num_matches"] < 7 or b["num_matches"] < 7:
                continue
            if a["cos_angle"] > 0.45 or b["cos_angle"] > 0.45:
                continue

            ca = grid_center(a)
            cb = grid_center(b)
            delta = cb - ca
            sep = float(np.linalg.norm(delta))
            dx = abs(float(delta[0]))
            dy = abs(float(delta[1]))
            if sep < min_sep:
                continue
            if dx < 0.55 * min_sep:
                continue
            if grid_overlap_ratio(a, b) > 0.08:
                continue
            if not (min(ca[0], cb[0]) < 0.45 * width and max(ca[0], cb[0]) > 0.55 * width):
                continue

            spacing_a = grid_spacing(a)
            spacing_b = grid_spacing(b)
            spacing_ratio = max(spacing_a, spacing_b) / max(1e-6, min(spacing_a, spacing_b))
            if spacing_ratio > 1.35:
                continue

            orientation_sim = grid_orientation_similarity(a, b)
            if orientation_sim < 0.60:
                continue

            rect_ab = rectified_grid_metrics(a, b)
            rect_ba = rectified_grid_metrics(b, a)
            if rect_ab is None or rect_ba is None:
                continue
            rect_ratio = max(rect_ab[0], rect_ba[0])
            rect_cos = max(rect_ab[1], rect_ba[1])
            if rect_ratio > 1.45 or rect_cos > 0.50:
                continue

            # Two side target fields should be separated mostly left-right and
            # should not both live in the same central/text region.
            side_bonus = 0.0
            if min(ca[0], cb[0]) < 0.48 * width and max(ca[0], cb[0]) > 0.52 * width:
                side_bonus += 800.0
            if dx > dy:
                side_bonus += 250.0

            pair_score = (
                a["score"]
                + b["score"]
                + 2.5 * sep
                + side_bonus
                - 900.0 * abs(math.log(spacing_ratio))
                - 350.0 * (1.0 - orientation_sim)
                - 1.2 * dy
            )

            if pair_score > best_score:
                best_score = pair_score
                best_pair = [a, b]

    if not best_pair:
        return []

    best_pair.sort(key=lambda grid: float(grid_center(grid)[0]))
    return best_pair


def select_grids(
    circles: List[Circle],
    num_grids: int,
    min_spacing_px: float,
    max_spacing_px: float,
    tolerance_ratio: float,
    min_matches: int,
    max_candidates: int,
    width: int,
    height: int,
) -> List[Dict]:
    if num_grids >= 2:
        candidates = enumerate_3x3_grids(
            circles=circles,
            min_spacing_px=min_spacing_px,
            max_spacing_px=max_spacing_px,
            tolerance_ratio=tolerance_ratio,
            min_matches=min_matches,
            max_candidates=max_candidates,
        )
        pair = select_side_grid_pair(candidates, width=width, height=height)
        if pair:
            return pair[:num_grids]
        return []

    remaining = circles[:]
    selected: List[Dict] = []

    for _ in range(num_grids):
        grid = find_best_3x3_grid(
            circles=remaining,
            min_spacing_px=min_spacing_px,
            max_spacing_px=max_spacing_px,
            tolerance_ratio=tolerance_ratio,
            min_matches=min_matches,
            max_candidates=max_candidates,
        )
        if grid is None:
            break

        selected.append(grid)
        matched_points = grid["matched_points"]
        new_remaining = []
        for circle in remaining:
            cxy = np.array([circle[0], circle[1]], dtype=np.float32)
            if matched_points.size and float(np.min(np.linalg.norm(matched_points - cxy, axis=1))) < 4.0:
                continue
            new_remaining.append(circle)
        remaining = new_remaining

    return selected


def transform_point(point: Point, homography: Optional[np.ndarray], mm_per_px: float) -> Tuple[float, float]:
    if homography is None:
        return float(point[0] * mm_per_px), float(point[1] * mm_per_px)

    pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, homography)[0, 0]
    return float(transformed[0]), float(transformed[1])


def quad_contains_points(quad: Optional[np.ndarray], points: np.ndarray) -> bool:
    if quad is None or points.size == 0:
        return False
    polygon = np.round(quad).astype(np.float32)
    for point in points.reshape(-1, 2):
        if cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) < 0:
            return False
    return True


def derive_quad_from_grids(grids: List[Dict], width: int, height: int) -> Optional[np.ndarray]:
    if not grids:
        return None
    points = np.vstack([grid["expected_points"] for grid in grids]).astype(np.float32)
    spacing = float(np.median([grid_spacing(grid) for grid in grids]))
    margin = max(25.0, 1.7 * spacing)
    x1 = max(0.0, float(np.min(points[:, 0]) - margin))
    y1 = max(0.0, float(np.min(points[:, 1]) - margin))
    x2 = min(float(width - 1), float(np.max(points[:, 0]) + margin))
    y2 = min(float(height - 1), float(np.max(points[:, 1]) + margin))
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def build_board_calibration(
    video_path: Path,
    frame_samples: int,
    hole_spacing_mm: float,
    num_grids: int,
    min_radius: int,
    max_radius: int,
    min_dist: int,
    hough_param1: float,
    hough_param2: float,
    dark_threshold: int,
    bright_threshold: int,
    min_spacing_px: float,
    max_spacing_px: float,
    grid_tolerance_ratio: float,
    min_grid_matches: int,
    max_grid_candidates: int,
    side_gray_threshold: int,
    side_value_threshold: int,
    side_min_area: float,
    side_max_area: float,
    side_min_circularity: float,
    side_cluster_link_px: float,
    side_min_grid_separation_px: float,
) -> BoardCalibration:
    median = read_median_frame(video_path, frame_samples)
    height, width = median.shape[:2]

    side_points, _ = detect_side_hole_candidates(
        image_bgr=median,
        gray_threshold=side_gray_threshold,
        value_threshold=side_value_threshold,
        min_radius=float(min_radius),
        max_radius=float(max_radius),
        min_area=side_min_area,
        max_area=side_max_area,
        min_circularity=side_min_circularity,
    )
    clusters = cluster_side_hole_candidates(side_points, link_distance_px=side_cluster_link_px)

    grid_args = argparse.Namespace(
        min_matches=min_grid_matches,
        min_spacing_px=min_spacing_px,
        max_spacing_px=max_spacing_px,
        min_grid_separation_px=side_min_grid_separation_px,
    )
    grids = select_two_side_hole_grids(side_points, clusters, grid_args)
    if num_grids > 0:
        grids = grids[:num_grids]

    if not grids:
        return BoardCalibration(
            calibrated=False,
            width=width,
            height=height,
            median_frame=median,
            board_quad=None,
            board_mask=np.full((height, width), 255, dtype=np.uint8),
            holes=[],
            mm_per_px=1.0,
            px_per_mm=1.0,
            image_to_mm_h=None,
            mm_to_image_h=None,
            grid_info=[
                {
                    "method": "side_bright_blob_clusters",
                    "candidate_count": len(side_points),
                    "cluster_sizes": [len(cluster) for cluster in clusters],
                    "reason": "No two valid side 3x3 hole grids were found.",
                }
            ],
            candidate_count=len(side_points),
        )

    all_grid_points = np.vstack([grid["expected_points"] for grid in grids]).astype(np.float32)
    derived_quad = derive_quad_from_grids(grids, width=width, height=height)
    board_quad = derived_quad
    board_mask = make_board_mask((height, width), board_quad)

    spacings = []
    for grid in grids:
        spacings.append(float(grid["spacing_u_px"]))
        spacings.append(float(grid["spacing_v_px"]))
    spacing_px = float(np.median(spacings))
    mm_per_px = float(hole_spacing_mm / spacing_px)
    px_per_mm = float(1.0 / mm_per_px)

    first_grid = grids[0]
    first_img = first_grid["expected_points"].astype(np.float32)
    first_world = []
    for row in range(3):
        for col in range(3):
            first_world.append([col * hole_spacing_mm, row * hole_spacing_mm])
    first_world_np = np.asarray(first_world, dtype=np.float32)
    image_to_mm_h, _ = cv2.findHomography(first_img, first_world_np, 0)
    mm_to_image_h, _ = cv2.findHomography(first_world_np, first_img, 0)

    holes: List[Hole] = []
    hole_id = 0
    grid_info: List[Dict] = []
    for grid_id, grid in enumerate(grids):
        point_indices = grid.get("point_indices", [])
        source_radii = [
            float(side_points[idx]["r"])
            for idx in point_indices
            if 0 <= int(idx) < len(side_points)
        ]
        default_radius = float(np.median(source_radii)) if source_radii else float(0.5 * (min_radius + max_radius))

        grid_info.append(
            {
                "grid_id": int(grid_id),
                "method": "side_bright_blob_clusters",
                "cluster_id": int(grid.get("cluster_id", -1)),
                "cluster_size": int(grid.get("cluster_size", 0)),
                "num_matches": int(grid.get("num_matches", grid.get("matches", 0))),
                "mean_error_px": float(grid["mean_error_px"]),
                "spacing_u_px": float(grid["spacing_u_px"]),
                "spacing_v_px": float(grid["spacing_v_px"]),
                "cos_angle": float(grid["cos_angle"]),
                "p0": np.asarray(grid["p0"], dtype=float).tolist(),
                "u": np.asarray(grid["u"], dtype=float).tolist(),
                "v": np.asarray(grid["v"], dtype=float).tolist(),
            }
        )

        for local_id, p in enumerate(grid["expected_points"]):
            row = local_id // 3
            col = local_id % 3
            radius = default_radius
            matched = False
            if side_points:
                dists = [
                    float(np.hypot(p[0] - side_point["x"], p[1] - side_point["y"]))
                    for side_point in side_points
                ]
                nearest_idx = int(np.argmin(dists))
                if dists[nearest_idx] <= max(7.0, 0.4 * grid_spacing(grid)):
                    matched = True
                    radius = float(side_points[nearest_idx]["r"])

            x_mm = float(col * hole_spacing_mm)
            y_mm = float((grid_id * 4 + row) * hole_spacing_mm)
            holes.append(
                Hole(
                    hole_id=hole_id,
                    grid_id=grid_id,
                    local_id=local_id,
                    row=row,
                    col=col,
                    x_px=float(p[0]),
                    y_px=float(p[1]),
                    r_px=radius,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    matched=matched,
                )
            )
            hole_id += 1

    return BoardCalibration(
        calibrated=True,
        width=width,
        height=height,
        median_frame=median,
        board_quad=board_quad,
        board_mask=board_mask,
        holes=holes,
        mm_per_px=mm_per_px,
        px_per_mm=px_per_mm,
        image_to_mm_h=image_to_mm_h,
        mm_to_image_h=mm_to_image_h,
        grid_info=grid_info,
        candidate_count=len(side_points),
    )


class CenterKalman:
    def __init__(self, process_noise: float, measurement_noise: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        self.initialized = False
        self.last_prediction: Optional[Tuple[float, float]] = None

    def _set_dt(self, dt: float) -> None:
        dt = max(1e-3, float(dt))
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

    def initialize(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        x, y = measurement
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
        self.last_prediction = (float(x), float(y))
        return self.last_prediction

    def predict(self, dt: float) -> Optional[Tuple[float, float]]:
        if not self.initialized:
            return None
        self._set_dt(dt)
        pred = self.kf.predict()
        self.last_prediction = (float(pred[0, 0]), float(pred[1, 0]))
        return self.last_prediction

    def correct(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        if not self.initialized:
            return self.initialize(measurement)
        z = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]], dtype=np.float32)
        corrected = self.kf.correct(z)
        result = (float(corrected[0, 0]), float(corrected[1, 0]))
        self.last_prediction = result
        return result


class TemporalPointFilters:
    def __init__(self, ema_alpha: float, ma_window: int):
        self.ema_alpha = float(ema_alpha)
        self.ema_value: Optional[Tuple[float, float]] = None
        self.samples: Deque[Tuple[float, float]] = deque(maxlen=max(1, ma_window))

    def update(self, measurement: Optional[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        if measurement is not None:
            if self.ema_value is None:
                self.ema_value = (float(measurement[0]), float(measurement[1]))
            else:
                self.ema_value = (
                    self.ema_alpha * float(measurement[0]) + (1.0 - self.ema_alpha) * self.ema_value[0],
                    self.ema_alpha * float(measurement[1]) + (1.0 - self.ema_alpha) * self.ema_value[1],
                )
            self.samples.append((float(measurement[0]), float(measurement[1])))

        if self.samples:
            arr = np.asarray(self.samples, dtype=np.float32)
            ma_value = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
        else:
            ma_value = None

        return self.ema_value, ma_value


class OccupancyStabilizer:
    def __init__(
        self,
        hole_count: int,
        alpha: float,
        occupied_threshold: float = 0.55,
        empty_threshold: float = 0.35,
        interaction_alpha_scale: float = 0.20,
        interaction_memory_frames: int = 18,
        require_interaction_to_occupy: bool = True,
        settle_frames: int = 3,
        stable_occupied_frames: int = 3,
        stable_empty_frames: int = 3,
    ):
        self.alpha = float(alpha)
        self.probabilities = np.zeros(hole_count, dtype=np.float32)
        self.states = np.zeros(hole_count, dtype=bool)
        self.recent_interactions = np.zeros(hole_count, dtype=np.int32)
        self.settle_counters = np.zeros(hole_count, dtype=np.int32)
        self.occupied_evidence = np.zeros(hole_count, dtype=np.int32)
        self.empty_evidence = np.zeros(hole_count, dtype=np.int32)
        self.occupied_threshold = float(occupied_threshold)
        self.empty_threshold = float(empty_threshold)
        self.interaction_alpha_scale = float(interaction_alpha_scale)
        self.interaction_memory_frames = int(max(0, interaction_memory_frames))
        self.require_interaction_to_occupy = bool(require_interaction_to_occupy)
        self.settle_frames = int(max(0, settle_frames))
        self.stable_occupied_frames = int(max(1, stable_occupied_frames))
        self.stable_empty_frames = int(max(1, stable_empty_frames))

    def update(self, raw_probs: Sequence[float], interaction_mask: Optional[Sequence[bool]] = None) -> np.ndarray:
        raw = np.asarray(raw_probs, dtype=np.float32)
        if raw.shape[0] != self.probabilities.shape[0]:
            self.probabilities = raw.copy()
            if interaction_mask is None or not self.require_interaction_to_occupy:
                initial_can_occupy = np.ones(raw.shape, dtype=bool)
            else:
                initial_can_occupy = np.asarray(interaction_mask, dtype=bool)
                if initial_can_occupy.shape[0] != raw.shape[0]:
                    initial_can_occupy = np.zeros(raw.shape, dtype=bool)
            self.states = (raw >= self.occupied_threshold) & initial_can_occupy
            self.recent_interactions = np.zeros(raw.shape[0], dtype=np.int32)
            self.settle_counters = np.zeros(raw.shape[0], dtype=np.int32)
            self.occupied_evidence = np.zeros(raw.shape[0], dtype=np.int32)
            self.empty_evidence = np.zeros(raw.shape[0], dtype=np.int32)
        else:
            if interaction_mask is None:
                alpha = np.full(raw.shape, self.alpha, dtype=np.float32)
            else:
                interactions = np.asarray(interaction_mask, dtype=bool)
                if interactions.shape[0] != raw.shape[0]:
                    interactions = np.zeros(raw.shape, dtype=bool)
                alpha = np.where(
                    interactions,
                    self.alpha * self.interaction_alpha_scale,
                    self.alpha,
                ).astype(np.float32)
            self.probabilities = alpha * raw + (1.0 - alpha) * self.probabilities
        return self.probabilities.copy()

    def update_stateful(
        self,
        raw_probs: Sequence[float],
        interaction_mask: Optional[Sequence[bool]] = None,
    ) -> Tuple[np.ndarray, List[bool]]:
        probabilities = self.update(raw_probs, interaction_mask=interaction_mask)
        if interaction_mask is None:
            interactions = np.zeros(probabilities.shape, dtype=bool)
        else:
            interactions = np.asarray(interaction_mask, dtype=bool)
            if interactions.shape[0] != probabilities.shape[0]:
                interactions = np.zeros(probabilities.shape, dtype=bool)

        self.recent_interactions = np.maximum(self.recent_interactions - 1, 0)
        self.recent_interactions[interactions] = self.interaction_memory_frames

        for i, prob in enumerate(probabilities):
            if interactions[i]:
                self.settle_counters[i] = self.settle_frames
                self.occupied_evidence[i] = 0
                self.empty_evidence[i] = 0
                continue

            if self.settle_counters[i] > 0:
                self.settle_counters[i] -= 1
                self.occupied_evidence[i] = 0
                self.empty_evidence[i] = 0
                continue

            can_mark_occupied = (
                (not self.require_interaction_to_occupy)
                or self.states[i]
                or self.recent_interactions[i] > 0
            )

            if prob >= self.occupied_threshold and can_mark_occupied:
                self.occupied_evidence[i] += 1
                self.empty_evidence[i] = 0
            else:
                self.occupied_evidence[i] = 0

            if prob <= self.empty_threshold:
                self.empty_evidence[i] += 1
            else:
                self.empty_evidence[i] = 0

            if self.occupied_evidence[i] >= self.stable_occupied_frames:
                self.states[i] = True
            elif self.empty_evidence[i] >= self.stable_empty_frames:
                self.states[i] = False
        return probabilities, [bool(v) for v in self.states]


def landmarks_to_pixels(hand_landmarks, width: int, height: int) -> np.ndarray:
    points = []
    for lm in hand_landmarks.landmark:
        x = float(np.clip(lm.x * width, 0, width - 1))
        y = float(np.clip(lm.y * height, 0, height - 1))
        z = float(lm.z)
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def compute_hand_center(points: np.ndarray, mode: str) -> Tuple[int, int]:
    if mode == "palm":
        xy = points[PALM_LANDMARKS, :2]
        center = np.mean(xy, axis=0)
    elif mode == "bbox":
        xy = points[:, :2]
        center = np.array([(np.min(xy[:, 0]) + np.max(xy[:, 0])) * 0.5, (np.min(xy[:, 1]) + np.max(xy[:, 1])) * 0.5])
    elif mode == "all":
        center = np.mean(points[:, :2], axis=0)
    else:
        raise ValueError("center_mode must be palm, bbox, or all")
    return int(round(center[0])), int(round(center[1]))


def bbox_from_points(points: np.ndarray, margin: int, width: int, height: int) -> Tuple[int, int, int, int]:
    xy = points[:, :2]
    x1 = int(max(0, np.min(xy[:, 0]) - margin))
    y1 = int(max(0, np.min(xy[:, 1]) - margin))
    x2 = int(min(width - 1, np.max(xy[:, 0]) + margin))
    y2 = int(min(height - 1, np.max(xy[:, 1]) + margin))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def select_active_hand(
    result,
    width: int,
    height: int,
    center_mode: str,
    previous_center: Optional[Tuple[float, float]],
    predicted_center: Optional[Tuple[float, float]],
    board_center: Optional[Tuple[float, float]],
    active_side: str,
) -> Optional[HandCandidate]:
    if not result.multi_hand_landmarks:
        return None

    candidates: List[HandCandidate] = []
    handedness_list = result.multi_handedness or []
    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
        points = landmarks_to_pixels(hand_landmarks, width, height)
        center = compute_hand_center(points, mode=center_mode)
        bbox = bbox_from_points(points, margin=18, width=width, height=height)

        handedness = "unknown"
        if idx < len(handedness_list) and handedness_list[idx].classification:
            handedness = handedness_list[idx].classification[0].label.lower()

        score = 0.0
        if previous_center is not None:
            score -= 0.035 * float(np.hypot(center[0] - previous_center[0], center[1] - previous_center[1]))
        elif predicted_center is not None:
            score -= 0.025 * float(np.hypot(center[0] - predicted_center[0], center[1] - predicted_center[1]))
        elif board_center is not None:
            score -= 0.004 * float(np.hypot(center[0] - board_center[0], center[1] - board_center[1]))

        x, y, bw, bh = bbox
        score += 0.002 * max(bw, bh)

        if active_side in {"left", "right"}:
            if handedness == active_side:
                score += 5.0
            if active_side == "left" and center[0] < width * 0.55:
                score += 1.5
            if active_side == "right" and center[0] > width * 0.45:
                score += 1.5

        candidates.append(
            HandCandidate(
                index=idx,
                landmarks=hand_landmarks,
                handedness=handedness,
                points=points,
                center=center,
                bbox=bbox,
                score=score,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0] if candidates else None


def detect_pins(
    frame_bgr: np.ndarray,
    median_frame: np.ndarray,
    prev_gray: Optional[np.ndarray],
    board_mask: np.ndarray,
    calibration: BoardCalibration,
    min_radius: int,
    max_radius: int,
    bright_threshold: int,
    diff_threshold: int,
    motion_threshold: int,
) -> Tuple[List[PinDetection], np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    median_gray = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    diff = cv2.absdiff(gray, median_gray)
    if prev_gray is None:
        motion = np.zeros_like(gray)
    else:
        motion = cv2.absdiff(gray, prev_gray)

    bright_mask = ((value >= bright_threshold) | ((value >= bright_threshold - 30) & (saturation >= 55))).astype(np.uint8) * 255
    diff_mask = (diff >= diff_threshold).astype(np.uint8) * 255
    motion_mask = (motion >= motion_threshold).astype(np.uint8) * 255

    moving_bright = cv2.bitwise_and(bright_mask, cv2.bitwise_or(diff_mask, motion_mask))
    changed_motion = cv2.bitwise_and(diff_mask, motion_mask)
    mask = cv2.bitwise_or(moving_bright, changed_motion)
    mask = cv2.bitwise_and(mask, board_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[PinDetection] = []
    min_area = math.pi * min_radius * min_radius
    max_area = math.pi * max_radius * max_radius

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.22:
            continue

        (x, y), r = cv2.minEnclosingCircle(contour)
        if not (min_radius <= r <= max_radius):
            continue
        if not point_in_mask(board_mask, x, y):
            continue

        roi_mask = np.zeros_like(gray)
        cv2.drawContours(roi_mask, [contour], -1, 255, -1)
        mean_value = float(cv2.mean(value, mask=roi_mask)[0])
        mean_diff = float(cv2.mean(diff, mask=roi_mask)[0])
        mean_motion = float(cv2.mean(motion, mask=roi_mask)[0])

        near_hole = any(
            float(np.hypot(x - hole.x_px, y - hole.y_px)) <= max(1.7 * hole.r_px, r + hole.r_px)
            for hole in calibration.holes
        )
        if near_hole and mean_diff < diff_threshold and mean_motion < motion_threshold:
            continue

        x_mm, y_mm = transform_point((float(x), float(y)), calibration.image_to_mm_h, calibration.mm_per_px)
        detections.append(
            PinDetection(
                x_px=float(x),
                y_px=float(y),
                r_px=float(r),
                area_px=float(area),
                mean_value=mean_value,
                mean_diff=mean_diff,
                mean_motion=mean_motion,
                near_hole=near_hole,
                x_mm=x_mm,
                y_mm=y_mm,
            )
        )

    detections.sort(key=lambda p: p.mean_value * p.area_px, reverse=True)
    return detections, gray


def hand_hole_interaction_distance(
    selected_hand: Optional[HandCandidate],
    hole: Hole,
) -> float:
    if selected_hand is None or selected_hand.points.size == 0:
        return float("inf")

    landmark_ids = [3, 4, 7, 8]
    points = selected_hand.points[landmark_ids, :2]
    dists = np.linalg.norm(points - np.asarray([[hole.x_px, hole.y_px]], dtype=np.float32), axis=1)
    return float(np.min(dists)) if dists.size else float("inf")


def estimate_hole_occupancy(
    frame_bgr: np.ndarray,
    median_frame: np.ndarray,
    holes: List[Hole],
    pins: List[PinDetection],
    selected_hand: Optional[HandCandidate],
    radius_multiplier: float,
    brightness_drop_threshold: float,
    dark_fraction_threshold: float,
    relative_dark_threshold: float,
    side_min_reference_value: float,
    hand_interaction_radius_px: float,
) -> Tuple[List[float], List[bool], List[Dict]]:
    if not holes:
        return [], [], []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    median_gray = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    median_hsv = cv2.cvtColor(median_frame, cv2.COLOR_BGR2HSV)
    median_value = median_hsv[:, :, 2]
    diff = cv2.absdiff(gray, median_gray)

    features = []
    h, w = gray.shape[:2]

    for hole in holes:
        radius = max(4, int(round(radius_multiplier * hole.r_px)))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(round(hole.x_px)), int(round(hole.y_px))), radius, 255, -1)
        pixels = max(1, int(np.count_nonzero(mask)))

        local_value = value[mask > 0]
        local_median_value = median_value[mask > 0]
        local_diff = diff[mask > 0]
        if local_value.size == 0 or local_median_value.size == 0:
            baseline_mean = 0.0
            current_mean = 0.0
            brightness_drop = 0.0
            brightness_ratio = 1.0
            dark_fraction = 0.0
            diff_fraction = 0.0
        else:
            baseline_mean = float(np.mean(local_median_value))
            current_mean = float(np.mean(local_value))
            brightness_drop = baseline_mean - current_mean
            brightness_ratio = current_mean / max(1.0, baseline_mean)
            per_pixel_drop = local_median_value.astype(np.float32) - local_value.astype(np.float32)
            dark_fraction = float(np.count_nonzero(per_pixel_drop >= brightness_drop_threshold) / pixels)
            diff_fraction = float(np.count_nonzero(local_diff >= brightness_drop_threshold) / pixels)

        pin_near = any(float(np.hypot(pin.x_px - hole.x_px, pin.y_px - hole.y_px)) <= max(radius, 1.4 * pin.r_px) for pin in pins)
        finger_dist = hand_hole_interaction_distance(selected_hand, hole)
        interacting = bool(finger_dist <= max(hand_interaction_radius_px, 2.2 * radius))

        features.append(
            {
                "grid_id": int(hole.grid_id),
                "baseline_value": float(baseline_mean),
                "current_value": float(current_mean),
                "brightness_drop": float(brightness_drop),
                "brightness_ratio": float(brightness_ratio),
                "dark_fraction": float(dark_fraction),
                "diff_fraction": float(diff_fraction),
                "pin_near": bool(pin_near),
                "finger_distance_px": float(finger_dist) if np.isfinite(finger_dist) else np.nan,
                "hand_interaction": bool(interacting),
            }
        )

    grid_reference: Dict[int, float] = {}
    for grid_id in sorted({int(h.grid_id) for h in holes}):
        values = [
            feature["current_value"]
            for feature in features
            if int(feature["grid_id"]) == grid_id and np.isfinite(feature["current_value"])
        ]
        grid_reference[grid_id] = float(np.percentile(values, 75)) if values else 0.0

    raw_probs = []
    occupied = []
    for feature in features:
        side_reference = grid_reference.get(int(feature["grid_id"]), 0.0)
        relative_dark_drop = float(side_reference - feature["current_value"])
        relative_allowed = bool(side_reference >= side_min_reference_value)

        drop_score = np.clip(feature["brightness_drop"] / max(1e-6, brightness_drop_threshold), 0.0, 1.0)
        ratio_score = np.clip((0.88 - feature["brightness_ratio"]) / 0.28, 0.0, 1.0)
        dark_fraction_score = np.clip(feature["dark_fraction"] / max(1e-6, dark_fraction_threshold), 0.0, 1.0)
        diff_score = np.clip(feature["diff_fraction"] / max(1e-6, dark_fraction_threshold), 0.0, 1.0)
        relative_score = np.clip(relative_dark_drop / max(1e-6, relative_dark_threshold), 0.0, 1.0) if relative_allowed else 0.0

        prob = 0.30 * float(drop_score)
        prob += 0.35 * float(relative_score)
        prob += 0.15 * float(ratio_score)
        prob += 0.15 * float(dark_fraction_score)
        prob += 0.05 * float(diff_score)
        prob += 0.05 if feature["pin_near"] else 0.0
        prob = float(np.clip(prob, 0.0, 1.0))

        feature["side_reference_value"] = float(side_reference)
        feature["relative_dark_drop"] = float(relative_dark_drop)
        feature["relative_dark_score"] = float(relative_score)
        feature["occupancy_raw_probability"] = prob
        raw_probs.append(prob)
        occupied.append(prob >= 0.55)

    return raw_probs, occupied, features


def draw_calibration_debug(calibration: BoardCalibration, output_path: Path) -> None:
    image = calibration.median_frame.copy()
    if calibration.board_quad is not None:
        cv2.polylines(image, [np.round(calibration.board_quad).astype(np.int32)], True, (255, 0, 0), 2)

    for hole in calibration.holes:
        color = (0, 255, 0) if hole.matched else (0, 180, 255)
        center = (int(round(hole.x_px)), int(round(hole.y_px)))
        cv2.circle(image, center, int(round(max(4, hole.r_px))), color, 2)
        cv2.circle(image, center, 2, (0, 0, 255), -1)
        cv2.putText(image, str(hole.hole_id), (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    text = f"calibrated={calibration.calibrated} holes={len(calibration.holes)} mm_per_px={calibration.mm_per_px:.4f}"
    cv2.putText(image, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(image, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.imwrite(str(output_path), image)


def draw_frame(
    frame_bgr: np.ndarray,
    calibration: BoardCalibration,
    selected_hand: Optional[HandCandidate],
    raw_center: Optional[Tuple[float, float]],
    final_center: Optional[Tuple[float, float]],
    ema_center: Optional[Tuple[float, float]],
    pins: List[PinDetection],
    occupancy_probs: Sequence[float],
    occupied: Sequence[bool],
    hole_interactions: Sequence[bool],
    trajectory: Sequence[Tuple[int, int]],
    frame_idx: int,
    time_s: float,
    speed_mm_s: float,
    cumulative_distance_mm: float,
    hand_status: str,
    pin_layer_enabled: bool = True,
) -> np.ndarray:
    annotated = frame_bgr.copy()

    if calibration.board_quad is not None:
        cv2.polylines(annotated, [np.round(calibration.board_quad).astype(np.int32)], True, (180, 180, 180), 2)

    for i, hole in enumerate(calibration.holes):
        is_occupied = bool(occupied[i]) if i < len(occupied) else False
        is_interacting = bool(hole_interactions[i]) if i < len(hole_interactions) else False
        prob = float(occupancy_probs[i]) if i < len(occupancy_probs) else 0.0
        if not pin_layer_enabled:
            color = (0, 220, 255)
        elif is_interacting:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255) if is_occupied else (0, 180, 0)
        center = (int(round(hole.x_px)), int(round(hole.y_px)))
        radius = int(round(max(5, hole.r_px * 1.25)))
        cv2.circle(annotated, center, radius, color, 2)
        if pin_layer_enabled and is_occupied:
            x_margin = max(4, int(round(radius * 0.55)))
            cv2.line(
                annotated,
                (center[0] - x_margin, center[1] - x_margin),
                (center[0] + x_margin, center[1] + x_margin),
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                annotated,
                (center[0] + x_margin, center[1] - x_margin),
                (center[0] - x_margin, center[1] + x_margin),
                color,
                2,
                cv2.LINE_AA,
            )
        if pin_layer_enabled:
            cv2.putText(
                annotated,
                f"{hole.hole_id}:{prob:.1f}",
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                annotated,
                str(hole.hole_id),
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    if pin_layer_enabled:
        for pin in pins:
            center = (int(round(pin.x_px)), int(round(pin.y_px)))
            cv2.circle(annotated, center, int(round(max(4, pin.r_px))), (255, 255, 0), 2)
            cv2.circle(annotated, center, 2, (0, 255, 255), -1)

    if selected_hand is not None:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        mp_drawing.draw_landmarks(
            annotated,
            selected_hand.landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )
        x, y, w, h = selected_hand.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 255), 1)

    if raw_center is not None:
        cv2.circle(annotated, (int(round(raw_center[0])), int(round(raw_center[1]))), 5, (255, 0, 255), -1)

    if ema_center is not None:
        cv2.circle(annotated, (int(round(ema_center[0])), int(round(ema_center[1]))), 5, (0, 165, 255), -1)

    if final_center is not None:
        center = (int(round(final_center[0])), int(round(final_center[1])))
        cv2.circle(annotated, center, 8, (0, 0, 255), -1)
        cv2.circle(annotated, center, 14, (255, 255, 255), 2)

    for a, b in zip(trajectory[:-1], trajectory[1:]):
        cv2.line(annotated, a, b, (0, 255, 255), 2)

    occupied_count = int(sum(bool(v) for v in occupied))
    speed_text = f"{speed_mm_s:.1f}" if np.isfinite(speed_mm_s) else "nan"
    if pin_layer_enabled:
        second_line = f"speed={speed_text} mm/s path={cumulative_distance_mm:.1f} mm pins={len(pins)} occupied={occupied_count}/{len(calibration.holes)}"
    else:
        second_line = f"speed={speed_text} mm/s path={cumulative_distance_mm:.1f} mm holes={len(calibration.holes)}"
    lines = [
        f"frame={frame_idx} t={time_s:.2f}s status={hand_status}",
        second_line,
    ]
    y0 = 24
    for line in lines:
        cv2.putText(annotated, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 25

    return annotated


def plot_results(df: pd.DataFrame, paths: Dict[str, Path], hole_count: int) -> None:
    valid = df.dropna(subset=["hand_x_mm", "hand_y_mm"]).copy()

    if not valid.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(df["time_s"], df["hand_x_mm"], label="x kalman")
        plt.plot(df["time_s"], df["hand_y_mm"], label="y kalman")
        plt.plot(df["time_s"], df["raw_x_mm"], label="x raw", alpha=0.35)
        plt.plot(df["time_s"], df["raw_y_mm"], label="y raw", alpha=0.35)
        plt.xlabel("time [s]")
        plt.ylabel("position [mm]")
        plt.title("Hand position in calibrated board coordinates")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths["position_plot"], dpi=150)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(valid["hand_x_mm"], valid["hand_y_mm"], marker=".", linewidth=1)
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.title("Hand trajectory")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(paths["trajectory_plot"], dpi=150)
        plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["speed_mm_s"], label="raw selected", alpha=0.35)
    plt.plot(df["time_s"], df["speed_ema_mm_s"], label="EMA speed", linewidth=2)
    plt.xlabel("time [s]")
    plt.ylabel("speed [mm/s]")
    plt.title("Hand speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["speed_plot"], dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["acceleration_mm_s2"])
    plt.xlabel("time [s]")
    plt.ylabel("acceleration [mm/s^2]")
    plt.title("Hand acceleration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths["accel_plot"], dpi=150)
    plt.close()

    occ_cols = [f"hole_{i:02d}_occupied" for i in range(hole_count) if f"hole_{i:02d}_occupied" in df.columns]
    if occ_cols:
        plt.figure(figsize=(10, 4))
        plt.plot(df["time_s"], df["occupied_count"], label="occupied count", linewidth=2)
        plt.xlabel("time [s]")
        plt.ylabel("occupied holes")
        plt.title("Hole occupancy over time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(paths["occupancy_plot"], dpi=150)
        plt.close()


def calibration_to_json(calibration: BoardCalibration) -> Dict:
    return {
        "calibrated": bool(calibration.calibrated),
        "width": int(calibration.width),
        "height": int(calibration.height),
        "mm_per_px": float(calibration.mm_per_px),
        "px_per_mm": float(calibration.px_per_mm),
        "candidate_count": int(calibration.candidate_count),
        "board_quad": calibration.board_quad.astype(float).tolist() if calibration.board_quad is not None else None,
        "image_to_mm_h": calibration.image_to_mm_h.astype(float).tolist() if calibration.image_to_mm_h is not None else None,
        "mm_to_image_h": calibration.mm_to_image_h.astype(float).tolist() if calibration.mm_to_image_h is not None else None,
        "grids": calibration.grid_info,
        "holes": [
            {
                "hole_id": int(h.hole_id),
                "grid_id": int(h.grid_id),
                "local_id": int(h.local_id),
                "row": int(h.row),
                "col": int(h.col),
                "x_px": float(h.x_px),
                "y_px": float(h.y_px),
                "r_px": float(h.r_px),
                "x_mm": float(h.x_mm),
                "y_mm": float(h.y_mm),
                "matched": bool(h.matched),
            }
            for h in calibration.holes
        ],
    }


def resolve_video_path(args: argparse.Namespace) -> Path:
    return Path(args.video) if args.video else get_default_video_path(Path(args.data_root))


def build_calibration_from_args(video_path: Path, args: argparse.Namespace) -> BoardCalibration:
    return build_board_calibration(
        video_path=video_path,
        frame_samples=args.frame_samples,
        hole_spacing_mm=args.hole_spacing_mm,
        num_grids=args.num_grids,
        min_radius=args.min_hole_radius,
        max_radius=args.max_hole_radius,
        min_dist=args.min_hole_dist,
        hough_param1=args.hough_param1,
        hough_param2=args.hough_param2,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        min_spacing_px=args.min_spacing_px,
        max_spacing_px=args.max_spacing_px,
        grid_tolerance_ratio=args.grid_tolerance_ratio,
        min_grid_matches=args.min_grid_matches,
        max_grid_candidates=args.max_grid_candidates,
        side_gray_threshold=args.side_gray_threshold,
        side_value_threshold=args.side_value_threshold,
        side_min_area=args.side_min_area,
        side_max_area=args.side_max_area,
        side_min_circularity=args.side_min_circularity,
        side_cluster_link_px=args.side_cluster_link_px,
        side_min_grid_separation_px=args.side_min_grid_separation_px,
    )


def write_calibration_outputs(calibration: BoardCalibration, paths: Dict[str, Path]) -> None:
    with open(paths["calibration_json"], "w", encoding="utf-8") as f:
        json.dump(calibration_to_json(calibration), f, indent=2, ensure_ascii=False)
    cv2.imwrite(str(paths["median_frame"]), calibration.median_frame)
    draw_calibration_debug(calibration, paths["calibration_debug"])


def calibration_summary(calibration: BoardCalibration, video_path: Path) -> Dict:
    cluster_sizes = [int(info["cluster_size"]) for info in calibration.grid_info if "cluster_size" in info]
    match_counts = [int(info["num_matches"]) for info in calibration.grid_info if "num_matches" in info]
    mean_errors = [float(info["mean_error_px"]) for info in calibration.grid_info if "mean_error_px" in info]

    return {
        "video": str(video_path),
        "calibrated": bool(calibration.calibrated),
        "holes_detected": int(len(calibration.holes)),
        "candidate_holes_before_grid_filter": int(calibration.candidate_count),
        "grid_count": int(len(calibration.grid_info)) if calibration.calibrated else 0,
        "cluster_sizes": cluster_sizes,
        "grid_match_counts": match_counts,
        "mean_grid_error_px": float(np.mean(mean_errors)) if mean_errors else np.nan,
        "mm_per_px": float(calibration.mm_per_px),
    }


def process_calibration_only(args: argparse.Namespace) -> Dict:
    video_path = resolve_video_path(args)
    output_root = Path(args.output_root)
    paths = create_output_paths(output_root, output_stem=args.output_stem)

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    print("Calibrating board and holes ...")
    calibration = build_calibration_from_args(video_path, args)

    if args.require_board and not calibration.calibrated:
        raise RuntimeError("Board calibration failed and --require-board was set.")

    write_calibration_outputs(calibration, paths)
    summary = calibration_summary(calibration, video_path)
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "Calibration: calibrated={} holes={} candidates={} mm_per_px={:.5f}".format(
            calibration.calibrated,
            len(calibration.holes),
            calibration.candidate_count,
            calibration.mm_per_px,
        )
    )
    print("\nSaved:")
    for key in ["summary_json", "calibration_json", "median_frame", "calibration_debug"]:
        print(f"  {key}: {paths[key]}")

    return summary


def process_video(args: argparse.Namespace) -> Dict:
    video_path = resolve_video_path(args)
    output_root = Path(args.output_root)
    paths = create_output_paths(output_root, output_stem=args.output_stem)

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    print("Calibrating board and holes ...")
    calibration = build_calibration_from_args(video_path, args)

    if args.require_board and not calibration.calibrated:
        raise RuntimeError("Board calibration failed and --require-board was set.")

    write_calibration_outputs(calibration, paths)

    print(
        "Calibration: calibrated={} holes={} candidates={} mm_per_px={:.5f}".format(
            calibration.calibrated,
            len(calibration.holes),
            calibration.candidate_count,
            calibration.mm_per_px,
        )
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or calibration.width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or calibration.height)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: {video_path}")
    print(f"Size: {width}x{height}, fps={fps:.2f}, frames={frame_count}")

    writer = cv2.VideoWriter(
        str(paths["video"]),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {paths['video']}")

    mp_hands = mp.solutions.hands
    kalman = CenterKalman(process_noise=args.kalman_process_noise, measurement_noise=args.kalman_measurement_noise)
    filters = TemporalPointFilters(ema_alpha=args.ema_alpha, ma_window=args.ma_window)
    pin_layer_enabled = not bool(args.disable_pins)
    occupancy_filter = OccupancyStabilizer(
        len(calibration.holes),
        alpha=args.occupancy_alpha,
        occupied_threshold=args.occupancy_probability_threshold,
        empty_threshold=args.occupancy_empty_threshold,
        interaction_alpha_scale=args.occupancy_interaction_alpha_scale,
        interaction_memory_frames=args.occupancy_interaction_memory_frames,
        require_interaction_to_occupy=not args.allow_occupancy_without_hand,
        settle_frames=args.occupancy_settle_frames,
        stable_occupied_frames=args.occupancy_stable_occupied_frames,
        stable_empty_frames=args.occupancy_stable_empty_frames,
    )

    board_center = None
    if calibration.board_quad is not None:
        board_center_arr = np.mean(calibration.board_quad, axis=0)
        board_center = (float(board_center_arr[0]), float(board_center_arr[1]))
    elif calibration.holes:
        board_center = (float(np.mean([h.x_px for h in calibration.holes])), float(np.mean([h.y_px for h in calibration.holes])))

    rows: List[Dict] = []
    trajectory_px: List[Tuple[int, int]] = []
    prev_gray = None
    prev_velocity_center_mm: Optional[Tuple[float, float]] = None
    prev_velocity_time: Optional[float] = None
    prev_frame_time: Optional[float] = None
    speed_ema = np.nan
    velocity_window_frames = max(1, int(args.velocity_window_frames))
    acceleration_window_frames = max(1, int(args.acceleration_window_frames))
    history_size = max(velocity_window_frames, acceleration_window_frames) + 8
    velocity_history: Deque[Tuple[int, float, float, float]] = deque(maxlen=history_size)
    speed_history: Deque[Tuple[int, float, float]] = deque(maxlen=history_size)
    cumulative_distance_mm = 0.0
    missing_frames = 0
    detected_frames = 0
    first_detect_time = np.nan
    last_detect_time = np.nan

    show_ok = bool(args.show)
    if show_ok:
        try:
            cv2.namedWindow("calibrated tracking", cv2.WINDOW_NORMAL)
        except cv2.error:
            show_ok = False
            print("GUI display is not available; continuing without --show.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            time_s = frame_idx / fps
            dt = 1.0 / fps if prev_frame_time is None else max(1.0 / fps, time_s - prev_frame_time)
            prev_frame_time = time_s
            predicted_center = kalman.predict(dt)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)

            previous_for_selection = None
            if trajectory_px:
                previous_for_selection = (float(trajectory_px[-1][0]), float(trajectory_px[-1][1]))

            selected_hand = select_active_hand(
                result=result,
                width=width,
                height=height,
                center_mode=args.center_mode,
                previous_center=previous_for_selection,
                predicted_center=predicted_center,
                board_center=board_center,
                active_side=args.active_side,
            )

            raw_center: Optional[Tuple[float, float]] = None
            hand_detected = selected_hand is not None
            measurement_rejected = False
            if selected_hand is not None:
                raw_center = (float(selected_hand.center[0]), float(selected_hand.center[1]))

            max_jump_px = max(
                args.max_jump_px,
                (args.max_hand_speed_mm_s * dt) / max(1e-6, calibration.mm_per_px),
            )

            accepted_measurement = raw_center
            if raw_center is not None and predicted_center is not None:
                jump = float(np.hypot(raw_center[0] - predicted_center[0], raw_center[1] - predicted_center[1]))
                if jump > max_jump_px:
                    measurement_rejected = True
                    accepted_measurement = None

            if accepted_measurement is not None:
                final_center = kalman.correct(accepted_measurement)
                missing_frames = 0
                detected_frames += 1
                if not np.isfinite(first_detect_time):
                    first_detect_time = time_s
                last_detect_time = time_s
            else:
                missing_frames += 1
                final_center = predicted_center if predicted_center is not None and missing_frames <= args.max_missing_frames else None

            ema_center, ma_center = filters.update(accepted_measurement)

            raw_mm = transform_point(raw_center, calibration.image_to_mm_h, calibration.mm_per_px) if raw_center is not None else (np.nan, np.nan)
            final_mm = transform_point(final_center, calibration.image_to_mm_h, calibration.mm_per_px) if final_center is not None else (np.nan, np.nan)
            ema_mm = transform_point(ema_center, calibration.image_to_mm_h, calibration.mm_per_px) if ema_center is not None else (np.nan, np.nan)
            ma_mm = transform_point(ma_center, calibration.image_to_mm_h, calibration.mm_per_px) if ma_center is not None else (np.nan, np.nan)

            speed_mm_s = np.nan
            acceleration_mm_s2 = np.nan
            segment_distance_mm = np.nan
            velocity_center = final_center
            velocity_center_mm = final_mm
            if args.velocity_center_source == "ema" and ema_center is not None:
                velocity_center = ema_center
                velocity_center_mm = ema_mm
            elif args.velocity_center_source == "ma" and ma_center is not None:
                velocity_center = ma_center
                velocity_center_mm = ma_mm

            if velocity_center is not None and np.isfinite(velocity_center_mm[0]) and np.isfinite(velocity_center_mm[1]):
                if prev_velocity_center_mm is not None and prev_velocity_time is not None:
                    segment_distance_mm = float(
                        np.hypot(
                            velocity_center_mm[0] - prev_velocity_center_mm[0],
                            velocity_center_mm[1] - prev_velocity_center_mm[1],
                        )
                    )
                    segment_dt = max(1e-6, time_s - prev_velocity_time)
                    segment_speed = segment_distance_mm / segment_dt
                    if segment_speed <= args.max_hand_speed_mm_s:
                        cumulative_distance_mm += segment_distance_mm
                    else:
                        segment_distance_mm = np.nan

                velocity_history.append(
                    (
                        int(frame_idx),
                        float(time_s),
                        float(velocity_center_mm[0]),
                        float(velocity_center_mm[1]),
                    )
                )

                if len(velocity_history) > velocity_window_frames:
                    old_frame, old_time, old_x, old_y = velocity_history[-(velocity_window_frames + 1)]
                    window_dt = max(1e-6, time_s - old_time)
                    window_distance = float(np.hypot(velocity_center_mm[0] - old_x, velocity_center_mm[1] - old_y))
                    candidate_speed = window_distance / window_dt
                    if candidate_speed <= args.max_hand_speed_mm_s:
                        speed_mm_s = float(candidate_speed)
                        if np.isnan(speed_ema):
                            speed_ema = speed_mm_s
                        else:
                            speed_ema = args.speed_ema_alpha * speed_mm_s + (1.0 - args.speed_ema_alpha) * speed_ema
                        speed_history.append((int(frame_idx), float(time_s), float(speed_ema)))

                        if len(speed_history) > acceleration_window_frames:
                            _, old_speed_time, old_speed = speed_history[-(acceleration_window_frames + 1)]
                            accel_dt = max(1e-6, time_s - old_speed_time)
                            candidate_accel = float((speed_ema - old_speed) / accel_dt)
                            if abs(candidate_accel) <= args.max_acceleration_mm_s2:
                                acceleration_mm_s2 = candidate_accel
                prev_velocity_center_mm = (float(velocity_center_mm[0]), float(velocity_center_mm[1]))
                prev_velocity_time = time_s

            if final_center is not None and np.isfinite(final_mm[0]) and np.isfinite(final_mm[1]):
                trajectory_px.append((int(round(final_center[0])), int(round(final_center[1]))))
                if len(trajectory_px) > args.max_trajectory_points:
                    trajectory_px = trajectory_px[-args.max_trajectory_points :]

            hand_for_pin_logic = selected_hand if accepted_measurement is not None else None
            if pin_layer_enabled:
                pins, prev_gray = detect_pins(
                    frame_bgr=frame,
                    median_frame=calibration.median_frame,
                    prev_gray=prev_gray,
                    board_mask=calibration.board_mask,
                    calibration=calibration,
                    min_radius=args.min_pin_radius,
                    max_radius=args.max_pin_radius,
                    bright_threshold=args.pin_bright_threshold,
                    diff_threshold=args.pin_diff_threshold,
                    motion_threshold=args.pin_motion_threshold,
                )

                raw_occ_probs, raw_occupied, occ_features = estimate_hole_occupancy(
                    frame_bgr=frame,
                    median_frame=calibration.median_frame,
                    holes=calibration.holes,
                    pins=pins,
                    selected_hand=hand_for_pin_logic,
                    radius_multiplier=args.occupancy_radius_multiplier,
                    brightness_drop_threshold=args.occupancy_brightness_drop,
                    dark_fraction_threshold=args.occupancy_dark_fraction,
                    relative_dark_threshold=args.occupancy_relative_dark_drop,
                    side_min_reference_value=args.occupancy_side_min_reference,
                    hand_interaction_radius_px=args.hand_hole_interaction_radius_px,
                )
                hole_interactions = [
                    bool(item.get("hand_interaction", False) or item.get("pin_near", False))
                    for item in occ_features
                ]
                if raw_occ_probs:
                    occ_probs, occupied = occupancy_filter.update_stateful(raw_occ_probs, interaction_mask=hole_interactions)
                else:
                    occ_probs = np.asarray([], dtype=np.float32)
                    occupied = []
            else:
                pins = []
                occ_features = []
                occ_probs = np.asarray([], dtype=np.float32)
                occupied = []
                hole_interactions = []

            if measurement_rejected:
                hand_status = "rejected_jump"
            elif hand_detected and accepted_measurement is not None:
                hand_status = "detected"
            elif final_center is not None:
                hand_status = "predicted"
            else:
                hand_status = "missing"

            annotated = draw_frame(
                frame_bgr=frame,
                calibration=calibration,
                selected_hand=selected_hand if accepted_measurement is not None else None,
                raw_center=raw_center,
                final_center=final_center,
                ema_center=ema_center,
                pins=pins,
                occupancy_probs=occ_probs,
                occupied=occupied,
                hole_interactions=hole_interactions,
                trajectory=trajectory_px,
                frame_idx=frame_idx,
                time_s=time_s,
                speed_mm_s=speed_ema,
                cumulative_distance_mm=cumulative_distance_mm,
                hand_status=hand_status,
                pin_layer_enabled=pin_layer_enabled,
            )
            writer.write(annotated)

            active_pin = None
            if final_center is not None and pins:
                active_pin = min(pins, key=lambda p: float(np.hypot(p.x_px - final_center[0], p.y_px - final_center[1])))

            row = {
                "frame": int(frame_idx),
                "time_s": float(time_s),
                "hand_status": hand_status,
                "hand_detected": int(hand_detected and accepted_measurement is not None),
                "hand_predicted": int(final_center is not None and accepted_measurement is None),
                "measurement_rejected": int(measurement_rejected),
                "raw_x_px": float(raw_center[0]) if raw_center is not None else np.nan,
                "raw_y_px": float(raw_center[1]) if raw_center is not None else np.nan,
                "kalman_x_px": float(final_center[0]) if final_center is not None else np.nan,
                "kalman_y_px": float(final_center[1]) if final_center is not None else np.nan,
                "ema_x_px": float(ema_center[0]) if ema_center is not None else np.nan,
                "ema_y_px": float(ema_center[1]) if ema_center is not None else np.nan,
                "ma_x_px": float(ma_center[0]) if ma_center is not None else np.nan,
                "ma_y_px": float(ma_center[1]) if ma_center is not None else np.nan,
                "raw_x_mm": float(raw_mm[0]),
                "raw_y_mm": float(raw_mm[1]),
                "hand_x_mm": float(final_mm[0]),
                "hand_y_mm": float(final_mm[1]),
                "ema_x_mm": float(ema_mm[0]),
                "ema_y_mm": float(ema_mm[1]),
                "ma_x_mm": float(ma_mm[0]),
                "ma_y_mm": float(ma_mm[1]),
                "velocity_center_source": str(args.velocity_center_source),
                "velocity_window_frames": int(velocity_window_frames),
                "acceleration_window_frames": int(acceleration_window_frames),
                "velocity_x_mm": float(velocity_center_mm[0]),
                "velocity_y_mm": float(velocity_center_mm[1]),
                "segment_distance_mm": float(segment_distance_mm) if np.isfinite(segment_distance_mm) else np.nan,
                "cumulative_distance_mm": float(cumulative_distance_mm),
                "speed_mm_s": float(speed_mm_s) if np.isfinite(speed_mm_s) else np.nan,
                "speed_ema_mm_s": float(speed_ema) if np.isfinite(speed_ema) else np.nan,
                "acceleration_mm_s2": float(acceleration_mm_s2) if np.isfinite(acceleration_mm_s2) else np.nan,
            }

            if pin_layer_enabled:
                row.update(
                    {
                        "pin_count": int(len(pins)),
                        "active_pin_x_px": float(active_pin.x_px) if active_pin is not None else np.nan,
                        "active_pin_y_px": float(active_pin.y_px) if active_pin is not None else np.nan,
                        "active_pin_x_mm": float(active_pin.x_mm) if active_pin is not None else np.nan,
                        "active_pin_y_mm": float(active_pin.y_mm) if active_pin is not None else np.nan,
                        "occupied_count": int(sum(occupied)),
                    }
                )

            for i, hole in enumerate(calibration.holes):
                row[f"hole_{i:02d}_x_mm"] = float(hole.x_mm)
                row[f"hole_{i:02d}_y_mm"] = float(hole.y_mm)
                if pin_layer_enabled:
                    prob = float(occ_probs[i]) if i < len(occ_probs) else np.nan
                    is_occupied = int(occupied[i]) if i < len(occupied) else 0
                    feature = occ_features[i] if i < len(occ_features) else {}
                    row[f"hole_{i:02d}_prob"] = prob
                    row[f"hole_{i:02d}_occupied"] = is_occupied
                    row[f"hole_{i:02d}_brightness_drop"] = float(feature.get("brightness_drop", np.nan))
                    row[f"hole_{i:02d}_relative_dark_drop"] = float(feature.get("relative_dark_drop", np.nan))
                    row[f"hole_{i:02d}_side_reference_value"] = float(feature.get("side_reference_value", np.nan))
                    row[f"hole_{i:02d}_dark_fraction"] = float(feature.get("dark_fraction", np.nan))
                    row[f"hole_{i:02d}_hand_interaction"] = int(bool(feature.get("hand_interaction", False)))
                    row[f"hole_{i:02d}_pin_near"] = int(bool(feature.get("pin_near", False)))
                    row[f"hole_{i:02d}_interaction_signal"] = int(bool(hole_interactions[i])) if i < len(hole_interactions) else 0
                    row[f"hole_{i:02d}_finger_distance_px"] = float(feature.get("finger_distance_px", np.nan))

            rows.append(row)

            if show_ok:
                try:
                    cv2.imshow("calibrated tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show_ok = False
                    print("GUI display failed; continuing without --show.")

            frame_idx += 1

    cap.release()
    writer.release()
    if show_ok:
        cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("No frames were processed.")

    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"], index=False)
    plot_results(df, paths, hole_count=len(calibration.holes))

    task_duration_s = float(last_detect_time - first_detect_time) if np.isfinite(first_detect_time) and np.isfinite(last_detect_time) else np.nan
    summary = {
        "video": str(video_path),
        "frames_processed": int(len(df)),
        "fps": float(fps),
        "calibrated": bool(calibration.calibrated),
        "holes_detected": int(len(calibration.holes)),
        "candidate_holes_before_grid_filter": int(calibration.candidate_count),
        "mm_per_px": float(calibration.mm_per_px),
        "detected_hand_frames": int(detected_frames),
        "hand_detection_rate": float(detected_frames / max(1, len(df))),
        "task_duration_s": task_duration_s,
        "total_distance_mm": float(cumulative_distance_mm),
        "velocity_center_source": str(args.velocity_center_source),
        "velocity_window_frames": int(velocity_window_frames),
        "acceleration_window_frames": int(acceleration_window_frames),
        "pin_layer_enabled": bool(pin_layer_enabled),
        "mean_speed_mm_s": float(df["speed_ema_mm_s"].dropna().mean()) if not df["speed_ema_mm_s"].dropna().empty else np.nan,
        "max_speed_mm_s": float(df["speed_ema_mm_s"].dropna().max()) if not df["speed_ema_mm_s"].dropna().empty else np.nan,
        "mean_pin_count": float(df["pin_count"].mean()) if pin_layer_enabled and "pin_count" in df else np.nan,
        "final_occupied_count": int(df["occupied_count"].iloc[-1]) if pin_layer_enabled and "occupied_count" in df else 0,
    }

    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    saved_keys = ["video", "csv", "summary_json", "calibration_json", "calibration_debug", "position_plot", "speed_plot", "accel_plot", "trajectory_plot"]
    if pin_layer_enabled:
        saved_keys.append("occupancy_plot")
    for key in saved_keys:
        print(f"  {key}: {paths[key]}")
    print("\nSummary:")
    print(f"  frames: {summary['frames_processed']}")
    print(f"  calibrated: {summary['calibrated']} holes={summary['holes_detected']}")
    print(f"  hand detection rate: {summary['hand_detection_rate']:.3f}")
    print(f"  total distance: {summary['total_distance_mm']:.2f} mm")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrated 9HPT board, hand, hole occupancy, and pin tracking in one pipeline."
    )
    parser.add_argument("--video", type=str, default=None, help="Input .mp4 video.")
    parser.add_argument("--data-root", type=str, default="../data", help="Data root used when --video is omitted.")
    parser.add_argument("--output-root", type=str, default="outputs_calibrated", help="Output directory.")
    parser.add_argument("--output-stem", type=str, default="calibrated_hand_pins", help="Base name for output video and CSV.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process. 0 means full video.")
    parser.add_argument("--calibration-only", action="store_true", help="Only calibrate the side 3x3 hole grids and save debug outputs.")
    parser.add_argument("--show", action="store_true", help="Display annotated video while processing.")

    parser.add_argument("--frame-samples", type=int, default=120, help="Initial frames used for median calibration.")
    parser.add_argument("--require-board", action="store_true", help="Fail if no valid 3x3 grid is found.")
    parser.add_argument("--hole-spacing-mm", type=float, default=32.0, help="Known spacing between neighboring holes.")
    parser.add_argument("--num-grids", type=int, default=2, help="Number of side 3x3 grids to find. Use 1 only for fallback/debug.")
    parser.add_argument("--min-hole-radius", type=float, default=2.8)
    parser.add_argument("--max-hole-radius", type=float, default=9.0)
    parser.add_argument("--min-hole-dist", type=int, default=24)
    parser.add_argument("--hough-param1", type=float, default=80)
    parser.add_argument("--hough-param2", type=float, default=12)
    parser.add_argument("--dark-threshold", type=int, default=90)
    parser.add_argument("--bright-threshold", type=int, default=170)
    parser.add_argument("--min-spacing-px", type=float, default=16)
    parser.add_argument("--max-spacing-px", type=float, default=44)
    parser.add_argument("--grid-tolerance-ratio", type=float, default=0.45)
    parser.add_argument("--min-grid-matches", type=int, default=7)
    parser.add_argument("--max-grid-candidates", type=int, default=160)
    parser.add_argument("--side-gray-threshold", type=int, default=170)
    parser.add_argument("--side-value-threshold", type=int, default=180)
    parser.add_argument("--side-min-area", type=float, default=15.0)
    parser.add_argument("--side-max-area", type=float, default=130.0)
    parser.add_argument("--side-min-circularity", type=float, default=0.25)
    parser.add_argument("--side-cluster-link-px", type=float, default=45.0)
    parser.add_argument("--side-min-grid-separation-px", type=float, default=90.0)

    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--active-side", type=str, default="auto", choices=["auto", "left", "right"])
    parser.add_argument("--center-mode", type=str, default="palm", choices=["palm", "bbox", "all"])
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.45)
    parser.add_argument("--kalman-process-noise", type=float, default=0.04)
    parser.add_argument("--kalman-measurement-noise", type=float, default=8.0)
    parser.add_argument("--ema-alpha", type=float, default=0.45)
    parser.add_argument("--ma-window", type=int, default=DEFAULT_HAND_CENTER_MA_WINDOW_FRAMES)
    parser.add_argument("--velocity-center-source", type=str, default="ma", choices=["kalman", "ema", "ma"], help="Center used for speed and distance; hand overlay still uses Kalman.")
    parser.add_argument("--velocity-window-frames", type=int, default=DEFAULT_VELOCITY_WINDOW_FRAMES, help="Frames used for speed calculation.")
    parser.add_argument("--acceleration-window-frames", type=int, default=DEFAULT_ACCELERATION_WINDOW_FRAMES, help="Frames used for acceleration calculation.")
    parser.add_argument("--max-missing-frames", type=int, default=5)
    parser.add_argument("--max-jump-px", type=float, default=90.0)
    parser.add_argument("--max-hand-speed-mm-s", type=float, default=2500.0)
    parser.add_argument("--max-acceleration-mm-s2", type=float, default=5000.0)
    parser.add_argument("--speed-ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-trajectory-points", type=int, default=1200)

    parser.add_argument("--disable-pins", action="store_true", help="Disable pin and occupancy detection; only track calibrated hand motion.")
    parser.add_argument("--min-pin-radius", type=int, default=3)
    parser.add_argument("--max-pin-radius", type=int, default=18)
    parser.add_argument("--pin-bright-threshold", type=int, default=165)
    parser.add_argument("--pin-diff-threshold", type=int, default=24)
    parser.add_argument("--pin-motion-threshold", type=int, default=14)

    parser.add_argument("--occupancy-radius-multiplier", type=float, default=1.6)
    parser.add_argument("--occupancy-alpha", type=float, default=0.35)
    parser.add_argument("--occupancy-brightness-drop", type=float, default=24.0, help="Mean value drop that suggests a pin blocks the illuminated hole.")
    parser.add_argument("--occupancy-dark-fraction", type=float, default=0.22, help="Fraction of pixels that must darken inside a hole.")
    parser.add_argument("--occupancy-relative-dark-drop", type=float, default=22.0, help="How much darker a hole must be than other holes on the same side grid.")
    parser.add_argument("--occupancy-side-min-reference", type=float, default=90.0, help="Minimum same-side reference brightness before relative darkness is trusted.")
    parser.add_argument("--occupancy-probability-threshold", type=float, default=0.55)
    parser.add_argument("--occupancy-empty-threshold", type=float, default=0.35)
    parser.add_argument("--occupancy-interaction-alpha-scale", type=float, default=0.20, help="Lower update speed while thumb/index are above a hole.")
    parser.add_argument("--occupancy-interaction-memory-frames", type=int, default=DEFAULT_OCCUPANCY_INTERACTION_MEMORY_FRAMES, help="Frames after thumb/index interaction where a hole may change to occupied.")
    parser.add_argument("--occupancy-settle-frames", type=int, default=DEFAULT_OCCUPANCY_SETTLE_FRAMES, help="Frames to ignore after thumb/index leave a hole.")
    parser.add_argument("--occupancy-stable-occupied-frames", type=int, default=DEFAULT_OCCUPANCY_STABLE_OCCUPIED_FRAMES, help="Stable frames needed before a hole is marked occupied.")
    parser.add_argument("--occupancy-stable-empty-frames", type=int, default=DEFAULT_OCCUPANCY_STABLE_EMPTY_FRAMES, help="Stable frames needed before a hole is marked empty.")
    parser.add_argument("--allow-occupancy-without-hand", action="store_true", help="Allow brightness-only occupancy changes without recent thumb/index interaction.")
    parser.add_argument("--hand-hole-interaction-radius-px", type=float, default=28.0)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.calibration_only:
        process_calibration_only(args)
    else:
        process_video(args)


if __name__ == "__main__":
    main()
