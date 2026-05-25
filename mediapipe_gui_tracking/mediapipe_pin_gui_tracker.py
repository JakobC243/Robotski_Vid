from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd


Point = Tuple[float, float]

INDEX_TIP = 8
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]
CSV_COLUMNS = [
    "frame",
    "time_s",
    "index_x_px",
    "index_y_px",
    "index_x_mm",
    "index_y_mm",
    "speed_mm_s",
    "acceleration_mm_s2",
    "total_distance_mm",
    "active_field",
    "hole_id",
    "hole_side",
    "hole_occupied",
    "hand_detected",
]


@dataclass
class Hole:
    hole_id: int
    side: str
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
class HandObservation:
    detected: bool
    index_px: Optional[Point]
    palm_px: Optional[Point]
    point_px: Optional[Point]
    landmarks: object
    points: np.ndarray
    handedness: str
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
    x_mm: float
    y_mm: float


def finite_float(value: float, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value) if np.isfinite(value) else float(default)
    except TypeError:
        return float(default)


def get_default_video_path(data_root: Optional[Path] = None) -> Path:
    candidates = []
    if data_root is not None:
        candidates.append(data_root)
    script_root = Path(__file__).resolve().parent
    candidates.extend([Path("../data"), Path("data"), script_root.parent / "data"])
    for root in candidates:
        videos = sorted(Path(root).expanduser().resolve().rglob("*.mp4")) if Path(root).exists() else []
        if videos:
            return videos[0]
    raise FileNotFoundError("No .mp4 video found. Pass --input or place videos under ../data or data.")


def create_output_paths(output_dir: Path, stem: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "video": output_dir / f"{stem}_gui.mp4",
        "video_avi": output_dir / f"{stem}_gui.avi",
        "csv": output_dir / f"{stem}.csv",
        "summary_json": output_dir / f"{stem}_summary.json",
        "calibration_json": output_dir / f"{stem}_calibration.json",
        "calibration_debug": output_dir / f"{stem}_calibration_debug.png",
        "median_frame": output_dir / f"{stem}_median_frame.png",
        "speed_plot": output_dir / f"{stem}_speed.png",
        "accel_plot": output_dir / f"{stem}_acceleration.png",
        "distance_plot": output_dir / f"{stem}_distance.png",
        "occupancy_plot": output_dir / f"{stem}_occupancy.png",
        "trajectory_plot": output_dir / f"{stem}_trajectory.png",
    }


def read_median_frame(video_path: Path, frame_samples: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    while len(frames) < max(1, frame_samples):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames could be read for calibration.")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    xi = int(round(x))
    yi = int(round(y))
    return 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi] > 0)


def make_board_mask(shape: Tuple[int, int], quad: Optional[np.ndarray]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if quad is None:
        mask[:, :] = 255
    else:
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask


def transform_point(point: Point, homography: Optional[np.ndarray], mm_per_px: float) -> Point:
    if homography is None:
        return float(point[0] * mm_per_px), float(point[1] * mm_per_px)
    pts = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, homography)[0, 0]
    return float(out[0]), float(out[1])


def merge_close_points(points: List[Dict], merge_dist_px: float) -> List[Dict]:
    merged: List[Dict] = []
    for point in points:
        for existing in merged:
            dist = float(np.hypot(point["x"] - existing["x"], point["y"] - existing["y"]))
            if dist <= merge_dist_px:
                existing["x"] = 0.5 * (existing["x"] + point["x"])
                existing["y"] = 0.5 * (existing["y"] + point["y"])
                existing["r"] = max(existing["r"], point["r"])
                existing["area"] = max(existing["area"], point["area"])
                existing["score"] = max(existing["score"], point["score"])
                break
        else:
            merged.append(dict(point))
    return merged


def detect_bright_hole_candidates(
    image_bgr: np.ndarray,
    gray_threshold: int,
    value_threshold: int,
    min_radius: float,
    max_radius: float,
    min_area: float,
    max_area: float,
    min_circularity: float,
) -> Tuple[List[Dict], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    mask = np.where((gray >= gray_threshold) | (value >= value_threshold), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points: List[Dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue
        local_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(local_mask, [contour], -1, 255, -1)
        mean_value = float(cv2.mean(value, mask=local_mask)[0])
        points.append(
            {
                "x": float(x),
                "y": float(y),
                "r": float(radius),
                "area": area,
                "circularity": circularity,
                "mean_value": mean_value,
                "score": float(mean_value * circularity * area),
            }
        )

    points = merge_close_points(points, merge_dist_px=max(3.5, 1.2 * min_radius))
    points.sort(key=lambda item: (item["y"], item["x"]))
    return points, mask


def cluster_candidates(points: List[Dict], link_distance_px: float) -> List[List[int]]:
    if not points:
        return []
    coords = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32)
    visited = np.zeros(len(points), dtype=bool)
    clusters: List[List[int]] = []
    for start in range(len(points)):
        if visited[start]:
            continue
        queue: Deque[int] = deque([start])
        visited[start] = True
        cluster = []
        while queue:
            idx = queue.popleft()
            cluster.append(idx)
            dists = np.linalg.norm(coords - coords[idx].reshape(1, 2), axis=1)
            for neighbor in np.where((dists <= link_distance_px) & (~visited))[0]:
                visited[int(neighbor)] = True
                queue.append(int(neighbor))
        clusters.append(cluster)
    clusters.sort(key=len, reverse=True)
    return clusters


def nearest_unused(expected: np.ndarray, points: np.ndarray, used: set, tolerance_px: float) -> Tuple[Optional[int], float]:
    dists = np.linalg.norm(points - expected.reshape(1, 2), axis=1)
    for idx in np.argsort(dists):
        idx = int(idx)
        if idx in used:
            continue
        dist = float(dists[idx])
        if dist <= tolerance_px:
            return idx, dist
        return None, float("inf")
    return None, float("inf")


def score_grid(points: np.ndarray, p0: np.ndarray, u: np.ndarray, v: np.ndarray, min_matches: int) -> Optional[Dict]:
    spacing_u = float(np.linalg.norm(u))
    spacing_v = float(np.linalg.norm(v))
    if spacing_u <= 1e-6 or spacing_v <= 1e-6:
        return None
    spacing_ratio = max(spacing_u, spacing_v) / max(1e-6, min(spacing_u, spacing_v))
    if spacing_ratio > 1.75:
        return None
    cos_angle = abs(float(np.dot(u, v) / (spacing_u * spacing_v)))
    if cos_angle > 0.70:
        return None

    tolerance_px = max(6.0, 0.32 * 0.5 * (spacing_u + spacing_v))
    expected_points = []
    matched_points = []
    matched_indices = []
    errors = []
    used = set()
    for row in range(3):
        for col in range(3):
            expected = p0 + col * u + row * v
            expected_points.append(expected)
            idx, err = nearest_unused(expected, points, used, tolerance_px)
            if idx is None:
                continue
            used.add(idx)
            matched_indices.append(idx)
            matched_points.append(points[idx])
            errors.append(err)

    matches = len(matched_indices)
    if matches < min_matches:
        return None
    mean_error = float(np.mean(errors)) if errors else float("inf")
    score = matches * 1000.0 - mean_error * 60.0 - abs(spacing_u - spacing_v) * 6.0 - cos_angle * 160.0
    return {
        "score": float(score),
        "num_matches": int(matches),
        "mean_error_px": mean_error,
        "spacing_u_px": spacing_u,
        "spacing_v_px": spacing_v,
        "cos_angle": cos_angle,
        "p0": p0.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "expected_points": np.asarray(expected_points, dtype=np.float32),
        "matched_points": np.asarray(matched_points, dtype=np.float32),
        "matched_indices": matched_indices,
    }


def fit_best_3x3_grid(
    cluster_points: np.ndarray,
    min_spacing_px: float,
    max_spacing_px: float,
    min_matches: int,
) -> Optional[Dict]:
    if len(cluster_points) < min_matches:
        return None
    best = None
    n = len(cluster_points)
    for i in range(n):
        p0 = cluster_points[i]
        for j in range(n):
            if j == i:
                continue
            u = cluster_points[j] - p0
            du = float(np.linalg.norm(u))
            if du < min_spacing_px or du > max_spacing_px:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                v = cluster_points[k] - p0
                dv = float(np.linalg.norm(v))
                if dv < min_spacing_px or dv > max_spacing_px:
                    continue
                candidate = score_grid(cluster_points, p0, u, v, min_matches=min_matches)
                if candidate is not None and (best is None or candidate["score"] > best["score"]):
                    best = candidate
    return best


def normalize_grid_orientation(grid: Dict) -> Dict:
    arr = np.asarray(grid["expected_points"], dtype=np.float32).reshape(3, 3, 2)
    variants = []
    base_variants = [arr, np.transpose(arr, (1, 0, 2))]
    for base in base_variants:
        variants.extend(
            [
                base,
                np.flip(base, axis=0),
                np.flip(base, axis=1),
                np.flip(np.flip(base, axis=0), axis=1),
            ]
        )

    def variant_score(a: np.ndarray) -> float:
        ux = a[0, 2] - a[0, 0]
        vy = a[2, 0] - a[0, 0]
        return float(ux[0] + vy[1] - 0.25 * abs(ux[1]) - 0.25 * abs(vy[0]))

    best = max(variants, key=variant_score).copy()
    out = dict(grid)
    out["expected_points"] = best.reshape(9, 2).astype(np.float32)
    out["p0"] = best[0, 0].astype(np.float32)
    out["u"] = (best[0, 1] - best[0, 0]).astype(np.float32)
    out["v"] = (best[1, 0] - best[0, 0]).astype(np.float32)
    center = np.mean(out["expected_points"], axis=0)
    out["center"] = [float(center[0]), float(center[1])]
    return out


def select_two_3x3_grids(
    points: List[Dict],
    clusters: List[List[int]],
    min_spacing_px: float,
    max_spacing_px: float,
    min_matches: int,
    min_grid_separation_px: float,
) -> List[Dict]:
    grids = []
    all_points = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32)
    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) < min_matches:
            continue
        cluster_points = all_points[cluster]
        grid = fit_best_3x3_grid(cluster_points, min_spacing_px, max_spacing_px, min_matches)
        if grid is None:
            continue
        grid["cluster_id"] = int(cluster_id)
        grid["cluster_size"] = int(len(cluster))
        grid["point_indices"] = [int(cluster[i]) for i in grid["matched_indices"]]
        grids.append(normalize_grid_orientation(grid))

    grids.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for grid in grids:
        center = np.asarray(grid["center"], dtype=np.float32)
        if any(float(np.linalg.norm(center - np.asarray(other["center"], dtype=np.float32))) < min_grid_separation_px for other in selected):
            continue
        selected.append(grid)
        if len(selected) == 2:
            break
    selected.sort(key=lambda item: item["center"][0])
    return selected


def derive_board_quad_from_grids(grids: Sequence[Dict], width: int, height: int) -> Optional[np.ndarray]:
    if not grids:
        return None
    points = np.vstack([g["expected_points"] for g in grids]).astype(np.float32)
    spacings = []
    for grid in grids:
        spacings.extend([float(grid["spacing_u_px"]), float(grid["spacing_v_px"])])
    margin = max(25.0, 1.8 * float(np.median(spacings) if spacings else 20.0))
    rect = cv2.minAreaRect(points)
    center, size, angle = rect
    expanded = (min(width * 1.2, size[0] + 2 * margin), min(height * 1.2, size[1] + 2 * margin))
    box = cv2.boxPoints((center, expanded, angle)).astype(np.float32)
    box[:, 0] = np.clip(box[:, 0], 0, width - 1)
    box[:, 1] = np.clip(box[:, 1], 0, height - 1)
    return box


def build_board_calibration(video_path: Path, args: argparse.Namespace) -> BoardCalibration:
    median = read_median_frame(video_path, args.frame_samples)
    height, width = median.shape[:2]
    side_points, _ = detect_bright_hole_candidates(
        median,
        gray_threshold=args.side_gray_threshold,
        value_threshold=args.side_value_threshold,
        min_radius=args.min_hole_radius,
        max_radius=args.max_hole_radius,
        min_area=args.side_min_area,
        max_area=args.side_max_area,
        min_circularity=args.side_min_circularity,
    )
    clusters = cluster_candidates(side_points, link_distance_px=args.side_cluster_link_px)
    grids = select_two_3x3_grids(
        side_points,
        clusters,
        min_spacing_px=args.min_spacing_px,
        max_spacing_px=args.max_spacing_px,
        min_matches=args.min_grid_matches,
        min_grid_separation_px=args.side_min_grid_separation_px,
    )
    if len(grids) != 2:
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
                    "method": "bright_blob_3x3_grid",
                    "candidate_count": len(side_points),
                    "cluster_sizes": [len(c) for c in clusters],
                    "reason": "Expected two 3x3 grids, but calibration did not find both.",
                }
            ],
            candidate_count=len(side_points),
        )

    spacings = []
    for grid in grids:
        spacings.extend([float(grid["spacing_u_px"]), float(grid["spacing_v_px"])])
    spacing_px = float(np.median(spacings))
    mm_per_px = float(args.hole_spacing_mm / max(1e-6, spacing_px))
    px_per_mm = float(1.0 / max(1e-6, mm_per_px))

    left_grid = grids[0]
    image_points = left_grid["expected_points"].astype(np.float32)
    world_points = np.asarray(
        [[col * args.hole_spacing_mm, row * args.hole_spacing_mm] for row in range(3) for col in range(3)],
        dtype=np.float32,
    )
    image_to_mm_h, _ = cv2.findHomography(image_points, world_points, 0)
    mm_to_image_h, _ = cv2.findHomography(world_points, image_points, 0)
    board_quad = derive_board_quad_from_grids(grids, width, height)
    board_mask = make_board_mask((height, width), board_quad)

    holes: List[Hole] = []
    grid_info: List[Dict] = []
    hole_id = 0
    for grid_id, grid in enumerate(grids):
        side = "left" if grid_id == 0 else "right"
        point_indices = grid.get("point_indices", [])
        source_radii = [float(side_points[idx]["r"]) for idx in point_indices if 0 <= int(idx) < len(side_points)]
        default_radius = float(np.median(source_radii)) if source_radii else float(0.5 * (args.min_hole_radius + args.max_hole_radius))
        grid_info.append(
            {
                "grid_id": int(grid_id),
                "side": side,
                "method": "bright_blob_3x3_grid",
                "cluster_id": int(grid.get("cluster_id", -1)),
                "cluster_size": int(grid.get("cluster_size", 0)),
                "num_matches": int(grid.get("num_matches", 0)),
                "mean_error_px": float(grid["mean_error_px"]),
                "spacing_u_px": float(grid["spacing_u_px"]),
                "spacing_v_px": float(grid["spacing_v_px"]),
                "cos_angle": float(grid["cos_angle"]),
                "center": [float(v) for v in grid["center"]],
                "expected_points": [[float(p[0]), float(p[1])] for p in grid["expected_points"]],
            }
        )
        for local_id, point in enumerate(grid["expected_points"]):
            row = local_id // 3
            col = local_id % 3
            radius = default_radius
            matched = False
            if side_points:
                dists = [float(np.hypot(point[0] - p["x"], point[1] - p["y"])) for p in side_points]
                nearest = int(np.argmin(dists))
                if dists[nearest] <= max(7.0, 0.4 * spacing_px):
                    matched = True
                    radius = float(side_points[nearest]["r"])
            x_mm, y_mm = transform_point((float(point[0]), float(point[1])), image_to_mm_h, mm_per_px)
            holes.append(
                Hole(
                    hole_id=hole_id,
                    side=side,
                    grid_id=grid_id,
                    local_id=local_id,
                    row=row,
                    col=col,
                    x_px=float(point[0]),
                    y_px=float(point[1]),
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
                "side": h.side,
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


def calibration_from_json(data: Dict, median_frame: np.ndarray) -> BoardCalibration:
    height, width = median_frame.shape[:2]
    board_quad = np.asarray(data["board_quad"], dtype=np.float32) if data.get("board_quad") is not None else None
    holes = [
        Hole(
            hole_id=int(h["hole_id"]),
            side=str(h.get("side", "left" if int(h.get("grid_id", 0)) == 0 else "right")),
            grid_id=int(h["grid_id"]),
            local_id=int(h["local_id"]),
            row=int(h["row"]),
            col=int(h["col"]),
            x_px=float(h["x_px"]),
            y_px=float(h["y_px"]),
            r_px=float(h["r_px"]),
            x_mm=float(h["x_mm"]),
            y_mm=float(h["y_mm"]),
            matched=bool(h.get("matched", True)),
        )
        for h in data.get("holes", [])
    ]
    return BoardCalibration(
        calibrated=bool(data.get("calibrated", False)),
        width=width,
        height=height,
        median_frame=median_frame,
        board_quad=board_quad,
        board_mask=make_board_mask((height, width), board_quad),
        holes=holes,
        mm_per_px=float(data.get("mm_per_px", 1.0)),
        px_per_mm=float(data.get("px_per_mm", 1.0)),
        image_to_mm_h=np.asarray(data["image_to_mm_h"], dtype=np.float32) if data.get("image_to_mm_h") is not None else None,
        mm_to_image_h=np.asarray(data["mm_to_image_h"], dtype=np.float32) if data.get("mm_to_image_h") is not None else None,
        grid_info=list(data.get("grids", [])),
        candidate_count=int(data.get("candidate_count", 0)),
    )


def load_or_build_calibration(video_path: Path, args: argparse.Namespace, paths: Dict[str, Path]) -> BoardCalibration:
    cache_path = Path(args.calibration_cache) if args.calibration_cache else None
    if cache_path is not None and cache_path.exists():
        median = read_median_frame(video_path, args.frame_samples)
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data.get("width", median.shape[1])) == median.shape[1] and int(data.get("height", median.shape[0])) == median.shape[0]:
            return calibration_from_json(data, median)
        print("Calibration cache exists, but frame size differs. Recalibrating.")

    calibration = build_board_calibration(video_path, args)
    with open(paths["calibration_json"], "w", encoding="utf-8") as f:
        json.dump(calibration_to_json(calibration), f, indent=2)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(calibration_to_json(calibration), f, indent=2)
    return calibration


def draw_calibration_debug(calibration: BoardCalibration, output_path: Path) -> None:
    image = calibration.median_frame.copy()
    if calibration.board_quad is not None:
        cv2.polylines(image, [np.round(calibration.board_quad).astype(np.int32)], True, (255, 0, 0), 2)
    for hole in calibration.holes:
        color = (0, 220, 0) if hole.matched else (0, 180, 255)
        center = (int(round(hole.x_px)), int(round(hole.y_px)))
        cv2.circle(image, center, int(round(max(4, hole.r_px))), color, 2)
        cv2.circle(image, center, 2, (0, 0, 255), -1)
        cv2.putText(image, f"{hole.side[0]}{hole.local_id}", (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    text = f"calibrated={calibration.calibrated} holes={len(calibration.holes)} candidates={calibration.candidate_count} mm_per_px={calibration.mm_per_px:.4f}"
    cv2.putText(image, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image)


class MediaPipeHandTracker:
    def __init__(self, args: argparse.Namespace):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=args.max_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        self.previous_point: Optional[Point] = None

    def close(self) -> None:
        self.hands.close()

    @staticmethod
    def _landmarks_to_pixels(hand_landmarks, width: int, height: int) -> np.ndarray:
        pts = []
        for lm in hand_landmarks.landmark:
            pts.append([float(np.clip(lm.x * width, 0, width - 1)), float(np.clip(lm.y * height, 0, height - 1)), float(lm.z)])
        return np.asarray(pts, dtype=np.float32)

    def detect(self, frame_bgr: np.ndarray, predicted_point: Optional[Point] = None) -> HandObservation:
        height, width = frame_bgr.shape[:2]
        result = self.hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return HandObservation(False, None, None, None, None, np.empty((0, 3), dtype=np.float32), "", 0.0)

        handedness_list = result.multi_handedness or []
        candidates: List[HandObservation] = []
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            points = self._landmarks_to_pixels(hand_landmarks, width, height)
            index_px = (float(points[INDEX_TIP, 0]), float(points[INDEX_TIP, 1]))
            palm = np.mean(points[PALM_LANDMARKS, :2], axis=0)
            palm_px = (float(palm[0]), float(palm[1]))
            handedness = "unknown"
            hand_score = 0.0
            if idx < len(handedness_list) and handedness_list[idx].classification:
                classification = handedness_list[idx].classification[0]
                handedness = classification.label.lower()
                hand_score += float(classification.score)
            ref = self.previous_point or predicted_point
            if ref is not None:
                hand_score -= 0.025 * float(np.hypot(index_px[0] - ref[0], index_px[1] - ref[1]))
            candidates.append(HandObservation(True, index_px, palm_px, index_px, hand_landmarks, points, handedness, hand_score))

        candidates.sort(key=lambda obs: obs.score, reverse=True)
        obs = candidates[0]
        self.previous_point = obs.point_px
        return obs


class PointSmoother:
    def __init__(self, window: int, ema_alpha: float = 0.55):
        self.samples: Deque[Point] = deque(maxlen=max(1, int(window)))
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self.ema_value: Optional[Point] = None

    def update(self, point: Optional[Point]) -> Optional[Point]:
        if point is None:
            return self.ema_value
        p = (float(point[0]), float(point[1]))
        self.samples.append(p)
        arr = np.asarray(self.samples, dtype=np.float32)
        ma = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
        if self.ema_value is None:
            self.ema_value = ma
        else:
            self.ema_value = (
                self.ema_alpha * ma[0] + (1.0 - self.ema_alpha) * self.ema_value[0],
                self.ema_alpha * ma[1] + (1.0 - self.ema_alpha) * self.ema_value[1],
            )
        return self.ema_value


class OpticalFlowBackup:
    def __init__(self, max_lost_frames: int):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_point: Optional[Point] = None
        self.lost_frames = 0
        self.max_lost_frames = int(max_lost_frames)

    def update(self, gray: np.ndarray, measurement: Optional[Point]) -> Tuple[Optional[Point], str]:
        if measurement is not None:
            self.prev_gray = gray.copy()
            self.prev_point = (float(measurement[0]), float(measurement[1]))
            self.lost_frames = 0
            return self.prev_point, "mediapipe"
        if self.prev_gray is None or self.prev_point is None or self.lost_frames >= self.max_lost_frames:
            self.prev_gray = gray.copy()
            return self.prev_point, "last" if self.prev_point is not None else "missing"
        p0 = np.asarray([[self.prev_point]], dtype=np.float32)
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            p0,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self.prev_gray = gray.copy()
        self.lost_frames += 1
        if p1 is not None and status is not None and int(status[0, 0]) == 1:
            point = (float(p1[0, 0, 0]), float(p1[0, 0, 1]))
            self.prev_point = point
            return point, "lk_backup"
        return self.prev_point, "last"


class MotionEstimator:
    def __init__(self, smooth_window: int, speed_alpha: float = 0.45, accel_alpha: float = 0.40):
        self.points: Deque[Tuple[float, Point]] = deque(maxlen=max(2, int(smooth_window)))
        self.last_point: Optional[Point] = None
        self.last_speed: Optional[float] = None
        self.speed_ema: Optional[float] = None
        self.accel_ema: Optional[float] = None
        self.total_distance = 0.0
        self.speed_alpha = float(speed_alpha)
        self.accel_alpha = float(accel_alpha)

    def update(self, time_s: float, point_mm: Optional[Point]) -> Tuple[float, float, float]:
        if point_mm is None or not np.all(np.isfinite(point_mm)):
            return finite_float(self.speed_ema, np.nan), finite_float(self.accel_ema, np.nan), self.total_distance
        point = (float(point_mm[0]), float(point_mm[1]))
        if self.last_point is not None:
            step = float(np.hypot(point[0] - self.last_point[0], point[1] - self.last_point[1]))
            if np.isfinite(step):
                self.total_distance += step
        self.last_point = point
        self.points.append((float(time_s), point))

        raw_speed = np.nan
        if len(self.points) >= 2:
            t0, p0 = self.points[0]
            t1, p1 = self.points[-1]
            dt = max(1e-6, t1 - t0)
            raw_speed = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt)
            self.speed_ema = raw_speed if self.speed_ema is None else self.speed_alpha * raw_speed + (1.0 - self.speed_alpha) * self.speed_ema

        raw_accel = np.nan
        if self.speed_ema is not None and self.last_speed is not None and len(self.points) >= 2:
            dt = max(1e-6, self.points[-1][0] - self.points[-2][0])
            raw_accel = float((self.speed_ema - self.last_speed) / dt)
            self.accel_ema = raw_accel if self.accel_ema is None else self.accel_alpha * raw_accel + (1.0 - self.accel_alpha) * self.accel_ema
        if self.speed_ema is not None:
            self.last_speed = self.speed_ema
        return finite_float(self.speed_ema, np.nan), finite_float(self.accel_ema, np.nan), self.total_distance


class OccupancyStabilizer:
    def __init__(self, hole_count: int, alpha: float, occupied_threshold: float, empty_threshold: float):
        self.alpha = float(alpha)
        self.occupied_threshold = float(occupied_threshold)
        self.empty_threshold = float(empty_threshold)
        self.probabilities = np.zeros(hole_count, dtype=np.float32)
        self.states = np.zeros(hole_count, dtype=bool)
        self.initialized = False

    def update(self, raw_probs: Sequence[float]) -> Tuple[np.ndarray, List[bool], List[int]]:
        raw = np.asarray(raw_probs, dtype=np.float32)
        if raw.shape[0] != self.probabilities.shape[0]:
            self.probabilities = np.zeros(raw.shape[0], dtype=np.float32)
            self.states = np.zeros(raw.shape[0], dtype=bool)
            self.initialized = False
        previous = self.states.copy()
        if not self.initialized:
            self.probabilities = raw.copy()
            self.initialized = True
        else:
            self.probabilities = self.alpha * raw + (1.0 - self.alpha) * self.probabilities
        self.states = np.where(self.probabilities >= self.occupied_threshold, True, self.states)
        self.states = np.where(self.probabilities <= self.empty_threshold, False, self.states)
        changed = [int(i) for i, (a, b) in enumerate(zip(previous, self.states)) if bool(a) != bool(b)]
        return self.probabilities.copy(), [bool(v) for v in self.states], changed


def hand_hole_distance(hand: Optional[HandObservation], hole: Hole) -> float:
    if hand is None or hand.points.size == 0:
        return float("inf")
    ids = [3, 4, 7, 8]
    pts = hand.points[ids, :2]
    dists = np.linalg.norm(pts - np.asarray([[hole.x_px, hole.y_px]], dtype=np.float32), axis=1)
    return float(np.min(dists)) if dists.size else float("inf")


def detect_pins(
    frame_bgr: np.ndarray,
    median_frame: np.ndarray,
    prev_gray: Optional[np.ndarray],
    board_mask: np.ndarray,
    calibration: BoardCalibration,
    args: argparse.Namespace,
) -> Tuple[List[PinDetection], np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    median_gray = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    diff = cv2.absdiff(gray, median_gray)
    motion = np.zeros_like(gray) if prev_gray is None else cv2.absdiff(gray, prev_gray)
    bright = ((value >= args.pin_bright_threshold) | ((value >= args.pin_bright_threshold - 30) & (saturation >= 55))).astype(np.uint8) * 255
    changed = ((diff >= args.pin_diff_threshold) | (motion >= args.pin_motion_threshold)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(cv2.bitwise_and(bright, changed), board_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[PinDetection] = []
    min_area = math.pi * args.min_pin_radius * args.min_pin_radius
    max_area = math.pi * args.max_pin_radius * args.max_pin_radius
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < 0.20:
            continue
        (x, y), r = cv2.minEnclosingCircle(contour)
        if not (args.min_pin_radius <= r <= args.max_pin_radius):
            continue
        if not point_in_mask(board_mask, x, y):
            continue
        local_mask = np.zeros_like(gray)
        cv2.drawContours(local_mask, [contour], -1, 255, -1)
        x_mm, y_mm = transform_point((float(x), float(y)), calibration.image_to_mm_h, calibration.mm_per_px)
        detections.append(
            PinDetection(
                x_px=float(x),
                y_px=float(y),
                r_px=float(r),
                area_px=area,
                mean_value=float(cv2.mean(value, mask=local_mask)[0]),
                mean_diff=float(cv2.mean(diff, mask=local_mask)[0]),
                mean_motion=float(cv2.mean(motion, mask=local_mask)[0]),
                x_mm=x_mm,
                y_mm=y_mm,
            )
        )
    detections.sort(key=lambda p: p.mean_value * p.area_px, reverse=True)
    return detections, gray


def estimate_hole_occupancy(
    frame_bgr: np.ndarray,
    median_frame: np.ndarray,
    holes: List[Hole],
    pins: List[PinDetection],
    hand: Optional[HandObservation],
    args: argparse.Namespace,
) -> Tuple[List[float], List[Dict]]:
    if not holes:
        return [], []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    median_value = cv2.cvtColor(median_frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    diff = cv2.absdiff(gray, cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY))
    h, w = gray.shape[:2]

    features = []
    for hole in holes:
        radius = max(4, int(round(args.occupancy_radius_multiplier * hole.r_px)))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(round(hole.x_px)), int(round(hole.y_px))), radius, 255, -1)
        pixels = max(1, int(np.count_nonzero(mask)))
        local_current = value[mask > 0]
        local_baseline = median_value[mask > 0]
        local_diff = diff[mask > 0]
        if local_current.size:
            baseline_mean = float(np.mean(local_baseline))
            current_mean = float(np.mean(local_current))
            per_pixel_drop = local_baseline.astype(np.float32) - local_current.astype(np.float32)
            brightness_drop = float(baseline_mean - current_mean)
            brightness_ratio = float(current_mean / max(1.0, baseline_mean))
            dark_fraction = float(np.count_nonzero(per_pixel_drop >= args.occupancy_brightness_drop) / pixels)
            diff_fraction = float(np.count_nonzero(local_diff >= args.occupancy_brightness_drop) / pixels)
        else:
            baseline_mean = current_mean = brightness_drop = dark_fraction = diff_fraction = 0.0
            brightness_ratio = 1.0
        pin_near = any(float(np.hypot(pin.x_px - hole.x_px, pin.y_px - hole.y_px)) <= max(radius, 1.5 * pin.r_px) for pin in pins)
        finger_dist = hand_hole_distance(hand, hole)
        features.append(
            {
                "side": hole.side,
                "baseline_value": baseline_mean,
                "current_value": current_mean,
                "brightness_drop": brightness_drop,
                "brightness_ratio": brightness_ratio,
                "dark_fraction": dark_fraction,
                "diff_fraction": diff_fraction,
                "pin_near": bool(pin_near),
                "finger_distance_px": finite_float(finger_dist, np.nan),
                "hand_interaction": bool(finger_dist <= max(args.hand_hole_interaction_radius_px, 2.2 * radius)),
            }
        )

    side_reference: Dict[str, float] = {}
    for side in ["left", "right"]:
        vals = [f["current_value"] for f in features if f["side"] == side and np.isfinite(f["current_value"])]
        side_reference[side] = float(np.percentile(vals, 75)) if vals else 0.0

    raw_probs = []
    for feature in features:
        ref = side_reference.get(feature["side"], 0.0)
        relative_dark_drop = float(ref - feature["current_value"])
        relative_allowed = ref >= args.occupancy_side_min_reference
        drop_score = np.clip(feature["brightness_drop"] / max(1e-6, args.occupancy_brightness_drop), 0.0, 1.0)
        ratio_score = np.clip((0.88 - feature["brightness_ratio"]) / 0.28, 0.0, 1.0)
        dark_fraction_score = np.clip(feature["dark_fraction"] / max(1e-6, args.occupancy_dark_fraction), 0.0, 1.0)
        diff_score = np.clip(feature["diff_fraction"] / max(1e-6, args.occupancy_dark_fraction), 0.0, 1.0)
        relative_score = np.clip(relative_dark_drop / max(1e-6, args.occupancy_relative_dark_drop), 0.0, 1.0) if relative_allowed else 0.0
        prob = 0.30 * drop_score + 0.35 * relative_score + 0.15 * ratio_score + 0.15 * dark_fraction_score + 0.05 * diff_score
        prob += 0.05 if feature["pin_near"] else 0.0
        feature["side_reference_value"] = float(ref)
        feature["relative_dark_drop"] = relative_dark_drop
        feature["occupancy_raw_probability"] = float(np.clip(prob, 0.0, 1.0))
        raw_probs.append(feature["occupancy_raw_probability"])
    return raw_probs, features


class ActiveFieldDetector:
    def __init__(self, calibration: BoardCalibration, value_threshold: float, diff_threshold: float):
        self.calibration = calibration
        self.value_threshold = float(value_threshold)
        self.diff_threshold = float(diff_threshold)

    def _side_mean(self, value: np.ndarray, side: str) -> float:
        mask = np.zeros(value.shape, dtype=np.uint8)
        for hole in self.calibration.holes:
            if hole.side != side:
                continue
            cv2.circle(mask, (int(round(hole.x_px)), int(round(hole.y_px))), int(round(max(5, 1.7 * hole.r_px))), 255, -1)
        vals = value[mask > 0]
        if vals.size == 0:
            return 0.0
        return float(np.percentile(vals, 70))

    def detect(self, frame_bgr: np.ndarray) -> Tuple[str, Dict[str, float]]:
        value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        left = self._side_mean(value, "left")
        right = self._side_mean(value, "right")
        board_vals = value[self.calibration.board_mask > 0]
        dynamic_threshold = max(self.value_threshold, float(np.percentile(board_vals, 65)) if board_vals.size else self.value_threshold)
        left_on = left >= dynamic_threshold
        right_on = right >= dynamic_threshold
        diff = left - right
        if left_on and right_on and abs(diff) <= self.diff_threshold:
            active = "both"
        elif left_on and (diff > self.diff_threshold or not right_on):
            active = "left"
        elif right_on and (-diff > self.diff_threshold or not left_on):
            active = "right"
        elif abs(diff) > self.diff_threshold and max(left, right) > 0.75 * dynamic_threshold:
            active = "left" if diff > 0 else "right"
        else:
            active = "unknown"
        return active, {"left_value": left, "right_value": right, "threshold": dynamic_threshold}


class GuiVideoRenderer:
    def __init__(self, frame_width: int, frame_height: int, graph_width: int = 420, status_height: int = 74):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.graph_width = int(graph_width)
        self.status_height = int(status_height)
        self.width = self.graph_width + self.frame_width
        self.height = self.status_height + self.frame_height
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        calibration: BoardCalibration,
        hand: Optional[HandObservation],
        raw_point: Optional[Point],
        final_point: Optional[Point],
        pins: Sequence[PinDetection],
        occupancy_probs: Sequence[float],
        occupied: Sequence[bool],
        trajectory: Sequence[Tuple[int, int]],
        active_field: str,
    ) -> np.ndarray:
        image = frame_bgr.copy()
        if calibration.board_quad is not None:
            cv2.polylines(image, [np.round(calibration.board_quad).astype(np.int32)], True, (180, 180, 180), 2)

        for side in ["left", "right"]:
            pts = np.asarray([[h.x_px, h.y_px] for h in calibration.holes if h.side == side], dtype=np.float32)
            if pts.size:
                x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))
                color = (0, 255, 255) if active_field in [side, "both"] else (90, 90, 90)
                cv2.rectangle(image, (x - 12, y - 12), (x + w + 12, y + h + 12), color, 2)

        for idx, hole in enumerate(calibration.holes):
            is_occupied = bool(occupied[idx]) if idx < len(occupied) else False
            prob = float(occupancy_probs[idx]) if idx < len(occupancy_probs) else 0.0
            color = (0, 0, 255) if is_occupied else (0, 180, 0)
            center = (int(round(hole.x_px)), int(round(hole.y_px)))
            radius = int(round(max(5, 1.25 * hole.r_px)))
            cv2.circle(image, center, radius, color, 2)
            if is_occupied:
                cv2.circle(image, center, max(3, radius // 2), color, -1)
            cv2.putText(image, f"{hole.hole_id}:{prob:.1f}", (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

        for pin in pins:
            center = (int(round(pin.x_px)), int(round(pin.y_px)))
            cv2.circle(image, center, int(round(max(4, pin.r_px))), (255, 255, 0), 2)
            cv2.circle(image, center, 2, (0, 255, 255), -1)

        for a, b in zip(trajectory[:-1], trajectory[1:]):
            cv2.line(image, a, b, (0, 255, 255), 2)

        if hand is not None and hand.detected and hand.landmarks is not None:
            self.mp_draw.draw_landmarks(
                image,
                hand.landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

        if raw_point is not None:
            cv2.circle(image, (int(round(raw_point[0])), int(round(raw_point[1]))), 5, (255, 0, 255), -1)
        if final_point is not None:
            p = (int(round(final_point[0])), int(round(final_point[1])))
            cv2.circle(image, p, 8, (0, 0, 255), -1)
            cv2.circle(image, p, 14, (255, 255, 255), 2)
        return image

    @staticmethod
    def _draw_graph(canvas: np.ndarray, x: int, y: int, w: int, h: int, values: Sequence[float], label: str, color: Tuple[int, int, int]) -> None:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (38, 38, 38), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 90), 1)
        cv2.putText(canvas, label, (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)
        vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
        if vals.size < 2:
            return
        recent = vals[-180:]
        lo = float(np.percentile(recent, 5))
        hi = float(np.percentile(recent, 95))
        if abs(hi - lo) < 1e-6:
            lo -= 1.0
            hi += 1.0
        cv2.putText(canvas, f"{recent[-1]:.1f}", (x + w - 92, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        plot_x0, plot_y0 = x + 12, y + 34
        plot_w, plot_h = w - 24, h - 46
        cv2.line(canvas, (plot_x0, plot_y0 + plot_h), (plot_x0 + plot_w, plot_y0 + plot_h), (80, 80, 80), 1)
        cv2.line(canvas, (plot_x0, plot_y0), (plot_x0, plot_y0 + plot_h), (80, 80, 80), 1)
        pts = []
        denom = max(1, len(recent) - 1)
        for i, val in enumerate(recent):
            px = int(plot_x0 + i * plot_w / denom)
            py = int(plot_y0 + plot_h - np.clip((float(val) - lo) / (hi - lo), 0.0, 1.0) * plot_h)
            pts.append((px, py))
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)

    def compose(
        self,
        overlay: np.ndarray,
        histories: Dict[str, Sequence[float]],
        status: Dict[str, object],
    ) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (self.width, self.status_height), (34, 34, 34), -1)
        status_line = (
            f"frame {status['frame']}  time {status['time_s']:.2f}s  hand {status['hand_detected']}  "
            f"active {status['active_field']}  occupied {status['occupied_count']}  "
            f"speed {status['speed_mm_s']:.1f} mm/s  accel {status['acceleration_mm_s2']:.1f} mm/s2  "
            f"path {status['total_distance_mm']:.1f} mm"
        )
        cv2.putText(canvas, status_line, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
        graph_h = self.frame_height // 3
        self._draw_graph(canvas, 0, self.status_height, self.graph_width, graph_h, histories["speed"], "speed_mm_s", (80, 220, 120))
        self._draw_graph(canvas, 0, self.status_height + graph_h, self.graph_width, graph_h, histories["acceleration"], "acceleration_mm_s2", (80, 170, 255))
        self._draw_graph(canvas, 0, self.status_height + 2 * graph_h, self.graph_width, self.frame_height - 2 * graph_h, histories["distance"], "total_distance_mm", (255, 210, 90))
        canvas[self.status_height : self.status_height + self.frame_height, self.graph_width : self.graph_width + self.frame_width] = overlay
        return canvas


def open_video_writer(path: Path, fallback_path: Path, fps: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, Path]:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, path
    writer = cv2.VideoWriter(str(fallback_path), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not create output video writer.")
    return writer, fallback_path


def nearest_hole(calibration: BoardCalibration, point_px: Optional[Point], assignment_radius_mm: float) -> Optional[Hole]:
    if point_px is None or not calibration.holes:
        return None
    px_radius = assignment_radius_mm * calibration.px_per_mm
    dists = [float(np.hypot(point_px[0] - h.x_px, point_px[1] - h.y_px)) for h in calibration.holes]
    idx = int(np.argmin(dists))
    if dists[idx] <= px_radius:
        return calibration.holes[idx]
    return None


def save_plots(df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    def simple_plot(column: str, path: Path, ylabel: str) -> None:
        plt.figure(figsize=(10, 4))
        plt.plot(df["time_s"], df[column])
        plt.xlabel("time [s]")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    simple_plot("speed_mm_s", paths["speed_plot"], "speed [mm/s]")
    simple_plot("acceleration_mm_s2", paths["accel_plot"], "acceleration [mm/s^2]")
    simple_plot("total_distance_mm", paths["distance_plot"], "distance [mm]")
    if "occupied_count" in df:
        simple_plot("occupied_count", paths["occupancy_plot"], "occupied holes")
    valid = df.dropna(subset=["index_x_mm", "index_y_mm"])
    if not valid.empty:
        plt.figure(figsize=(6, 6))
        plt.plot(valid["index_x_mm"], valid["index_y_mm"], linewidth=1)
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(paths["trajectory_plot"], dpi=150)
        plt.close()


def resolve_input(args: argparse.Namespace) -> Path:
    return Path(args.input).expanduser().resolve() if args.input else get_default_video_path(Path(args.data_root))


def process_video(args: argparse.Namespace) -> Dict:
    video_path = resolve_input(args)
    paths = create_output_paths(Path(args.output_dir), args.output_stem)
    calibration = load_or_build_calibration(video_path, args, paths)
    cv2.imwrite(str(paths["median_frame"]), calibration.median_frame)
    draw_calibration_debug(calibration, paths["calibration_debug"])
    with open(paths["calibration_json"], "w", encoding="utf-8") as f:
        json.dump(calibration_to_json(calibration), f, indent=2)

    if not calibration.calibrated or len(calibration.holes) != 18:
        summary = {
            "video": str(video_path),
            "calibrated": bool(calibration.calibrated),
            "holes_detected": int(len(calibration.holes)),
            "candidate_count": int(calibration.candidate_count),
            "error": "Calibration failed: expected 18 holes from two 3x3 side grids.",
        }
        with open(paths["summary_json"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        raise RuntimeError(f"Calibration failed. Debug image saved to {paths['calibration_debug']}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or calibration.width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or calibration.height)
    renderer = GuiVideoRenderer(width, height, graph_width=args.graph_width)
    writer, actual_video_path = open_video_writer(paths["video"], paths["video_avi"], fps, (renderer.width, renderer.height))

    hand_tracker = MediaPipeHandTracker(args)
    point_smoother = PointSmoother(args.smooth_window, ema_alpha=args.position_ema_alpha)
    backup_tracker = OpticalFlowBackup(args.max_missing_frames)
    motion = MotionEstimator(args.smooth_window, speed_alpha=args.speed_ema_alpha, accel_alpha=args.acceleration_ema_alpha)
    occupancy_filter = OccupancyStabilizer(len(calibration.holes), args.occupancy_alpha, args.occupancy_probability_threshold, args.occupancy_empty_threshold)
    active_detector = ActiveFieldDetector(calibration, args.active_value_threshold, args.active_diff_threshold)

    rows = []
    speed_history: List[float] = []
    accel_history: List[float] = []
    distance_history: List[float] = []
    trajectory: Deque[Tuple[int, int]] = deque(maxlen=args.max_trajectory_points)
    prev_pin_gray: Optional[np.ndarray] = None
    last_valid_point: Optional[Point] = None
    last_time_s = 0.0
    detected_frames = 0
    occupancy_changes: List[Dict] = []
    show_ok = bool(args.show)

    try:
        frame_idx = 0
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            time_s = frame_idx / fps
            dt = max(1e-6, time_s - last_time_s) if frame_idx > 0 else 1.0 / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hand_obs = hand_tracker.detect(frame, predicted_point=last_valid_point)
            mp_point = hand_obs.point_px if hand_obs.detected else None
            detected_frames += int(hand_obs.detected)
            candidate_point, tracking_source = backup_tracker.update(gray, mp_point)

            rejected_jump = False
            if candidate_point is not None and last_valid_point is not None:
                max_step_px = max(args.max_jump_px, (args.max_hand_speed_mm_s * dt) / max(1e-6, calibration.mm_per_px))
                if float(np.hypot(candidate_point[0] - last_valid_point[0], candidate_point[1] - last_valid_point[1])) > max_step_px:
                    rejected_jump = True
                    candidate_point = last_valid_point
                    tracking_source = "rejected_jump"

            final_point = point_smoother.update(candidate_point)
            if final_point is not None:
                last_valid_point = final_point
                trajectory.append((int(round(final_point[0])), int(round(final_point[1]))))
            final_mm = transform_point(final_point, calibration.image_to_mm_h, calibration.mm_per_px) if final_point is not None else None
            speed, accel, total_distance = motion.update(time_s, final_mm)

            pins, prev_pin_gray = detect_pins(frame, calibration.median_frame, prev_pin_gray, calibration.board_mask, calibration, args)
            raw_occ_probs, occ_features = estimate_hole_occupancy(frame, calibration.median_frame, calibration.holes, pins, hand_obs if hand_obs.detected else None, args)
            occ_probs, occupied, changed = occupancy_filter.update(raw_occ_probs)
            for hole_idx in changed:
                occupancy_changes.append(
                    {
                        "frame": int(frame_idx),
                        "time_s": float(time_s),
                        "hole_id": int(calibration.holes[hole_idx].hole_id),
                        "side": calibration.holes[hole_idx].side,
                        "occupied": bool(occupied[hole_idx]),
                    }
                )

            active_field, active_info = active_detector.detect(frame)
            assigned_hole = nearest_hole(calibration, final_point, args.hole_assignment_radius_mm)
            if assigned_hole is None:
                hole_id = ""
                hole_side = ""
                hole_occupied = ""
            else:
                hole_id = int(assigned_hole.hole_id)
                hole_side = assigned_hole.side
                hole_occupied = int(bool(occupied[assigned_hole.hole_id]))

            row = {
                "frame": int(frame_idx),
                "time_s": float(time_s),
                "index_x_px": float(final_point[0]) if final_point is not None else np.nan,
                "index_y_px": float(final_point[1]) if final_point is not None else np.nan,
                "index_x_mm": float(final_mm[0]) if final_mm is not None else np.nan,
                "index_y_mm": float(final_mm[1]) if final_mm is not None else np.nan,
                "speed_mm_s": float(speed) if np.isfinite(speed) else np.nan,
                "acceleration_mm_s2": float(accel) if np.isfinite(accel) else np.nan,
                "total_distance_mm": float(total_distance),
                "active_field": active_field,
                "hole_id": hole_id,
                "hole_side": hole_side,
                "hole_occupied": hole_occupied,
                "hand_detected": int(hand_obs.detected and not rejected_jump),
                "tracking_source": tracking_source,
                "occupied_count": int(sum(occupied)),
                "pin_count": int(len(pins)),
                "active_left_value": float(active_info["left_value"]),
                "active_right_value": float(active_info["right_value"]),
            }
            for hole, prob, state in zip(calibration.holes, occ_probs, occupied):
                row[f"hole_{hole.hole_id:02d}_occupied"] = int(bool(state))
                row[f"hole_{hole.hole_id:02d}_prob"] = float(prob)
            rows.append(row)

            speed_history.append(float(speed) if np.isfinite(speed) else np.nan)
            accel_history.append(float(accel) if np.isfinite(accel) else np.nan)
            distance_history.append(float(total_distance))
            overlay = renderer.draw_overlay(
                frame,
                calibration,
                hand_obs if hand_obs.detected else None,
                raw_point=mp_point,
                final_point=final_point,
                pins=pins,
                occupancy_probs=occ_probs,
                occupied=occupied,
                trajectory=list(trajectory),
                active_field=active_field,
            )
            gui = renderer.compose(
                overlay,
                {"speed": speed_history, "acceleration": accel_history, "distance": distance_history},
                {
                    "frame": frame_idx,
                    "time_s": time_s,
                    "hand_detected": bool(hand_obs.detected),
                    "active_field": active_field,
                    "occupied_count": int(sum(occupied)),
                    "speed_mm_s": finite_float(speed, 0.0),
                    "acceleration_mm_s2": finite_float(accel, 0.0),
                    "total_distance_mm": total_distance,
                },
            )
            writer.write(gui)
            if show_ok:
                try:
                    cv2.imshow("mediapipe pin GUI tracker", gui)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show_ok = False
            last_time_s = time_s
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        hand_tracker.close()
        if show_ok:
            cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("No frames were processed.")

    df = pd.DataFrame(rows)
    for col in CSV_COLUMNS:
        if col not in df:
            df[col] = np.nan
    df.to_csv(paths["csv"], index=False, columns=CSV_COLUMNS + [c for c in df.columns if c not in CSV_COLUMNS])
    save_plots(df, paths)

    summary = {
        "video": str(video_path),
        "gui_video": str(actual_video_path),
        "frames_processed": int(len(df)),
        "fps": float(fps),
        "calibrated": bool(calibration.calibrated),
        "holes_detected": int(len(calibration.holes)),
        "candidate_holes_before_grid_filter": int(calibration.candidate_count),
        "mm_per_px": float(calibration.mm_per_px),
        "hand_detection_rate": float(detected_frames / max(1, len(df))),
        "total_distance_mm": float(df["total_distance_mm"].iloc[-1]),
        "mean_speed_mm_s": float(df["speed_mm_s"].dropna().mean()) if not df["speed_mm_s"].dropna().empty else np.nan,
        "max_speed_mm_s": float(df["speed_mm_s"].dropna().max()) if not df["speed_mm_s"].dropna().empty else np.nan,
        "final_occupied_count": int(df["occupied_count"].iloc[-1]) if "occupied_count" in df else 0,
        "occupancy_changes": occupancy_changes,
        "outputs": {k: str(v) for k, v in paths.items() if k != "video_avi"},
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    for key in ["video", "csv", "summary_json", "calibration_json", "calibration_debug", "speed_plot", "accel_plot", "distance_plot"]:
        path = actual_video_path if key == "video" else paths[key]
        print(f"  {key}: {path}")
    print(f"Summary: frames={summary['frames_processed']} hand_detection_rate={summary['hand_detection_rate']:.3f} total_distance={summary['total_distance_mm']:.1f} mm")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe GUI video tracker for two 3x3 peg fields.")
    parser.add_argument("--input", type=str, default=None, help="Input video. If omitted, the first mp4 under ../data or data is used.")
    parser.add_argument("--data-root", type=str, default="../data", help="Fallback data root used when --input is omitted.")
    parser.add_argument("--output-dir", type=str, default="mediapipe_gui_tracking/outputs")
    parser.add_argument("--output-stem", type=str, default="mediapipe_pin_gui_tracker")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process. 0 means all frames.")
    parser.add_argument("--debug", action="store_true", help="Keep verbose debug-oriented outputs.")
    parser.add_argument("--show", action="store_true", help="Preview GUI frames while processing.")
    parser.add_argument("--calibration-cache", type=str, default=None, help="Optional calibration JSON cache.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Short moving-average window for points and metrics.")

    parser.add_argument("--frame-samples", type=int, default=120)
    parser.add_argument("--hole-spacing-mm", type=float, default=32.0)
    parser.add_argument("--min-hole-radius", type=float, default=2.8)
    parser.add_argument("--max-hole-radius", type=float, default=9.0)
    parser.add_argument("--side-gray-threshold", type=int, default=170)
    parser.add_argument("--side-value-threshold", type=int, default=180)
    parser.add_argument("--side-min-area", type=float, default=15.0)
    parser.add_argument("--side-max-area", type=float, default=130.0)
    parser.add_argument("--side-min-circularity", type=float, default=0.25)
    parser.add_argument("--side-cluster-link-px", type=float, default=45.0)
    parser.add_argument("--side-min-grid-separation-px", type=float, default=90.0)
    parser.add_argument("--min-spacing-px", type=float, default=16.0)
    parser.add_argument("--max-spacing-px", type=float, default=44.0)
    parser.add_argument("--min-grid-matches", type=int, default=7)

    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.45)
    parser.add_argument("--position-ema-alpha", type=float, default=0.55)
    parser.add_argument("--speed-ema-alpha", type=float, default=0.40)
    parser.add_argument("--acceleration-ema-alpha", type=float, default=0.35)
    parser.add_argument("--max-missing-frames", type=int, default=6)
    parser.add_argument("--max-jump-px", type=float, default=95.0)
    parser.add_argument("--max-hand-speed-mm-s", type=float, default=2500.0)
    parser.add_argument("--max-trajectory-points", type=int, default=1400)

    parser.add_argument("--min-pin-radius", type=int, default=3)
    parser.add_argument("--max-pin-radius", type=int, default=18)
    parser.add_argument("--pin-bright-threshold", type=int, default=165)
    parser.add_argument("--pin-diff-threshold", type=int, default=24)
    parser.add_argument("--pin-motion-threshold", type=int, default=14)

    parser.add_argument("--occupancy-radius-multiplier", type=float, default=1.6)
    parser.add_argument("--occupancy-alpha", type=float, default=0.35)
    parser.add_argument("--occupancy-brightness-drop", type=float, default=24.0)
    parser.add_argument("--occupancy-dark-fraction", type=float, default=0.22)
    parser.add_argument("--occupancy-relative-dark-drop", type=float, default=22.0)
    parser.add_argument("--occupancy-side-min-reference", type=float, default=90.0)
    parser.add_argument("--occupancy-probability-threshold", type=float, default=0.55)
    parser.add_argument("--occupancy-empty-threshold", type=float, default=0.35)
    parser.add_argument("--hand-hole-interaction-radius-px", type=float, default=28.0)
    parser.add_argument("--hole-assignment-radius-mm", type=float, default=28.0)

    parser.add_argument("--active-value-threshold", type=float, default=145.0)
    parser.add_argument("--active-diff-threshold", type=float, default=18.0)
    parser.add_argument("--graph-width", type=int, default=420)
    return parser.parse_args(argv)


def main() -> None:
    process_video(parse_args())


if __name__ == "__main__":
    main()
