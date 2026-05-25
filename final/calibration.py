from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from utils import Point, ensure_parent, finite_point


BOARD_COORD_MARGIN = 0.0
ImageRegion = Union[Tuple[int, int, int, int], np.ndarray]


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def board_dimensions(corners: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = corners
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    right = np.linalg.norm(br - tr)
    left = np.linalg.norm(bl - tl)
    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    return max(1, width), max(1, height)


def roi_from_corners(corners: np.ndarray, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    x_min = int(max(0, np.floor(np.min(corners[:, 0]))))
    y_min = int(max(0, np.floor(np.min(corners[:, 1]))))
    x_max = int(min(frame_width - 1, np.ceil(np.max(corners[:, 0]))))
    y_max = int(min(frame_height - 1, np.ceil(np.max(corners[:, 1]))))
    return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)


def clip_roi_from_points(points: np.ndarray, frame_width: int, frame_height: int, margin: float) -> Tuple[int, int, int, int]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x_min = int(max(0, np.floor(float(np.min(pts[:, 0])) - margin)))
    y_min = int(max(0, np.floor(float(np.min(pts[:, 1])) - margin)))
    x_max = int(min(frame_width - 1, np.ceil(float(np.max(pts[:, 0])) + margin)))
    y_max = int(min(frame_height - 1, np.ceil(float(np.max(pts[:, 1])) + margin)))
    return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)


def start_region_from_hole_grids(
    grids: List[Dict],
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> Optional[np.ndarray]:
    valid_grids = []
    for grid in grids:
        points = np.asarray(grid.get("expected_points", []), dtype=np.float32).reshape(-1, 2)
        if points.shape[0] >= 9 and np.all(np.isfinite(points)):
            valid_grids.append((grid, points))
    if len(valid_grids) < 2:
        return None

    selected = valid_grids[:2]
    pts_by_grid = [points for _, points in selected]
    centers = np.asarray([np.mean(points, axis=0) for points in pts_by_grid], dtype=np.float32)
    center_delta = centers[1] - centers[0]
    center_distance = float(np.linalg.norm(center_delta))

    spacings = []
    for grid, _ in selected:
        for key in ("spacing_u_px", "spacing_v_px"):
            value = float(grid.get(key, 0.0))
            if np.isfinite(value) and value > 0.0:
                spacings.append(value)
    spacing = float(np.median(spacings)) if spacings else max(16.0, center_distance / 8.0)
    if center_distance <= max(1.0, 0.5 * spacing):
        return None

    along = (center_delta / center_distance).astype(np.float32)
    across = np.asarray([-along[1], along[0]], dtype=np.float32)
    all_points = np.vstack(pts_by_grid).astype(np.float32)
    along_proj = all_points @ along
    across_proj = all_points @ across
    along_span = float(np.ptp(along_proj))
    across_span = float(np.ptp(across_proj))
    padding = float(max(0.0, padding_ratio))

    along_pad = max(0.60 * spacing, padding * max(spacing, along_span))
    across_pad = max(0.85 * spacing, padding * max(spacing, across_span))
    along_pad = min(along_pad, max(2.20 * spacing, 0.25 * along_span))
    across_pad = min(across_pad, max(1.60 * spacing, 0.50 * across_span))

    along_min = float(np.min(along_proj) - along_pad)
    along_max = float(np.max(along_proj) + along_pad)
    across_min = float(np.min(across_proj) - across_pad)
    across_max = float(np.max(across_proj) + across_pad)

    polygon = np.asarray(
        [
            along_min * along + across_min * across,
            along_max * along + across_min * across,
            along_max * along + across_max * across,
            along_min * along + across_max * across,
        ],
        dtype=np.float32,
    )
    polygon[:, 0] = np.clip(polygon[:, 0], 0, frame_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, frame_height - 1)
    return polygon


def hole_grid_region_from_points(points: np.ndarray, spacing: float, padding_scale: float) -> Optional[np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4 or not np.all(np.isfinite(pts)):
        return None
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)
    center = np.mean(box, axis=0)
    pad = max(2.0, float(spacing) * float(max(0.0, padding_scale)))
    expanded = []
    for point in box:
        direction = point - center
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            expanded.append(point)
        else:
            expanded.append(point + direction / norm * pad)
    return order_corners(np.asarray(expanded, dtype=np.float32))


@dataclass
class CalibrationHole:
    hole_id: int
    grid_id: int
    local_id: int
    row: int
    col: int
    x_px: float
    y_px: float
    r_px: float
    matched: bool

    def to_dict(self) -> Dict:
        return {
            "hole_id": int(self.hole_id),
            "grid_id": int(self.grid_id),
            "local_id": int(self.local_id),
            "row": int(self.row),
            "col": int(self.col),
            "x_px": float(self.x_px),
            "y_px": float(self.y_px),
            "r_px": float(self.r_px),
            "matched": bool(self.matched),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationHole":
        return cls(
            hole_id=int(data.get("hole_id", 0)),
            grid_id=int(data.get("grid_id", 0)),
            local_id=int(data.get("local_id", 0)),
            row=int(data.get("row", 0)),
            col=int(data.get("col", 0)),
            x_px=float(data.get("x_px", float("nan"))),
            y_px=float(data.get("y_px", float("nan"))),
            r_px=float(data.get("r_px", 4.0)),
            matched=bool(data.get("matched", False)),
        )


def merge_close_points(points: List[Dict], merge_dist_px: float) -> List[Dict]:
    merged: List[Dict] = []
    for point in points:
        found = False
        for existing in merged:
            dist = float(np.hypot(point["x"] - existing["x"], point["y"] - existing["y"]))
            if dist <= merge_dist_px:
                existing["x"] = 0.5 * (existing["x"] + point["x"])
                existing["y"] = 0.5 * (existing["y"] + point["y"])
                existing["r"] = max(existing["r"], point["r"])
                existing["area"] = max(existing["area"], point["area"])
                existing["score"] = max(existing["score"], point["score"])
                found = True
                break
        if not found:
            merged.append(dict(point))
    return merged


def detect_bright_hole_candidates(
    image_bgr: np.ndarray,
    gray_threshold: int = 170,
    value_threshold: int = 180,
    min_radius: float = 2.8,
    max_radius: float = 9.0,
    min_area: float = 15.0,
    max_area: float = 130.0,
    min_circularity: float = 0.25,
) -> Tuple[List[Dict], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
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
                "score": mean_value * circularity * area,
            }
        )
    points = merge_close_points(points, merge_dist_px=max(3.5, 1.2 * min_radius))
    points.sort(key=lambda item: (item["y"], item["x"]))
    return points, mask


def cluster_hole_candidates(points: List[Dict], link_distance_px: float = 45.0) -> List[List[int]]:
    if not points:
        return []
    coords = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32)
    visited = np.zeros(len(points), dtype=bool)
    clusters: List[List[int]] = []
    for start in range(len(points)):
        if visited[start]:
            continue
        queue: deque[int] = deque([start])
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


def score_hole_grid(points: np.ndarray, p0: np.ndarray, u: np.ndarray, v: np.ndarray, min_matches: int) -> Optional[Dict]:
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
        "matches": int(matches),
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


def fit_best_hole_grid(cluster_points: np.ndarray, min_spacing_px: float, max_spacing_px: float, min_matches: int) -> Optional[Dict]:
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
                candidate = score_hole_grid(cluster_points, p0, u, v, min_matches=min_matches)
                if candidate is not None and (best is None or candidate["score"] > best["score"]):
                    best = candidate
    return best


def select_two_hole_grids(
    points: List[Dict],
    clusters: List[List[int]],
    min_spacing_px: float = 16.0,
    max_spacing_px: float = 44.0,
    min_matches: int = 7,
    min_grid_separation_px: float = 90.0,
) -> List[Dict]:
    grids = []
    all_points = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32)
    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) < min_matches:
            continue
        grid = fit_best_hole_grid(
            cluster_points=all_points[cluster],
            min_spacing_px=min_spacing_px,
            max_spacing_px=max_spacing_px,
            min_matches=min_matches,
        )
        if grid is None:
            continue
        grid["cluster_id"] = int(cluster_id)
        grid["cluster_size"] = int(len(cluster))
        center = np.mean(grid["expected_points"], axis=0)
        grid["center"] = np.asarray(center, dtype=np.float32)
        grid["point_indices"] = [int(cluster[i]) for i in grid["matched_indices"]]
        grids.append(grid)

    grids.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for grid in grids:
        center = np.asarray(grid["center"], dtype=np.float32)
        if any(float(np.linalg.norm(center - np.asarray(other["center"], dtype=np.float32))) < min_grid_separation_px for other in selected):
            continue
        selected.append(grid)
        if len(selected) == 2:
            break
    selected.sort(key=lambda item: (float(item["center"][0]), float(item["center"][1])))
    return selected


def grid_json_safe(grid: Dict) -> Dict:
    return {
        "grid_id": int(grid.get("grid_id", 0)),
        "cluster_id": int(grid.get("cluster_id", -1)),
        "cluster_size": int(grid.get("cluster_size", 0)),
        "matches": int(grid.get("matches", grid.get("num_matches", 0))),
        "mean_error_px": float(grid.get("mean_error_px", float("nan"))),
        "spacing_u_px": float(grid.get("spacing_u_px", float("nan"))),
        "spacing_v_px": float(grid.get("spacing_v_px", float("nan"))),
        "cos_angle": float(grid.get("cos_angle", float("nan"))),
        "center": np.asarray(grid.get("center", [float("nan"), float("nan")]), dtype=float).tolist(),
        "p0": np.asarray(grid.get("p0", [float("nan"), float("nan")]), dtype=float).tolist(),
        "u": np.asarray(grid.get("u", [float("nan"), float("nan")]), dtype=float).tolist(),
        "v": np.asarray(grid.get("v", [float("nan"), float("nan")]), dtype=float).tolist(),
        "expected_points": np.asarray(grid.get("expected_points", []), dtype=float).reshape(-1, 2).tolist(),
        "point_indices": [int(i) for i in grid.get("point_indices", [])],
    }


def grid_from_json(data: Dict) -> Dict:
    out = dict(data)
    for key in ("center", "p0", "u", "v"):
        if key in out:
            out[key] = np.asarray(out[key], dtype=np.float32)
    if "expected_points" in out:
        out["expected_points"] = np.asarray(out["expected_points"], dtype=np.float32).reshape(-1, 2)
    return out


def derive_quad_from_hole_grids(grids: List[Dict], width: int, height: int) -> Optional[np.ndarray]:
    if not grids:
        return None
    points = np.vstack([grid["expected_points"] for grid in grids]).astype(np.float32)
    spacings = []
    for grid in grids:
        spacings.extend([float(grid["spacing_u_px"]), float(grid["spacing_v_px"])])
    spacing = float(np.median(spacings)) if spacings else 20.0
    margin = max(22.0, 1.7 * spacing)
    roi = clip_roi_from_points(points, width, height, margin)
    x, y, w, h = roi
    return np.asarray([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


@dataclass
class BoardCalibration:
    frame_width: int
    frame_height: int
    board_corners_px: Optional[np.ndarray]
    homography_matrix: Optional[np.ndarray]
    inverse_homography_matrix: Optional[np.ndarray]
    board_roi: Optional[Tuple[int, int, int, int]]
    board_width: float
    board_height: float
    zones: Dict[str, Tuple[float, float, float, float]]
    holes: List[CalibrationHole]
    hole_grids: List[Dict]
    hole_candidate_count: int
    status: str
    source: str
    created_at: str
    video: str

    @property
    def calibrated(self) -> bool:
        return self.board_corners_px is not None and self.homography_matrix is not None

    def image_to_board(self, point: Optional[Point]) -> Point:
        if not self.calibrated or not finite_point(point):
            return float("nan"), float("nan")
        src = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.homography_matrix)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    def board_points_to_image(self, points: np.ndarray) -> Optional[np.ndarray]:
        if self.inverse_homography_matrix is None:
            return None
        pts = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        transformed = cv2.perspectiveTransform(pts, self.inverse_homography_matrix)[0]
        return transformed.astype(np.float32)

    def zone_image_roi(self, zone_name: str, padding_ratio: float = 0.15) -> Optional[Tuple[int, int, int, int]]:
        if not self.calibrated or zone_name not in self.zones:
            return None
        x, y, w, h = self.zones[zone_name]
        pad_x = float(w) * float(max(0.0, padding_ratio))
        pad_y = float(h) * float(max(0.0, padding_ratio)) * 0.35
        x1 = max(0.0, float(x) - pad_x)
        y1 = max(0.0, float(y) - pad_y)
        x2 = min(float(self.board_width), float(x + w) + pad_x)
        y2 = min(float(self.board_height), float(y + h) + pad_y)
        board_poly = np.asarray(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )
        image_poly = self.board_points_to_image(board_poly)
        if image_poly is None:
            return None
        x_min = int(max(0, np.floor(np.min(image_poly[:, 0]))))
        y_min = int(max(0, np.floor(np.min(image_poly[:, 1]))))
        x_max = int(min(self.frame_width - 1, np.ceil(np.max(image_poly[:, 0]))))
        y_max = int(min(self.frame_height - 1, np.ceil(np.max(image_poly[:, 1]))))
        return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)

    def start_zone_image_roi(self, padding_ratio: float = 0.30) -> Optional[ImageRegion]:
        if len(self.hole_grids) >= 2:
            region = start_region_from_hole_grids(
                self.hole_grids,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                padding_ratio=padding_ratio,
            )
            if region is not None:
                return region
        return self.zone_image_roi("center_zone", padding_ratio=padding_ratio)

    def hole_grid_regions(self, padding_scale: float = 0.75) -> List[Dict]:
        regions = []
        for grid in self.hole_grids:
            points = np.asarray(grid.get("expected_points", []), dtype=np.float32).reshape(-1, 2)
            if points.shape[0] < 9:
                continue
            spacings = [
                float(grid.get("spacing_u_px", 0.0)),
                float(grid.get("spacing_v_px", 0.0)),
            ]
            valid_spacings = [value for value in spacings if np.isfinite(value) and value > 0.0]
            spacing = float(np.median(valid_spacings)) if valid_spacings else 20.0
            polygon = hole_grid_region_from_points(points, spacing, padding_scale)
            if polygon is None:
                continue
            polygon[:, 0] = np.clip(polygon[:, 0], 0, self.frame_width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, self.frame_height - 1)
            center = np.mean(polygon, axis=0)
            regions.append(
                {
                    "grid_id": int(grid.get("grid_id", len(regions))),
                    "side": "",
                    "polygon": polygon.astype(np.float32),
                    "center": center.astype(np.float32),
                }
            )
        regions.sort(key=lambda item: (float(item["center"][0]), float(item["center"][1])))
        for idx, region in enumerate(regions):
            if len(regions) == 1:
                region["side"] = "unknown"
            elif idx == 0:
                region["side"] = "left"
            elif idx == len(regions) - 1:
                region["side"] = "right"
            else:
                region["side"] = f"grid_{idx}"
        return regions

    def to_dict(self) -> Dict:
        return {
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
            "board_corners_px": self.board_corners_px.tolist() if self.board_corners_px is not None else None,
            "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            "inverse_homography_matrix": self.inverse_homography_matrix.tolist() if self.inverse_homography_matrix is not None else None,
            "board_roi": list(self.board_roi) if self.board_roi is not None else None,
            "board_width": float(self.board_width),
            "board_height": float(self.board_height),
            "zones": {key: list(value) for key, value in self.zones.items()},
            "holes": [hole.to_dict() for hole in self.holes],
            "hole_grids": [grid_json_safe(grid) for grid in self.hole_grids],
            "hole_candidate_count": int(self.hole_candidate_count),
            "status": self.status,
            "source": self.source,
            "created_at": self.created_at,
            "video": self.video,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BoardCalibration":
        return cls(
            frame_width=int(data.get("frame_width", 0)),
            frame_height=int(data.get("frame_height", 0)),
            board_corners_px=np.asarray(data["board_corners_px"], dtype=np.float32) if data.get("board_corners_px") is not None else None,
            homography_matrix=np.asarray(data["homography_matrix"], dtype=np.float32) if data.get("homography_matrix") is not None else None,
            inverse_homography_matrix=np.asarray(data["inverse_homography_matrix"], dtype=np.float32) if data.get("inverse_homography_matrix") is not None else None,
            board_roi=tuple(int(v) for v in data["board_roi"]) if data.get("board_roi") is not None else None,
            board_width=float(data.get("board_width", 0.0)),
            board_height=float(data.get("board_height", 0.0)),
            zones={key: tuple(float(v) for v in value) for key, value in data.get("zones", {}).items()},
            holes=[CalibrationHole.from_dict(item) for item in data.get("holes", [])],
            hole_grids=[grid_from_json(item) for item in data.get("hole_grids", [])],
            hole_candidate_count=int(data.get("hole_candidate_count", 0)),
            status=str(data.get("status", "unknown")),
            source=str(data.get("source", "json")),
            created_at=str(data.get("created_at", "")),
            video=str(data.get("video", "")),
        )


def empty_calibration(frame: np.ndarray, status: str, source: str, video: str = "") -> BoardCalibration:
    h, w = frame.shape[:2]
    return BoardCalibration(
        frame_width=w,
        frame_height=h,
        board_corners_px=None,
        homography_matrix=None,
        inverse_homography_matrix=None,
        board_roi=None,
        board_width=0.0,
        board_height=0.0,
        zones={},
        holes=[],
        hole_grids=[],
        hole_candidate_count=0,
        status=status,
        source=source,
        created_at=datetime.now().isoformat(timespec="seconds"),
        video=video,
    )


def build_calibration_from_corners(frame: np.ndarray, corners: np.ndarray, source: str, video: str = "") -> BoardCalibration:
    h, w = frame.shape[:2]
    corners = order_corners(corners)
    board_w, board_h = board_dimensions(corners)
    dst = np.asarray(
        [
            [BOARD_COORD_MARGIN, BOARD_COORD_MARGIN],
            [board_w - BOARD_COORD_MARGIN, BOARD_COORD_MARGIN],
            [board_w - BOARD_COORD_MARGIN, board_h - BOARD_COORD_MARGIN],
            [BOARD_COORD_MARGIN, board_h - BOARD_COORD_MARGIN],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners, dst)
    inverse = cv2.getPerspectiveTransform(dst, corners)
    roi = roi_from_corners(corners, w, h)
    third = board_w / 3.0
    zones = {
        "left_target_zone": (0.0, 0.0, third, float(board_h)),
        "center_zone": (third, 0.0, third, float(board_h)),
        "right_target_zone": (2.0 * third, 0.0, third, float(board_h)),
    }
    return BoardCalibration(
        frame_width=w,
        frame_height=h,
        board_corners_px=corners,
        homography_matrix=homography,
        inverse_homography_matrix=inverse,
        board_roi=roi,
        board_width=float(board_w),
        board_height=float(board_h),
        zones=zones,
        holes=[],
        hole_grids=[],
        hole_candidate_count=0,
        status="calibrated",
        source=source,
        created_at=datetime.now().isoformat(timespec="seconds"),
        video=video,
    )


def build_calibration_from_hole_grids(frame: np.ndarray, video: str = "") -> Optional[BoardCalibration]:
    h, w = frame.shape[:2]
    side_points, _ = detect_bright_hole_candidates(frame)
    clusters = cluster_hole_candidates(side_points, link_distance_px=45.0)
    grids = select_two_hole_grids(
        side_points,
        clusters,
        min_spacing_px=16.0,
        max_spacing_px=44.0,
        min_matches=7,
        min_grid_separation_px=90.0,
    )
    if len(grids) < 2:
        return None

    quad = derive_quad_from_hole_grids(grids, width=w, height=h)
    if quad is None:
        return None
    calibration = build_calibration_from_corners(frame, quad, source="hole_grid", video=video)

    holes: List[CalibrationHole] = []
    hole_id = 0
    for grid_id, grid in enumerate(grids):
        grid["grid_id"] = int(grid_id)
        point_indices = [int(idx) for idx in grid.get("point_indices", [])]
        source_radii = [
            float(side_points[idx]["r"])
            for idx in point_indices
            if 0 <= int(idx) < len(side_points)
        ]
        default_radius = float(np.median(source_radii)) if source_radii else 5.0
        matched_indices = set(point_indices)
        for local_id, point in enumerate(np.asarray(grid["expected_points"], dtype=np.float32).reshape(-1, 2)):
            row = int(local_id // 3)
            col = int(local_id % 3)
            radius = default_radius
            matched = False
            if side_points:
                candidate_centers = np.asarray([[p["x"], p["y"]] for p in side_points], dtype=np.float32)
                dists = np.linalg.norm(candidate_centers - point.reshape(1, 2), axis=1)
                nearest_idx = int(np.argmin(dists))
                if nearest_idx in matched_indices or float(dists[nearest_idx]) <= max(7.0, 0.35 * default_radius + 5.0):
                    matched = True
                    radius = float(side_points[nearest_idx]["r"])
            holes.append(
                CalibrationHole(
                    hole_id=hole_id,
                    grid_id=int(grid_id),
                    local_id=int(local_id),
                    row=row,
                    col=col,
                    x_px=float(point[0]),
                    y_px=float(point[1]),
                    r_px=float(radius),
                    matched=matched,
                )
            )
            hole_id += 1

    calibration.holes = holes
    calibration.hole_grids = grids
    calibration.hole_candidate_count = len(side_points)
    calibration.source = "hole_grid"
    calibration.status = "calibrated"
    return calibration


def auto_detect_board_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    frame_area = float(h * w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (7, 7), 0)
    median = float(np.median(blurred))
    low = int(max(20, 0.66 * median))
    high = int(min(255, max(low + 30, 1.33 * median)))
    edges = cv2.Canny(blurred, low, high)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.06 * frame_area or area > 0.98 * frame_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
        corners = order_corners(corners)
        board_w, board_h = board_dimensions(corners)
        aspect = board_w / max(1.0, float(board_h))
        if not (0.25 <= aspect <= 4.0):
            continue
        rect_area = float(board_w * board_h)
        fill_ratio = area / max(1.0, rect_area)
        if fill_ratio < 0.25:
            continue
        score = area * min(1.0, fill_ratio) * (1.0 - min(0.5, abs(np.log(max(1e-6, aspect))) * 0.08))
        candidates.append((score, corners))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def manual_select_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    points = []
    display = frame.copy()
    window = "manual board calibration - click 4 corners, ENTER to accept, R to reset, ESC to cancel"

    def redraw() -> None:
        nonlocal display
        display = frame.copy()
        for idx, point in enumerate(points):
            cv2.circle(display, point, 5, (0, 255, 255), -1)
            cv2.putText(display, str(idx + 1), (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        if len(points) >= 2:
            cv2.polylines(display, [np.asarray(points, dtype=np.int32)], len(points) == 4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, "Click 4 board corners. ENTER=accept, R=reset, ESC=cancel", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x), int(y)))
            redraw()

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        redraw()
        while True:
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and len(points) == 4:
                return np.asarray(points, dtype=np.float32)
            if key in (27, ord("q")):
                return None
            if key in (ord("r"), ord("R")):
                points.clear()
                redraw()
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def calibrate_frame(frame: np.ndarray, video: str = "", allow_manual: bool = True) -> BoardCalibration:
    hole_calibration = build_calibration_from_hole_grids(frame, video=video)
    if hole_calibration is not None:
        return hole_calibration
    corners = auto_detect_board_corners(frame)
    if corners is not None:
        return build_calibration_from_corners(frame, corners, source="auto_contour", video=video)
    if allow_manual:
        manual = manual_select_corners(frame)
        if manual is not None:
            return build_calibration_from_corners(frame, manual, source="manual_click", video=video)
    return empty_calibration(frame, status="no_calibration", source="failed", video=video)


def load_calibration(path: Path) -> BoardCalibration:
    with open(path, "r", encoding="utf-8") as f:
        return BoardCalibration.from_dict(json.load(f))


def save_calibration(calibration: BoardCalibration, path: Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration.to_dict(), f, indent=2, allow_nan=True)


def draw_calibration(frame: np.ndarray, calibration: BoardCalibration) -> None:
    if calibration.calibrated and calibration.board_corners_px is not None:
        corners = np.round(calibration.board_corners_px).astype(np.int32)
        cv2.polylines(frame, [corners], True, (0, 220, 255), 2, cv2.LINE_AA)
        for idx, point in enumerate(corners):
            cv2.circle(frame, tuple(point), 4, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, str(idx + 1), (int(point[0]) + 5, int(point[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "CALIBRATED", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        if calibration.inverse_homography_matrix is not None:
            axes = np.asarray([[[0, 0], [80, 0], [0, 80]]], dtype=np.float32)
            image_axes = cv2.perspectiveTransform(axes, calibration.inverse_homography_matrix)[0]
            origin = tuple(np.round(image_axes[0]).astype(int))
            x_axis = tuple(np.round(image_axes[1]).astype(int))
            y_axis = tuple(np.round(image_axes[2]).astype(int))
            cv2.arrowedLine(frame, origin, x_axis, (0, 80, 255), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.arrowedLine(frame, origin, y_axis, (80, 255, 80), 2, cv2.LINE_AA, tipLength=0.25)
        for grid in calibration.hole_grids:
            pts = np.asarray(grid.get("expected_points", []), dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] != 9:
                continue
            color = (60, 255, 120) if int(grid.get("grid_id", 0)) == 0 else (255, 180, 60)
            for row in range(3):
                p1 = tuple(np.round(pts[row * 3]).astype(int))
                p2 = tuple(np.round(pts[row * 3 + 2]).astype(int))
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
            for col in range(3):
                p1 = tuple(np.round(pts[col]).astype(int))
                p2 = tuple(np.round(pts[6 + col]).astype(int))
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
        for hole in calibration.holes:
            center = (int(round(hole.x_px)), int(round(hole.y_px)))
            color = (60, 255, 120) if hole.matched else (0, 190, 255)
            cv2.circle(frame, center, int(round(max(4.0, hole.r_px))), color, 1, cv2.LINE_AA)
            cv2.circle(frame, center, 2, (0, 60, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{hole.grid_id}.{hole.local_id}",
                (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
        if calibration.holes:
            cv2.putText(
                frame,
                f"HOLES {len(calibration.holes)} candidates={calibration.hole_candidate_count}",
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (60, 255, 120),
                1,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(frame, "NO CALIBRATION", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
