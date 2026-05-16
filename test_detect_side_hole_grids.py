import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


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
    gray_threshold: int,
    value_threshold: int,
    min_radius: float,
    max_radius: float,
    min_area: float,
    max_area: float,
    min_circularity: float,
) -> Tuple[List[Dict], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    # The true holes are bright circular spots. The white text plate becomes one
    # large component and is rejected by area/radius; black text is not selected.
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


def cluster_candidates(points: List[Dict], link_distance_px: float) -> List[List[int]]:
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
            neighbors = np.where((dists <= link_distance_px) & (~visited))[0]
            for neighbor in neighbors:
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
    score = (
        matches * 1000.0
        - mean_error * 60.0
        - abs(spacing_u - spacing_v) * 6.0
        - cos_angle * 160.0
    )

    return {
        "score": float(score),
        "matches": int(matches),
        "mean_error_px": mean_error,
        "spacing_u_px": spacing_u,
        "spacing_v_px": spacing_v,
        "cos_angle": cos_angle,
        "p0": p0.astype(float).tolist(),
        "u": u.astype(float).tolist(),
        "v": v.astype(float).tolist(),
        "expected_points": np.asarray(expected_points, dtype=np.float32),
        "matched_points": np.asarray(matched_points, dtype=np.float32),
        "matched_indices": matched_indices,
    }


def fit_best_3x3_grid(cluster_points: np.ndarray, min_spacing_px: float, max_spacing_px: float, min_matches: int) -> Optional[Dict]:
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
                if candidate is None:
                    continue
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

    return best


def select_two_grids(points: List[Dict], clusters: List[List[int]], args: argparse.Namespace) -> List[Dict]:
    grids = []
    all_points = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32)

    for cluster_id, cluster in enumerate(clusters):
        if len(cluster) < args.min_matches:
            continue

        cluster_points = all_points[cluster]
        grid = fit_best_3x3_grid(
            cluster_points=cluster_points,
            min_spacing_px=args.min_spacing_px,
            max_spacing_px=args.max_spacing_px,
            min_matches=args.min_matches,
        )
        if grid is None:
            continue

        grid["cluster_id"] = int(cluster_id)
        grid["cluster_size"] = int(len(cluster))
        center = np.mean(grid["expected_points"], axis=0)
        grid["center"] = [float(center[0]), float(center[1])]
        grid["point_indices"] = [int(cluster[i]) for i in grid["matched_indices"]]
        grids.append(grid)

    grids.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    for grid in grids:
        center = np.asarray(grid["center"], dtype=np.float32)
        if any(float(np.linalg.norm(center - np.asarray(other["center"], dtype=np.float32))) < args.min_grid_separation_px for other in selected):
            continue
        selected.append(grid)
        if len(selected) == 2:
            break

    selected.sort(key=lambda item: item["center"][1])
    return selected


def draw_debug(image_bgr: np.ndarray, points: List[Dict], clusters: List[List[int]], grids: List[Dict], output_path: Path) -> None:
    debug = image_bgr.copy()

    for point in points:
        center = (int(round(point["x"])), int(round(point["y"])))
        cv2.circle(debug, center, int(round(max(3.0, point["r"]))), (180, 180, 180), 1)

    colors = [(0, 255, 0), (0, 180, 255)]
    for grid_id, grid in enumerate(grids):
        color = colors[grid_id % len(colors)]
        expected = grid["expected_points"]

        for row in range(3):
            p1 = tuple(np.round(expected[row * 3]).astype(int))
            p2 = tuple(np.round(expected[row * 3 + 2]).astype(int))
            cv2.line(debug, p1, p2, color, 2)
        for col in range(3):
            p1 = tuple(np.round(expected[col]).astype(int))
            p2 = tuple(np.round(expected[6 + col]).astype(int))
            cv2.line(debug, p1, p2, color, 2)

        for local_id, point in enumerate(expected):
            center = tuple(np.round(point).astype(int))
            cv2.circle(debug, center, 9, color, 2)
            cv2.circle(debug, center, 2, (0, 0, 255), -1)
            cv2.putText(debug, f"G{grid_id}_{local_id}", (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        label = "G{} matches={}/9 err={:.2f}px sp={:.1f}/{:.1f}px".format(
            grid_id,
            grid["matches"],
            grid["mean_error_px"],
            grid["spacing_u_px"],
            grid["spacing_v_px"],
        )
        y = 25 + grid_id * 25
        cv2.putText(debug, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(debug, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

    summary = f"bright candidates={len(points)} clusters={len(clusters)} selected_grids={len(grids)}"
    cv2.putText(debug, summary, (12, debug.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(debug, summary, (12, debug.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), debug)


def grid_to_json(grid: Dict) -> Dict:
    return {
        "cluster_id": int(grid["cluster_id"]),
        "cluster_size": int(grid["cluster_size"]),
        "center": grid["center"],
        "matches": int(grid["matches"]),
        "mean_error_px": float(grid["mean_error_px"]),
        "spacing_u_px": float(grid["spacing_u_px"]),
        "spacing_v_px": float(grid["spacing_v_px"]),
        "cos_angle": float(grid["cos_angle"]),
        "p0": grid["p0"],
        "u": grid["u"],
        "v": grid["v"],
        "expected_points": [
            {"local_id": int(i), "x_px": float(p[0]), "y_px": float(p[1])}
            for i, p in enumerate(grid["expected_points"])
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test detector for the two visible 3x3 hole grids on one median image.")
    parser.add_argument(
        "--image",
        type=str,
        default="outputs_old_hole_calib_test/median_frame.png",
        help="Input median image.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs_side_hole_test")
    parser.add_argument("--gray-threshold", type=int, default=170)
    parser.add_argument("--value-threshold", type=int, default=180)
    parser.add_argument("--min-radius", type=float, default=2.8)
    parser.add_argument("--max-radius", type=float, default=9.0)
    parser.add_argument("--min-area", type=float, default=15.0)
    parser.add_argument("--max-area", type=float, default=130.0)
    parser.add_argument("--min-circularity", type=float, default=0.25)
    parser.add_argument("--cluster-link-px", type=float, default=45.0)
    parser.add_argument("--min-spacing-px", type=float, default=16.0)
    parser.add_argument("--max-spacing-px", type=float, default=44.0)
    parser.add_argument("--min-matches", type=int, default=7)
    parser.add_argument("--min-grid-separation-px", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points, mask = detect_bright_hole_candidates(
        image_bgr=image,
        gray_threshold=args.gray_threshold,
        value_threshold=args.value_threshold,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
    )
    clusters = cluster_candidates(points, link_distance_px=args.cluster_link_px)
    grids = select_two_grids(points, clusters, args)

    debug_path = output_dir / "side_hole_grids_debug.png"
    mask_path = output_dir / "bright_hole_mask.png"
    json_path = output_dir / "side_hole_grids.json"
    draw_debug(image, points, clusters, grids, debug_path)
    cv2.imwrite(str(mask_path), mask)

    result = {
        "image": str(image_path),
        "candidate_count": len(points),
        "cluster_sizes": [len(cluster) for cluster in clusters],
        "selected_grid_count": len(grids),
        "grids": [grid_to_json(grid) for grid in grids],
        "parameters": vars(args),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Candidates: {len(points)}")
    print(f"Clusters: {[len(cluster) for cluster in clusters]}")
    print(f"Selected 3x3 grids: {len(grids)}")
    for i, grid in enumerate(grids):
        print(
            "  grid {}: center=({:.1f},{:.1f}) matches={}/9 err={:.2f}px spacing={:.1f}/{:.1f}px".format(
                i,
                grid["center"][0],
                grid["center"][1],
                grid["matches"],
                grid["mean_error_px"],
                grid["spacing_u_px"],
                grid["spacing_v_px"],
            )
        )
    print(f"Debug image: {debug_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
