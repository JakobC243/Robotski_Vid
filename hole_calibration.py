import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Circle = Tuple[float, float, float]


# ============================================================
# BRANJE REFERENČNE SLIKE IZ VIDEA
# ============================================================

def read_median_frame(video_path: Path, frame_samples: int) -> np.ndarray:
    """
    Iz prvih N slik naredi median frame.
    To zmanjša vpliv roke, če se premika čez ploščo.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti videa: {video_path}")

    frames = []
    count = 0

    while count < frame_samples:
        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame)
        count += 1

    cap.release()

    if not frames:
        raise RuntimeError("Ni prebranih slik iz videa.")

    median = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return median


# ============================================================
# ZAZNAVA KANDIDATOV ZA LUKNJE
# ============================================================

def merge_close_circles(
    circles: List[Circle],
    merge_dist_px: float = 8.0,
) -> List[Circle]:
    """
    Združi kroge, ki so zelo blizu skupaj.
    Hough + blob detektor lahko isto luknjo zaznata večkrat.
    """

    merged: List[Circle] = []

    for x, y, r in circles:
        found = False

        for i, (ox, oy, orad) in enumerate(merged):
            d = float(np.hypot(x - ox, y - oy))

            if d < merge_dist_px:
                merged[i] = (
                    0.5 * (x + ox),
                    0.5 * (y + oy),
                    0.5 * (r + orad),
                )
                found = True
                break

        if not found:
            merged.append((float(x), float(y), float(r)))

    return merged


def detect_holes_hough(
    frame_bgr: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_dist: int,
    param1: float,
    param2: float,
) -> List[Circle]:
    """
    Hough detekcija krogov.
    Dobra za luknje in prižgane LED, kadar so krožne.
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

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

    result: List[Circle] = []

    if circles is not None:
        circles = np.round(circles[0, :], 2)

        for x, y, r in circles:
            result.append((float(x), float(y), float(r)))

    return result


def _detect_circular_blobs_from_mask(
    mask: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_circularity: float,
) -> List[Circle]:
    """
    Najde približno krožne blob-e v binarni maski.
    """

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles: List[Circle] = []

    min_area = np.pi * min_radius * min_radius
    max_area = np.pi * max_radius * max_radius

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        if circularity < min_circularity:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)

        if min_radius <= r <= max_radius:
            circles.append((float(x), float(y), float(r)))

    return circles


def detect_dark_circular_blobs(
    frame_bgr: np.ndarray,
    min_radius: int,
    max_radius: int,
    dark_threshold: int,
) -> List[Circle]:
    """
    Luknje so lahko temne.
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < dark_threshold).astype(np.uint8) * 255

    return _detect_circular_blobs_from_mask(
        mask=mask,
        min_radius=min_radius,
        max_radius=max_radius,
        min_circularity=0.35,
    )


def detect_bright_circular_blobs(
    frame_bgr: np.ndarray,
    min_radius: int,
    max_radius: int,
    bright_threshold: int,
) -> List[Circle]:
    """
    Prižgane luknje / LED so lahko zelo svetle.
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > bright_threshold).astype(np.uint8) * 255

    return _detect_circular_blobs_from_mask(
        mask=mask,
        min_radius=min_radius,
        max_radius=max_radius,
        min_circularity=0.30,
    )


def detect_hole_candidates(
    frame_bgr: np.ndarray,
    min_radius: int,
    max_radius: int,
    min_dist: int,
    hough_param1: float,
    hough_param2: float,
    dark_threshold: int,
    bright_threshold: int,
) -> List[Circle]:
    """
    Skupna detekcija kandidatov:
    - Hough krogi,
    - temni krožni blob-i,
    - svetli krožni blob-i.
    """

    hough_circles = detect_holes_hough(
        frame_bgr=frame_bgr,
        min_radius=min_radius,
        max_radius=max_radius,
        min_dist=min_dist,
        param1=hough_param1,
        param2=hough_param2,
    )

    dark_circles = detect_dark_circular_blobs(
        frame_bgr=frame_bgr,
        min_radius=min_radius,
        max_radius=max_radius,
        dark_threshold=dark_threshold,
    )

    bright_circles = detect_bright_circular_blobs(
        frame_bgr=frame_bgr,
        min_radius=min_radius,
        max_radius=max_radius,
        bright_threshold=bright_threshold,
    )

    all_circles = merge_close_circles(
        hough_circles + dark_circles + bright_circles,
        merge_dist_px=8.0,
    )

    return all_circles


# ============================================================
# ISKANJE 3 x 3 MREŽE
# ============================================================

def nearest_candidate(
    expected: np.ndarray,
    centers: np.ndarray,
    used_indices: set,
    tolerance_px: float,
) -> Tuple[Optional[int], float]:
    """
    Za pričakovano mrežno točko najde najbližji zaznan kandidat.
    """

    if len(centers) == 0:
        return None, float("inf")

    diffs = centers - expected.reshape(1, 2)
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))

    order = np.argsort(dists)

    for idx in order:
        idx = int(idx)

        if idx in used_indices:
            continue

        d = float(dists[idx])

        if d <= tolerance_px:
            return idx, d

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
    """
    Oceni, ali p0 + col*u + row*v tvori 3 x 3 mrežo.
    """

    matched_indices = []
    matched_points = []
    expected_points = []
    errors = []
    used_indices = set()

    for row in range(3):
        for col in range(3):
            expected = p0 + col * u + row * v
            expected_points.append(expected)

            idx, err = nearest_candidate(
                expected=expected,
                centers=centers,
                used_indices=used_indices,
                tolerance_px=tolerance_px,
            )

            if idx is not None:
                used_indices.add(idx)
                matched_indices.append(idx)
                matched_points.append(centers[idx])
                errors.append(err)

    num_matches = len(matched_indices)

    if num_matches < min_matches:
        return None

    spacing_u = float(np.linalg.norm(u))
    spacing_v = float(np.linalg.norm(v))

    if spacing_u <= 1e-6 or spacing_v <= 1e-6:
        return None

    spacing_ratio = max(spacing_u, spacing_v) / min(spacing_u, spacing_v)

    # Mreža ne sme biti preveč raztegnjena.
    if spacing_ratio > 2.0:
        return None

    # Smeri naj bosta približno pravokotni.
    cos_angle = abs(float(np.dot(u, v) / (spacing_u * spacing_v)))

    if cos_angle > 0.70:
        return None

    mean_error = float(np.mean(errors)) if errors else float("inf")

    # Višji score je boljši.
    # Glavni kriterij: čim več zadetkov v 3x3 vzorcu.
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
        "p0": p0.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "spacing_u_px": spacing_u,
        "spacing_v_px": spacing_v,
        "cos_angle": cos_angle,
    }


def find_best_3x3_grid(
    circles: List[Circle],
    min_spacing_px: float,
    max_spacing_px: float,
    tolerance_ratio: float,
    min_matches: int,
) -> Optional[Dict]:
    """
    Poišče najboljšo 3 x 3 mrežo med kandidati.
    Dovoljeno je, da nekaj lukenj manjka, npr. zaradi roke.
    """

    if len(circles) < min_matches:
        return None

    centers = np.array([[c[0], c[1]] for c in circles], dtype=np.float32)
    n = len(centers)

    best = None

    for i in range(n):
        p0 = centers[i]

        for j in range(n):
            if j == i:
                continue

            u = centers[j] - p0
            du = float(np.linalg.norm(u))

            if du < min_spacing_px or du > max_spacing_px:
                continue

            for k in range(n):
                if k == i or k == j:
                    continue

                v = centers[k] - p0
                dv = float(np.linalg.norm(v))

                if dv < min_spacing_px or dv > max_spacing_px:
                    continue

                tolerance_px = max(5.0, tolerance_ratio * 0.5 * (du + dv))

                candidate = score_grid_candidate(
                    centers=centers,
                    p0=p0,
                    u=u,
                    v=v,
                    tolerance_px=tolerance_px,
                    min_matches=min_matches,
                )

                if candidate is None:
                    continue

                if best is None or candidate["score"] > best["score"]:
                    best = candidate

    return best


def select_grids(
    circles: List[Circle],
    num_grids: int,
    min_spacing_px: float,
    max_spacing_px: float,
    tolerance_ratio: float,
    min_matches: int,
) -> List[Dict]:
    """
    Najde eno ali več 3x3 mrež.
    Za metrično kalibracijo je že ena pravilna mreža dovolj.
    """

    remaining = circles[:]
    selected = []

    for _ in range(num_grids):
        grid = find_best_3x3_grid(
            circles=remaining,
            min_spacing_px=min_spacing_px,
            max_spacing_px=max_spacing_px,
            tolerance_ratio=tolerance_ratio,
            min_matches=min_matches,
        )

        if grid is None:
            break

        selected.append(grid)

        matched_set = set(grid["matched_indices"])

        remaining = [
            c for idx, c in enumerate(remaining)
            if idx not in matched_set
        ]

    return selected


# ============================================================
# MERILO mm/px
# ============================================================

def compute_scale_from_grids(
    grids: List[Dict],
    hole_spacing_mm: float,
) -> Dict:
    """
    Razdalja med sosednjima luknjama je znana: 32 mm.
    Iz ocenjene razdalje v px dobimo mm/px.
    """

    spacings = []

    for g in grids:
        spacings.append(float(g["spacing_u_px"]))
        spacings.append(float(g["spacing_v_px"]))

    if not spacings:
        raise RuntimeError("Ni mrež za izračun merila.")

    spacing_px = float(np.median(spacings))
    mm_per_px = float(hole_spacing_mm / spacing_px)

    return {
        "hole_spacing_mm": float(hole_spacing_mm),
        "estimated_spacing_px": spacing_px,
        "mm_per_px": mm_per_px,
        "px_per_mm": float(1.0 / mm_per_px),
        "num_grids_detected": int(len(grids)),
        "grid_spacings_px": spacings,
    }


# ============================================================
# DEBUG RISANJE
# ============================================================

def draw_debug_image(
    frame_bgr: np.ndarray,
    all_circles: List[Circle],
    grids: List[Dict],
    scale_info: Dict,
    output_path: Path,
) -> None:
    """
    Nariše:
    - sive kroge = vsi kandidati,
    - barvno 3x3 mrežo = izbrana pravilna mreža,
    - merilo mm/px.
    """

    debug = frame_bgr.copy()

    # Vsi kandidati sivo.
    for x, y, r in all_circles:
        center = (int(round(x)), int(round(y)))
        radius = int(round(r))
        cv2.circle(debug, center, radius, (120, 120, 120), 1)

    colors = [
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
    ]

    for grid_id, grid in enumerate(grids):
        color = colors[grid_id % len(colors)]

        p0 = grid["p0"]
        u = grid["u"]
        v = grid["v"]
        expected = grid["expected_points"]
        matched = grid["matched_points"]

        # Mrežne črte.
        for row in range(3):
            p_start = p0 + row * v
            p_end = p0 + 2 * u + row * v

            cv2.line(
                debug,
                tuple(np.round(p_start).astype(int)),
                tuple(np.round(p_end).astype(int)),
                color,
                2,
            )

        for col in range(3):
            p_start = p0 + col * u
            p_end = p0 + col * u + 2 * v

            cv2.line(
                debug,
                tuple(np.round(p_start).astype(int)),
                tuple(np.round(p_end).astype(int)),
                color,
                2,
            )

        # Pričakovane točke mreže.
        for i, p in enumerate(expected):
            center = tuple(np.round(p).astype(int))

            cv2.circle(debug, center, 9, color, 2)
            cv2.circle(debug, center, 3, (0, 0, 255), -1)

            cv2.putText(
                debug,
                f"G{grid_id}_{i}",
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Dejanski kandidati, ki so se ujemali z mrežo.
        for p in matched:
            center = tuple(np.round(p).astype(int))
            cv2.circle(debug, center, 12, (0, 0, 255), 2)

        text_lines = [
            f"grid {grid_id}: matches={grid['num_matches']}/9, err={grid['mean_error_px']:.2f}px",
            f"spacing u={grid['spacing_u_px']:.2f}px, v={grid['spacing_v_px']:.2f}px",
        ]

        y_grid = 130 + grid_id * 55

        for line in text_lines:
            cv2.putText(
                debug,
                line,
                (20, y_grid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug,
                line,
                (20, y_grid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_grid += 24

    text_lines = [
        f"3x3 grids detected: {scale_info['num_grids_detected']}",
        f"spacing: {scale_info['estimated_spacing_px']:.2f} px = {scale_info['hole_spacing_mm']:.1f} mm",
        f"scale: {scale_info['mm_per_px']:.4f} mm/px",
        f"candidates: {len(all_circles)}",
    ]

    y0 = 30

    for line in text_lines:
        cv2.putText(
            debug,
            line,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            line,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y0 += 30

    cv2.imwrite(str(output_path), debug)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zaznava 3x3 mreže lukenj 9HPT plošče in samodejna kalibracija mm/px."
    )

    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--hole-spacing-mm", type=float, default=32.0)

    # Za tvoje podatke je ena mreža dovolj. Če sta vidni dve, lahko daš --num-grids 2.
    parser.add_argument("--num-grids", type=int, default=1)

    # Več frame-ov pomaga odstraniti roko z median frame-om.
    parser.add_argument("--frame-samples", type=int, default=120)

    # Velikost lukenj/LED v pikslih.
    parser.add_argument("--min-radius", type=int, default=3)
    parser.add_argument("--max-radius", type=int, default=20)
    parser.add_argument("--min-dist", type=int, default=12)

    # Hough parametri.
    parser.add_argument("--hough-param1", type=float, default=80)
    parser.add_argument("--hough-param2", type=float, default=10)

    # Temne luknje in svetle LED.
    parser.add_argument("--dark-threshold", type=int, default=90)
    parser.add_argument("--bright-threshold", type=int, default=170)

    # Pričakovani razmik med sosednjimi luknjami v pikslih.
    # Če izbira napačno mrežo, zožaj ta interval.
    parser.add_argument("--min-spacing-px", type=float, default=15)
    parser.add_argument("--max-spacing-px", type=float, default=70)

    # Toleranca za delno zakrite / rahlo napačno zaznane luknje.
    parser.add_argument("--grid-tolerance-ratio", type=float, default=0.45)

    # Sprejmi mrežo, tudi če je vidnih samo 5 od 9 lukenj.
    parser.add_argument("--min-grid-matches", type=int, default=5)

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video ne obstaja: {video_path}")

    print("Berem median frame ...")
    frame = read_median_frame(
        video_path=video_path,
        frame_samples=args.frame_samples,
    )

    print("Zaznavam kandidate za luknje / LED ...")
    all_circles = detect_hole_candidates(
        frame_bgr=frame,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_dist=args.min_dist,
        hough_param1=args.hough_param1,
        hough_param2=args.hough_param2,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
    )

    print(f"Kandidatov pred geometrijskim filtriranjem: {len(all_circles)}")

    print("Iščem 3x3 mrežo ...")
    grids = select_grids(
        circles=all_circles,
        num_grids=args.num_grids,
        min_spacing_px=args.min_spacing_px,
        max_spacing_px=args.max_spacing_px,
        tolerance_ratio=args.grid_tolerance_ratio,
        min_matches=args.min_grid_matches,
    )

    if not grids:
        raise RuntimeError(
            "Ni najdene pravilne 3x3 mreže. "
            "Poskusi spremeniti --min-spacing-px, --max-spacing-px, "
            "--grid-tolerance-ratio, --min-grid-matches ali --hough-param2."
        )

    scale_info = compute_scale_from_grids(
        grids=grids,
        hole_spacing_mm=args.hole_spacing_mm,
    )

    calibration = {
        **scale_info,
        "candidate_circles_count": len(all_circles),
        "parameters": {
            "hole_spacing_mm": args.hole_spacing_mm,
            "num_grids_requested": args.num_grids,
            "frame_samples": args.frame_samples,
            "min_radius": args.min_radius,
            "max_radius": args.max_radius,
            "min_dist": args.min_dist,
            "hough_param1": args.hough_param1,
            "hough_param2": args.hough_param2,
            "dark_threshold": args.dark_threshold,
            "bright_threshold": args.bright_threshold,
            "min_spacing_px": args.min_spacing_px,
            "max_spacing_px": args.max_spacing_px,
            "grid_tolerance_ratio": args.grid_tolerance_ratio,
            "min_grid_matches": args.min_grid_matches,
        },
        "grids": [],
    }

    for grid_id, grid in enumerate(grids):
        grid_data = {
            "grid_id": grid_id,
            "num_matches": int(grid["num_matches"]),
            "mean_error_px": float(grid["mean_error_px"]),
            "spacing_u_px": float(grid["spacing_u_px"]),
            "spacing_v_px": float(grid["spacing_v_px"]),
            "cos_angle": float(grid["cos_angle"]),
            "holes_expected_px": [],
            "holes_matched_px": [],
        }

        for i, p in enumerate(grid["expected_points"]):
            grid_data["holes_expected_px"].append(
                {
                    "local_id": int(i),
                    "x_px": float(p[0]),
                    "y_px": float(p[1]),
                }
            )

        for i, p in enumerate(grid["matched_points"]):
            grid_data["holes_matched_px"].append(
                {
                    "matched_id": int(i),
                    "x_px": float(p[0]),
                    "y_px": float(p[1]),
                }
            )

        calibration["grids"].append(grid_data)

    json_path = output_dir / "hole_calibration_grid.json"
    debug_path = output_dir / "hole_detection_grid_debug.png"
    median_path = output_dir / "median_frame.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)

    cv2.imwrite(str(median_path), frame)

    draw_debug_image(
        frame_bgr=frame,
        all_circles=all_circles,
        grids=grids,
        scale_info=scale_info,
        output_path=debug_path,
    )

    print("\nKalibracija končana.")
    print(f"  kandidatov: {len(all_circles)}")
    print(f"  najdenih 3x3 mrež: {scale_info['num_grids_detected']}")
    print(f"  razdalja med luknjami: {scale_info['estimated_spacing_px']:.2f} px")
    print(f"  merilo: {scale_info['mm_per_px']:.5f} mm/px")
    print(f"  json: {json_path}")
    print(f"  debug slika: {debug_path}")
    print(f"  median frame: {median_path}")


if __name__ == "__main__":
    main()