from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from utils import Point, ensure_parent, finite_point


BOARD_COORD_MARGIN = 0.0


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
        status="calibrated",
        source=source,
        created_at=datetime.now().isoformat(timespec="seconds"),
        video=video,
    )


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
    else:
        cv2.putText(frame, "NO CALIBRATION", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
