from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from calibration import BoardCalibration, hole_grid_region_from_points


SIDES = ("left", "right")


@dataclass
class PegGrid:
    side: str
    points: np.ndarray
    spacing: float
    sample_radius: float
    polygon: np.ndarray


@dataclass
class PegPatch:
    descriptor: np.ndarray
    value_median: float
    edge_mean: float
    texture_std: float


@dataclass
class PegOccupancyInfo:
    states: Dict[str, List[int]]
    scores: Dict[str, List[float]]
    visible: Dict[str, List[int]]
    updated: Dict[str, bool]
    covered: Dict[str, bool]
    target_side: str
    target_phase: str
    measurement_active: bool

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "peg_target_side": self.target_side,
            "peg_target_phase": self.target_phase,
            "peg_measurement_active": int(self.measurement_active),
        }
        for side in SIDES:
            states = self.states.get(side, [0] * 9)
            scores = self.scores.get(side, [float("nan")] * 9)
            visible = self.visible.get(side, [0] * 9)
            row[f"peg_{side}_count"] = int(sum(int(v) for v in states))
            row[f"peg_{side}_updated"] = int(bool(self.updated.get(side, False)))
            row[f"peg_{side}_covered"] = int(bool(self.covered.get(side, False)))
            for idx in range(9):
                row[f"peg_{side}_{idx}"] = int(states[idx]) if idx < len(states) else 0
                row[f"peg_{side}_{idx}_score"] = float(scores[idx]) if idx < len(scores) else float("nan")
                row[f"peg_{side}_{idx}_visible"] = int(visible[idx]) if idx < len(visible) else 0
        target_states = self.states.get(self.target_side, []) if self.target_side in SIDES else []
        row["peg_target_count"] = int(sum(int(v) for v in target_states))
        return row


class PegOccupancyDetector:
    def __init__(
        self,
        calibration: BoardCalibration,
        stable_frames: int = 5,
        sample_radius_scale: float = 0.30,
        change_threshold: float = 0.32,
        reference_alpha: float = 0.06,
    ) -> None:
        self.grids = self._build_grids(calibration, sample_radius_scale)
        self.stable_frames = int(max(1, stable_frames))
        self.change_threshold = float(max(0.05, change_threshold))
        self.reference_alpha = float(np.clip(reference_alpha, 0.0, 0.30))
        self.references: Dict[str, List[Optional[PegPatch]]] = {side: [None] * 9 for side in SIDES}
        self.states: Dict[str, List[int]] = {side: [0] * 9 for side in SIDES}
        self.scores: Dict[str, List[float]] = {side: [float("nan")] * 9 for side in SIDES}
        self.visible: Dict[str, List[int]] = {side: [0] * 9 for side in SIDES}
        self.candidate_states: Dict[str, List[Optional[int]]] = {side: [None] * 9 for side in SIDES}
        self.candidate_counts: Dict[str, List[int]] = {side: [0] * 9 for side in SIDES}
        self.reference_bootstrap_done: Dict[str, bool] = {side: False for side in SIDES}
        self.last_target_side = ""
        self.target_phase = "tracking"

    def _build_grids(self, calibration: BoardCalibration, sample_radius_scale: float) -> Dict[str, PegGrid]:
        grids = []
        for grid in calibration.hole_grids:
            points = np.asarray(grid.get("expected_points", []), dtype=np.float32).reshape(-1, 2)
            if points.shape[0] < 9 or not np.all(np.isfinite(points)):
                continue
            spacings = [float(grid.get("spacing_u_px", 0.0)), float(grid.get("spacing_v_px", 0.0))]
            valid_spacings = [value for value in spacings if np.isfinite(value) and value > 0.0]
            spacing = float(np.median(valid_spacings)) if valid_spacings else 24.0
            sample_radius = float(np.clip(sample_radius_scale * spacing, 7.0, 14.0))
            polygon = hole_grid_region_from_points(points, spacing, padding_scale=0.30)
            if polygon is None:
                rect = cv2.minAreaRect(points)
                polygon = cv2.boxPoints(rect).astype(np.float32)
            grids.append(
                {
                    "center": np.mean(points, axis=0),
                    "points": points,
                    "spacing": spacing,
                    "sample_radius": sample_radius,
                    "polygon": polygon.astype(np.float32),
                }
            )
        grids.sort(key=lambda item: (float(item["center"][0]), float(item["center"][1])))
        if len(grids) < 2:
            return {}
        selected = [grids[0], grids[-1]]
        out: Dict[str, PegGrid] = {}
        for side, item in zip(SIDES, selected):
            out[side] = PegGrid(
                side=side,
                points=np.asarray(item["points"], dtype=np.float32),
                spacing=float(item["spacing"]),
                sample_radius=float(item["sample_radius"]),
                polygon=np.asarray(item["polygon"], dtype=np.float32),
            )
        return out

    def _patch(self, frame_bgr: np.ndarray, point: np.ndarray, radius: float) -> Optional[PegPatch]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cx = float(point[0])
        cy = float(point[1])
        h, w = gray.shape[:2]
        r = float(radius)
        x0 = int(max(0, np.floor(cx - r)))
        x1 = int(min(w, np.ceil(cx + r + 1.0)))
        y0 = int(max(0, np.floor(cy - r)))
        y1 = int(min(h, np.ceil(cy + r + 1.0)))
        if x1 <= x0 or y1 <= y0:
            return None

        patch = gray[y0:y1, x0:x1].astype(np.float32)
        resized = cv2.resize(patch, (17, 17), interpolation=cv2.INTER_AREA)
        median = float(np.median(resized))
        std = float(np.std(resized))
        normalized = (resized - median) / max(12.0, std)
        normalized = np.clip(normalized, -3.0, 3.0).astype(np.float32)
        edges = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        edges = np.abs(edges) + np.abs(cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3))
        return PegPatch(
            descriptor=normalized,
            value_median=median,
            edge_mean=float(np.mean(edges)),
            texture_std=std,
        )

    def _blend_reference(self, reference: PegPatch, patch: PegPatch) -> PegPatch:
        alpha = self.reference_alpha
        return PegPatch(
            descriptor=(1.0 - alpha) * reference.descriptor + alpha * patch.descriptor,
            value_median=(1.0 - alpha) * reference.value_median + alpha * patch.value_median,
            edge_mean=(1.0 - alpha) * reference.edge_mean + alpha * patch.edge_mean,
            texture_std=(1.0 - alpha) * reference.texture_std + alpha * patch.texture_std,
        )

    def _change_score(self, reference: PegPatch, patch: PegPatch) -> float:
        texture_change = float(np.mean(np.abs(reference.descriptor - patch.descriptor)))
        value_change = abs(reference.value_median - patch.value_median) / 48.0
        edge_change = abs(reference.edge_mean - patch.edge_mean) / 55.0
        std_change = abs(reference.texture_std - patch.texture_std) / 22.0
        return float(0.42 * texture_change + 0.38 * value_change + 0.12 * edge_change + 0.08 * std_change)

    def _reference_count(self, side: str) -> int:
        return int(sum(1 for reference in self.references.get(side, []) if reference is not None))

    def _median_reference(self, patches: List[PegPatch]) -> PegPatch:
        descriptors = np.stack([patch.descriptor for patch in patches], axis=0)
        return PegPatch(
            descriptor=np.median(descriptors, axis=0).astype(np.float32),
            value_median=float(np.median([patch.value_median for patch in patches])),
            edge_mean=float(np.median([patch.edge_mean for patch in patches])),
            texture_std=float(np.median([patch.texture_std for patch in patches])),
        )

    def _bootstrap_from_visible_field(
        self,
        frame_bgr: np.ndarray,
        side: str,
        grid: PegGrid,
        target_side: str,
    ) -> Optional[bool]:
        patch_items: List[Tuple[int, PegPatch]] = []
        visible = [0] * 9
        for idx, point in enumerate(grid.points[:9]):
            patch = self._patch(frame_bgr, point, grid.sample_radius)
            if patch is None:
                continue
            visible[idx] = 1
            patch_items.append((idx, patch))

        self.visible[side] = visible
        if len(patch_items) < 7:
            self.target_phase = "waiting_clear_reference"
            return None

        patches = [patch for _, patch in patch_items]
        pair_scores: List[float] = []
        for i, patch in enumerate(patches):
            distances = [
                self._change_score(other_patch, patch)
                for j, other_patch in enumerate(patches)
                if i != j
            ]
            pair_scores.append(float(np.median(distances)) if distances else 0.0)

        score_array = np.asarray(pair_scores, dtype=np.float32)
        median_score = float(np.median(score_array))
        mad = float(np.median(np.abs(score_array - median_score)))
        robust_sigma = 1.4826 * mad
        threshold = max(0.60 * self.change_threshold, median_score + max(0.06, 2.5 * robust_sigma))
        occupied_flags = [bool(score >= threshold) for score in pair_scores]

        if sum(occupied_flags) == 0 and len(pair_scores) >= 2:
            ordered = sorted(pair_scores, reverse=True)
            gap = ordered[0] - ordered[1]
            strongest_idx = int(np.argmax(score_array))
            if ordered[0] >= max(0.70 * self.change_threshold, median_score + 0.08) and gap >= 0.05:
                occupied_flags[strongest_idx] = True

        if sum(occupied_flags) > 4:
            occupied_flags = [False] * len(occupied_flags)

        empty_patches = [patch for occupied, patch in zip(occupied_flags, patches) if not occupied]
        if len(empty_patches) < 5:
            self.target_phase = "waiting_clear_reference"
            return None

        empty_reference = self._median_reference(empty_patches)
        baseline_state = self._baseline_state(side, target_side)
        changed_state = int(1 - baseline_state)
        boot_updated = False

        self.scores[side] = [float("nan")] * 9
        for (idx, patch), score, occupied in zip(patch_items, pair_scores, occupied_flags):
            raw_state = changed_state if occupied else baseline_state
            self.references[side][idx] = empty_reference if occupied else patch
            self.scores[side][idx] = float(score)
            self.candidate_states[side][idx] = raw_state
            self.candidate_counts[side][idx] = self.stable_frames
            if self.states[side][idx] != raw_state:
                self.states[side][idx] = raw_state
                boot_updated = True

        self.reference_bootstrap_done[side] = True
        self.target_phase = "non_led_bootstrap" if boot_updated else "non_led_reference"
        return boot_updated

    def _baseline_state(self, side: str, target_side: str) -> int:
        return 0

    def _reset_for_target_side(self, target_side: str) -> None:
        if target_side not in SIDES or target_side == self.last_target_side:
            return
        self.last_target_side = target_side
        self.target_phase = "tracking"
        for side in SIDES:
            baseline = self._baseline_state(side, target_side)
            self.states[side] = [baseline] * 9
            self.references[side] = [None] * 9
            self.scores[side] = [float("nan")] * 9
            self.visible[side] = [0] * 9
            self.candidate_states[side] = [None] * 9
            self.candidate_counts[side] = [0] * 9
            self.reference_bootstrap_done[side] = False

    def update(
        self,
        frame_bgr: np.ndarray,
        covered_sides: Set[str],
        target_side: str = "",
        measurement_active: bool = True,
        reference_bootstrap: bool = False,
    ) -> PegOccupancyInfo:
        if target_side in SIDES:
            self._reset_for_target_side(target_side)

        updated = {side: False for side in SIDES}
        covered = {side: side in covered_sides or "both" in covered_sides for side in SIDES}

        for side, grid in self.grids.items():
            if side != target_side:
                self.visible[side] = [0] * 9
                self.scores[side] = [float("nan")] * 9
                continue

            if covered.get(side, False):
                self.visible[side] = [0] * 9
                if reference_bootstrap and measurement_active and not self.reference_bootstrap_done.get(side, False):
                    self.target_phase = "waiting_clear_reference"
                continue

            if reference_bootstrap and measurement_active and not self.reference_bootstrap_done.get(side, False):
                if self._reference_count(side) >= 7:
                    self.reference_bootstrap_done[side] = True
                else:
                    boot_updated = self._bootstrap_from_visible_field(frame_bgr, side, grid, target_side)
                    if boot_updated is not None:
                        updated[side] = bool(updated[side] or boot_updated)
                    continue

            baseline_state = self._baseline_state(side, target_side)
            for idx, point in enumerate(grid.points[:9]):
                patch = self._patch(frame_bgr, point, grid.sample_radius)
                if patch is None:
                    self.visible[side][idx] = 0
                    continue
                self.visible[side][idx] = 1
                reference = self.references[side][idx]
                if reference is None:
                    self.references[side][idx] = patch
                    self.scores[side][idx] = 0.0
                    continue

                score = self._change_score(reference, patch)
                self.scores[side][idx] = score
                changed = bool(score >= self.change_threshold)
                raw_state = int(1 - baseline_state) if changed else int(baseline_state)

                if not measurement_active:
                    self.references[side][idx] = self._blend_reference(reference, patch)
                    self.states[side][idx] = baseline_state
                    continue

                if self.candidate_states[side][idx] != raw_state:
                    self.candidate_states[side][idx] = raw_state
                    self.candidate_counts[side][idx] = 1
                else:
                    self.candidate_counts[side][idx] += 1

                previous_state = self.states[side][idx]
                if self.candidate_counts[side][idx] >= self.stable_frames and previous_state != raw_state:
                    self.states[side][idx] = raw_state
                    updated[side] = True

                if raw_state == baseline_state and self.candidate_counts[side][idx] >= self.stable_frames:
                    self.references[side][idx] = self._blend_reference(reference, patch)

        return self.info(updated=updated, covered=covered, target_side=target_side, measurement_active=measurement_active)

    def info(
        self,
        updated: Optional[Dict[str, bool]] = None,
        covered: Optional[Dict[str, bool]] = None,
        target_side: str = "",
        measurement_active: bool = True,
    ) -> PegOccupancyInfo:
        return PegOccupancyInfo(
            states={side: list(self.states.get(side, [0] * 9)) for side in SIDES},
            scores={side: list(self.scores.get(side, [float("nan")] * 9)) for side in SIDES},
            visible={side: list(self.visible.get(side, [0] * 9)) for side in SIDES},
            updated=updated or {side: False for side in SIDES},
            covered=covered or {side: False for side in SIDES},
            target_side=target_side if target_side in SIDES else "",
            target_phase=self.target_phase if target_side in SIDES else "",
            measurement_active=bool(measurement_active),
        )

    def draw(self, frame_bgr: np.ndarray, info: PegOccupancyInfo) -> None:
        for side, grid in self.grids.items():
            if info.target_side not in SIDES or side != info.target_side:
                continue
            states = info.states.get(side, [0] * 9)
            visible = info.visible.get(side, [0] * 9)
            for idx, point in enumerate(grid.points[:9]):
                center = (int(round(float(point[0]))), int(round(float(point[1]))))
                radius = max(5, int(round(0.27 * grid.spacing)))
                occupied = bool(idx < len(states) and states[idx])
                is_visible = bool(idx < len(visible) and visible[idx])
                if occupied:
                    cv2.circle(frame_bgr, center, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, center, radius, (255, 120, 20), -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, center, radius, (245, 245, 245), 1, cv2.LINE_AA)
                else:
                    color = (245, 245, 245) if is_visible else (135, 135, 135)
                    cv2.circle(frame_bgr, center, radius + 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.circle(frame_bgr, center, radius, color, 2, cv2.LINE_AA)
