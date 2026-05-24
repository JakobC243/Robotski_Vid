import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from track_calibrated_hand_and_pins import parse_args as parse_pipeline_args
from track_calibrated_hand_and_pins import process_video


SIDE_VALUES = ("auto", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final quick-check runner: tracking + corner graph + live test warnings in the output video."
    )
    parser.add_argument("--video", required=True, help="Input MP4.")
    parser.add_argument("--output-root", default="outputs_final_quick", help="Output folder.")
    parser.add_argument("--output-stem", default="final_quick_check", help="Base output name.")
    parser.add_argument("--expected-hand", choices=SIDE_VALUES, default="auto", help="Expected MediaPipe hand label.")
    parser.add_argument("--target-side", choices=SIDE_VALUES, default="auto", help="Only this board side should receive pins.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video.")
    parser.add_argument("--frame-samples", type=int, default=120)
    parser.add_argument("--show", action="store_true", help="Show base tracker window while tracking.")
    parser.add_argument("--show-final", action="store_true", help="Show final quick-check video while it is written.")
    parser.add_argument("--skip-tracker", action="store_true", help="Reuse an existing tracker CSV/video in output-root.")

    parser.add_argument("--ma-window", type=int, default=7, help="Moving average window for hand center.")
    parser.add_argument("--velocity-window-frames", type=int, default=8, help="Longer window reduces speed spikes.")
    parser.add_argument("--acceleration-window-frames", type=int, default=12)
    parser.add_argument("--speed-ema-alpha", type=float, default=0.18, help="Lower value smooths speed more.")
    parser.add_argument("--display-speed-alpha", type=float, default=0.12, help="Extra smoothing for the corner graph.")
    parser.add_argument("--max-hand-speed-mm-s", type=float, default=1800.0)
    parser.add_argument("--max-acceleration-mm-s2", type=float, default=3500.0)

    parser.add_argument("--brightness-jump-threshold", type=float, default=28.0, help="Mean-frame brightness jump warning.")
    parser.add_argument("--missing-hand-seconds", type=float, default=0.80, help="Warn after this many seconds without hand.")
    parser.add_argument("--spike-ratio", type=float, default=2.4, help="Raw speed must exceed smoothed speed by this ratio.")
    parser.add_argument("--min-spike-speed-mm-s", type=float, default=450.0)
    parser.add_argument("--graph-seconds", type=float, default=8.0, help="History shown in corner graph.")
    return parser.parse_args()


def build_pipeline_args(args: argparse.Namespace) -> argparse.Namespace:
    pipeline_args = parse_pipeline_args([])
    pipeline_args.video = args.video
    pipeline_args.output_root = args.output_root
    pipeline_args.output_stem = args.output_stem
    pipeline_args.max_frames = args.max_frames
    pipeline_args.frame_samples = args.frame_samples
    pipeline_args.show = args.show

    pipeline_args.disable_pins = False
    pipeline_args.velocity_center_source = "ma"
    pipeline_args.ma_window = args.ma_window
    pipeline_args.velocity_window_frames = args.velocity_window_frames
    pipeline_args.acceleration_window_frames = args.acceleration_window_frames
    pipeline_args.speed_ema_alpha = args.speed_ema_alpha
    pipeline_args.max_hand_speed_mm_s = args.max_hand_speed_mm_s
    pipeline_args.max_acceleration_mm_s2 = args.max_acceleration_mm_s2
    pipeline_args.output_stem = args.output_stem
    if args.expected_hand in {"left", "right"}:
        pipeline_args.active_side = args.expected_hand
    return pipeline_args


def output_paths(output_root: Path, output_stem: str) -> Dict[str, Path]:
    return {
        "tracker_video": output_root / f"{output_stem}_preview.avi",
        "tracker_csv": output_root / f"{output_stem}.csv",
        "live_video": output_root / f"{output_stem}_live_check.avi",
        "check_csv": output_root / f"{output_stem}_check.csv",
        "check_json": output_root / f"{output_stem}_check_summary.json",
        "check_txt": output_root / f"{output_stem}_check_report.txt",
    }


def finite_or_zero(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def hole_indices(df: pd.DataFrame) -> List[int]:
    indices: List[int] = []
    for col in df.columns:
        if col.startswith("hole_") and col.endswith("_x_mm"):
            try:
                indices.append(int(col.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(set(indices))


def infer_hole_sides(df: pd.DataFrame) -> Dict[int, str]:
    indices = hole_indices(df)
    if not indices:
        return {}
    first = df.iloc[0]
    xs = [finite_or_zero(first.get(f"hole_{idx:02d}_x_mm")) for idx in indices]
    midpoint = float(np.median(xs))
    return {
        idx: "left" if finite_or_zero(first.get(f"hole_{idx:02d}_x_mm")) < midpoint else "right"
        for idx in indices
    }


def side_activity_score(df: pd.DataFrame, hole_sides: Dict[int, str], side: str) -> float:
    score = 0.0
    for idx, hole_side in hole_sides.items():
        if hole_side != side:
            continue
        for suffix, weight in (
            ("interaction_signal", 2.0),
            ("pin_near", 1.5),
            ("occupied", 3.0),
            ("prob", 0.25),
        ):
            col = f"hole_{idx:02d}_{suffix}"
            if col in df.columns:
                score += weight * float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    return score


def infer_target_side(df: pd.DataFrame, hole_sides: Dict[int, str], requested_side: str) -> str:
    if requested_side in {"left", "right"}:
        return requested_side
    left_score = side_activity_score(df, hole_sides, "left")
    right_score = side_activity_score(df, hole_sides, "right")
    if left_score == 0 and right_score == 0 and "hand_x_mm" in df.columns:
        indices = hole_indices(df)
        if indices:
            first = df.iloc[0]
            hole_mid = np.median([finite_or_zero(first.get(f"hole_{idx:02d}_x_mm")) for idx in indices])
            hand_x = pd.to_numeric(df["hand_x_mm"], errors="coerce").dropna()
            if not hand_x.empty:
                return "left" if float((hand_x < hole_mid).mean()) >= 0.5 else "right"
    return "left" if left_score >= right_score else "right"


def compute_brightness_flags(video_path: Path, frame_count: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    brightness: List[float] = []
    while len(brightness) < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))
    cap.release()

    values = np.asarray(brightness, dtype=np.float32)
    flags = np.zeros(frame_count, dtype=bool)
    if len(values) >= 2:
        smooth = pd.Series(values).rolling(window=15, min_periods=1).mean().to_numpy()
        deltas = np.diff(smooth, prepend=smooth[0])
        flags[: len(deltas)] = np.abs(deltas) >= threshold
    if len(values) < frame_count:
        values = np.pad(values, (0, frame_count - len(values)), constant_values=np.nan)
    return flags, values[:frame_count]


def side_frame_counts(row: pd.Series, hole_sides: Dict[int, str]) -> Dict[str, Dict[str, int]]:
    counts = {
        "left": {"occupied": 0, "pin_near": 0, "interaction": 0},
        "right": {"occupied": 0, "pin_near": 0, "interaction": 0},
    }
    for idx, side in hole_sides.items():
        counts[side]["occupied"] += int(finite_or_zero(row.get(f"hole_{idx:02d}_occupied")) > 0.5)
        counts[side]["pin_near"] += int(finite_or_zero(row.get(f"hole_{idx:02d}_pin_near")) > 0.5)
        counts[side]["interaction"] += int(finite_or_zero(row.get(f"hole_{idx:02d}_interaction_signal")) > 0.5)
    return counts


def smoothed_series(series: pd.Series, alpha: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()


def build_check_table(
    df: pd.DataFrame,
    video_path: Path,
    target_side: str,
    expected_hand: str,
    hole_sides: Dict[int, str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    check = pd.DataFrame()
    check["frame"] = df["frame"].astype(int)
    check["time_s"] = pd.to_numeric(df["time_s"], errors="coerce").fillna(0)
    check["hand_detected"] = pd.to_numeric(df.get("hand_detected", 0), errors="coerce").fillna(0).astype(int)
    check["speed_raw_mm_s"] = pd.to_numeric(df.get("speed_mm_s", np.nan), errors="coerce")
    check["speed_ema_mm_s"] = pd.to_numeric(df.get("speed_ema_mm_s", np.nan), errors="coerce")
    check["speed_final_mm_s"] = smoothed_series(check["speed_ema_mm_s"], args.display_speed_alpha)

    brightness_flags, brightness = compute_brightness_flags(video_path, len(df), args.brightness_jump_threshold)
    check["brightness_mean"] = brightness
    check["light_change_warning"] = brightness_flags.astype(int)

    last_seen_time = 0.0
    missing_flags: List[int] = []
    for detected, time_s in zip(check["hand_detected"], check["time_s"]):
        if int(detected) == 1:
            last_seen_time = float(time_s)
        missing_flags.append(int(float(time_s) - last_seen_time >= args.missing_hand_seconds))
    check["missing_hand_warning"] = missing_flags

    if "selected_handedness" in df.columns and expected_hand in {"left", "right"}:
        labels = df["selected_handedness"].fillna("").astype(str).str.lower()
        check["wrong_hand_warning"] = (
            (check["hand_detected"] == 1)
            & labels.isin(["left", "right"])
            & (labels != expected_hand)
        ).astype(int)
    else:
        check["wrong_hand_warning"] = 0

    wrong_side_flags: List[int] = []
    target_occupied: List[int] = []
    other_activity: List[int] = []
    target_pin_near: List[int] = []
    for _, row in df.iterrows():
        counts = side_frame_counts(row, hole_sides)
        other_side = "right" if target_side == "left" else "left"
        target_occupied.append(counts[target_side]["occupied"])
        target_pin_near.append(counts[target_side]["pin_near"])
        other_active = counts[other_side]["pin_near"] + counts[other_side]["interaction"]
        other_activity.append(other_active)
        wrong_side_flags.append(int(other_active > 0))
    check["target_side"] = target_side
    check["target_occupied_count"] = target_occupied
    check["target_pin_near_count"] = target_pin_near
    check["other_side_activity_count"] = other_activity
    check["wrong_side_warning"] = wrong_side_flags

    speed_raw = check["speed_raw_mm_s"].fillna(0)
    speed_final = check["speed_final_mm_s"].fillna(0)
    check["speed_spike_warning"] = (
        (speed_raw >= args.min_spike_speed_mm_s)
        & (speed_raw >= np.maximum(args.min_spike_speed_mm_s, args.spike_ratio * np.maximum(speed_final, 1.0)))
    ).astype(int)

    warning_cols = [
        "wrong_hand_warning",
        "wrong_side_warning",
        "light_change_warning",
        "missing_hand_warning",
        "speed_spike_warning",
    ]
    check["warning_count"] = check[warning_cols].sum(axis=1).astype(int)
    check["ok"] = (check["warning_count"] == 0).astype(int)
    return check


def count_segments(flags: Iterable[int]) -> int:
    count = 0
    prev = 0
    for flag in flags:
        cur = int(flag) > 0
        if cur and not prev:
            count += 1
        prev = cur
    return count


def draw_text(frame: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: Tuple[int, int, int]) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_corner_panel(
    frame: np.ndarray,
    row: pd.Series,
    speed_history: Sequence[float],
    target_side: str,
    expected_hand: str,
    graph_max: float,
) -> None:
    h, w = frame.shape[:2]
    panel_w = min(270, max(230, w // 3))
    panel_h = 158
    x0 = max(8, w - panel_w - 10)
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 24, 28), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 1)

    warning_count = int(row.get("warning_count", 0))
    status = "OK" if warning_count == 0 else "CHECK"
    status_color = (90, 230, 90) if warning_count == 0 else (0, 210, 255)
    draw_text(frame, f"{status}  side={target_side}", (x0 + 10, y0 + 24), 0.55, status_color)
    if expected_hand in {"left", "right"}:
        draw_text(frame, f"hand={expected_hand}", (x0 + 10, y0 + 47), 0.46, (235, 235, 235))
    else:
        draw_text(frame, "hand=auto", (x0 + 10, y0 + 47), 0.46, (235, 235, 235))

    graph_x = x0 + 12
    graph_y = y0 + 62
    graph_w = panel_w - 24
    graph_h = 52
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (60, 64, 70), 1)
    cv2.line(frame, (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), (130, 130, 130), 1)

    values = [v for v in speed_history if math.isfinite(v)]
    if len(values) >= 2:
        max_y = max(graph_max, float(np.nanpercentile(values, 95)) * 1.2, 1.0)
        points: List[Tuple[int, int]] = []
        last_values = values[-graph_w:]
        for i, value in enumerate(last_values):
            px = graph_x + int(round(i * graph_w / max(1, len(last_values) - 1)))
            py = graph_y + graph_h - int(round(min(value, max_y) / max_y * (graph_h - 4))) - 2
            points.append((px, py))
        for p1, p2 in zip(points[:-1], points[1:]):
            cv2.line(frame, p1, p2, (80, 220, 160), 2, cv2.LINE_AA)

    speed = finite_or_zero(row.get("speed_final_mm_s"))
    target_count = int(row.get("target_occupied_count", 0))
    draw_text(frame, f"v={speed:5.1f} mm/s  pins={target_count}", (x0 + 10, y0 + 134), 0.45, (245, 245, 245))

    warnings: List[str] = []
    if int(row.get("wrong_hand_warning", 0)):
        warnings.append("wrong hand")
    if int(row.get("wrong_side_warning", 0)):
        warnings.append("wrong side")
    if int(row.get("light_change_warning", 0)):
        warnings.append("lights")
    if int(row.get("missing_hand_warning", 0)):
        warnings.append("hand lost")
    if int(row.get("speed_spike_warning", 0)):
        warnings.append("speed spike")
    if warnings:
        draw_text(frame, " / ".join(warnings[:2]), (x0 + 10, y0 + 153), 0.40, (0, 210, 255))


def write_overlay_video(
    tracker_video: Path,
    output_video: Path,
    check: pd.DataFrame,
    target_side: str,
    expected_hand: str,
    graph_seconds: float,
    show_final: bool,
) -> None:
    cap = cv2.VideoCapture(str(tracker_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open tracker video: {tracker_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open final video writer: {output_video}")

    max_history = max(2, int(round(graph_seconds * fps)))
    speed_history: deque = deque(maxlen=max_history)
    graph_max = max(500.0, float(check["speed_final_mm_s"].dropna().quantile(0.95)) * 1.35 if not check["speed_final_mm_s"].dropna().empty else 500.0)

    if show_final:
        try:
            cv2.namedWindow("final quick check", cv2.WINDOW_NORMAL)
        except cv2.error:
            show_final = False

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(check):
            break
        row = check.iloc[frame_idx]
        speed_history.append(finite_or_zero(row.get("speed_final_mm_s")))
        draw_corner_panel(frame, row, list(speed_history), target_side, expected_hand, graph_max)
        writer.write(frame)
        if show_final:
            try:
                cv2.imshow("final quick check", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                show_final = False
        frame_idx += 1

    cap.release()
    writer.release()
    if show_final:
        cv2.destroyAllWindows()


def write_summary(paths: Dict[str, Path], check: pd.DataFrame, target_side: str, expected_hand: str) -> Dict:
    warning_cols = [
        "wrong_hand_warning",
        "wrong_side_warning",
        "light_change_warning",
        "missing_hand_warning",
        "speed_spike_warning",
    ]
    summary = {
        "target_side": target_side,
        "expected_hand": expected_hand,
        "frames": int(len(check)),
        "ok_frame_rate": float(check["ok"].mean()) if len(check) else 0.0,
        "final_target_occupied_count": int(check["target_occupied_count"].iloc[-1]) if len(check) else 0,
        "mean_final_speed_mm_s": float(check["speed_final_mm_s"].dropna().mean()) if not check["speed_final_mm_s"].dropna().empty else None,
        "max_final_speed_mm_s": float(check["speed_final_mm_s"].dropna().max()) if not check["speed_final_mm_s"].dropna().empty else None,
        "warnings_by_frame": {col: int(check[col].sum()) for col in warning_cols},
        "warning_segments": {col: int(count_segments(check[col])) for col in warning_cols},
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    with open(paths["check_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "Final quick check",
        f"target_side: {target_side}",
        f"expected_hand: {expected_hand}",
        f"frames: {summary['frames']}",
        f"ok_frame_rate: {summary['ok_frame_rate']:.3f}",
        f"final_target_occupied_count: {summary['final_target_occupied_count']}",
        f"mean_final_speed_mm_s: {summary['mean_final_speed_mm_s']}",
        f"max_final_speed_mm_s: {summary['max_final_speed_mm_s']}",
        "warnings_by_frame:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in summary["warnings_by_frame"].items())
    paths["check_txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_root, args.output_stem)
    video_path = Path(args.video)

    if not args.skip_tracker:
        process_video(build_pipeline_args(args))

    if not paths["tracker_csv"].exists():
        raise FileNotFoundError(f"Missing tracker CSV: {paths['tracker_csv']}")
    if not paths["tracker_video"].exists():
        raise FileNotFoundError(f"Missing tracker video: {paths['tracker_video']}")

    df = pd.read_csv(paths["tracker_csv"])
    hole_sides = infer_hole_sides(df)
    target_side = infer_target_side(df, hole_sides, args.target_side)
    check = build_check_table(
        df=df,
        video_path=video_path,
        target_side=target_side,
        expected_hand=args.expected_hand,
        hole_sides=hole_sides,
        args=args,
    )
    check.to_csv(paths["check_csv"], index=False)

    write_overlay_video(
        tracker_video=paths["tracker_video"],
        output_video=paths["live_video"],
        check=check,
        target_side=target_side,
        expected_hand=args.expected_hand,
        graph_seconds=args.graph_seconds,
        show_final=args.show_final,
    )
    summary = write_summary(paths, check, target_side, args.expected_hand)

    print("\nFinal quick-check saved:")
    print(f"  live video: {paths['live_video']}")
    print(f"  check csv:  {paths['check_csv']}")
    print(f"  summary:    {paths['check_json']}")
    print(f"  report:     {paths['check_txt']}")
    print("\nQuick summary:")
    print(f"  target side: {summary['target_side']}")
    print(f"  ok frame rate: {summary['ok_frame_rate']:.3f}")
    print(f"  final target occupied count: {summary['final_target_occupied_count']}")


if __name__ == "__main__":
    main()
