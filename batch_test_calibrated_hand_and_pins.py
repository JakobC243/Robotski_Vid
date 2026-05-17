import argparse
import csv
import json
import traceback
from pathlib import Path
from typing import Dict, List

# This batch runner intentionally reuses the single-video pipeline parser.
# That keeps all calibration, hand tracking, and pin parameters consistent
# between one-off tests and multi-video evaluation runs.
from track_calibrated_hand_and_pins import (
    parse_args as parse_pipeline_args,
    process_calibration_only,
    process_video,
)


def safe_case_name(input_root: Path, video_path: Path) -> str:
    # Use the relative video path so files from different patients never
    # overwrite each other even if the video names are similar.
    try:
        relative = video_path.relative_to(input_root)
    except ValueError:
        relative = video_path.name

    if isinstance(relative, Path):
        parts = relative.with_suffix("").parts
    else:
        parts = (str(relative),)

    return "__".join(parts).replace(" ", "_")


def flatten_for_csv(row: Dict) -> Dict:
    # CSV cells cannot store nested Python objects cleanly; JSON keeps the
    # per-video diagnostics readable while preserving machine-parsable data.
    flat = dict(row)
    for key, value in list(flat.items()):
        if isinstance(value, (list, tuple, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


def collect_videos(input_root: Path, pattern: str, limit: int) -> List[Path]:
    # Sorting makes repeated batch runs deterministic, which is important when
    # comparing threshold changes across the same subset of videos.
    videos = sorted(input_root.rglob(pattern))
    if limit > 0:
        videos = videos[:limit]
    return videos


def make_pipeline_args(batch_args: argparse.Namespace, video_path: Path, output_dir: Path) -> argparse.Namespace:
    # Start from the main script defaults, then override only the values that
    # the batch CLI exposes. This avoids a second, drifting configuration set.
    args = parse_pipeline_args([])
    args.video = str(video_path)
    args.output_root = str(output_dir)
    args.data_root = str(batch_args.input_root)
    args.show = False
    args.calibration_only = batch_args.mode == "calibration"
    args.disable_pins = batch_args.mode == "hand"
    if batch_args.mode == "hand":
        args.output_stem = "hand_ai_tracking_calibrated"
        args.velocity_center_source = "ma"
        args.ma_window = 3
        args.speed_ema_alpha = 0.30
    elif batch_args.mode in {"pins-v2", "full"}:
        args.output_stem = "calibrated_hand_pins_v2"
        args.velocity_center_source = "ma"
        args.ma_window = 3

    # Parameters below are forwarded directly to the underlying one-video
    # pipeline so batch tests can tune calibration, speed, and occupancy logic.
    passthrough_names = [
        "frame_samples",
        "require_board",
        "hole_spacing_mm",
        "num_grids",
        "min_hole_radius",
        "max_hole_radius",
        "min_spacing_px",
        "max_spacing_px",
        "min_grid_matches",
        "side_gray_threshold",
        "side_value_threshold",
        "side_min_area",
        "side_max_area",
        "side_min_circularity",
        "side_cluster_link_px",
        "side_min_grid_separation_px",
        "max_frames",
        "active_side",
        "center_mode",
        "min_detection_confidence",
        "min_tracking_confidence",
        "velocity_window_frames",
        "acceleration_window_frames",
        "occupancy_probability_threshold",
        "occupancy_brightness_drop",
        "occupancy_dark_fraction",
        "occupancy_relative_dark_drop",
        "occupancy_side_min_reference",
    ]
    for name in passthrough_names:
        setattr(args, name, getattr(batch_args, name))

    return args


def write_batch_outputs(rows: List[Dict], output_root: Path) -> None:
    # Rewriting the summary after every video gives a useful partial result if
    # a long run is interrupted or a later case fails.
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "batch_summary.json"
    csv_path = output_root / "batch_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    if rows:
        flat_rows = [flatten_for_csv(row) for row in rows]
        keys = sorted({key for row in flat_rows for key in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_rows)

    print(f"\nBatch JSON: {json_path}")
    print(f"Batch CSV:  {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for calibrated hand/pin tracking or fast calibration-only checks."
    )
    parser.add_argument("--input-root", type=str, default="../data", help="Root folder with patient videos.")
    parser.add_argument("--pattern", type=str, default="*camP_1*.mp4", help="Recursive video pattern.")
    parser.add_argument("--output-root", type=str, default="outputs_batch_calibrated")
    parser.add_argument("--mode", choices=["calibration", "hand", "pins-v2", "full"], default="calibration")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos. 0 means all matching videos.")
    parser.add_argument("--max-frames", type=int, default=0, help="Used only in full mode. 0 means full video.")

    parser.add_argument("--frame-samples", type=int, default=120)
    parser.add_argument("--require-board", action="store_true")
    parser.add_argument("--hole-spacing-mm", type=float, default=32.0)
    parser.add_argument("--num-grids", type=int, default=2)
    parser.add_argument("--min-hole-radius", type=float, default=2.8)
    parser.add_argument("--max-hole-radius", type=float, default=9.0)
    parser.add_argument("--min-spacing-px", type=float, default=16.0)
    parser.add_argument("--max-spacing-px", type=float, default=44.0)
    parser.add_argument("--min-grid-matches", type=int, default=7)
    parser.add_argument("--side-gray-threshold", type=int, default=170)
    parser.add_argument("--side-value-threshold", type=int, default=180)
    parser.add_argument("--side-min-area", type=float, default=15.0)
    parser.add_argument("--side-max-area", type=float, default=130.0)
    parser.add_argument("--side-min-circularity", type=float, default=0.25)
    parser.add_argument("--side-cluster-link-px", type=float, default=45.0)
    parser.add_argument("--side-min-grid-separation-px", type=float, default=90.0)

    parser.add_argument("--active-side", type=str, default="auto", choices=["auto", "left", "right"])
    parser.add_argument("--center-mode", type=str, default="palm", choices=["palm", "bbox", "all"])
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.45)
    parser.add_argument("--velocity-window-frames", type=int, default=3)
    parser.add_argument("--acceleration-window-frames", type=int, default=5)
    parser.add_argument("--occupancy-probability-threshold", type=float, default=0.55)
    parser.add_argument("--occupancy-brightness-drop", type=float, default=24.0)
    parser.add_argument("--occupancy-dark-fraction", type=float, default=0.22)
    parser.add_argument("--occupancy-relative-dark-drop", type=float, default=22.0)
    parser.add_argument("--occupancy-side-min-reference", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    batch_args = parse_args()
    input_root = Path(batch_args.input_root)
    output_root = Path(batch_args.output_root)
    videos = collect_videos(input_root, batch_args.pattern, batch_args.limit)

    if not videos:
        raise FileNotFoundError(f"No videos found under {input_root} with pattern {batch_args.pattern}")

    rows: List[Dict] = []
    print(f"Found {len(videos)} videos. Mode={batch_args.mode}")

    for idx, video_path in enumerate(videos, start=1):
        case_name = safe_case_name(input_root, video_path)
        case_output = output_root / case_name
        print(f"\n[{idx}/{len(videos)}] {video_path}")

        row: Dict = {
            "video": str(video_path),
            "case": case_name,
            "output_dir": str(case_output),
            "mode": batch_args.mode,
            "status": "ok",
        }

        try:
            # Failures are captured per video so one problematic recording does
            # not hide results from the rest of the batch.
            pipeline_args = make_pipeline_args(batch_args, video_path, case_output)
            if batch_args.mode == "calibration":
                summary = process_calibration_only(pipeline_args)
            else:
                summary = process_video(pipeline_args)
            row.update(summary)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc()
            print(f"ERROR: {exc}")

        rows.append(row)
        write_batch_outputs(rows, output_root)

    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    calibrated_count = sum(1 for row in rows if row.get("calibrated") is True)
    print(f"\nDone. OK={ok_count}/{len(rows)} calibrated={calibrated_count}/{len(rows)}")


if __name__ == "__main__":
    main()
