from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization import PANEL_WIDTH


def finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df.get(column, pd.Series(index=df.index, dtype=float)), errors="coerce")


def measurement_df(df: pd.DataFrame) -> pd.DataFrame:
    time = finite_series(df, "measurement_time_s")
    running = finite_series(df, "measurement_running").fillna(0).astype(int)
    out = df[(running > 0) & np.isfinite(time)].copy()
    return out.reset_index(drop=True)


def save_annotated_frame(video_path: Path, frame_idx: int, output_path: Path, crop_panel: bool = True) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_idx)))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        if crop_panel and frame.shape[1] > 2 * PANEL_WIDTH:
            frame = frame[:, PANEL_WIDTH : frame.shape[1] - PANEL_WIDTH]
        elif crop_panel and frame.shape[1] > PANEL_WIDTH:
            frame = frame[:, : frame.shape[1] - PANEL_WIDTH]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
    finally:
        cap.release()


def setup_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d8dee5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_trajectory(df: pd.DataFrame, output_path: Path) -> None:
    x = finite_series(df, "hand_center_mm_x")
    y = finite_series(df, "hand_center_mm_y")
    valid = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(5.0, 4.1), constrained_layout=True)
    ax.plot(x[valid], y[valid], color="#1f77b4", linewidth=1.6)
    ax.scatter(x[valid].iloc[:1], y[valid].iloc[:1], color="#2ca02c", s=28, label="zacetek")
    ax.scatter(x[valid].iloc[-1:], y[valid].iloc[-1:], color="#d62728", s=28, label="konec")
    ax.set_aspect("equal", adjustable="box")
    setup_axes(ax, "x [mm]", "y [mm]")
    ax.set_title("Trajektorija sredisca roke")
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_kinematics(df: pd.DataFrame, output_path: Path, title: str, columns: Tuple[str, str, str]) -> None:
    t = finite_series(df, "measurement_time_s")
    labels = ("d(t) [mm]", "v(t) [mm/s]", "a(t) [mm/s2]")
    colors = ("#1f77b4", "#2ca02c", "#d17c0f")
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.4), sharex=True, constrained_layout=True)
    for ax, column, label, color in zip(axes, columns, labels, colors):
        values = finite_series(df, column)
        ax.plot(t, values, color=color, linewidth=1.4)
        setup_axes(ax, "", label)
    axes[-1].set_xlabel("cas od zacetka [s]")
    axes[0].set_title(title)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_thumb_index_kinematics(df: pd.DataFrame, output_path: Path) -> None:
    t = finite_series(df, "measurement_time_s")
    pairs = [
        ("thumb_tip_path_mm_cumulative", "index_tip_path_mm_cumulative", "d(t) [mm]"),
        ("thumb_tip_speed_mm_s_smooth", "index_tip_speed_mm_s_smooth", "v(t) [mm/s]"),
        ("thumb_tip_acceleration_mm_s2_smooth", "index_tip_acceleration_mm_s2_smooth", "a(t) [mm/s2]"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.4), sharex=True, constrained_layout=True)
    for ax, (thumb_col, index_col, ylabel) in zip(axes, pairs):
        ax.plot(t, finite_series(df, thumb_col), color="#17becf", linewidth=1.35, label="palec")
        ax.plot(t, finite_series(df, index_col), color="#9467bd", linewidth=1.35, label="kazalec")
        setup_axes(ax, "", ylabel)
    axes[0].set_title("Kinematika palca in kazalca")
    axes[0].legend(frameon=False, loc="best")
    axes[-1].set_xlabel("cas od zacetka [s]")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_led_sequence(df: pd.DataFrame, output_path: Path) -> None:
    t = finite_series(df, "time_s")
    left_score = finite_series(df, "light_left_score")
    right_score = finite_series(df, "light_right_score")
    left_on = finite_series(df, "light_left_on").fillna(0)
    right_on = finite_series(df, "light_right_on").fillna(0)
    trial_start = finite_series(df, "trial_start_frame").dropna()
    fps = 1.0 / np.nanmedian(np.diff(t[np.isfinite(t)])) if np.isfinite(t).sum() > 2 else 25.0
    start_t = float(trial_start.iloc[0]) / fps if not trial_start.empty and fps > 0 else float("nan")

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, constrained_layout=True)
    axes[0].plot(t, left_score, color="#1f77b4", linewidth=1.3, label="levo polje")
    axes[0].plot(t, right_score, color="#d62728", linewidth=1.3, label="desno polje")
    setup_axes(axes[0], "", "LED ocena [-]")
    axes[0].legend(frameon=False, loc="best")
    axes[1].step(t, left_on, where="post", color="#1f77b4", linewidth=1.2, label="levo sveti")
    axes[1].step(t, right_on + 1.15, where="post", color="#d62728", linewidth=1.2, label="desno sveti")
    axes[1].set_yticks([0, 1, 1.15, 2.15])
    axes[1].set_yticklabels(["L off", "L on", "D off", "D on"])
    setup_axes(axes[1], "cas videa [s]", "stanje")
    if np.isfinite(start_t):
        for ax in axes:
            ax.axvline(start_t, color="#111111", linestyle="--", linewidth=1.1)
        axes[0].text(start_t, axes[0].get_ylim()[1], " zacetek", va="top", ha="left", fontsize=9)
    axes[0].set_title("Zaznava LED sekvence in zacetka meritve")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_peg_detection(df: pd.DataFrame, output_path: Path) -> None:
    t = finite_series(df, "measurement_time_s")
    count = finite_series(df, "peg_target_count")
    covered = finite_series(df, "peg_right_covered").fillna(0)
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    ax.step(t, count, where="post", color="#1f77b4", linewidth=1.5, label="zaznano stevilo zaticov")
    ax.fill_between(t, 0, count.max() if np.isfinite(count.max()) else 1, where=covered > 0, color="#d8dee5", alpha=0.45, label="polje zakrito")
    setup_axes(ax, "cas od zacetka [s]", "stevilo [-]")
    ax.set_title("Eksperimentalna zaznava odlaganja zaticov")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def metric_summary(df: pd.DataFrame, run_df: pd.DataFrame) -> dict:
    def max_value(column: str) -> float:
        values = finite_series(run_df, column).dropna()
        return float(values.max()) if not values.empty else float("nan")

    def mean_value(column: str) -> float:
        values = finite_series(run_df, column).dropna()
        return float(values.mean()) if not values.empty else float("nan")

    completed = finite_series(df, "measurement_completed").fillna(0)
    trial_start = finite_series(df, "trial_start_frame").dropna()
    end_frame = finite_series(df, "measurement_end_frame").dropna()
    return {
        "processed_frames": int(len(df)),
        "detection_rate": float(finite_series(df, "hand_detected").fillna(0).mean()),
        "trial_side": str(df.loc[finite_series(df, "trial_started").fillna(0) > 0, "trial_side"].iloc[-1])
        if (finite_series(df, "trial_started").fillna(0) > 0).any()
        else "",
        "trial_start_frame": float(trial_start.iloc[0]) if not trial_start.empty else float("nan"),
        "measurement_completed": bool(completed.max() > 0),
        "measurement_end_frame": float(end_frame.iloc[0]) if not end_frame.empty else float("nan"),
        "measurement_duration_s": max_value("measurement_time_s"),
        "path_length_mm": max_value("path_length_mm_cumulative"),
        "mean_speed_mm_s": mean_value("speed_mm_s_smooth"),
        "max_speed_mm_s": max_value("speed_mm_s_smooth"),
        "mean_acceleration_mm_s2": mean_value("acceleration_mm_s2_smooth"),
        "max_acceleration_mm_s2": max_value("acceleration_mm_s2_smooth"),
        "thumb_path_length_mm": max_value("thumb_tip_path_mm_cumulative"),
        "index_path_length_mm": max_value("index_tip_path_mm_cumulative"),
        "max_peg_target_count": max_value("peg_target_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export article-ready figures from a 9HPT pipeline run.")
    parser.add_argument("--csv", required=True, help="Pipeline CSV output.")
    parser.add_argument("--video", required=True, help="Annotated pipeline video output.")
    parser.add_argument("--output-dir", required=True, help="Directory for article figures.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    video_path = Path(args.video)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    run_df = measurement_df(df)
    if run_df.empty:
        raise RuntimeError("No measurement rows available for plotting.")

    trial_start_frame = finite_series(df, "trial_start_frame").dropna()
    start_frame = int(trial_start_frame.iloc[0]) if not trial_start_frame.empty else int(run_df["frame_idx"].iloc[0])
    hand_frame = int(min(int(df["frame_idx"].max()), start_frame + 55))
    peg_count = finite_series(df, "peg_target_count")
    peg_frame = int(df.loc[peg_count.idxmax(), "frame_idx"]) if peg_count.notna().any() else hand_frame

    save_annotated_frame(video_path, max(0, start_frame - 78), out_dir / "calibrated_frame.png")
    save_annotated_frame(video_path, hand_frame, out_dir / "hand_landmarks_frame.png")
    save_annotated_frame(video_path, peg_frame, out_dir / "peg_detection_frame.png")
    plot_trajectory(run_df, out_dir / "hand_trajectory.png")
    plot_kinematics(
        run_df,
        out_dir / "hand_kinematics.png",
        "Kinematika sredisca roke",
        ("path_length_mm_cumulative", "speed_mm_s_smooth", "acceleration_mm_s2_smooth"),
    )
    plot_thumb_index_kinematics(run_df, out_dir / "thumb_index_kinematics.png")
    plot_led_sequence(df, out_dir / "led_sequence.png")
    plot_peg_detection(run_df, out_dir / "peg_detection_experiment.png")

    summary = metric_summary(df, run_df)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
