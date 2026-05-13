import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_default_video(data_root):
    videos = sorted(data_root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError("No .mp4 videos found in /data/Data")
    return videos[0]


def parse_roi(roi_text, width, height):
    if roi_text is None:
        return 0, 0, width, height

    parts = [int(v.strip()) for v in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be in format x,y,w,h")

    x, y, w, h = parts
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def make_side_roi_mask(width, height, side, roi):
    mask = np.zeros((height, width), dtype=np.uint8)

    x, y, w, h = roi
    mask[y:y + h, x:x + w] = 255

    if side == "right":
        side_mask = np.zeros_like(mask)
        side_mask[:, int(0.25 * width):] = 255
        mask = cv2.bitwise_and(mask, side_mask)
    elif side == "left":
        side_mask = np.zeros_like(mask)
        side_mask[:, :int(0.75 * width)] = 255
        mask = cv2.bitwise_and(mask, side_mask)
    elif side == "all":
        pass
    else:
        raise ValueError("--side must be right, left or all")

    return mask


def skin_mask_bgr(frame):
    blur = cv2.GaussianBlur(frame, (7, 7), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)

    # HSV skin ranges. OpenCV H is [0, 179].
    mask_hsv_1 = cv2.inRange(
        hsv,
        np.array([0, 20, 35], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8),
    )

    # Additional range for slightly different lighting.
    mask_hsv_2 = cv2.inRange(
        hsv,
        np.array([160, 20, 35], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    mask_hsv = cv2.bitwise_or(mask_hsv_1, mask_hsv_2)

    # YCrCb skin range.
    mask_ycrcb = cv2.inRange(
        ycrcb,
        np.array([0, 133, 75], dtype=np.uint8),
        np.array([255, 180, 135], dtype=np.uint8),
    )

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((11, 11), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask


def motion_mask_bgr(frame, prev_gray, bg_subtractor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mog = bg_subtractor.apply(frame)
    _, mog = cv2.threshold(mog, 180, 255, cv2.THRESH_BINARY)

    if prev_gray is None:
        diff_mask = np.zeros_like(gray)
    else:
        diff = cv2.absdiff(gray, prev_gray)
        _, diff_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)

    motion = cv2.bitwise_or(mog, diff_mask)

    kernel = np.ones((7, 7), np.uint8)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
    motion = cv2.dilate(motion, kernel, iterations=2)

    return motion, gray


def contour_mask(shape, contour):
    m = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    return m


def contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def select_hand_candidate(base_mask, skin, motion, roi_mask, prev_center, width, height, min_area, max_area):
    search_mask = cv2.bitwise_and(base_mask, roi_mask)

    contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1e18

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        center = contour_center(contour)
        if center is None:
            continue

        cx, cy = center
        x, y, w, h = cv2.boundingRect(contour)

        if w <= 2 or h <= 2:
            continue

        # Ignore objects touching image borders too much.
        border_penalty = 0.0
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            border_penalty = 1.5

        cmask = contour_mask(skin.shape, contour)
        contour_pixels = max(1, int(np.count_nonzero(cmask)))

        skin_overlap = np.count_nonzero(cv2.bitwise_and(cmask, skin)) / contour_pixels
        motion_overlap = np.count_nonzero(cv2.bitwise_and(cmask, motion)) / contour_pixels

        hull = cv2.convexHull(contour)
        hull_area = max(1.0, cv2.contourArea(hull))
        solidity = area / hull_area

        # Penalize very long thin regions, like table/wood edges.
        aspect = max(float(w) / float(h), float(h) / float(w))
        aspect_penalty = max(0.0, aspect - 4.0) * 0.6

        distance_penalty = 0.0
        if prev_center is not None:
            dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
            distance_penalty = 0.018 * dist

        # Candidate score:
        # motion is important to avoid static false positives,
        # skin helps find hand,
        # distance term prevents jumps,
        # aspect/border terms reject wrong regions.
        score = (
            3.0 * motion_overlap
            + 2.0 * skin_overlap
            + 0.9 * solidity
            + 0.00008 * area
            - distance_penalty
            - aspect_penalty
            - border_penalty
        )

        # If there is no previous center, require at least some skin or motion.
        if prev_center is None and skin_overlap < 0.08 and motion_overlap < 0.05:
            continue

        if score > best_score:
            best_score = score
            best = {
                "center": center,
                "contour": contour,
                "bbox": (x, y, w, h),
                "area": area,
                "score": score,
                "skin_overlap": skin_overlap,
                "motion_overlap": motion_overlap,
            }

    return best


def smooth_center(measured, previous, alpha):
    if previous is None:
        return measured
    x = alpha * measured[0] + (1.0 - alpha) * previous[0]
    y = alpha * measured[1] + (1.0 - alpha) * previous[1]
    return int(round(x)), int(round(y))


def plot_outputs(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_s"], df["x_px"], label="x")
    plt.plot(df["time_s"], df["y_px"], label="y")
    plt.xlabel("time (s)")
    plt.ylabel("position (px)")
    plt.title("Hand center position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_x_y.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["speed_px_s"])
    plt.xlabel("time (s)")
    plt.ylabel("speed (px/s)")
    plt.title("Hand speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "speed_over_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["acceleration_px_s2"])
    plt.xlabel("time (s)")
    plt.ylabel("acceleration (px/s^2)")
    plt.title("Hand acceleration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "acceleration_over_time.png", dpi=150)
    plt.close()

    valid = df.dropna(subset=["x_px", "y_px"])
    plt.figure(figsize=(6, 6))
    plt.plot(valid["x_px"], valid["y_px"], marker=".", linewidth=1)
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.title("Hand trajectory")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_xy.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--side", type=str, default="right", choices=["right", "left", "all"])
    parser.add_argument("--roi", type=str, default=None, help="Optional ROI: x,y,w,h")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--min-area", type=float, default=700.0)
    parser.add_argument("--max-area-frac", type=float, default=0.28)
    parser.add_argument("--max-jump", type=float, default=95.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    data_root = Path("/data/Data")
    output_dir = Path("/workspace/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video) if args.video else find_default_video(data_root)

    print("Input video:", video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {}".format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Size: {}x{}, fps={}, frames={}".format(width, height, fps, frame_count))

    roi = parse_roi(args.roi, width, height)
    roi_mask = make_side_roi_mask(width, height, args.side, roi)

    max_area = args.max_area_frac * width * height

    out_video = output_dir / "hand_tracking_v2.avi"
    out_csv = output_dir / "hand_tracking_v2.csv"

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open VideoWriter for {}".format(out_video))

    bg = cv2.createBackgroundSubtractorMOG2(
        history=80,
        varThreshold=20,
        detectShadows=False,
    )

    prev_gray = None
    prev_center_raw = None
    prev_center_smooth = None
    prev_time = None
    prev_speed = np.nan
    trajectory = []
    rows = []

    show_ok = args.show
    if show_ok:
        try:
            cv2.namedWindow("hand tracking v2", cv2.WINDOW_NORMAL)
            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        except cv2.error:
            show_ok = False
            print("GUI unavailable; saving video only.")

    frame_idx = 0
    missed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        t = frame_idx / fps

        skin = skin_mask_bgr(frame)
        motion, prev_gray = motion_mask_bgr(frame, prev_gray, bg)

        # Main candidate mask:
        # skin OR motion, then restrict by ROI/side.
        # This can still follow a hand when lighting makes skin segmentation imperfect.
        base = cv2.bitwise_or(skin, motion)
        base = cv2.bitwise_and(base, roi_mask)

        kernel = np.ones((9, 9), np.uint8)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)

        candidate = select_hand_candidate(
            base_mask=base,
            skin=skin,
            motion=motion,
            roi_mask=roi_mask,
            prev_center=prev_center_smooth,
            width=width,
            height=height,
            min_area=args.min_area,
            max_area=max_area,
        )

        detected = False
        rejected_jump = False
        raw_center = None
        smooth = None
        contour = None
        bbox = None
        area = 0.0
        score = np.nan
        skin_overlap = np.nan
        motion_overlap = np.nan

        if candidate is not None:
            raw_center = candidate["center"]
            contour = candidate["contour"]
            bbox = candidate["bbox"]
            area = candidate["area"]
            score = candidate["score"]
            skin_overlap = candidate["skin_overlap"]
            motion_overlap = candidate["motion_overlap"]

            if prev_center_smooth is not None:
                jump = np.hypot(raw_center[0] - prev_center_smooth[0], raw_center[1] - prev_center_smooth[1])
                if jump > args.max_jump:
                    rejected_jump = True
                else:
                    detected = True
            else:
                detected = True

        if detected:
            smooth = smooth_center(raw_center, prev_center_smooth, args.alpha)
            prev_center_raw = raw_center
            prev_center_smooth = smooth
            missed = 0
        else:
            # Do not jump to a false detection.
            # If detection is missing, keep last center for a short time.
            missed += 1
            if prev_center_smooth is not None and missed <= int(fps * 0.5):
                smooth = prev_center_smooth
            else:
                smooth = None

        speed = np.nan
        acceleration = np.nan

        if smooth is not None:
            if prev_time is not None and len(trajectory) > 0:
                dt = t - prev_time
                if dt > 0:
                    prev = trajectory[-1]
                    speed = float(np.hypot(smooth[0] - prev[0], smooth[1] - prev[1]) / dt)
                    if not np.isnan(prev_speed):
                        acceleration = float((speed - prev_speed) / dt)

            trajectory.append(smooth)
            prev_time = t
            prev_speed = speed
        else:
            prev_speed = np.nan

        annotated = frame.copy()

        # Show selected ROI.
        rx, ry, rw, rh = roi
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (80, 80, 80), 1)

        if contour is not None and not rejected_jump:
            cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)

        if bbox is not None and not rejected_jump:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if raw_center is not None and not rejected_jump:
            cv2.circle(annotated, raw_center, 5, (0, 165, 255), -1)

        if smooth is not None:
            cv2.circle(annotated, smooth, 8, (0, 0, 255), -1)
            cv2.circle(annotated, smooth, 14, (255, 255, 255), 2)

        for a, b in zip(trajectory[:-1], trajectory[1:]):
            cv2.line(annotated, a, b, (0, 255, 255), 2)

        status = "detected" if detected else "tracking/hold" if smooth is not None else "missing"
        if rejected_jump:
            status = "rejected jump"

        text1 = "frame={} t={:.2f}s status={} area={:.0f}".format(frame_idx, t, status, area)
        text2 = "pos={} speed={} acc={}".format(
            smooth if smooth is not None else "nan",
            "{:.1f}px/s".format(speed) if not np.isnan(speed) else "nan",
            "{:.1f}px/s2".format(acceleration) if not np.isnan(acceleration) else "nan",
        )
        text3 = "score={} skin={} motion={}".format(
            "{:.2f}".format(score) if not np.isnan(score) else "nan",
            "{:.2f}".format(skin_overlap) if not np.isnan(skin_overlap) else "nan",
            "{:.2f}".format(motion_overlap) if not np.isnan(motion_overlap) else "nan",
        )

        cv2.putText(annotated, text1, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated, text2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated, text3, (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        writer.write(annotated)

        if frame_idx in [0, 25, 50, 100, 200, 400]:
            cv2.imwrite(str(output_dir / "v2_debug_frame_{:04d}.jpg".format(frame_idx)), annotated)
            cv2.imwrite(str(output_dir / "v2_debug_mask_{:04d}.jpg".format(frame_idx)), base)

        rows.append({
            "frame": frame_idx,
            "time_s": t,
            "detected": detected,
            "status": status,
            "x_px": float(smooth[0]) if smooth is not None else np.nan,
            "y_px": float(smooth[1]) if smooth is not None else np.nan,
            "x_raw_px": float(raw_center[0]) if raw_center is not None else np.nan,
            "y_raw_px": float(raw_center[1]) if raw_center is not None else np.nan,
            "area_px": float(area),
            "score": float(score) if not np.isnan(score) else np.nan,
            "skin_overlap": float(skin_overlap) if not np.isnan(skin_overlap) else np.nan,
            "motion_overlap": float(motion_overlap) if not np.isnan(motion_overlap) else np.nan,
            "speed_px_s": float(speed) if not np.isnan(speed) else np.nan,
            "acceleration_px_s2": float(acceleration) if not np.isnan(acceleration) else np.nan,
        })

        if show_ok:
            try:
                cv2.imshow("hand tracking v2", annotated)
                cv2.imshow("mask", base)
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                    break
            except cv2.error:
                show_ok = False
                print("GUI failed; continuing without display.")

        frame_idx += 1

    cap.release()
    writer.release()

    if show_ok:
        cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    plot_outputs(df, output_dir)

    print("Saved video:", out_video)
    print("Saved CSV:", out_csv)
    print("Saved graphs in:", output_dir)
    print("Processed frames:", len(df))
    print("Detections:", int(df["detected"].sum()), "/", len(df))
    print("Video size:", out_video.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
