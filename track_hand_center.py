from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import argparse


def find_hand_mask(frame_bgr):
    """
    Osnovna segmentacija kože v HSV + YCrCb prostoru.
    Deluje kot začetna rešitev za roko, ni še končna robustna metoda.
    """

    # Malo zgladimo sliko
    blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

    # HSV segmentacija kože
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Tipično območje za kožo v HSV.
    # H je v OpenCV na intervalu [0,179], ne [0,360].
    lower_hsv = np.array([0, 25, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb segmentacija kože
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Združimo oba pogoja
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # Odstranimo šum in zapolnimo luknje
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def find_hand_center(mask, min_area=1500):
    """
    Poišče največjo konturo v maski in izračuna njeno središče.
    Vrne:
      center = (cx, cy) ali None
      contour = največja kontura ali None
      bbox = (x, y, w, h) ali None
      area = površina konture
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < min_area:
        return None, None, None, area

    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None, contour, cv2.boundingRect(contour), area

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    bbox = cv2.boundingRect(contour)

    return (cx, cy), contour, bbox, area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Pot do videa. Če ni podana, vzame prvi .mp4 iz /data/Data.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Prikaže okno s cv2.imshow. Deluje samo, če ima okolje GUI/X11.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Največ frame-ov za obdelavo. 0 pomeni cel video.",
    )
    args = parser.parse_args()

    data_dir = Path("/data/Data")
    out_dir = Path("/workspace/outputs")
    out_dir.mkdir(exist_ok=True)

    if args.video is None:
        videos = sorted(data_dir.glob("patient_*/*.mp4"))
        if not videos:
            raise FileNotFoundError("Ni najdenih .mp4 videov v /data/Data.")
        video_path = videos[0]
    else:
        video_path = Path(args.video)

    print("Video:", video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti videa: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("FPS:", fps)
    print("Frames:", n_frames)
    print("Size:", width, "x", height)

    out_video_path = out_dir / "hand_tracking_preview.mp4"
    out_csv_path = out_dir / "hand_center_tracking.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    records = []

    prev_center = None
    prev_time = None
    path_points = []

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        t = frame_idx / fps

        mask = find_hand_mask(frame)
        center, contour, bbox, area = find_hand_center(mask)

        speed_px_s = np.nan

        annotated = frame.copy()

        # Nariši zaznano roko
        if contour is not None:
            cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if center is not None:
            cx, cy = center
            path_points.append(center)

            # Izračun hitrosti v pix/s
            if prev_center is not None and prev_time is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                dt = t - prev_time
                if dt > 0:
                    speed_px_s = float(np.sqrt(dx * dx + dy * dy) / dt)

            prev_center = center
            prev_time = t

            # Središče roke
            cv2.circle(annotated, center, 8, (0, 0, 255), -1)
            cv2.circle(annotated, center, 14, (255, 255, 255), 2)
            cv2.putText(
                annotated,
                f"center=({cx},{cy})",
                (cx + 15, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Nariši trajektorijo do trenutnega frame-a
        for i in range(1, len(path_points)):
            cv2.line(annotated, path_points[i - 1], path_points[i], (0, 255, 255), 2)

        # Informacije na sliki
        cv2.putText(
            annotated,
            f"frame={frame_idx}  t={t:.2f}s  area={area:.0f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if not np.isnan(speed_px_s):
            cv2.putText(
                annotated,
                f"speed={speed_px_s:.1f} px/s",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(annotated)

        if args.show:
            cv2.imshow("Hand tracking", annotated)
            cv2.imshow("Hand mask", mask)

            # q za izhod
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        records.append(
            {
                "frame": frame_idx,
                "time_s": t,
                "x_px": center[0] if center is not None else np.nan,
                "y_px": center[1] if center is not None else np.nan,
                "area_px": area,
                "speed_px_s": speed_px_s,
            }
        )

        frame_idx += 1

    cap.release()
    writer.release()

    if args.show:
        cv2.destroyAllWindows()

    df = pd.DataFrame(records)

    # Dodamo še pospešek iz hitrosti
    df["acceleration_px_s2"] = df["speed_px_s"].diff() * fps

    df.to_csv(out_csv_path, index=False)

    print()
    print("Saved annotated video:", out_video_path)
    print("Saved tracking CSV:", out_csv_path)
    print("Processed frames:", len(df))

    if len(df) > 0:
        valid = df.dropna(subset=["x_px", "y_px"])
        print("Detected frames:", len(valid), "/", len(df))
        if len(valid) > 1:
            xy = valid[["x_px", "y_px"]].to_numpy()
            diffs = np.diff(xy, axis=0)
            path_length_px = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
            print("Approx. path length:", round(float(path_length_px), 2), "px")


if __name__ == "__main__":
    main()
