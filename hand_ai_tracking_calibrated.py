from track_calibrated_hand_and_pins import (
    parse_args,
    process_calibration_only,
    process_video,
)


def main() -> None:
    args = parse_args()

    if args.output_root == "outputs_calibrated":
        args.output_root = "outputs_hand_ai_calibrated"
    if args.output_stem == "calibrated_hand_pins":
        args.output_stem = "hand_ai_tracking_calibrated"

    args.disable_pins = True
    args.velocity_center_source = "ma"
    if args.ma_window == 5:
        args.ma_window = 3
    if args.speed_ema_alpha == 0.35:
        args.speed_ema_alpha = 0.30

    if args.calibration_only:
        process_calibration_only(args)
    else:
        process_video(args)


if __name__ == "__main__":
    main()
