from track_calibrated_hand_and_pins import (
    parse_args,
    process_calibration_only,
    process_video,
)


def main() -> None:
    args = parse_args()

    if args.output_root == "outputs_calibrated":
        args.output_root = "outputs_calibrated_pins_v2"
    if args.output_stem == "calibrated_hand_pins":
        args.output_stem = "calibrated_hand_pins_v2"

    args.disable_pins = False
    args.velocity_center_source = "ma"

    if args.calibration_only:
        process_calibration_only(args)
    else:
        process_video(args)


if __name__ == "__main__":
    main()
