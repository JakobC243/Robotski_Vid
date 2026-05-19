"""Compatibility entry point for board hole calibration.

This script intentionally uses the final calibrated hand/pin pipeline for
calibration, so running ``hole_calibration.py`` produces the same board
calibration outputs as ``track_calibrated_hand_and_pins.py --calibration-only``.
Legacy option names from the old standalone script are still accepted.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence

from track_calibrated_hand_and_pins import parse_args, process_calibration_only


LEGACY_ARG_ALIASES = {
    "--output-dir": "--output-root",
    "--min-radius": "--min-hole-radius",
    "--max-radius": "--max-hole-radius",
    "--min-dist": "--min-hole-dist",
}


def normalize_legacy_args(argv: Sequence[str]) -> List[str]:
    normalized: List[str] = []

    for token in argv:
        if token in LEGACY_ARG_ALIASES:
            normalized.append(LEGACY_ARG_ALIASES[token])
            continue

        for legacy_name, final_name in LEGACY_ARG_ALIASES.items():
            legacy_prefix = f"{legacy_name}="
            if token.startswith(legacy_prefix):
                normalized.append(f"{final_name}={token[len(legacy_prefix):]}")
                break
        else:
            normalized.append(token)

    if "--calibration-only" not in normalized:
        normalized.append("--calibration-only")

    return normalized


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(normalize_legacy_args(raw_args))
    args.calibration_only = True
    return process_calibration_only(args)


if __name__ == "__main__":
    main()
