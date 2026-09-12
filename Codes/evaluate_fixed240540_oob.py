#!/usr/bin/env python3
"""Evaluate an E50 backbone under the fixed 240--540 FINAL_260802 protocol."""

from fixed240540_protocol import activate

activate()

from evaluate_oob import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
