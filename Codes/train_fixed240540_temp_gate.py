#!/usr/bin/env python3
"""Calibrate and OOB-evaluate the Temp-MLP Gate on fixed 240--540 training."""

from fixed240540_protocol import activate

activate()

from train_residual_verification_gate import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
