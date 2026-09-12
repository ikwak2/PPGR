#!/usr/bin/env python3
"""Train one or more FINAL_260802 fixed-240--540 backbone folds."""

from fixed240540_protocol import activate

activate()

from train import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
