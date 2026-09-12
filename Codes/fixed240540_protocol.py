"""Environment activation for the FINAL_260802 fixed 240--540 protocol."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEED_OFFSET = int(os.environ.get("FINAL_SEED_OFFSET", "0"))
BASE_PROTOCOL_ID = "final_260802_fixed_240_540_full60_claimed_target_v1"
PROTOCOL_ID = (
    BASE_PROTOCOL_ID
    if SEED_OFFSET == 0
    else f"{BASE_PROTOCOL_ID}_seed_offset_{SEED_OFFSET}"
)
RESULT_ROOT = HERE / "results" / "FINAL_260802_FULL_240_540"
ENROLL_SECONDS = (10, 20, 30)


def activate():
    """Set protocol variables before importing ``protocol`` or its runners."""
    settings = {
        "FINAL_PROTOCOL_ID": PROTOCOL_ID,
        "FINAL_TRAIN_START_SECONDS": "0",
        "FINAL_TRAIN_END_MINUTES": "60",
        "FINAL_ENROLL_START_SECONDS": "120",
        "FINAL_PROBE_START_MINUTES": "48",
        "FINAL_PROBE_END_MINUTES": "58",
        "FINAL_ENROLL_SECONDS": "10,20,30",
        "FINAL_HORIZONS": "50",
        "FINAL_SEED_OFFSET": str(SEED_OFFSET),
    }
    for key, value in settings.items():
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(f"Conflicting {key}={existing!r}; expected {value!r}")
        os.environ[key] = value

    import protocol

    if protocol.PROTOCOL_ID != PROTOCOL_ID:
        raise RuntimeError("protocol was imported before fixed240540 activation")
    if protocol.train_regions()[0] != (240 * 60 * protocol.FS, 300 * 60 * protocol.FS):
        raise RuntimeError("fixed 240--540 training-region activation failed")
    if tuple(protocol.ENROLL_SECONDS) != ENROLL_SECONDS:
        raise RuntimeError("10/20/30-second enrollment activation failed")
    return protocol
