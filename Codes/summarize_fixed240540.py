#!/usr/bin/env python3
"""Summarize the six fixed-240--540 FINAL_260802 OOB systems."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fixed240540_protocol import PROTOCOL_ID, RESULT_ROOT


MODELS = (
    ("PPG", "ppg", "epoch", 50),
    ("PPG+ACC", "ppg_acc", "epoch", 50),
    ("PPG+ACC+Gate C10", "ppg_acc_residual_verification_gate", "calibration_epoch", 10),
    ("PPG+ACC+Temp (ECAPA)", "baseline", "epoch", 50),
    ("PPG+ACC+Temp (MLP)", "ppg_acc_temp_mlp", "epoch", 50),
    (
        "PPG+ACC+Temp (Gate C10)",
        "ppg_acc_temp_mlp_residual_verification_gate",
        "calibration_epoch",
        10,
    ),
)
FOLD_COLS = [f"fold{i}_eer" for i in range(1, 5)]


def atomic_write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    rows = []
    for display, directory, checkpoint_column, checkpoint in MODELS:
        path = args.result_root / directory / "oob_cross_fold_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame = frame[
            frame[checkpoint_column].eq(checkpoint)
            & frame.enroll_seconds.isin([10, 20, 30])
        ].copy()
        if len(frame) != 12:
            raise RuntimeError(f"{path}: expected 12 primary cells, found {len(frame)}")
        fold_values = np.asarray([frame[column].mean() for column in FOLD_COLS])
        rows.append(
            {
                "display_name": display,
                "result_directory": directory,
                "checkpoint": f"{'C' if checkpoint_column == 'calibration_epoch' else 'E'}{checkpoint:02d}",
                "enrollment_seconds": "10,20,30",
                "top_m_values": "1,3,5,10",
                "eer_fold_mean": float(fold_values.mean()),
                "eer_fold_std_ddof1": float(fold_values.std(ddof=1)),
                **{f"fold{i}_eer": float(value) for i, value in enumerate(fold_values, 1)},
                "protocol_id": PROTOCOL_ID,
            }
        )
    output = pd.DataFrame(rows)
    args.result_root.mkdir(parents=True, exist_ok=True)
    atomic_write(output, args.result_root / "six_model_main_summary.csv")
    payload = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "training_minutes": [240, 540],
        "enrollment_seconds": [10, 20, 30],
        "top_m_values": [1, 3, 5, 10],
        "models": rows,
    }
    path = args.result_root / "six_model_main_summary.json"
    temporary = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    print(output[["display_name", "eer_fold_mean", "eer_fold_std_ddof1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
