#!/usr/bin/env python3
"""Aggregate FINAL_260802 offset 0/5000/10000 OOB details.

The reporting unit is one complete four-fold Macro-12 run per seed offset.
All standard deviations in this file are sample SD across the three run-level
values (ddof=1), never SD across folds or reporting cells.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "results" / "FINAL_260802_THREE_SEED_SUMMARY"
ENROLLMENTS = (10, 20, 30)
TOP_M_VALUES = (1, 3, 5, 10)
BLOCKS = tuple(f"Temporal Block {index}" for index in range(1, 6))

RUNS = (
    {
        "key": "offset_0",
        "offset": 0,
        "root": HERE / "results" / "FINAL_260802_FULL_240_540",
        "protocol_id": "final_260802_fixed_240_540_full60_claimed_target_v1",
    },
    {
        "key": "offset_5000",
        "offset": 5000,
        "root": HERE / "results" / "FINAL_260802_FULL_240_540_SEED1_OFFSET5000",
        "protocol_id": (
            "final_260802_fixed_240_540_full60_claimed_target_v1_seed_offset_5000"
        ),
    },
    {
        "key": "offset_10000",
        "offset": 10000,
        "root": HERE / "results" / "FINAL_260802_FULL_240_540_SEED1",
        "protocol_id": (
            "final_260802_fixed_240_540_full60_claimed_target_v1_seed_offset_10000"
        ),
    },
)

MODELS = (
    {
        "key": "ppg",
        "label": "PPG",
        "subdir": "ppg",
        "kind": "backbone",
        "primary": 50,
    },
    {
        "key": "ppg_acc",
        "label": "PPG+ACC",
        "subdir": "ppg_acc",
        "kind": "backbone",
        "primary": 50,
    },
    {
        "key": "ppg_acc_gate",
        "label": "PPG+ACC+Gate",
        "subdir": "ppg_acc_residual_verification_gate",
        "kind": "gate",
        "primary": 10,
    },
    {
        "key": "temp_ecapa",
        "label": "PPG+ACC+Temp ECAPA",
        "subdir": "baseline",
        "kind": "backbone",
        "primary": 50,
    },
    {
        "key": "temp_mlp",
        "label": "PPG+ACC+Temp MLP",
        "subdir": "ppg_acc_temp_mlp",
        "kind": "backbone",
        "primary": 50,
    },
    {
        "key": "temp_mlp_gate",
        "label": "PPG+ACC+Temp MLP+Gate",
        "subdir": "ppg_acc_temp_mlp_residual_verification_gate",
        "kind": "gate",
        "primary": 10,
    },
)

CONTRASTS = (
    ("PPG+ACC - PPG", "ppg_acc", "ppg"),
    ("PPG+ACC+Gate - PPG+ACC", "ppg_acc_gate", "ppg_acc"),
    ("Temp ECAPA - PPG+ACC", "temp_ecapa", "ppg_acc"),
    ("Temp MLP - PPG+ACC", "temp_mlp", "ppg_acc"),
    ("Temp MLP+Gate - Temp MLP", "temp_mlp_gate", "temp_mlp"),
    ("Temp MLP+Gate - PPG+ACC+Gate", "temp_mlp_gate", "ppg_acc_gate"),
    ("Temp MLP+Gate - PPG+ACC", "temp_mlp_gate", "ppg_acc"),
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def cells_for(payload: dict, model: dict, horizon: int) -> dict:
    horizon_payload = payload["cross_fold"][str(horizon)]
    return horizon_payload["cells"] if model["kind"] == "gate" else horizon_payload


def cell_eer(cells: dict, enrollment: int, top_m: int) -> float:
    return float(cells[str(enrollment)][str(top_m)]["overall"]["eer_mean"])


def macro_eer(cells: dict) -> float:
    return float(
        np.mean(
            [
                cell_eer(cells, enrollment, top_m)
                for enrollment in ENROLLMENTS
                for top_m in TOP_M_VALUES
            ]
        )
    )


def summarize_values(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"Expected three finite run values, received {array}")
    return {
        "offset_0_eer_percent": float(array[0] * 100.0),
        "offset_5000_eer_percent": float(array[1] * 100.0),
        "offset_10000_eer_percent": float(array[2] * 100.0),
        "mean_eer_percent": float(array.mean() * 100.0),
        "seed_sd_eer_percentage_points_ddof1": float(array.std(ddof=1) * 100.0),
    }


def main() -> int:
    payloads: Dict[str, Dict[str, dict]] = {}
    primary_cells: Dict[str, Dict[str, dict]] = {}
    for run in RUNS:
        payloads[run["key"]] = {}
        primary_cells[run["key"]] = {}
        official = load_json(run["root"] / "six_model_main_summary.json")
        if official.get("status") != "complete":
            raise RuntimeError(f"Incomplete six-model summary: {run['root']}")
        if official.get("protocol_id") != run["protocol_id"]:
            raise RuntimeError(f"Protocol mismatch: {run['root']}")
        official_by_subdir = {
            item["result_directory"]: float(item["eer_fold_mean"])
            for item in official["models"]
        }
        for model in MODELS:
            path = run["root"] / model["subdir"] / "oob_cross_fold.json"
            payload = load_json(path)
            if payload.get("status") != "complete":
                raise RuntimeError(f"Incomplete OOB result: {path}")
            if payload.get("protocol", {}).get("protocol_id") != run["protocol_id"]:
                raise RuntimeError(f"OOB protocol mismatch: {path}")
            cells = cells_for(payload, model, model["primary"])
            calculated = macro_eer(cells)
            expected = official_by_subdir[model["subdir"]]
            if abs(calculated - expected) > 1e-12:
                raise RuntimeError(
                    f"Macro-12 parity failure {run['key']} {model['label']}: "
                    f"calculated={calculated}, official={expected}"
                )
            payloads[run["key"]][model["key"]] = payload
            primary_cells[run["key"]][model["key"]] = cells

    macro_rows = []
    enrollment_rows = []
    top_m_rows = []
    block_rows = []
    for model in MODELS:
        model_key = model["key"]
        run_cells = [primary_cells[run["key"]][model_key] for run in RUNS]
        macro_rows.append(
            {
                "model": model["label"],
                **summarize_values([macro_eer(cells) for cells in run_cells]),
            }
        )
        for enrollment in ENROLLMENTS:
            values = [
                float(
                    np.mean(
                        [cell_eer(cells, enrollment, top_m) for top_m in TOP_M_VALUES]
                    )
                )
                for cells in run_cells
            ]
            enrollment_rows.append(
                {
                    "model": model["label"],
                    "enrollment_seconds": enrollment,
                    **summarize_values(values),
                }
            )
        for top_m in TOP_M_VALUES:
            values = [
                float(
                    np.mean(
                        [cell_eer(cells, enrollment, top_m) for enrollment in ENROLLMENTS]
                    )
                )
                for cells in run_cells
            ]
            top_m_rows.append(
                {
                    "model": model["label"],
                    "top_m": top_m,
                    **summarize_values(values),
                }
            )
        for block in BLOCKS:
            values = [
                float(
                    np.mean(
                        [
                            cells[str(enrollment)][str(top_m)]["activity"][block][
                                "eer_mean"
                            ]
                            for enrollment in ENROLLMENTS
                            for top_m in TOP_M_VALUES
                        ]
                    )
                )
                for cells in run_cells
            ]
            block_rows.append(
                {
                    "model": model["label"],
                    "temporal_block": block,
                    **summarize_values(values),
                }
            )

    trajectory_rows = []
    for model in [item for item in MODELS if item["kind"] == "gate"]:
        for horizon in (0, 1, 2, 5, 10):
            values = [
                macro_eer(cells_for(payloads[run["key"]][model["key"]], model, horizon))
                for run in RUNS
            ]
            trajectory_rows.append(
                {
                    "model": model["label"],
                    "calibration_epoch": horizon,
                    **summarize_values(values),
                }
            )

    macro_by_model = {
        model["key"]: [
            macro_eer(primary_cells[run["key"]][model["key"]]) for run in RUNS
        ]
        for model in MODELS
    }
    paired_rows = []
    for label, left, right in CONTRASTS:
        values = [
            left_value - right_value
            for left_value, right_value in zip(
                macro_by_model[left], macro_by_model[right]
            )
        ]
        summary = summarize_values(values)
        paired_rows.append(
            {
                "comparison_left_minus_right": label,
                "offset_0_delta_eer_percentage_points": summary[
                    "offset_0_eer_percent"
                ],
                "offset_5000_delta_eer_percentage_points": summary[
                    "offset_5000_eer_percent"
                ],
                "offset_10000_delta_eer_percentage_points": summary[
                    "offset_10000_eer_percent"
                ],
                "mean_delta_eer_percentage_points": summary[
                    "mean_eer_percent"
                ],
                "seed_sd_delta_eer_percentage_points_ddof1": summary[
                    "seed_sd_eer_percentage_points_ddof1"
                ],
                "lower_eer_runs": int(np.sum(np.asarray(values) < 0.0)),
                "n_runs": 3,
            }
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "macro12_by_seed.csv", macro_rows)
    write_csv(OUTPUT_ROOT / "enrollment_by_seed.csv", enrollment_rows)
    write_csv(OUTPUT_ROOT / "top_m_by_seed.csv", top_m_rows)
    write_csv(OUTPUT_ROOT / "temporal_block_by_seed.csv", block_rows)
    write_csv(OUTPUT_ROOT / "gate_trajectory_by_seed.csv", trajectory_rows)
    write_csv(OUTPUT_ROOT / "paired_delta_eer_by_seed.csv", paired_rows)
    summary = {
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_unit": "one complete four-fold Macro-12 run per seed offset",
        "seed_offsets": [run["offset"] for run in RUNS],
        "seed_sd_definition": "sample SD across three run-level values; ddof=1",
        "enrollment_seconds": list(ENROLLMENTS),
        "top_m_values": list(TOP_M_VALUES),
        "temporal_blocks": list(BLOCKS),
        "runs": [
            {
                "offset": run["offset"],
                "protocol_id": run["protocol_id"],
                "result_root": str(run["root"]),
            }
            for run in RUNS
        ],
        "outputs": {
            "macro12": "macro12_by_seed.csv",
            "enrollment": "enrollment_by_seed.csv",
            "top_m": "top_m_by_seed.csv",
            "temporal_block": "temporal_block_by_seed.csv",
            "gate_trajectory": "gate_trajectory_by_seed.csv",
            "paired_delta_eer": "paired_delta_eer_by_seed.csv",
        },
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"three-seed detail summary complete: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
