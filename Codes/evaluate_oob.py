"""Evaluate raw E5-E50 checkpoints without selecting an OOB-best epoch."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from models import (
    CONTEXT_NORM_CHOICES,
    MODEL_SPECS,
    build_model,
    model_result_subdir,
)
from protocol import (
    ACTIVITY_NAMES,
    ENROLL_SECONDS,
    EPOCHS,
    FOLD_CONFIG,
    HORIZONS,
    PROTOCOL_ID,
    TOP_M_VALUES,
    atomic_json_dump,
    build_claimed_target_cases,
    config_payload,
    evaluate_claimed_target_grid,
)
from train import DEFAULT_DATA_ROOT, DEFAULT_RESULT_ROOT, VARIANTS


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def checkpoint_path(result_root: Path, variant: str, fold_id: int, epoch: int) -> Path:
    return (
        result_root
        / model_result_subdir(variant)
        / "checkpoints"
        / f"fold{fold_id}_E{epoch:02d}.pt"
    )


def evaluate_fold(
    variant: str,
    fold_id: int,
    data_root: Path,
    result_root: Path,
    device: torch.device,
    expected_context_norm: str,
) -> dict:
    variant_dir = result_root / model_result_subdir(variant)
    output_path = variant_dir / f"oob_fold{fold_id}.json"
    log = Logger(variant_dir / f"oob_fold{fold_id}.log")
    paths = {
        epoch: checkpoint_path(result_root, variant, fold_id, epoch)
        for epoch in HORIZONS
    }
    missing = [epoch for epoch, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing raw checkpoints for fold {fold_id}: {missing}")

    payload = {
        "status": "evaluating",
        "variant": variant,
        "fold_id": fold_id,
        "oob_users": FOLD_CONFIG[fold_id]["oob"],
        "primary_epoch": EPOCHS,
        "checkpoint_selection": "none",
        "protocol": config_payload("claimed_target"),
        "epochs": {},
    }
    if output_path.is_file():
        import json

        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("variant") == variant
            and int(existing.get("fold_id", -1)) == fold_id
            and existing.get("protocol", {}).get("protocol_id") == PROTOCOL_ID
        ):
            payload = existing
            payload["status"] = "evaluating"

    log("")
    log("=" * 88)
    log(
        f"protocol={PROTOCOL_ID} variant={variant} fold={fold_id} "
        f"oob={FOLD_CONFIG[fold_id]['oob']} device={device}"
    )
    pending_epochs = [epoch for epoch in HORIZONS if str(epoch) not in payload["epochs"]]
    if not pending_epochs:
        payload["status"] = "complete"
        atomic_json_dump(payload, output_path)
        log("All raw epochs were already evaluated.")
        return payload

    log("Building claimed-target OOB cases from enrollment and probe data only...")
    oob_users = FOLD_CONFIG[fold_id]["oob"]
    cases_by_enrollment = {
        seconds: build_claimed_target_cases(data_root, oob_users, seconds)
        for seconds in ENROLL_SECONDS
    }
    model = build_model(variant, context_norm=expected_context_norm).to(device)
    started = time.time()
    for epoch in pending_epochs:
        path = paths[epoch]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("variant") != variant
            or int(checkpoint.get("fold_id", -1)) != fold_id
            or int(checkpoint.get("epoch", -1)) != epoch
            or checkpoint.get("checkpoint_selection") != "raw_epoch_no_validation"
            or checkpoint.get("protocol", {}).get("protocol_id") != PROTOCOL_ID
        ):
            raise RuntimeError(f"Checkpoint metadata mismatch: {path}")
        checkpoint_norm = checkpoint.get("context_norm", "layernorm")
        if checkpoint_norm != expected_context_norm:
            raise RuntimeError(
                f"Context norm mismatch in {path}: {checkpoint_norm} != {expected_context_norm}"
            )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        macro_eer, cells = evaluate_claimed_target_grid(
            model, cases_by_enrollment, device=device
        )
        payload["epochs"][str(epoch)] = {
            "epoch": epoch,
            "primary": epoch == EPOCHS,
            "checkpoint": str(path.resolve()),
            "model_state_sha256": checkpoint["model_state_sha256"],
            "oob_macro_eer_across_16_cells": macro_eer,
            "cells": cells,
        }
        payload["elapsed_hours"] = (time.time() - started) / 3600.0
        atomic_json_dump(payload, output_path)
        log(f"evaluated raw E{epoch:02d} macro_eer={100 * macro_eer:.3f}%")

    payload["status"] = "complete"
    payload["elapsed_hours"] = (time.time() - started) / 3600.0
    atomic_json_dump(payload, output_path)
    log(f"OOB complete; primary=E{EPOCHS:02d} output={output_path}")
    del model, cases_by_enrollment
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _cell(payload: dict, epoch: int, enroll_seconds: int, top_m: int) -> dict:
    return payload["epochs"][str(epoch)]["cells"][str(enroll_seconds)][str(top_m)]


def _block_result(cell: dict, block_index: int) -> dict:
    """Read current block labels and historical positional labels."""
    block_results = cell["activity"]
    block_name = ACTIVITY_NAMES[block_index]
    if block_name in block_results:
        return block_results[block_name]
    if len(block_results) != len(ACTIVITY_NAMES):
        raise RuntimeError("Unexpected number of Temporal Block results")
    return list(block_results.values())[block_index]


def aggregate_variant_if_complete(result_root: Path, variant: str) -> dict | None:
    import json

    variant_dir = result_root / model_result_subdir(variant)
    fold_payloads = []
    for fold_id in range(1, 5):
        path = variant_dir / f"oob_fold{fold_id}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            return None
        if payload.get("protocol", {}).get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"Protocol mismatch in {path}")
        fold_payloads.append(payload)

    cross_fold: Dict[str, dict] = {}
    overall_rows = []
    activity_rows = []
    for epoch in HORIZONS:
        epoch_output: Dict[str, dict] = {}
        for seconds in ENROLL_SECONDS:
            seconds_output: Dict[str, dict] = {}
            for top_m in TOP_M_VALUES:
                cells = [_cell(payload, epoch, seconds, top_m) for payload in fold_payloads]
                overall_values = np.asarray(
                    [cell["overall"]["eer"] for cell in cells], dtype=np.float64
                )
                activities = {}
                for block_index, activity in enumerate(ACTIVITY_NAMES):
                    values = np.asarray(
                        [
                            _block_result(cell, block_index)["eer"]
                            for cell in cells
                        ],
                        dtype=np.float64,
                    )
                    activities[activity] = {
                        "eer_mean": float(values.mean()),
                        "eer_std": float(values.std()),
                        "per_fold": values.tolist(),
                    }
                    activity_rows.append(
                        {
                            "variant": variant,
                            "epoch": epoch,
                            "primary": epoch == EPOCHS,
                            "enroll_seconds": seconds,
                            "top_m": top_m,
                            "activity": activity,
                            "eer_mean": float(values.mean()),
                            "eer_std": float(values.std()),
                            **{f"fold{i}_eer": float(value) for i, value in enumerate(values, 1)},
                        }
                    )
                seconds_output[str(top_m)] = {
                    "overall": {
                        "eer_mean": float(overall_values.mean()),
                        "eer_std": float(overall_values.std()),
                        "per_fold": overall_values.tolist(),
                    },
                    "activity": activities,
                }
                overall_rows.append(
                    {
                        "variant": variant,
                        "epoch": epoch,
                        "primary": epoch == EPOCHS,
                        "enroll_seconds": seconds,
                        "top_m": top_m,
                        "eer_mean": float(overall_values.mean()),
                        "eer_std": float(overall_values.std()),
                        **{
                            f"fold{i}_eer": float(value)
                            for i, value in enumerate(overall_values, 1)
                        },
                    }
                )
            epoch_output[str(seconds)] = seconds_output
        cross_fold[str(epoch)] = epoch_output

    output = {
        "status": "complete",
        "variant": variant,
        "primary_epoch": EPOCHS,
        "checkpoint_selection": "none; E50 is prespecified primary",
        "protocol": config_payload("claimed_target"),
        "cross_fold": cross_fold,
    }
    atomic_json_dump(output, variant_dir / "oob_cross_fold.json")
    _write_csv_atomic(variant_dir / "oob_cross_fold_summary.csv", overall_rows)
    _write_csv_atomic(variant_dir / "oob_activity_summary.csv", activity_rows)
    return output


def _write_csv_atomic(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--context-norm",
        choices=CONTEXT_NORM_CHOICES,
        default="layernorm",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Evaluate one fold without cross-fold aggregation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    evaluate_fold(
        variant=args.variant,
        fold_id=args.fold,
        data_root=args.data_root,
        result_root=args.result_root,
        device=torch.device(args.device),
        expected_context_norm=args.context_norm,
    )
    if not args.skip_aggregate:
        aggregate_variant_if_complete(args.result_root, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
