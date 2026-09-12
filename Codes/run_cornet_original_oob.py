#!/usr/bin/env python3
"""Train, evaluate, validate, and summarize the redesigned CorNET OOB model."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.utils.data import DataLoader

from cornet_original_oob import (
    AAM_MARGIN,
    AAM_SCALE,
    BATCH_SIZE,
    ENROLL_SECONDS,
    EPOCHS,
    FOLD_CONFIG,
    FOLD_SEEDS,
    FS,
    LEARNING_RATE,
    PROTOCOL_ID,
    RESULT_SUBDIR,
    RMSPROP_ALPHA,
    RMSPROP_EPS,
    SEED_OFFSET,
    TOP_M_VALUES,
    VARIANT,
    WINDOW_SAMPLES,
    AAMSoftmax,
    CorNETOriginalEncoder,
    CorNETTrainDataset,
    atomic_json_dump,
    atomic_torch_save,
    build_oob_windows,
    cpu_clone_state,
    evaluate_oob_users,
    model_payload,
    parameter_count,
    protocol_payload,
    set_seed,
    state_digest,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/Data/CRS25/PPG_Certifiation/data/Final_Data")
DEFAULT_RESULT_ROOT = (
    HERE
    / "results"
    / "FINAL_260904_CORNET_ORIGINAL_BACKBONE_OOB"
    / f"offset_{SEED_OFFSET}"
)


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def variant_dir(result_root: Path) -> Path:
    return result_root / RESULT_SUBDIR


def _loader_rng_payload(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def _restore_loader_rng(payload: dict, generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])
    generator.set_state(payload["loader_generator"])


def _optimizer(model: torch.nn.Module, criterion: torch.nn.Module) -> RMSprop:
    return RMSprop(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        alpha=RMSPROP_ALPHA,
        eps=RMSPROP_EPS,
        weight_decay=0.0,
        momentum=0.0,
        centered=False,
    )


def _checkpoint_payload(
    model: CorNETOriginalEncoder,
    fold_id: int,
    epoch: int,
    dataset_metadata: dict,
) -> dict:
    state = cpu_clone_state(model.state_dict())
    return {
        "format_version": 1,
        "variant": VARIANT,
        "protocol_id": PROTOCOL_ID,
        "fold_id": int(fold_id),
        "seed": int(FOLD_SEEDS[fold_id]),
        "seed_offset": int(SEED_OFFSET),
        "epoch": int(epoch),
        "checkpoint_selection": "raw_epoch_no_validation",
        "primary_checkpoint": epoch == EPOCHS,
        "train_users": list(FOLD_CONFIG[fold_id]["train"]),
        "model_state_sha256": state_digest(state),
        "model_state": state,
        "model": model_payload(),
        "protocol": protocol_payload(),
        "training_dataset": dataset_metadata,
    }


def train_fold(
    fold_id: int,
    data_root: Path,
    result_root: Path,
    device: torch.device,
    resume: bool,
) -> dict:
    output_dir = variant_dir(result_root)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(output_dir / f"train_fold{fold_id}.log")
    train_users = FOLD_CONFIG[fold_id]["train"]

    log("")
    log("=" * 88)
    log(
        f"protocol={PROTOCOL_ID} variant={VARIANT} fold={fold_id} "
        f"train={train_users} device={device}"
    )
    build_started = time.time()
    dataset = CorNETTrainDataset(data_root, train_users)
    build_seconds = time.time() - build_started
    log(
        f"train_windows={len(dataset)} batch_size={BATCH_SIZE} "
        f"dataset_build_seconds={build_seconds:.1f} validation=disabled"
    )

    seed = FOLD_SEEDS[fold_id]
    set_seed(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + 100_000)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=loader_generator,
    )
    model = CorNETOriginalEncoder().to(device)
    criterion = AAMSoftmax(
        in_features=128,
        n_classes=len(train_users),
        margin=AAM_MARGIN,
        scale=AAM_SCALE,
    ).to(device)
    optimizer = _optimizer(model, criterion)

    latest_path = checkpoint_dir / f"fold{fold_id}_latest.pt"
    raw_path = checkpoint_dir / f"fold{fold_id}_E{EPOCHS:02d}.pt"
    history: list[dict] = []
    start_epoch = 1
    if resume and latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        if (
            latest.get("variant") != VARIANT
            or latest.get("protocol_id") != PROTOCOL_ID
            or int(latest.get("fold_id", -1)) != fold_id
            or latest.get("train_users") != list(train_users)
        ):
            raise RuntimeError(f"Resume metadata mismatch: {latest_path}")
        model.load_state_dict(latest["model_state"])
        criterion.load_state_dict(latest["criterion_state"])
        optimizer.load_state_dict(latest["optimizer_state"])
        history = list(latest["history"])
        start_epoch = int(latest["epoch"]) + 1
        _restore_loader_rng(latest["rng"], loader_generator)
        log(f"resuming_after_epoch={start_epoch - 1}")

    if start_epoch > EPOCHS:
        if not raw_path.is_file():
            raise RuntimeError(f"Completed latest checkpoint but missing {raw_path}")
        log(f"training already complete raw_checkpoint={raw_path}")
        return {
            "fold_id": fold_id,
            "completed_epoch": EPOCHS,
            "checkpoint": str(raw_path.resolve()),
        }

    training_started = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_started = time.time()
        model.train()
        criterion.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for ppg, labels in loader:
            ppg = ppg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embedding = model(ppg)
            loss = criterion(embedding, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            with torch.no_grad():
                class_weights = F.normalize(criterion.weight, p=2, dim=1)
                logits = (
                    F.normalize(embedding, p=2, dim=1)
                    @ class_weights.T
                    * criterion.scale
                )
                correct += int((logits.argmax(dim=1) == labels).sum())
                total += len(labels)

        epoch_seconds = time.time() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, len(loader)),
            "train_accuracy": correct / max(1, total),
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)
        log(
            f"epoch={epoch:02d} loss={record['train_loss']:.6f} "
            f"train_acc={100 * record['train_accuracy']:.3f}% "
            f"seconds={epoch_seconds:.1f}"
        )

        state = cpu_clone_state(model.state_dict())
        latest_payload = {
            "format_version": 1,
            "variant": VARIANT,
            "protocol_id": PROTOCOL_ID,
            "fold_id": fold_id,
            "seed": seed,
            "seed_offset": SEED_OFFSET,
            "epoch": epoch,
            "train_users": list(train_users),
            "model_state": state,
            "criterion_state": cpu_clone_state(criterion.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "dataset_metadata": dataset.metadata,
            "rng": _loader_rng_payload(loader_generator),
        }
        atomic_torch_save(latest_payload, latest_path)
        atomic_json_dump(
            {
                "status": "complete" if epoch == EPOCHS else "training",
                "variant": VARIANT,
                "protocol_id": PROTOCOL_ID,
                "fold_id": fold_id,
                "seed": seed,
                "seed_offset": SEED_OFFSET,
                "completed_epoch": epoch,
                "primary_epoch": EPOCHS,
                "history": history,
                "training_dataset": dataset.metadata,
                "dataset_build_seconds": build_seconds,
            },
            output_dir / f"fold{fold_id}_training.json",
        )

        if epoch == EPOCHS:
            raw_payload = _checkpoint_payload(
                model, fold_id, epoch, dataset.metadata
            )
            atomic_torch_save(raw_payload, raw_path)
            log(
                f"saved raw E{EPOCHS:02d} "
                f"sha256={raw_payload['model_state_sha256'][:12]} "
                f"path={raw_path}"
            )

    elapsed_hours = (time.time() - training_started) / 3600.0
    log(f"training complete fold={fold_id} elapsed={elapsed_hours:.3f}h")
    return {
        "fold_id": fold_id,
        "completed_epoch": EPOCHS,
        "checkpoint": str(raw_path.resolve()),
        "elapsed_hours": elapsed_hours,
    }


def benchmark_fold(
    fold_id: int,
    data_root: Path,
    device: torch.device,
    batches: int,
) -> dict:
    started = time.time()
    dataset = CorNETTrainDataset(data_root, FOLD_CONFIG[fold_id]["train"])
    build_seconds = time.time() - started
    seed = FOLD_SEEDS[fold_id]
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed + 100_000)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    model = CorNETOriginalEncoder().to(device)
    criterion = AAMSoftmax(n_classes=12).to(device)
    optimizer = _optimizer(model, criterion)
    measured = 0
    warmup = min(10, max(1, batches // 5))
    measure_started = None
    for index, (ppg, labels) in enumerate(loader):
        if index == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            measure_started = time.time()
        ppg = ppg.to(device)
        labels = labels.to(device)
        loss = criterion(model(ppg), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if index >= warmup:
            measured += 1
        if measured >= batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if measure_started is None or measured == 0:
        raise RuntimeError("Benchmark did not measure any batches")
    measured_seconds = time.time() - measure_started
    seconds_per_batch = measured_seconds / measured
    epoch_seconds = seconds_per_batch * len(loader)
    result = {
        "fold_id": fold_id,
        "device": str(device),
        "windows": len(dataset),
        "batches_per_epoch": len(loader),
        "measured_batches": measured,
        "dataset_build_seconds": build_seconds,
        "seconds_per_batch": seconds_per_batch,
        "estimated_epoch_seconds": epoch_seconds,
        "estimated_50_epoch_hours": epoch_seconds * EPOCHS / 3600.0,
    }
    print(json.dumps(result, indent=2))
    return result


def evaluate_fold(
    fold_id: int,
    data_root: Path,
    result_root: Path,
    device: torch.device,
) -> dict:
    output_dir = variant_dir(result_root)
    output_path = output_dir / f"oob_fold{fold_id}.json"
    checkpoint_path = output_dir / "checkpoints" / f"fold{fold_id}_E{EPOCHS:02d}.pt"
    log = Logger(output_dir / f"oob_fold{fold_id}.log")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("variant") != VARIANT
        or checkpoint.get("protocol_id") != PROTOCOL_ID
        or int(checkpoint.get("fold_id", -1)) != fold_id
        or int(checkpoint.get("epoch", -1)) != EPOCHS
        or checkpoint.get("checkpoint_selection") != "raw_epoch_no_validation"
    ):
        raise RuntimeError(f"Checkpoint metadata mismatch: {checkpoint_path}")
    if state_digest(checkpoint["model_state"]) != checkpoint["model_state_sha256"]:
        raise RuntimeError(f"Checkpoint digest mismatch: {checkpoint_path}")
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("protocol_id") == PROTOCOL_ID
            and existing.get("model_state_sha256")
            == checkpoint["model_state_sha256"]
        ):
            log(f"OOB already complete output={output_path}")
            return existing

    log("")
    log("=" * 88)
    log(
        f"protocol={PROTOCOL_ID} variant={VARIANT} fold={fold_id} "
        f"oob={FOLD_CONFIG[fold_id]['oob']} device={device}"
    )
    model = CorNETOriginalEncoder().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    started = time.time()
    users = build_oob_windows(data_root, FOLD_CONFIG[fold_id]["oob"])
    macro_eer, cells, window_metadata = evaluate_oob_users(model, users, device)
    payload = {
        "status": "complete",
        "variant": VARIANT,
        "protocol_id": PROTOCOL_ID,
        "fold_id": fold_id,
        "seed": FOLD_SEEDS[fold_id],
        "seed_offset": SEED_OFFSET,
        "oob_users": list(FOLD_CONFIG[fold_id]["oob"]),
        "primary_epoch": EPOCHS,
        "checkpoint_selection": "none; prespecified raw E50",
        "checkpoint": str(checkpoint_path.resolve()),
        "model_state_sha256": checkpoint["model_state_sha256"],
        "macro_eer_across_12_cells": macro_eer,
        "cells": cells,
        "window_metadata": window_metadata,
        "protocol": protocol_payload(),
        "model": model_payload(),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    atomic_json_dump(payload, output_path)
    log(f"OOB complete macro12_eer={100 * macro_eer:.3f}% output={output_path}")
    del model, users
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate_folds(result_root: Path) -> dict:
    output_dir = variant_dir(result_root)
    folds = []
    for fold_id in range(1, 5):
        path = output_dir / f"oob_fold{fold_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(f"Incomplete OOB payload: {path}")
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"Protocol mismatch: {path}")
        folds.append(payload)

    cross_fold = {}
    rows = []
    fold_macro = np.asarray(
        [fold["macro_eer_across_12_cells"] for fold in folds], dtype=np.float64
    )
    for seconds in ENROLL_SECONDS:
        length_cells = {}
        for top_m in TOP_M_VALUES:
            cells = [fold["cells"][str(seconds)][str(top_m)] for fold in folds]
            values = np.asarray(
                [cell["overall"]["eer"] for cell in cells], dtype=np.float64
            )
            activity = {}
            for activity_name in cells[0]["activity"]:
                activity_values = np.asarray(
                    [cell["activity"][activity_name]["eer"] for cell in cells],
                    dtype=np.float64,
                )
                activity[activity_name] = {
                    "eer_mean": float(activity_values.mean()),
                    "eer_std": float(activity_values.std()),
                    "per_fold": activity_values.tolist(),
                }
            length_cells[str(top_m)] = {
                "overall": {
                    "eer_mean": float(values.mean()),
                    "eer_std": float(values.std()),
                    "per_fold": values.tolist(),
                },
                "activity": activity,
            }
            rows.append(
                {
                    "seed_offset": SEED_OFFSET,
                    "enrollment_seconds": seconds,
                    "top_m": top_m,
                    "eer_mean": float(values.mean()),
                    "eer_std": float(values.std()),
                    **{
                        f"fold{index}_eer": float(value)
                        for index, value in enumerate(values, 1)
                    },
                }
            )
        cross_fold[str(seconds)] = length_cells
    payload = {
        "status": "complete",
        "variant": VARIANT,
        "protocol_id": PROTOCOL_ID,
        "seed_offset": SEED_OFFSET,
        "macro12_eer": float(np.mean([row["eer_mean"] for row in rows])),
        "fold_macro12_eer_mean": float(fold_macro.mean()),
        "fold_macro12_eer_std": float(fold_macro.std()),
        "fold_macro12_eer": fold_macro.tolist(),
        "cross_fold": cross_fold,
        "protocol": protocol_payload(),
        "model": model_payload(),
    }
    atomic_json_dump(payload, output_dir / "oob_cross_fold.json")
    _write_csv(output_dir / "oob_cross_fold_summary.csv", rows)
    print(
        f"offset={SEED_OFFSET} macro12_eer={100 * payload['macro12_eer']:.3f}% "
        f"output={output_dir / 'oob_cross_fold.json'}",
        flush=True,
    )
    return payload


def summarize_three_seeds(base_root: Path) -> dict:
    offsets = (0, 5000, 10000)
    runs = []
    for offset in offsets:
        path = base_root / f"offset_{offset}" / RESULT_SUBDIR / "oob_cross_fold.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("seed_offset") != offset:
            raise RuntimeError(f"Invalid three-seed input: {path}")
        runs.append(payload)

    run_macro = np.asarray([run["macro12_eer"] for run in runs], dtype=np.float64)
    cell_rows = []
    for seconds in ENROLL_SECONDS:
        for top_m in TOP_M_VALUES:
            values = np.asarray(
                [
                    run["cross_fold"][str(seconds)][str(top_m)]["overall"][
                        "eer_mean"
                    ]
                    for run in runs
                ],
                dtype=np.float64,
            )
            cell_rows.append(
                {
                    "enrollment_seconds": seconds,
                    "top_m": top_m,
                    "eer_mean": float(values.mean()),
                    "eer_seed_sd_ddof1": float(values.std(ddof=1)),
                    **{
                        f"offset_{offset}_eer": float(value)
                        for offset, value in zip(offsets, values)
                    },
                }
            )
    output_dir = base_root / "three_seed_summary"
    output = {
        "status": "complete",
        "variant": VARIANT,
        "seed_offsets": list(offsets),
        "macro12_eer_mean": float(run_macro.mean()),
        "macro12_eer_seed_sd_ddof1": float(run_macro.std(ddof=1)),
        "macro12_eer_by_seed": {
            str(offset): float(value) for offset, value in zip(offsets, run_macro)
        },
        "cells": cell_rows,
    }
    atomic_json_dump(output, output_dir / "three_seed_summary.json")
    _write_csv(output_dir / "three_seed_cells.csv", cell_rows)
    print(
        f"three-seed macro12={100 * output['macro12_eer_mean']:.3f}% "
        f"+/- {100 * output['macro12_eer_seed_sd_ddof1']:.3f}%",
        flush=True,
    )
    return output


def validate(device: torch.device) -> dict:
    for fold_id, fold in FOLD_CONFIG.items():
        if set(fold["train"]) & set(fold["oob"]):
            raise RuntimeError(f"Fold {fold_id} train/OOB overlap")
        if len(fold["train"]) != 12 or len(fold["oob"]) != 4:
            raise RuntimeError(f"Fold {fold_id} membership count mismatch")
    model = CorNETOriginalEncoder().to(device)
    model.train()
    sample = torch.randn(2, 1, WINDOW_SAMPLES, device=device)
    embedding, shapes = model.forward_with_intermediates(sample)
    expected = {
        "input": (2, 1, 1000),
        "conv1": (2, 32, 961),
        "pool1": (2, 32, 240),
        "conv2": (2, 32, 201),
        "pool2": (2, 32, 50),
        "lstm1": (2, 50, 128),
        "lstm2": (2, 50, 128),
        "embedding": (2, 128),
    }
    if shapes != expected:
        raise RuntimeError(f"Shape regression failed: {shapes}")
    criterion = AAMSoftmax(n_classes=12).to(device)
    loss = criterion(embedding, torch.tensor([0, 1], device=device))
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite synthetic loss")
    result = {
        "status": "pass",
        "protocol_id": PROTOCOL_ID,
        "device": str(device),
        "shapes": {key: list(value) for key, value in shapes.items()},
        "encoder_parameters": parameter_count(model),
        "training_head_parameters": parameter_count(criterion),
        "synthetic_loss": float(loss.detach().cpu()),
        "protocol": protocol_payload(),
        "model": model_payload(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--device", default="cpu")
    validate_parser.add_argument("--output", type=Path)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4], default=1)
    benchmark_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    benchmark_parser.add_argument("--device", default="cuda")
    benchmark_parser.add_argument("--batches", type=int, default=100)
    benchmark_parser.add_argument("--output", type=Path)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4], required=True)
    train_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    train_parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--resume", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4], required=True)
    evaluate_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    evaluate_parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    evaluate_parser.add_argument("--device", default="cuda")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument(
        "--base-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT.parent,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "validate":
        result = validate(torch.device(args.device))
        if args.output:
            atomic_json_dump(result, args.output)
        return 0
    if args.command == "benchmark":
        if not args.data_root.is_dir():
            raise FileNotFoundError(args.data_root)
        result = benchmark_fold(
            args.fold, args.data_root, torch.device(args.device), args.batches
        )
        if args.output:
            atomic_json_dump(result, args.output)
        return 0
    if args.command == "train":
        if not args.data_root.is_dir():
            raise FileNotFoundError(args.data_root)
        train_fold(
            args.fold,
            args.data_root,
            args.result_root,
            torch.device(args.device),
            args.resume,
        )
        return 0
    if args.command == "evaluate":
        if not args.data_root.is_dir():
            raise FileNotFoundError(args.data_root)
        evaluate_fold(
            args.fold,
            args.data_root,
            args.result_root,
            torch.device(args.device),
        )
        return 0
    if args.command == "aggregate":
        aggregate_folds(args.result_root)
        return 0
    if args.command == "summarize":
        summarize_three_seeds(args.base_root)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
