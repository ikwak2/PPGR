"""Warm-start and verification-calibrate an identity residual quality gate.

This experiment deliberately freezes the E50 Compact Temp MLP backbone.  It
trains only a signed residual gate on subject-disjoint in-fold claimed-target
enrollment/query pairs, then reports every prespecified calibration horizon on
the untouched OOB participants.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import (
    EMBED_DIM,
    CompactResidualGateWarmStartModel,
    build_compact_residual_gate_warmstart_model,
    build_model,
)
from protocol import (
    ACTIVITY_NAMES,
    ENROLL_SECONDS,
    FOLD_CONFIG,
    FOLD_SEEDS,
    PROTOCOL_ID,
    TOP_M_VALUES,
    ClaimedTargetCase,
    atomic_json_dump,
    build_claimed_target_cases,
    config_payload,
    evaluate_claimed_target_grid,
    set_seed,
)
from train import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULT_ROOT,
    atomic_torch_save,
    cpu_clone_state,
    state_digest,
)


METHOD = "ppg_acc_temp_mlp_residual_verification_gate"
SOURCE_VARIANT = "ppg_acc_temp_mlp"
SOURCE_RESULT_SUBDIR = SOURCE_VARIANT
METHOD_RESULT_SUBDIR = METHOD
GATE_MODEL_VARIANT = "ppg_acc_gate_temp_mlp"
SOURCE_EPOCH = 50
CALIBRATION_EPOCHS = 10
CALIBRATION_HORIZONS = (0, 1, 2, 5, 10)
PRIMARY_CALIBRATION_EPOCH = CALIBRATION_EPOCHS
CONTEXT_NORM = "layernorm"
MAX_GATE_DELTA = 0.5
LEARNING_RATE = 1e-3
RANK_MARGIN = 0.05
RANK_TEMPERATURE = 0.05
IDENTITY_REGULARIZATION = 0.01
IMPOSTORS_PER_STEP = 2
FEATURE_BATCH_SIZE = 512
# C00 and its frozen source produce identical embeddings, but repeated GPU
# evaluation can move the discrete EER operating point by a few trials because
# of floating-point reduction order.  This is 0.01 percentage point in EER.
C00_OOB_PARITY_TOLERANCE = 1e-4


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        line = f"[{stamp} UTC] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


@dataclass
class FrozenFeatures:
    ppg_embedding: torch.Tensor
    acc_embedding: torch.Tensor
    context: torch.Tensor
    temperature_correction: torch.Tensor

    def __len__(self) -> int:
        return int(self.context.size(0))


@dataclass
class FrozenTargetCase:
    target_id: int
    enrollment: FrozenFeatures
    probes_by_owner: Dict[int, Dict[int, FrozenFeatures]]


def source_checkpoint_path(
    result_root: Path, fold_id: int
) -> Path:
    return (
        result_root
        / SOURCE_RESULT_SUBDIR
        / "checkpoints"
        / f"fold{fold_id}_E{SOURCE_EPOCH:02d}.pt"
    )


def calibration_checkpoint_path(
    result_root: Path, fold_id: int, epoch: int
) -> Path:
    return (
        result_root
        / METHOD_RESULT_SUBDIR
        / "checkpoints"
        / f"fold{fold_id}_C{epoch:02d}.pt"
    )


def _protocol_payload() -> dict:
    payload = config_payload("claimed_target")
    payload["verification_gate_calibration"] = {
        "source_variant": SOURCE_VARIANT,
        "source_result_subdir": SOURCE_RESULT_SUBDIR,
        "method_result_subdir": METHOD_RESULT_SUBDIR,
        "gate_model_variant": GATE_MODEL_VARIANT,
        "source_epoch": SOURCE_EPOCH,
        "trainable_parameters": "gate only",
        "frozen_parameters": [
            "ppg_encoder",
            "acc_encoder",
            "fusion_projector",
            "compact_temperature_mlp",
        ],
        "gate": "1 + 0.5*tanh(delta), modality-specific expansion",
        "gate_context_features": [
            "ppg_low_high_log_power_ratio",
            "acc_low_power",
            "acc_high_power",
            "ppg_acc_corr",
        ],
        "calibration_epochs": CALIBRATION_EPOCHS,
        "reported_horizons": list(CALIBRATION_HORIZONS),
        "primary_horizon": PRIMARY_CALIBRATION_EPOCH,
        "enrollment_seconds_used_for_calibration": list(ENROLL_SECONDS),
        "top_m_values_used_in_loss": list(TOP_M_VALUES),
        "impostor_owners_per_target_activity_step": IMPOSTORS_PER_STEP,
        "ranking_loss": "pairwise softplus over genuine/impostor Top-M scores",
        "ranking_margin": RANK_MARGIN,
        "ranking_temperature": RANK_TEMPERATURE,
        "identity_regularization": IDENTITY_REGULARIZATION,
        "learning_rate": LEARNING_RATE,
        "oob_used_for_training_or_selection": False,
    }
    return payload


def _load_warmstarted_model(
    fold_id: int,
    result_root: Path,
    device: torch.device,
) -> Tuple[CompactResidualGateWarmStartModel, dict, float]:
    source_path = source_checkpoint_path(result_root, fold_id)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if (
        source.get("variant") != SOURCE_VARIANT
        or int(source.get("fold_id", -1)) != fold_id
        or int(source.get("epoch", -1)) != SOURCE_EPOCH
        or source.get("protocol", {}).get("protocol_id") != PROTOCOL_ID
    ):
        raise RuntimeError(f"Source checkpoint metadata mismatch: {source_path}")

    model = build_compact_residual_gate_warmstart_model(
        context_norm=CONTEXT_NORM,
        max_delta=MAX_GATE_DELTA,
        gate_variant=GATE_MODEL_VARIANT,
        output_variant=METHOD,
    )
    incompatible = model.load_state_dict(source["model_state"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_gate_missing = [
        key for key in incompatible.missing_keys if not key.startswith("gate.")
    ]
    if unexpected or non_gate_missing or not incompatible.missing_keys:
        raise RuntimeError(
            "Unsafe warm start: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )

    # Prove that C00 has exactly the same output as the source model before
    # allowing any calibration work to proceed.
    baseline = build_model(SOURCE_VARIANT, context_norm=CONTEXT_NORM)
    baseline.load_state_dict(source["model_state"])
    baseline.to(device).eval()
    model.to(device).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(10_000 + fold_id)
    ppg = torch.randn(3, 1, 512, generator=generator).to(device)
    temp = torch.randn(3, 1, 512, generator=generator).to(device)
    acc = torch.randn(3, 3, 512, generator=generator).to(device)
    with torch.no_grad():
        source_output = baseline(ppg, temp, acc)
        warm_output = model(ppg, temp, acc)
    max_abs_error = float((source_output - warm_output).abs().max().item())
    if max_abs_error != 0.0:
        raise RuntimeError(
            f"Identity warm-start check failed: max_abs_error={max_abs_error}"
        )
    del baseline, ppg, temp, acc, source_output, warm_output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, source, max_abs_error


def _freeze_backbone(model: CompactResidualGateWarmStartModel) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.gate.parameters():
        parameter.requires_grad_(True)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable <= 0:
        raise RuntimeError("The residual gate has no trainable parameters")
    model.eval()
    model.gate.train()
    return trainable


def _extract_frozen_features(
    model: CompactResidualGateWarmStartModel,
    tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> FrozenFeatures:
    ppg_all, temp_all, acc_all = tensors
    ppg_outputs = []
    acc_outputs = []
    contexts = []
    corrections = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ppg_all), FEATURE_BATCH_SIZE):
            end = start + FEATURE_BATCH_SIZE
            ppg = ppg_all[start:end].to(device)
            temp = temp_all[start:end].to(device)
            acc = acc_all[start:end].to(device)
            ppg_outputs.append(model.encoders["ppg"](ppg).cpu())
            acc_outputs.append(model.encoders["acc"](acc).cpu())
            contexts.append(model.gate._context(ppg, temp, acc).cpu())
            zeros = torch.zeros(
                len(ppg), EMBED_DIM, device=device, dtype=ppg.dtype
            )
            if model.temperature_corrector is None:
                corrections.append(zeros.cpu())
            else:
                corrections.append(
                    model.temperature_corrector(zeros, temp).cpu()
                )
    return FrozenFeatures(
        ppg_embedding=torch.cat(ppg_outputs),
        acc_embedding=torch.cat(acc_outputs),
        context=torch.cat(contexts),
        temperature_correction=torch.cat(corrections),
    )


def _split_frozen_features(
    merged: FrozenFeatures, lengths: Sequence[int]
) -> Sequence[FrozenFeatures]:
    boundaries = np.cumsum(lengths)[:-1].tolist()
    ppg_parts = torch.tensor_split(merged.ppg_embedding, boundaries)
    acc_parts = torch.tensor_split(merged.acc_embedding, boundaries)
    context_parts = torch.tensor_split(merged.context, boundaries)
    correction_parts = torch.tensor_split(
        merged.temperature_correction, boundaries
    )
    return [
        FrozenFeatures(ppg, acc, context, correction)
        for ppg, acc, context, correction in zip(
            ppg_parts, acc_parts, context_parts, correction_parts
        )
    ]


def _freeze_case(
    model: CompactResidualGateWarmStartModel,
    case: ClaimedTargetCase,
    device: torch.device,
) -> FrozenTargetCase:
    groups = [case.enrollment]
    probe_keys = []
    for owner_id, activity_tensors in case.probes_by_owner.items():
        for activity, tensors in activity_tensors.items():
            groups.append(tensors)
            probe_keys.append((owner_id, activity))
    lengths = [len(group[0]) for group in groups]
    merged_tensors = tuple(
        torch.cat([group[index] for group in groups], dim=0)
        for index in range(3)
    )
    frozen_groups = _split_frozen_features(
        _extract_frozen_features(model, merged_tensors, device), lengths
    )
    probes: Dict[int, Dict[int, FrozenFeatures]] = {}
    for (owner_id, activity), features in zip(
        probe_keys, frozen_groups[1:]
    ):
        probes.setdefault(owner_id, {})[activity] = features
    return FrozenTargetCase(
        target_id=case.target_id,
        enrollment=frozen_groups[0],
        probes_by_owner=probes,
    )


def _precompute_training_cases(
    model: CompactResidualGateWarmStartModel,
    fold_id: int,
    data_root: Path,
    device: torch.device,
    log: Logger,
) -> Dict[int, Sequence[FrozenTargetCase]]:
    output: Dict[int, Sequence[FrozenTargetCase]] = {}
    train_users = FOLD_CONFIG[fold_id]["train"]
    for seconds in ENROLL_SECONDS:
        started = time.time()
        log(
            f"Building claimed-target training cases: enrollment={seconds}s "
            f"targets={len(train_users)}"
        )
        raw_cases = build_claimed_target_cases(
            data_root, train_users, seconds
        )
        frozen = []
        for index, case in enumerate(raw_cases, 1):
            frozen.append(_freeze_case(model, case, device))
            log(
                f"Frozen features enrollment={seconds}s target="
                f"{case.target_id} ({index}/{len(raw_cases)})"
            )
        output[seconds] = frozen
        del raw_cases
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(
            f"Enrollment={seconds}s feature pass complete in "
            f"{(time.time() - started) / 60.0:.1f} min"
        )
    return output


def _embedding_from_frozen(
    model: CompactResidualGateWarmStartModel,
    features: FrozenFeatures,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float]]:
    ppg_embedding = features.ppg_embedding.to(device)
    acc_embedding = features.acc_embedding.to(device)
    context = features.context.to(device)
    correction = features.temperature_correction.to(device)
    normalized_context = model.gate.context_norm(context)
    codes = model.gate.gate_code(normalized_context).view(
        len(features), 2, model.gate.gate_rank
    )
    ppg_gate = 1.0 + model.gate.max_delta * torch.tanh(
        model.gate.gate_expand["ppg"](codes[:, 0])
    )
    acc_gate = 1.0 + model.gate.max_delta * torch.tanh(
        model.gate.gate_expand["acc"](codes[:, 1])
    )
    fused = torch.cat(
        (ppg_embedding * ppg_gate, acc_embedding * acc_gate), dim=1
    )
    embedding = F.normalize(
        model.projector(fused) + correction, p=2, dim=1
    )
    identity_penalty = 0.5 * (
        (ppg_gate - 1.0).square().mean()
        + (acc_gate - 1.0).square().mean()
    )
    gate_range = (
        float(torch.minimum(ppg_gate.min(), acc_gate.min()).detach()),
        float(torch.maximum(ppg_gate.max(), acc_gate.max()).detach()),
    )
    return embedding, identity_penalty, gate_range


def _top_m_scores(
    probes: torch.Tensor, enrollment: torch.Tensor, top_m: int
) -> torch.Tensor:
    similarities = probes @ enrollment.T
    effective_m = min(int(top_m), similarities.size(1))
    return similarities.topk(effective_m, dim=1).values.mean(dim=1)


def _ranking_loss(
    enrollment: torch.Tensor,
    genuine: torch.Tensor,
    impostor: torch.Tensor,
) -> torch.Tensor:
    loss = genuine.new_zeros(())
    for top_m in TOP_M_VALUES:
        genuine_scores = _top_m_scores(
            genuine, enrollment=enrollment, top_m=top_m
        )
        impostor_scores = _top_m_scores(
            impostor, enrollment=enrollment, top_m=top_m
        )
        differences = (
            impostor_scores[:, None]
            - genuine_scores[None, :]
            + RANK_MARGIN
        ) / RANK_TEMPERATURE
        loss = loss + F.softplus(differences).mean()
    return loss / len(TOP_M_VALUES)


def _save_checkpoint(
    model: CompactResidualGateWarmStartModel,
    optimizer: Adam,
    source: Mapping[str, object],
    fold_id: int,
    epoch: int,
    history: Sequence[dict],
    result_root: Path,
    trainable_parameters: int,
    identity_max_abs_error: float,
) -> Path:
    model_state = cpu_clone_state(model.state_dict())
    path = calibration_checkpoint_path(result_root, fold_id, epoch)
    payload = {
        "format_version": 1,
        "method": METHOD,
        "variant": METHOD,
        "fold_id": fold_id,
        "calibration_epoch": epoch,
        "source_variant": SOURCE_VARIANT,
        "source_epoch": SOURCE_EPOCH,
        "source_checkpoint": str(
            source_checkpoint_path(result_root, fold_id).resolve()
        ),
        "source_model_state_sha256": source["model_state_sha256"],
        "model_state_sha256": state_digest(model_state),
        "model_state": model_state,
        "gate_optimizer_state": optimizer.state_dict(),
        "calibration_history": list(history),
        "primary_checkpoint": epoch == PRIMARY_CALIBRATION_EPOCH,
        "checkpoint_selection": "prespecified calibration horizon; no OOB selection",
        "trainable_parameters": trainable_parameters,
        "identity_warmstart_max_abs_error": identity_max_abs_error,
        "protocol": _protocol_payload(),
    }
    atomic_torch_save(payload, path)
    return path


def _calibrate(
    model: CompactResidualGateWarmStartModel,
    cases_by_enrollment: Mapping[int, Sequence[FrozenTargetCase]],
    optimizer: Adam,
    source: Mapping[str, object],
    fold_id: int,
    result_root: Path,
    log: Logger,
    trainable_parameters: int,
    identity_max_abs_error: float,
) -> Sequence[dict]:
    history = []
    _save_checkpoint(
        model,
        optimizer,
        source,
        fold_id,
        0,
        history,
        result_root,
        trainable_parameters,
        identity_max_abs_error,
    )
    log("Saved C00; it is exactly the ungated Compact Temp MLP E50")

    device = next(model.parameters()).device
    seed = FOLD_SEEDS[fold_id] + 700_000
    for epoch in range(1, CALIBRATION_EPOCHS + 1):
        epoch_started = time.time()
        model.eval()
        model.gate.train()
        steps = [
            (seconds, target_index, activity)
            for seconds in ENROLL_SECONDS
            for target_index, case in enumerate(cases_by_enrollment[seconds])
            for activity in range(len(ACTIVITY_NAMES))
            if activity
            in case.probes_by_owner.get(case.target_id, {})
        ]
        random.Random(seed + epoch).shuffle(steps)
        loss_sum = 0.0
        rank_sum = 0.0
        regularization_sum = 0.0
        gate_min = float("inf")
        gate_max = float("-inf")
        for seconds, target_index, activity in steps:
            case = cases_by_enrollment[seconds][target_index]
            available_impostors = sorted(
                owner_id
                for owner_id, probes in case.probes_by_owner.items()
                if owner_id != case.target_id and activity in probes
            )
            if not available_impostors:
                continue
            rotation = (
                epoch
                + target_index
                + activity
                + ENROLL_SECONDS.index(seconds)
            ) % len(available_impostors)
            selected_owners = [
                available_impostors[(rotation + offset) % len(available_impostors)]
                for offset in range(
                    min(IMPOSTORS_PER_STEP, len(available_impostors))
                )
            ]
            impostor_features = FrozenFeatures(
                ppg_embedding=torch.cat(
                    [
                        case.probes_by_owner[owner][activity].ppg_embedding
                        for owner in selected_owners
                    ]
                ),
                acc_embedding=torch.cat(
                    [
                        case.probes_by_owner[owner][activity].acc_embedding
                        for owner in selected_owners
                    ]
                ),
                context=torch.cat(
                    [
                        case.probes_by_owner[owner][activity].context
                        for owner in selected_owners
                    ]
                ),
                temperature_correction=torch.cat(
                    [
                        case.probes_by_owner[owner][activity].temperature_correction
                        for owner in selected_owners
                    ]
                ),
            )
            enrollment, enroll_reg, enroll_range = _embedding_from_frozen(
                model, case.enrollment, device
            )
            genuine, genuine_reg, genuine_range = _embedding_from_frozen(
                model,
                case.probes_by_owner[case.target_id][activity],
                device,
            )
            impostor, impostor_reg, impostor_range = _embedding_from_frozen(
                model, impostor_features, device
            )
            rank_loss = _ranking_loss(enrollment, genuine, impostor)
            identity_penalty = (
                enroll_reg + genuine_reg + impostor_reg
            ) / 3.0
            loss = rank_loss + IDENTITY_REGULARIZATION * identity_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            rank_sum += float(rank_loss.detach())
            regularization_sum += float(identity_penalty.detach())
            gate_min = min(
                gate_min,
                enroll_range[0],
                genuine_range[0],
                impostor_range[0],
            )
            gate_max = max(
                gate_max,
                enroll_range[1],
                genuine_range[1],
                impostor_range[1],
            )

        record = {
            "calibration_epoch": epoch,
            "steps": len(steps),
            "loss": loss_sum / len(steps),
            "ranking_loss": rank_sum / len(steps),
            "identity_penalty": regularization_sum / len(steps),
            "gate_min_observed": gate_min,
            "gate_max_observed": gate_max,
            "elapsed_minutes": (time.time() - epoch_started) / 60.0,
        }
        history.append(record)
        log(
            f"C{epoch:02d}/{CALIBRATION_EPOCHS:02d} "
            f"loss={record['loss']:.5f} rank={record['ranking_loss']:.5f} "
            f"gate=[{gate_min:.3f},{gate_max:.3f}] "
            f"time={record['elapsed_minutes']:.1f} min"
        )
        if epoch in CALIBRATION_HORIZONS:
            _save_checkpoint(
                model,
                optimizer,
                source,
                fold_id,
                epoch,
                history,
                result_root,
                trainable_parameters,
                identity_max_abs_error,
            )
    return history


def _evaluate_fold(
    model: CompactResidualGateWarmStartModel,
    fold_id: int,
    data_root: Path,
    result_root: Path,
    device: torch.device,
    log: Logger,
    history: Sequence[dict],
) -> dict:
    output_path = result_root / METHOD_RESULT_SUBDIR / f"oob_fold{fold_id}.json"
    payload = {
        "status": "evaluating",
        "method": METHOD,
        "variant": METHOD,
        "fold_id": fold_id,
        "train_users": FOLD_CONFIG[fold_id]["train"],
        "oob_users": FOLD_CONFIG[fold_id]["oob"],
        "source_epoch": SOURCE_EPOCH,
        "primary_calibration_epoch": PRIMARY_CALIBRATION_EPOCH,
        "checkpoint_selection": "none; C10 is prespecified primary",
        "protocol": _protocol_payload(),
        "calibration_history": list(history),
        "calibration_epochs": {},
    }
    atomic_json_dump(payload, output_path)
    log("Building untouched claimed-target OOB cases for final reporting")
    cases_by_enrollment = {
        seconds: build_claimed_target_cases(
            data_root, FOLD_CONFIG[fold_id]["oob"], seconds
        )
        for seconds in ENROLL_SECONDS
    }
    evaluation_started = time.time()
    for epoch in CALIBRATION_HORIZONS:
        path = calibration_checkpoint_path(result_root, fold_id, epoch)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("method") != METHOD
            or int(checkpoint.get("fold_id", -1)) != fold_id
            or int(checkpoint.get("calibration_epoch", -1)) != epoch
        ):
            raise RuntimeError(f"Calibration checkpoint mismatch: {path}")
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        macro_eer, cells = evaluate_claimed_target_grid(
            model, cases_by_enrollment, device=device
        )
        payload["calibration_epochs"][str(epoch)] = {
            "calibration_epoch": epoch,
            "primary": epoch == PRIMARY_CALIBRATION_EPOCH,
            "checkpoint": str(path.resolve()),
            "model_state_sha256": checkpoint["model_state_sha256"],
            "oob_macro_eer_across_16_cells": macro_eer,
            "cells": cells,
        }
        payload["elapsed_evaluation_hours"] = (
            time.time() - evaluation_started
        ) / 3600.0
        atomic_json_dump(payload, output_path)
        log(f"Evaluated C{epoch:02d}: OOB macro EER={100 * macro_eer:.3f}%")

    source_oob_path = (
        result_root / SOURCE_RESULT_SUBDIR / f"oob_fold{fold_id}.json"
    )
    if source_oob_path.is_file():
        source_oob = json.loads(source_oob_path.read_text(encoding="utf-8"))
        source_macro = float(
            source_oob["epochs"][str(SOURCE_EPOCH)][
                "oob_macro_eer_across_16_cells"
            ]
        )
        c00_macro = float(
            payload["calibration_epochs"]["0"][
                "oob_macro_eer_across_16_cells"
            ]
        )
        parity_error = abs(source_macro - c00_macro)
        payload["c00_source_oob_macro_eer"] = source_macro
        payload["c00_oob_macro_parity_abs_error"] = parity_error
        if parity_error > C00_OOB_PARITY_TOLERANCE:
            raise RuntimeError(
                f"C00 OOB parity failed: absolute EER error={parity_error}"
            )
    payload["status"] = "complete"
    payload["elapsed_evaluation_hours"] = (
        time.time() - evaluation_started
    ) / 3600.0
    atomic_json_dump(payload, output_path)
    log(f"Fold {fold_id} complete: {output_path}")
    del cases_by_enrollment
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def train_and_evaluate_fold(
    fold_id: int,
    data_root: Path,
    result_root: Path,
    device: torch.device,
) -> dict:
    method_dir = result_root / METHOD_RESULT_SUBDIR
    log = Logger(method_dir / f"fold{fold_id}.log")
    log("=" * 88)
    log(
        f"method={METHOD} fold={fold_id} device={device} "
        f"train={FOLD_CONFIG[fold_id]['train']} oob={FOLD_CONFIG[fold_id]['oob']}"
    )
    set_seed(FOLD_SEEDS[fold_id] + 700_000)
    model, source, identity_error = _load_warmstarted_model(
        fold_id, result_root, device
    )
    trainable_parameters = _freeze_backbone(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    log(
        f"Warm start verified exactly (max_abs_error={identity_error}); "
        f"trainable gate parameters={trainable_parameters:,}/"
        f"{total_parameters:,}"
    )
    optimizer = Adam(model.gate.parameters(), lr=LEARNING_RATE)
    frozen_cases = _precompute_training_cases(
        model, fold_id, data_root, device, log
    )
    history = _calibrate(
        model,
        frozen_cases,
        optimizer,
        source,
        fold_id,
        result_root,
        log,
        trainable_parameters,
        identity_error,
    )
    del frozen_cases
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _evaluate_fold(
        model,
        fold_id,
        data_root,
        result_root,
        device,
        log,
        history,
    )


def _cell(payload: dict, epoch: int, seconds: int, top_m: int) -> dict:
    return payload["calibration_epochs"][str(epoch)]["cells"][str(seconds)][
        str(top_m)
    ]


def _block_result(cell: dict, block_index: int) -> dict:
    """Read current block labels and historical positional labels."""
    block_results = cell["activity"]
    block_name = ACTIVITY_NAMES[block_index]
    if block_name in block_results:
        return block_results[block_name]
    if len(block_results) != len(ACTIVITY_NAMES):
        raise RuntimeError("Unexpected number of Temporal Block results")
    return list(block_results.values())[block_index]


def _write_csv_atomic(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate_if_complete(result_root: Path) -> dict | None:
    method_dir = result_root / METHOD_RESULT_SUBDIR
    folds = []
    for fold_id in range(1, 5):
        path = method_dir / f"oob_fold{fold_id}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            return None
        folds.append(payload)

    cross_fold = {}
    rows = []
    activity_rows = []
    for epoch in CALIBRATION_HORIZONS:
        epoch_output = {}
        macro_values = np.asarray(
            [
                fold["calibration_epochs"][str(epoch)][
                    "oob_macro_eer_across_16_cells"
                ]
                for fold in folds
            ],
            dtype=np.float64,
        )
        for seconds in ENROLL_SECONDS:
            seconds_output = {}
            for top_m in TOP_M_VALUES:
                cells = [
                    _cell(fold, epoch, seconds, top_m) for fold in folds
                ]
                values = np.asarray(
                    [cell["overall"]["eer"] for cell in cells],
                    dtype=np.float64,
                )
                activities = {}
                for block_index, activity in enumerate(ACTIVITY_NAMES):
                    activity_values = np.asarray(
                        [
                            _block_result(cell, block_index)["eer"]
                            for cell in cells
                        ],
                        dtype=np.float64,
                    )
                    activities[activity] = {
                        "eer_mean": float(activity_values.mean()),
                        "eer_std": float(activity_values.std()),
                        "per_fold": activity_values.tolist(),
                    }
                    activity_rows.append(
                        {
                            "method": METHOD,
                            "calibration_epoch": epoch,
                            "primary": epoch == PRIMARY_CALIBRATION_EPOCH,
                            "enroll_seconds": seconds,
                            "top_m": top_m,
                            "activity": activity,
                            "eer_mean": float(activity_values.mean()),
                            "eer_std": float(activity_values.std()),
                            **{
                                f"fold{index}_eer": float(value)
                                for index, value in enumerate(activity_values, 1)
                            },
                        }
                    )
                seconds_output[str(top_m)] = {
                    "overall": {
                        "eer_mean": float(values.mean()),
                        "eer_std": float(values.std()),
                        "per_fold": values.tolist(),
                    },
                    "activity": activities,
                }
                rows.append(
                    {
                        "method": METHOD,
                        "calibration_epoch": epoch,
                        "primary": epoch == PRIMARY_CALIBRATION_EPOCH,
                        "enroll_seconds": seconds,
                        "top_m": top_m,
                        "eer_mean": float(values.mean()),
                        "eer_std": float(values.std()),
                        **{
                            f"fold{index}_eer": float(value)
                            for index, value in enumerate(values, 1)
                        },
                    }
                )
            epoch_output[str(seconds)] = seconds_output
        cross_fold[str(epoch)] = {
            "macro_eer_mean_across_folds": float(macro_values.mean()),
            "macro_eer_std_across_folds": float(macro_values.std()),
            "macro_eer_per_fold": macro_values.tolist(),
            "cells": epoch_output,
        }

    output = {
        "status": "complete",
        "method": METHOD,
        "source_variant": SOURCE_VARIANT,
        "source_epoch": SOURCE_EPOCH,
        "primary_calibration_epoch": PRIMARY_CALIBRATION_EPOCH,
        "checkpoint_selection": "none; C10 is prespecified primary",
        "protocol": _protocol_payload(),
        "cross_fold": cross_fold,
    }
    atomic_json_dump(output, method_dir / "oob_cross_fold.json")
    _write_csv_atomic(method_dir / "oob_cross_fold_summary.csv", rows)
    _write_csv_atomic(method_dir / "oob_activity_summary.csv", activity_rows)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate_only:
        output = aggregate_if_complete(args.result_root)
        if output is None:
            raise RuntimeError("All four completed fold outputs are required")
        return 0
    if args.fold is None:
        raise ValueError("--fold is required unless --aggregate-only is used")
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)

    # A previous run may have completed all expensive OOB evaluations and then
    # stopped only at the parity assertion.  Validate and finalize that output
    # instead of rebuilding frozen features and recalibrating the same Gate.
    output_path = (
        args.result_root / METHOD_RESULT_SUBDIR / f"oob_fold{args.fold}.json"
    )
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        completed_horizons = payload.get("calibration_epochs", {})
        if all(str(epoch) in completed_horizons for epoch in CALIBRATION_HORIZONS):
            source_oob_path = (
                args.result_root
                / SOURCE_RESULT_SUBDIR
                / f"oob_fold{args.fold}.json"
            )
            source_oob = json.loads(source_oob_path.read_text(encoding="utf-8"))
            source_macro = float(
                source_oob["epochs"][str(SOURCE_EPOCH)][
                    "oob_macro_eer_across_16_cells"
                ]
            )
            c00_macro = float(
                completed_horizons["0"]["oob_macro_eer_across_16_cells"]
            )
            parity_error = abs(source_macro - c00_macro)
            if parity_error <= C00_OOB_PARITY_TOLERANCE:
                payload["c00_source_oob_macro_eer"] = source_macro
                payload["c00_oob_macro_parity_abs_error"] = parity_error
                payload["status"] = "complete"
                atomic_json_dump(payload, output_path)
                print(
                    f"Reusing completed fold {args.fold} evaluation; "
                    f"C00 parity error={parity_error}",
                    flush=True,
                )
                aggregate_if_complete(args.result_root)
                return 0
    train_and_evaluate_fold(
        args.fold,
        args.data_root,
        args.result_root,
        torch.device(args.device),
    )
    aggregate_if_complete(args.result_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
