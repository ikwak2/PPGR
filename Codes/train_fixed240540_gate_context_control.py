#!/usr/bin/env python3
"""Run controlled Seed/fold Gate calibrations with ablated context alignment.

The source E50 model, residual Gate architecture, optimizer, pairwise ranking
loss, and C00/C01/C02/C05/C10 horizons are inherited unchanged from
``train_residual_verification_gate``.  Only the four-dimensional context fed
to the Gate is controlled:

* ``context_free`` uses one train-fold mean context for every window.
* ``context_shuffled`` jointly permutes context rows within every extraction
  batch, preserving each four-feature vector while breaking its row pairing.

The control is selected with ``GATE_CONTEXT_CONTROL_MODE`` and the E50 source
with ``GATE_CONTEXT_CONTROL_SOURCE``.  This file intentionally writes to new
method directories and never overwrites the existing real-context results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, Mapping, Sequence

import torch

from fixed240540_protocol import activate


protocol = activate()

import train_residual_verification_gate as gate_runner  # noqa: E402
from models import (  # noqa: E402
    PPGACCResidualGateWarmStartModel,
    build_ppg_acc_residual_gate_warmstart_model,
)


CONTROL_MODES = {"real", "context_free", "context_shuffled"}
CONTROL_SOURCES = {"ppg_acc", "temp_mlp"}
CONTROL_MODE = os.environ.get(
    "GATE_CONTEXT_CONTROL_MODE", "context_free"
).strip()
CONTROL_SOURCE = os.environ.get(
    "GATE_CONTEXT_CONTROL_SOURCE", "ppg_acc"
).strip()
CONTROL_INITIALIZATION = os.environ.get(
    "GATE_CONTEXT_CONTROL_INITIALIZATION", "seeded"
).strip()
if CONTROL_INITIALIZATION not in {"seeded", "real_c00"}:
    raise ValueError(f"Unknown control initialization: {CONTROL_INITIALIZATION}")
if CONTROL_MODE not in CONTROL_MODES:
    raise ValueError(
        f"Unknown GATE_CONTEXT_CONTROL_MODE={CONTROL_MODE!r}; "
        f"choose from {sorted(CONTROL_MODES)}"
    )
if CONTROL_SOURCE not in CONTROL_SOURCES:
    raise ValueError(
        f"Unknown GATE_CONTEXT_CONTROL_SOURCE={CONTROL_SOURCE!r}; "
        f"choose from {sorted(CONTROL_SOURCES)}"
    )

SHUFFLE_SEED_OFFSET = 880_000
_FOLD_CONTROL_METADATA: Dict[int, dict] = {}
_FOLD_INITIALIZATION_METADATA: Dict[int, dict] = {}


def _configure_runner() -> None:
    if CONTROL_SOURCE == "ppg_acc":
        base_method = "ppg_acc_residual_verification_gate"
        gate_runner.SOURCE_VARIANT = "ppg_acc"
        gate_runner.SOURCE_RESULT_SUBDIR = "ppg_acc"
        gate_runner.GATE_MODEL_VARIANT = "ppg_acc_gate"
        gate_runner.CompactResidualGateWarmStartModel = (
            PPGACCResidualGateWarmStartModel
        )

        def ppg_acc_builder(
            context_norm: str = "layernorm",
            max_delta: float = 0.5,
            gate_variant: str = "ppg_acc_gate",
            output_variant: str = base_method,
        ) -> PPGACCResidualGateWarmStartModel:
            if gate_variant != "ppg_acc_gate":
                raise ValueError(f"Unexpected PPG+ACC gate variant: {gate_variant}")
            model = build_ppg_acc_residual_gate_warmstart_model(
                context_norm=context_norm,
                max_delta=max_delta,
            )
            model.variant = output_variant
            return model

        gate_runner.build_compact_residual_gate_warmstart_model = (
            ppg_acc_builder
        )
    else:
        base_method = "ppg_acc_temp_mlp_residual_verification_gate"
        gate_runner.SOURCE_VARIANT = "ppg_acc_temp_mlp"
        gate_runner.SOURCE_RESULT_SUBDIR = "ppg_acc_temp_mlp"
        gate_runner.GATE_MODEL_VARIANT = "ppg_acc_gate_temp_mlp"

    suffix = "" if CONTROL_MODE == "real" else f"_{CONTROL_MODE}"
    gate_runner.METHOD = f"{base_method}{suffix}"
    gate_runner.METHOD_RESULT_SUBDIR = gate_runner.METHOD


_configure_runner()


def _deterministic_permutation(length: int, fold_id: int) -> torch.Tensor:
    """Return the same non-identity row permutation for every equal batch."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if length <= 1:
        return torch.arange(length, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    seed = (
        int(protocol.FOLD_SEEDS[fold_id])
        + SHUFFLE_SEED_OFFSET
        + 1_009 * int(length)
    )
    generator.manual_seed(seed)
    permutation = torch.randperm(length, generator=generator)
    identity = torch.arange(length, dtype=torch.long)
    if torch.equal(permutation, identity):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


def _install_context_transform(
    model: torch.nn.Module,
    fold_id: int,
    train_fold_mean: torch.Tensor | None = None,
) -> None:
    if getattr(model.gate, "_context_control_installed", False):
        raise RuntimeError("Context control was installed more than once")
    original_context = model.gate._context

    if CONTROL_MODE == "context_free":
        if train_fold_mean is None:
            raise ValueError("context_free requires a train-fold context mean")
        fixed_mean = train_fold_mean.detach().cpu().clone()

        def controlled_context(ppg, temp, acc):
            context = original_context(ppg, temp, acc)
            mean = fixed_mean.to(device=context.device, dtype=context.dtype)
            return mean.view(1, -1).expand(context.size(0), -1)

    elif CONTROL_MODE == "context_shuffled":

        def controlled_context(ppg, temp, acc):
            context = original_context(ppg, temp, acc)
            permutation = _deterministic_permutation(
                context.size(0), fold_id
            ).to(context.device)
            return context.index_select(0, permutation)

    elif CONTROL_MODE == "real":

        def controlled_context(ppg, temp, acc):
            return original_context(ppg, temp, acc)

    else:  # pragma: no cover - validated at import
        raise AssertionError(CONTROL_MODE)

    model.gate._context = controlled_context
    model.gate._context_control_installed = True


def _iter_features(
    cases_by_enrollment: Mapping[
        int, Sequence[gate_runner.FrozenTargetCase]
    ],
) -> Iterator[gate_runner.FrozenFeatures]:
    for seconds in sorted(cases_by_enrollment):
        for case in cases_by_enrollment[seconds]:
            yield case.enrollment
            for owner_id in sorted(case.probes_by_owner):
                probes = case.probes_by_owner[owner_id]
                for activity in sorted(probes):
                    yield probes[activity]


def _train_fold_context_mean(
    cases_by_enrollment: Mapping[
        int, Sequence[gate_runner.FrozenTargetCase]
    ],
) -> tuple[torch.Tensor, int]:
    total = None
    row_count = 0
    output_dtype = None
    for features in _iter_features(cases_by_enrollment):
        context = features.context
        if context.ndim != 2 or context.size(1) != 4:
            raise RuntimeError(
                f"Expected four-dimensional Gate context, got {context.shape}"
            )
        output_dtype = context.dtype
        partial = context.to(dtype=torch.float64).sum(dim=0)
        total = partial if total is None else total + partial
        row_count += int(context.size(0))
    if total is None or row_count == 0 or output_dtype is None:
        raise RuntimeError("No train-fold Gate contexts were generated")
    mean = (total / row_count).to(dtype=output_dtype)
    if not bool(torch.isfinite(mean).all()):
        raise RuntimeError(f"Non-finite train-fold context mean: {mean}")
    return mean, row_count


def _replace_context_with_mean(
    cases_by_enrollment: Mapping[
        int, Sequence[gate_runner.FrozenTargetCase]
    ],
    mean: torch.Tensor,
) -> None:
    for features in _iter_features(cases_by_enrollment):
        features.context = mean.view(1, -1).expand(
            len(features), -1
        ).clone()


def _control_metadata(fold_id: int) -> dict:
    metadata = {
        "mode": CONTROL_MODE,
        "source": CONTROL_SOURCE,
        "fold_id": fold_id,
        "oob_used_to_construct_control": False,
        "context_features_permuted_jointly": (
            CONTROL_MODE == "context_shuffled"
        ),
    }
    if fold_id in _FOLD_INITIALIZATION_METADATA:
        metadata["matched_initialization"] = _FOLD_INITIALIZATION_METADATA[fold_id]
    if CONTROL_MODE == "context_free":
        metadata.update(
            {
                "definition": (
                    "Every training and OOB window receives the mean raw "
                    "four-feature context computed using this fold's training "
                    "claimed-target cases only."
                ),
                "train_fold_context_mean": _FOLD_CONTROL_METADATA[fold_id][
                    "train_fold_context_mean"
                ],
                "train_context_rows_used_for_mean": _FOLD_CONTROL_METADATA[
                    fold_id
                ]["train_context_rows_used_for_mean"],
            }
        )
    elif CONTROL_MODE == "context_shuffled":
        metadata.update(
            {
                "definition": (
                    "Raw four-feature context rows are jointly permuted "
                    "within each context-extraction forward batch, breaking "
                    "window/context alignment while preserving complete "
                    "context vectors and batch marginals."
                ),
                "maximum_forward_batch_rows": gate_runner.FEATURE_BATCH_SIZE,
                "permutation_seed": (
                    "FOLD_SEEDS[fold] + 880000 + 1009 * batch_rows"
                ),
                "same_permutation_for_equal_batch_sizes": True,
            }
        )
    else:
        metadata["definition"] = "Identity transform; real aligned context."
    return metadata


def protocol_payload() -> dict:
    payload = protocol.config_payload("claimed_target")
    payload["verification_gate_calibration"] = {
        "source_variant": gate_runner.SOURCE_VARIANT,
        "source_result_subdir": gate_runner.SOURCE_RESULT_SUBDIR,
        "method_result_subdir": gate_runner.METHOD_RESULT_SUBDIR,
        "gate_model_variant": gate_runner.GATE_MODEL_VARIANT,
        "source_epoch": gate_runner.SOURCE_EPOCH,
        "source_training_objective": (
            "participant-ID classification with AAM-Softmax through E50"
        ),
        "gate_calibration_objective": (
            "verification-aware pairwise ranking from C01 through C10"
        ),
        "classification_head_updated_during_gate_calibration": False,
        "trainable_parameters": "gate only",
        "frozen_parameters": (
            ["ppg_encoder", "acc_encoder", "fusion_projector"]
            if CONTROL_SOURCE == "ppg_acc"
            else [
                "ppg_encoder",
                "acc_encoder",
                "fusion_projector",
                "compact_temperature_mlp",
            ]
        ),
        "gate": "1 + 0.5*tanh(delta), modality-specific expansion",
        "gate_context_features": [
            "ppg_low_high_log_power_ratio",
            "acc_low_power",
            "acc_high_power",
            "ppg_acc_corr",
        ],
        "context_control": CONTROL_MODE,
        "context_control_initialization": CONTROL_INITIALIZATION,
        "context_control_scope": (
            "fold-training mean reused for OOB"
            if CONTROL_MODE == "context_free"
            else "joint row permutation within each forward batch"
            if CONTROL_MODE == "context_shuffled"
            else "none"
        ),
        "temperature_feature_used_by_gate": False,
        "temperature_mlp_present": CONTROL_SOURCE == "temp_mlp",
        "calibration_epochs": gate_runner.CALIBRATION_EPOCHS,
        "reported_horizons": list(gate_runner.CALIBRATION_HORIZONS),
        "primary_horizon": gate_runner.PRIMARY_CALIBRATION_EPOCH,
        "enrollment_seconds_used_for_calibration": list(
            protocol.ENROLL_SECONDS
        ),
        "top_m_values_used_in_loss": list(protocol.TOP_M_VALUES),
        "impostor_owners_per_target_activity_step": (
            gate_runner.IMPOSTORS_PER_STEP
        ),
        "ranking_loss": (
            "pairwise softplus over genuine/impostor Top-M scores"
        ),
        "ranking_margin": gate_runner.RANK_MARGIN,
        "ranking_temperature": gate_runner.RANK_TEMPERATURE,
        "identity_regularization": gate_runner.IDENTITY_REGULARIZATION,
        "learning_rate": gate_runner.LEARNING_RATE,
        "oob_used_for_training_or_selection": False,
        "controlled_against_real_context_method": (
            "ppg_acc_residual_verification_gate"
            if CONTROL_SOURCE == "ppg_acc"
            else "ppg_acc_temp_mlp_residual_verification_gate"
        ),
    }
    return payload


_original_load_warmstarted_model = gate_runner._load_warmstarted_model
_original_precompute_training_cases = gate_runner._precompute_training_cases
_original_evaluate_fold = gate_runner._evaluate_fold


def controlled_load_warmstarted_model(fold_id, result_root, device):
    model, source, identity_error = _original_load_warmstarted_model(
        fold_id, result_root, device
    )
    if CONTROL_INITIALIZATION == "real_c00":
        # Match the existing real-context experiment exactly even if constructor
        # RNG consumption has changed since that experiment was produced.
        # Only the prespecified, untrained C00 is consulted; C10/OOB scores are not.
        real_method = f"{gate_runner.SOURCE_VARIANT}_residual_verification_gate"
        reference_path = (
            Path(result_root) / real_method / "checkpoints" / f"fold{fold_id}_C00.pt"
        )
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        if (
            reference.get("fold_id") != fold_id
            or reference.get("calibration_epoch") != 0
            or reference.get("source_variant") != gate_runner.SOURCE_VARIANT
            or reference.get("source_epoch") != gate_runner.SOURCE_EPOCH
            or reference.get("protocol", {}).get("protocol_id") != protocol.PROTOCOL_ID
            or reference.get("source_model_state_sha256") != source["model_state_sha256"]
        ):
            raise RuntimeError(f"Matched C00 metadata mismatch: {reference_path}")
        state = reference["model_state"]
        if gate_runner.state_digest(state) != reference["model_state_sha256"]:
            raise RuntimeError(f"Matched C00 digest mismatch: {reference_path}")
        if any(not torch.equal(value, state[key]) for key, value in source["model_state"].items()):
            raise RuntimeError(f"Matched C00 has changed source weights: {reference_path}")
        if any(bool(torch.count_nonzero(value)) for key, value in state.items() if key.startswith("gate.gate_expand.")):
            raise RuntimeError(f"Matched C00 is not an identity Gate: {reference_path}")
        model.load_state_dict(state, strict=True)
        if gate_runner.state_digest(model.state_dict()) != reference["model_state_sha256"]:
            raise RuntimeError("Control did not reproduce the matched C00 weights")
        _FOLD_INITIALIZATION_METADATA[fold_id] = {
            "mode": "real_c00",
            "checkpoint": str(reference_path.resolve()),
            "model_state_sha256": reference["model_state_sha256"],
            "source_model_state_sha256": source["model_state_sha256"],
            "untrained_identity_gate_only": True,
            "oob_scores_used": False,
        }
    # The C00 exact-output assertion in the original loader runs first.
    if CONTROL_MODE in {"real", "context_shuffled"}:
        _install_context_transform(model, fold_id)
    return model, source, identity_error


def controlled_precompute_training_cases(
    model, fold_id, data_root, device, log
):
    cases = _original_precompute_training_cases(
        model, fold_id, data_root, device, log
    )
    if CONTROL_MODE == "context_free":
        mean, row_count = _train_fold_context_mean(cases)
        _replace_context_with_mean(cases, mean)
        _FOLD_CONTROL_METADATA[fold_id] = {
            "train_fold_context_mean": [float(value) for value in mean],
            "train_context_rows_used_for_mean": row_count,
        }
        _install_context_transform(model, fold_id, train_fold_mean=mean)
        log(
            "Installed context_free control from train fold only: "
            f"rows={row_count:,} mean={mean.tolist()}"
        )
    else:
        _FOLD_CONTROL_METADATA[fold_id] = {}
        log(f"Installed {CONTROL_MODE} Gate-context control")

    metadata = _control_metadata(fold_id)
    gate_runner.atomic_json_dump(
        metadata,
        Path(log.path).parent / f"fold{fold_id}_context_control.json",
    )
    return cases


def controlled_evaluate_fold(
    model, fold_id, data_root, result_root, device, log, history
):
    payload = _original_evaluate_fold(
        model,
        fold_id,
        data_root,
        result_root,
        device,
        log,
        history,
    )
    payload["context_control"] = _control_metadata(fold_id)
    output_path = (
        Path(result_root)
        / gate_runner.METHOD_RESULT_SUBDIR
        / f"oob_fold{fold_id}.json"
    )
    gate_runner.atomic_json_dump(payload, output_path)
    return payload


gate_runner._protocol_payload = protocol_payload
gate_runner._load_warmstarted_model = controlled_load_warmstarted_model
gate_runner._precompute_training_cases = controlled_precompute_training_cases
gate_runner._evaluate_fold = controlled_evaluate_fold


if __name__ == "__main__":
    if os.environ.get("GATE_CONTEXT_VALIDATE_INITIALIZATION_ONLY") == "1":
        args = gate_runner.parse_args()
        if args.fold is None:
            raise ValueError("--fold is required for initialization validation")
        gate_runner.set_seed(protocol.FOLD_SEEDS[args.fold] + 700_000)
        model, _, _ = controlled_load_warmstarted_model(
            args.fold, args.result_root, torch.device(args.device)
        )
        print(f"Matched C00 validation passed: source={CONTROL_SOURCE} fold={args.fold} protocol={protocol.PROTOCOL_ID}", flush=True)
        raise SystemExit(0)
    raise SystemExit(gate_runner.main())
