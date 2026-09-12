#!/usr/bin/env python3
"""Warm-start PPG+ACC E50, train its Gate to C10, and run OOB."""

from fixed240540_protocol import activate

protocol = activate()

import train_residual_verification_gate as gate_runner  # noqa: E402
from models import (  # noqa: E402
    PPGACCResidualGateWarmStartModel,
    build_ppg_acc_residual_gate_warmstart_model,
)


gate_runner.METHOD = "ppg_acc_residual_verification_gate"
gate_runner.SOURCE_VARIANT = "ppg_acc"
gate_runner.SOURCE_RESULT_SUBDIR = "ppg_acc"
gate_runner.METHOD_RESULT_SUBDIR = gate_runner.METHOD
gate_runner.GATE_MODEL_VARIANT = "ppg_acc_gate"
gate_runner.CompactResidualGateWarmStartModel = PPGACCResidualGateWarmStartModel


def build_ppg_acc_gate_model(
    context_norm: str = "layernorm",
    max_delta: float = 0.5,
    gate_variant: str = "ppg_acc_gate",
    output_variant: str = gate_runner.METHOD,
) -> PPGACCResidualGateWarmStartModel:
    """Adapt the PPG+ACC builder to the generic Gate runner contract."""
    if gate_variant != "ppg_acc_gate":
        raise ValueError(f"Unexpected PPG+ACC gate variant: {gate_variant}")
    model = build_ppg_acc_residual_gate_warmstart_model(
        context_norm=context_norm,
        max_delta=max_delta,
    )
    model.variant = output_variant
    return model


gate_runner.build_compact_residual_gate_warmstart_model = build_ppg_acc_gate_model


def protocol_payload():
    payload = protocol.config_payload("claimed_target")
    payload["verification_gate_calibration"] = {
        "source_variant": "ppg_acc",
        "source_result_subdir": "ppg_acc",
        "method_result_subdir": "ppg_acc_residual_verification_gate",
        "gate_model_variant": "ppg_acc_gate",
        "source_epoch": 50,
        "trainable_parameters": "gate only",
        "frozen_parameters": ["ppg_encoder", "acc_encoder", "fusion_projector"],
        "gate": "1 + 0.5*tanh(delta), modality-specific channel expansion",
        "gate_context_features": [
            "ppg_low_high_log_power_ratio",
            "acc_low_power",
            "acc_high_power",
            "ppg_acc_corr",
        ],
        "temperature_feature_used": False,
        "calibration_epochs": 10,
        "reported_horizons": [0, 1, 2, 5, 10],
        "primary_horizon": 10,
        "enrollment_seconds_used_for_calibration": list(protocol.ENROLL_SECONDS),
        "top_m_values_used_in_loss": list(protocol.TOP_M_VALUES),
        "impostor_owners_per_target_activity_step": 2,
        "ranking_loss": "pairwise softplus over genuine/impostor Top-M scores",
        "identity_regularization": 0.01,
        "learning_rate": 1e-3,
        "oob_used_for_training_or_selection": False,
    }
    return payload


gate_runner._protocol_payload = protocol_payload


if __name__ == "__main__":
    raise SystemExit(gate_runner.main())
