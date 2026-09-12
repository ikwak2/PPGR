#!/usr/bin/env python3
"""PulseID under strict subject-disjoint OOB PPG verification.

PulseID (Wei et al., BSPC 2024) contributes PPG-specific multi-scale
augmentation, a multi-scale CNN feature-fusion encoder, and a curriculum
combining identity cross entropy with triplet loss.  Its original evaluation is
identification-oriented; this module leaves the OOB users out of all fitting and
uses its frozen encoder only for enrollment--probe template matching.

The article does not provide an executable reference implementation nor every
numeric implementation detail required for the present data.  The fixed,
pre-registered completions are deliberately collected in ``model_payload`` and
``protocol_payload``.  They are never selected using OOB enrollment or probe.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

from oob_input_cache import build_cached
from macro12_reporting import aggregate_macro12, summarize_macro12, evaluation_lock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from protocol import (
    ACTIVITY_NAMES,
    ACTIVITY_START_MINUTES,
    EXCLUSIONS,
    FOLD_CONFIG,
    FOLD_CONFIG_SOURCE,
    compute_eer,
)


HERE = Path(__file__).resolve().parent
VARIANT = "pulseid_green_oob"
RESULT_SUBDIR = Path("comparison_model_redesign") / VARIANT
DEFAULT_DATA_ROOT = Path("/Data/CRS25/PPG_Certifiation/data/Final_Data")
SEED_OFFSET = int(os.environ.get("FINAL_SEED_OFFSET", "0"))
DEFAULT_RESULT_ROOT = (
    HERE / "results" / "FINAL_260906_PULSEID_GREEN_OOB" / f"offset_{SEED_OFFSET}"
)

# The current data rate is deliberately retained; all durations below are in
# seconds so no source-dataset sampling rate leaks into the present protocol.
FS = 128
MINUTE_SAMPLES = 60 * FS
BLOCK_MINUTES = 60
BLOCK_SAMPLES = BLOCK_MINUTES * MINUTE_SAMPLES
WINDOW_SECONDS = 10
WINDOW_SAMPLES = WINDOW_SECONDS * FS
STRIDE_SECONDS = WINDOW_SECONDS
STRIDE_SAMPLES = STRIDE_SECONDS * FS
ENROLL_START_SECONDS = 120
ENROLL_SECONDS = (10, 20, 30)
PROBE_START_MINUTES = 48
PROBE_END_MINUTES = 58
TOP_M_VALUES = (1, 3, 5, 10)

# Fixed current-study training completion.  E50 is used across the redesigned
# comparators; no validation or OOB score can select a different checkpoint.
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TRIPLET_MARGIN = 0.2
CURRICULUM_CE_ONLY_EPOCHS = 10
CE_WEIGHT = 1.0
TRIPLET_WEIGHT = 1.0
MULTISCALE_CROP_FRACTIONS = (0.50, 0.75, 1.00)

# Multi-scale CNN completion.  The layer widths are fixed before running OOB
# data, not tuned to OOB performance.  A global temporal pool makes the encoder
# insensitive to the crop length used during PulseID augmentation.
STEM_CHANNELS = 96
BRANCH_MID_CHANNELS = 192
BRANCH_OUT_CHANNELS = 256
FUSION_CHANNELS = 320
EMBED_DIM = 512
SCALE_KERNELS = (3, 5, 7)

BASE_FOLD_SEEDS = {1: 42, 2: 123, 3: 456, 4: 789}
FOLD_SEEDS = {fold: seed + SEED_OFFSET for fold, seed in BASE_FOLD_SEEDS.items()}
BASE_PROTOCOL_ID = "pulseid_green_oob_fixed240540_v1"
PROTOCOL_ID = (
    BASE_PROTOCOL_ID if SEED_OFFSET == 0 else f"{BASE_PROTOCOL_ID}_seed_offset_{SEED_OFFSET}"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_json_dump(payload: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_clone_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def protocol_payload() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "variant": VARIANT,
        "seed_offset": SEED_OFFSET,
        "fold_config_source": FOLD_CONFIG_SOURCE,
        "folds": FOLD_CONFIG,
        "input": "single-channel green wrist PPG only; PPG/acc/temperature CSV column PPG",
        "excluded_modalities": ["red PPG", "IR PPG", "ACC", "temperature", "ECG"],
        "sampling_rate_hz": FS,
        "resampling": "none; native current-data 128 Hz",
        "preprocessing": {
            "bandpass_or_lowpass": "none; PulseID is used without an added signal filter",
            "window_normalization": "per-window zero mean and unit variance",
            "quality_rejection": "none beyond invalid/missing samples and existing protocol exclusions",
        },
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "stride_seconds": STRIDE_SECONDS,
        "stride_samples": STRIDE_SAMPLES,
        "overlap_seconds": 0,
        "training_augmentation": {
            "name": "PPG multi-scale random time crop followed by linear resampling",
            "crop_fractions": list(MULTISCALE_CROP_FRACTIONS),
            "scope": "background training inputs only",
            "oob_enrollment_and_probe": "disabled",
        },
        "enrollment_start_seconds": ENROLL_START_SECONDS,
        "enrollment_seconds": list(ENROLL_SECONDS),
        "probe_relative_minutes": [PROBE_START_MINUTES, PROBE_END_MINUTES],
        "support_shots": {"10": 5, "20": 10, "30": 15},
        "top_m_values": list(TOP_M_VALUES),
        "trial_score": "separate Top-M mean score for M=1,3,5,10; M>N uses all N references",
        "evaluation_aggregation": "macro12_v1",
        "epochs": EPOCHS,
        "checkpoint_selection": "none; prespecified raw E50",
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE},
        "batch_size": BATCH_SIZE,
        "training_objective": {
            "stage_1": f"E1-E{CURRICULUM_CE_ONLY_EPOCHS}: background 12-class cross entropy",
            "stage_2": (
                f"E{CURRICULUM_CE_ONLY_EPOCHS + 1}-E{EPOCHS}: "
                f"{CE_WEIGHT}*cross_entropy + {TRIPLET_WEIGHT}*triplet_margin"
            ),
            "triplet_margin": TRIPLET_MARGIN,
            "triplet_construction": "anchor/positive same background identity, negative different background identity",
            "sampler": "all anchor windows, DataLoader shuffle=True; no identity-balanced or weighted sampler",
        },
        "oob_separation": (
            "OOB enrollment and probe are excluded from model fitting, identity-head classes, "
            "multi-scale augmentation, triplet construction, normalization/statistic fitting, "
            "configuration selection, and checkpoint selection; enrollment only supplies "
            "frozen-encoder inference templates."
        ),
    }


def model_payload() -> dict:
    return {
        "name": "PulseID multi-scale CNN adapted to strict subject-disjoint OOB verification",
        "input_shape": ["B", 1, WINDOW_SAMPLES],
        "encoder": [
            "Conv1d(1,96,k=7,same) -> BN -> ReLU",
            "parallel branches k=3,5,7: Conv1d(96,192,k) -> BN -> ReLU -> Conv1d(192,256,k=3) -> BN -> ReLU -> MaxPool1d(4)",
            "concat(3x256) -> Conv1d(768,320,k=3,same) -> BN -> ReLU -> MaxPool1d(2)",
            "Conv1d(320,320,k=3,same) -> BN -> ReLU",
            "global average pool + global max pool -> Linear(640,512) -> ReLU -> L2-normalized embedding",
        ],
        "training_only_head": "Linear(512, background_class_count=12)",
        "evaluation": "remove 12-class head; cosine similarity of frozen L2-normalized 512-D embeddings",
        "implementation_completion": {
            "scale_kernels": list(SCALE_KERNELS),
            "stem_channels": STEM_CHANNELS,
            "branch_mid_channels": BRANCH_MID_CHANNELS,
            "branch_out_channels": BRANCH_OUT_CHANNELS,
            "fusion_channels": FUSION_CHANNELS,
            "embedding_dim": EMBED_DIM,
            "source_code_available": False,
            "numeric_widths_and_curriculum_schedule_selected_without_OOB": True,
        },
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    return [
        (int(start), int(end))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
    ]


def _normalize_window(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean, std = float(values.mean()), float(values.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-8:
        raise ValueError("non-finite or constant PPG window")
    return ((values - mean) / std).astype(np.float32, copy=False)


def _resample(values: np.ndarray, target_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == target_size:
        return values.copy()
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, target_size, dtype=np.float64)
    return np.interp(target, source, values).astype(np.float32)


class RawPPGUser:
    """Green-only current-data loader with the pre-existing exclusion rule."""

    def __init__(self, data_root: str | Path, user_id: int) -> None:
        self.user_id = int(user_id)
        path = Path(data_root) / f"user_{self.user_id}_final.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["PPG"])
        self.ppg = frame["PPG"].to_numpy(dtype=np.float64)
        self.n_samples = len(self.ppg)

    def block(self, block_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = ACTIVITY_START_MINUTES[int(block_index)] * MINUTE_SAMPLES
        end = min(start + BLOCK_SAMPLES, self.n_samples)
        if end - start < WINDOW_SAMPLES:
            raise RuntimeError(f"user {self.user_id}: missing Temporal Block {block_index + 1}")
        values = self.ppg[start:end].copy()
        valid = np.isfinite(values)
        excluded = EXCLUSIONS.get(self.user_id)
        if excluded is not None:
            overlap_start, overlap_end = max(start, excluded[0]), min(end, excluded[1])
            if overlap_start < overlap_end:
                valid[overlap_start - start : overlap_end - start] = False
        values[~valid] = np.nan
        return values, valid


def fixed_windows(values: np.ndarray, valid: np.ndarray, start: int, end: int) -> tuple[list[np.ndarray], dict]:
    """Return fully valid 10-second non-overlap normalized PPG windows."""
    start, end = max(0, int(start)), min(len(values), int(end))
    values, valid = np.asarray(values[start:end]), np.asarray(valid[start:end], dtype=bool)
    windows: list[np.ndarray] = []
    run_meta: list[dict] = []
    for run_start, run_end in _runs(valid):
        count = 0
        for left in range(run_start, run_end - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            segment = values[left : left + WINDOW_SAMPLES]
            if np.isfinite(segment).all():
                try:
                    windows.append(_normalize_window(segment)[None, :])
                    count += 1
                except ValueError:
                    # A zero-variance waveform cannot be converted to the source
                    # model input; this is not an additional morphology/SNR gate.
                    pass
        run_meta.append({"start": run_start, "end": run_end, "windows": count})
    return windows, {"requested_start": start, "requested_end": end, "runs": run_meta, "windows": len(windows)}


class PulseIDTripletDataset(Dataset):
    """Background-only anchors with deterministic multi-scale triplets each epoch."""

    def __init__(self, data_root: str | Path, user_ids: Sequence[int]) -> None:
        self.user_ids = [int(user) for user in user_ids]
        arrays, label_arrays = [], []
        per_user: dict[str, dict] = {}
        for label, user_id in enumerate(self.user_ids):
            raw = RawPPGUser(data_root, user_id)
            block_counts, block_metadata = [], []
            for block_index in range(len(ACTIVITY_START_MINUTES)):
                values, valid = raw.block(block_index)
                windows, metadata = fixed_windows(values, valid, 0, len(values))
                if windows:
                    arrays.append(np.stack(windows))
                    label_arrays.append(np.full(len(windows), label, dtype=np.int64))
                block_counts.append(len(windows))
                block_metadata.append(metadata)
            per_user[str(user_id)] = {
                "label": label,
                "block_window_counts": block_counts,
                "total_windows": int(sum(block_counts)),
                "block_metadata": block_metadata,
            }
        if not arrays:
            raise RuntimeError("No background PulseID windows were generated")
        self.inputs = torch.from_numpy(np.concatenate(arrays, axis=0).astype(np.float32, copy=False))
        self.labels = torch.from_numpy(np.concatenate(label_arrays, axis=0))
        self.indices_by_label = {
            label: np.flatnonzero(self.labels.numpy() == label).astype(np.int64)
            for label in range(len(self.user_ids))
        }
        if any(len(indices) < 2 for indices in self.indices_by_label.values()):
            raise RuntimeError("Every background participant needs at least two windows")
        self.position_in_label = np.empty(len(self.inputs), dtype=np.int64)
        for indices in self.indices_by_label.values():
            self.position_in_label[indices] = np.arange(len(indices), dtype=np.int64)
        self.epoch = 0
        self.seed = 0
        self.metadata = {
            "user_ids": self.user_ids,
            "per_user": per_user,
            "available_background_windows": int(len(self.inputs)),
            "triplets_per_epoch": int(len(self.inputs)),
            "identity_balanced_sampler": False,
            "input_shape": [1, WINDOW_SAMPLES],
        }

    def set_epoch(self, epoch: int, seed: int) -> None:
        self.epoch, self.seed = int(epoch), int(seed)

    def __len__(self) -> int:
        return len(self.inputs)

    def _mix(self, index: int, salt: int) -> int:
        value = (
            int(index) * 6364136223846793005
            + self.epoch * 1442695040888963407
            + self.seed * 22695477
            + int(salt)
        )
        return int(value & ((1 << 63) - 1))

    def _augment(self, values: torch.Tensor, index: int, salt: int) -> torch.Tensor:
        signal = values.squeeze(0).numpy()
        fraction = MULTISCALE_CROP_FRACTIONS[self._mix(index, salt) % len(MULTISCALE_CROP_FRACTIONS)]
        crop_length = max(3, int(round(WINDOW_SAMPLES * fraction)))
        start_limit = WINDOW_SAMPLES - crop_length
        start = 0 if start_limit == 0 else self._mix(index, salt + 11) % (start_limit + 1)
        crop = signal[start : start + crop_length]
        try:
            return torch.from_numpy(_normalize_window(_resample(crop, WINDOW_SAMPLES))[None, :])
        except ValueError:
            # A short crop can be constant even though its parent 10-second
            # window is valid and non-constant.  Keep that sample in the epoch
            # without inventing a quality-rejection rule: this augmentation draw
            # deterministically falls back to the already normalized source
            # window rather than excluding or replacing the sample.
            return values.clone()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_index = int(index)
        label = int(self.labels[anchor_index])
        same = self.indices_by_label[label]
        position = int(self.position_in_label[anchor_index])
        positive_position = self._mix(anchor_index, 29) % (len(same) - 1)
        if positive_position >= position:
            positive_position += 1
        positive_index = int(same[positive_position])
        other_label = self._mix(anchor_index, 43) % (len(self.user_ids) - 1)
        if other_label >= label:
            other_label += 1
        other = self.indices_by_label[int(other_label)]
        negative_index = int(other[self._mix(anchor_index, 59) % len(other)])
        return (
            self._augment(self.inputs[anchor_index], anchor_index, 101),
            self._augment(self.inputs[positive_index], positive_index, 211),
            self._augment(self.inputs[negative_index], negative_index, 307),
            torch.tensor(label, dtype=torch.long),
        )


@dataclass
class OOBUserWindows:
    user_id: int
    enrollment: Dict[int, torch.Tensor]
    probes: Dict[int, torch.Tensor]
    metadata: dict


def build_oob_windows(data_root: str | Path, user_ids: Sequence[int]) -> list[OOBUserWindows]:
    users: list[OOBUserWindows] = []
    for user_id in user_ids:
        raw = RawPPGUser(data_root, int(user_id))
        enrollment_parts = {seconds: [] for seconds in ENROLL_SECONDS}
        probes: dict[int, torch.Tensor] = {}
        metadata: dict = {"enrollment": {}, "probe": {}}
        for block_index in range(len(ACTIVITY_START_MINUTES)):
            values, valid = raw.block(block_index)
            for seconds in ENROLL_SECONDS:
                windows, details = fixed_windows(
                    values,
                    valid,
                    ENROLL_START_SECONDS * FS,
                    (ENROLL_START_SECONDS + seconds) * FS,
                )
                enrollment_parts[seconds].extend(windows)
                metadata["enrollment"].setdefault(str(seconds), []).append(details)
            probe_windows, probe_details = fixed_windows(
                values,
                valid,
                PROBE_START_MINUTES * MINUTE_SAMPLES,
                PROBE_END_MINUTES * MINUTE_SAMPLES,
            )
            metadata["probe"][str(block_index)] = probe_details
            if probe_windows:
                probes[block_index] = torch.from_numpy(np.stack(probe_windows))
        enrollment: dict[int, torch.Tensor] = {}
        for seconds, windows in enrollment_parts.items():
            if not windows:
                raise RuntimeError(f"user {user_id}: no valid {seconds}s OOB enrollment window")
            enrollment[seconds] = torch.from_numpy(np.stack(windows))
        if not probes:
            raise RuntimeError(f"user {user_id}: no valid OOB probe window")
        users.append(OOBUserWindows(int(user_id), enrollment, probes, metadata))
    return users


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(values)))


class _ScaleBranch(nn.Module):
    def __init__(self, kernel: int) -> None:
        super().__init__()
        self.scale = _ConvBNReLU(STEM_CHANNELS, BRANCH_MID_CHANNELS, kernel)
        self.local = _ConvBNReLU(BRANCH_MID_CHANNELS, BRANCH_OUT_CHANNELS, 3)
        self.pool = nn.MaxPool1d(4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.pool(self.local(self.scale(values)))


class PulseIDEncoder(nn.Module):
    """Multi-scale temporal CNN with a training-only 12-class identity head."""

    def __init__(self, class_count: int = 12) -> None:
        super().__init__()
        self.stem = _ConvBNReLU(1, STEM_CHANNELS, 7)
        self.branches = nn.ModuleList([_ScaleBranch(kernel) for kernel in SCALE_KERNELS])
        self.fuse = _ConvBNReLU(len(SCALE_KERNELS) * BRANCH_OUT_CHANNELS, FUSION_CHANNELS, 3)
        self.pool = nn.MaxPool1d(2)
        self.refine = _ConvBNReLU(FUSION_CHANNELS, FUSION_CHANNELS, 3)
        self.projection = nn.Linear(2 * FUSION_CHANNELS, EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, class_count)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 1 or values.shape[2] != WINDOW_SAMPLES:
            raise ValueError(f"expected [B,1,{WINDOW_SAMPLES}], got {tuple(values.shape)}")
        stem = self.stem(values)
        fused = torch.cat([branch(stem) for branch in self.branches], dim=1)
        fused = self.refine(self.pool(self.fuse(fused)))
        average = F.adaptive_avg_pool1d(fused, 1).squeeze(-1)
        maximum = F.adaptive_max_pool1d(fused, 1).squeeze(-1)
        return F.normalize(F.relu(self.projection(torch.cat([average, maximum], dim=1))), dim=1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(values)
        return embedding, self.classifier(embedding)

    def forward_with_shapes(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        shapes: dict[str, tuple[int, ...]] = {"input": tuple(values.shape)}
        stem = self.stem(values)
        shapes["stem"] = tuple(stem.shape)
        branches = [branch(stem) for branch in self.branches]
        shapes["branches"] = tuple(branches[0].shape)
        fused = torch.cat(branches, dim=1)
        shapes["concat"] = tuple(fused.shape)
        fused = self.refine(self.pool(self.fuse(fused)))
        shapes["fused"] = tuple(fused.shape)
        average = F.adaptive_avg_pool1d(fused, 1).squeeze(-1)
        maximum = F.adaptive_max_pool1d(fused, 1).squeeze(-1)
        embedding = F.normalize(F.relu(self.projection(torch.cat([average, maximum], dim=1))), dim=1)
        logits = self.classifier(embedding)
        shapes["embedding"] = tuple(embedding.shape)
        shapes["logits"] = tuple(logits.shape)
        return embedding, logits, shapes


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


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    return device


def _rng_payload(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def _restore_rng(payload: dict, generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])
    generator.set_state(payload["loader_generator"])


def _checkpoint_payload(model: PulseIDEncoder, fold_id: int, dataset_metadata: dict) -> dict:
    state = cpu_clone_state(model.state_dict())
    return {
        "format_version": 1,
        "variant": VARIANT,
        "protocol_id": PROTOCOL_ID,
        "fold_id": fold_id,
        "seed": FOLD_SEEDS[fold_id],
        "seed_offset": SEED_OFFSET,
        "epoch": EPOCHS,
        "checkpoint_selection": "raw_epoch_no_validation",
        "primary_checkpoint": True,
        "train_users": list(FOLD_CONFIG[fold_id]["train"]),
        "model_state": state,
        "model_state_sha256": state_digest(state),
        "model": model_payload(),
        "protocol": protocol_payload(),
        "training_dataset": dataset_metadata,
    }


def train_fold(fold_id: int, data_root: Path, result_root: Path, device: torch.device, resume: bool) -> dict:
    output, checkpoints = variant_dir(result_root), variant_dir(result_root) / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    log = Logger(output / f"train_fold{fold_id}.log")
    train_users, seed = FOLD_CONFIG[fold_id]["train"], FOLD_SEEDS[fold_id]
    log(f"protocol={PROTOCOL_ID} variant={VARIANT} fold={fold_id} train={train_users} device={device}")
    started = time.time()
    dataset = PulseIDTripletDataset(data_root, train_users)
    build_seconds = time.time() - started
    log(f"background_windows={len(dataset)} triplets_per_epoch={len(dataset)} batch_size={BATCH_SIZE} build_seconds={build_seconds:.1f}")
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed + 100_000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False,
                        generator=generator, pin_memory=device.type == "cuda")
    model = PulseIDEncoder(class_count=len(train_users)).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    latest_path = checkpoints / f"fold{fold_id}_latest.pt"
    raw_path = checkpoints / f"fold{fold_id}_E{EPOCHS:02d}.pt"
    history: list[dict] = []
    start_epoch = 1
    if resume and latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        if latest.get("variant") != VARIANT or latest.get("protocol_id") != PROTOCOL_ID or latest.get("fold_id") != fold_id:
            raise RuntimeError(f"resume metadata mismatch: {latest_path}")
        model.load_state_dict(latest["model_state"])
        optimizer.load_state_dict(latest["optimizer_state"])
        history = list(latest["history"])
        start_epoch = int(latest["epoch"]) + 1
        _restore_rng(latest["rng"], generator)
        log(f"resuming_after_epoch={start_epoch - 1}")
    if start_epoch > EPOCHS:
        if not raw_path.is_file():
            raise RuntimeError(f"completed latest checkpoint but missing {raw_path}")
        return {"fold_id": fold_id, "checkpoint": str(raw_path), "completed_epoch": EPOCHS}
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_started = time.time()
        dataset.set_epoch(epoch, seed)
        model.train()
        total_loss = total_ce = total_triplet = 0.0
        correct = total = 0
        for anchor, positive, negative, labels in loader:
            anchor, positive, negative, labels = (
                anchor.to(device, non_blocking=True),
                positive.to(device, non_blocking=True),
                negative.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            combined = torch.cat([anchor, positive, negative], dim=0)
            embeddings, logits = model(combined)
            anchor_embedding, positive_embedding, negative_embedding = torch.split(embeddings, len(anchor), dim=0)
            anchor_logits = logits[: len(anchor)]
            ce = F.cross_entropy(anchor_logits, labels)
            triplet = F.triplet_margin_loss(anchor_embedding, positive_embedding, negative_embedding, margin=TRIPLET_MARGIN, p=2)
            loss = ce if epoch <= CURRICULUM_CE_ONLY_EPOCHS else CE_WEIGHT * ce + TRIPLET_WEIGHT * triplet
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            total_ce += float(ce.detach())
            total_triplet += float(triplet.detach())
            correct += int((anchor_logits.argmax(dim=1) == labels).sum())
            total += len(labels)
        record = {
            "epoch": epoch,
            "phase": "ce_only" if epoch <= CURRICULUM_CE_ONLY_EPOCHS else "ce_plus_triplet",
            "loss": total_loss / max(1, len(loader)),
            "cross_entropy": total_ce / max(1, len(loader)),
            "triplet_loss": total_triplet / max(1, len(loader)),
            "anchor_class_accuracy": correct / max(1, total),
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(record)
        log(f"epoch={epoch:02d} phase={record['phase']} loss={record['loss']:.6f} ce={record['cross_entropy']:.6f} triplet={record['triplet_loss']:.6f} class_acc={100*record['anchor_class_accuracy']:.3f}% seconds={record['epoch_seconds']:.1f}")
        latest = {
            "format_version": 1, "variant": VARIANT, "protocol_id": PROTOCOL_ID,
            "fold_id": fold_id, "seed": seed, "seed_offset": SEED_OFFSET, "epoch": epoch,
            "train_users": list(train_users), "model_state": cpu_clone_state(model.state_dict()),
            "optimizer_state": optimizer.state_dict(), "history": history,
            "dataset_metadata": dataset.metadata, "rng": _rng_payload(generator),
        }
        atomic_torch_save(latest, latest_path)
        atomic_json_dump({
            "status": "complete" if epoch == EPOCHS else "training", "variant": VARIANT,
            "protocol_id": PROTOCOL_ID, "fold_id": fold_id, "seed": seed, "seed_offset": SEED_OFFSET,
            "completed_epoch": epoch, "primary_epoch": EPOCHS, "history": history,
            "training_dataset": dataset.metadata, "dataset_build_seconds": build_seconds,
        }, output / f"fold{fold_id}_training.json")
        if epoch == EPOCHS:
            payload = _checkpoint_payload(model, fold_id, dataset.metadata)
            atomic_torch_save(payload, raw_path)
            log(f"saved raw E{EPOCHS:02d} sha256={payload['model_state_sha256'][:12]} path={raw_path}")
    return {"fold_id": fold_id, "checkpoint": str(raw_path), "completed_epoch": EPOCHS,
            "elapsed_hours": (time.time() - started) / 3600.0}


def _encode(model: PulseIDEncoder, values: torch.Tensor, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            pieces.append(model.encode(values[start : start + batch_size].to(device)).cpu())
    return torch.cat(pieces, dim=0)


def _summary(score_lists: Mapping[int, Mapping[str, Sequence[float]]]) -> dict:
    activities, genuine_all, impostor_all = {}, [], []
    for block_index, name in enumerate(ACTIVITY_NAMES):
        genuine = list(score_lists[block_index]["genuine"])
        impostor = list(score_lists[block_index]["impostor"])
        eer, threshold = compute_eer(genuine, impostor) if genuine and impostor else (float("nan"), float("nan"))
        activities[name] = {"eer": eer, "threshold": threshold, "genuine_trials": len(genuine), "impostor_trials": len(impostor)}
        genuine_all.extend(genuine)
        impostor_all.extend(impostor)
    eer, threshold = compute_eer(genuine_all, impostor_all)
    return {"overall": {"eer": eer, "threshold": threshold, "genuine_trials": len(genuine_all), "impostor_trials": len(impostor_all)}, "activity": activities}


def evaluate_oob_users(model: PulseIDEncoder, users: Sequence[OOBUserWindows], device: torch.device) -> tuple[float, dict, dict]:
    enrollment_embeddings: dict[int, dict[int, torch.Tensor]] = {
        user.user_id: {seconds: _encode(model, user.enrollment[seconds], device) for seconds in ENROLL_SECONDS}
        for user in users
    }
    probe_embeddings: dict[int, dict[int, torch.Tensor]] = {
        user.user_id: {block: _encode(model, probes, device) for block, probes in user.probes.items()}
        for user in users
    }
    cells, eers = {}, []
    for seconds in ENROLL_SECONDS:
        score_lists = {m: {index: {"genuine": [], "impostor": []} for index in range(len(ACTIVITY_NAMES))} for m in TOP_M_VALUES}
        for target in users:
            references = enrollment_embeddings[target.user_id][seconds]
            for owner in users:
                kind = "genuine" if target.user_id == owner.user_id else "impostor"
                for block_index, probes in probe_embeddings[owner.user_id].items():
                    similarity = probes @ references.T
                    top_scores = []
                    for top_m in TOP_M_VALUES:
                        effective = min(top_m, similarity.shape[1])
                        top_scores.append(np.partition(similarity.numpy(), -effective, axis=1)[:, -effective:].mean(axis=1))
                    for top_m, scores in zip(TOP_M_VALUES, top_scores):
                        score_lists[top_m][block_index][kind].extend(scores.tolist())
        cells[str(seconds)] = {str(m): _summary(score_lists[m]) for m in TOP_M_VALUES}
        eers.extend(cell["overall"]["eer"] for cell in cells[str(seconds)].values())
    metadata = {str(user.user_id): user.metadata for user in users}
    return float(np.mean(eers)), cells, metadata


@evaluation_lock
def evaluate_fold(fold_id: int, data_root: Path, result_root: Path, device: torch.device) -> dict:
    output = variant_dir(result_root)
    checkpoint_path = output / "checkpoints" / f"fold{fold_id}_E{EPOCHS:02d}.pt"
    result_path = output / f"oob_fold{fold_id}.json"
    log = Logger(output / f"oob_fold{fold_id}.log")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != VARIANT or checkpoint.get("protocol_id") != PROTOCOL_ID or checkpoint.get("fold_id") != fold_id:
        raise RuntimeError(f"checkpoint metadata mismatch: {checkpoint_path}")
    if state_digest(checkpoint["model_state"]) != checkpoint["model_state_sha256"]:
        raise RuntimeError(f"checkpoint digest mismatch: {checkpoint_path}")
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("evaluation_aggregation") == "macro12_v1" and existing.get("model_state_sha256") == checkpoint["model_state_sha256"]:
            log(f"OOB already complete output={result_path}")
            return existing
    log(f"protocol={PROTOCOL_ID} variant={VARIANT} fold={fold_id} oob={FOLD_CONFIG[fold_id]['oob']} device={device}")
    model = PulseIDEncoder(class_count=len(FOLD_CONFIG[fold_id]["train"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    started = time.time()
    users = build_cached(build_oob_windows, data_root, FOLD_CONFIG[fold_id]["oob"], PROTOCOL_ID)
    macro_eer, cells, metadata = evaluate_oob_users(model, users, device)
    payload = {
        "status": "complete", "variant": VARIANT, "protocol_id": PROTOCOL_ID,
        "fold_id": fold_id, "seed": FOLD_SEEDS[fold_id], "seed_offset": SEED_OFFSET,
        "oob_users": list(FOLD_CONFIG[fold_id]["oob"]), "primary_epoch": EPOCHS,
        "checkpoint_selection": "none; prespecified raw E50", "checkpoint": str(checkpoint_path.resolve()),
        "model_state_sha256": checkpoint["model_state_sha256"],
        "evaluation_aggregation": "macro12_v1", "macro_eer_across_12_cells": macro_eer, "cells": cells,
        "window_metadata": metadata, "protocol": protocol_payload(), "model": model_payload(),
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    atomic_json_dump(payload, result_path)
    log(f"OOB complete macro12_eer={100*macro_eer:.3f}% output={result_path}")
    del model, users
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_folds(result_root: Path) -> dict:
    return aggregate_macro12(variant_dir(result_root), VARIANT, PROTOCOL_ID, SEED_OFFSET, protocol_payload(), model_payload())


def summarize_three_seeds(base_root: Path) -> dict:
    return summarize_macro12(base_root, RESULT_SUBDIR, VARIANT, PROTOCOL_ID)


def validate(device: torch.device) -> dict:
    for fold_id, fold in FOLD_CONFIG.items():
        if set(fold["train"]) & set(fold["oob"]) or len(fold["train"]) != 12 or len(fold["oob"]) != 4:
            raise RuntimeError(f"invalid strict-OOB fold {fold_id}")
    model = PulseIDEncoder().to(device).train()
    values = torch.randn(3, 1, WINDOW_SAMPLES, device=device)
    embedding, logits, shapes = model.forward_with_shapes(values)
    expected = {
        "input": (3, 1, WINDOW_SAMPLES), "stem": (3, STEM_CHANNELS, WINDOW_SAMPLES),
        "branches": (3, BRANCH_OUT_CHANNELS, WINDOW_SAMPLES // 4),
        "concat": (3, len(SCALE_KERNELS) * BRANCH_OUT_CHANNELS, WINDOW_SAMPLES // 4),
        "fused": (3, FUSION_CHANNELS, WINDOW_SAMPLES // 8), "embedding": (3, EMBED_DIM), "logits": (3, 12),
    }
    if shapes != expected:
        raise RuntimeError(f"shape regression: {shapes}")
    if not torch.allclose(embedding.norm(dim=1), torch.ones(len(embedding), device=device), atol=1e-5):
        raise RuntimeError("encoder output is not unit normalized")
    loss = F.cross_entropy(logits, torch.tensor([0, 1, 2], device=device)) + F.triplet_margin_loss(embedding, embedding.roll(1, 0), embedding.roll(2, 0), margin=TRIPLET_MARGIN)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite synthetic loss")
    encoder_parameters = parameter_count(model) - parameter_count(model.classifier)
    payload = {"status": "pass", "protocol_id": PROTOCOL_ID, "device": str(device),
               "shapes": {key: list(value) for key, value in shapes.items()},
               "encoder_parameters": encoder_parameters, "training_model_parameters": parameter_count(model),
               "synthetic_loss": float(loss.detach().cpu()), "protocol": protocol_payload(), "model": model_payload()}
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--fold", type=int, choices=range(1, 5), required=True)
    train.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    train.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    train.add_argument("--device", default="cuda")
    train.add_argument("--resume", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--fold", type=int, choices=range(1, 5), required=True)
    evaluate.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    evaluate.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    evaluate.add_argument("--device", default="cuda")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--base-root", type=Path, required=True)
    check = subparsers.add_parser("validate")
    check.add_argument("--device", default="cpu")
    check.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "train":
        train_fold(args.fold, args.data_root, args.result_root, _device(args.device), args.resume)
    elif args.command == "evaluate":
        evaluate_fold(args.fold, args.data_root, args.result_root, _device(args.device))
    elif args.command == "aggregate":
        aggregate_folds(args.result_root)
    elif args.command == "summarize":
        summarize_three_seeds(args.base_root)
    elif args.command == "validate":
        payload = validate(_device(args.device))
        if args.output is not None:
            atomic_json_dump(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
