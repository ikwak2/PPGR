"""CorNET original-backbone data, model, and strict-OOB evaluation helpers.

This module is deliberately independent from the completed ``cornet_ppg_acc``
experiment.  It keeps the CorNET single-channel 8-second CNN-LSTM backbone,
uses a training-only AAM head on the 12 development identities, and evaluates
the frozen 128-D encoder on the four subject-disjoint OOB identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from torch.utils.data import Dataset

from protocol import (
    ACTIVITY_NAMES,
    ACTIVITY_START_MINUTES,
    EXCLUSIONS,
    FOLD_CONFIG,
    FOLD_CONFIG_SOURCE,
    compute_eer,
)


VARIANT = "cornet_original_backbone_oob"
RESULT_SUBDIR = Path("comparison_model_redesign") / VARIANT
RAW_FS = 128
FS = 125
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * FS
STRIDE_SECONDS = 2
STRIDE_SAMPLES = STRIDE_SECONDS * FS
MINUTE_RAW_SAMPLES = RAW_FS * 60
BLOCK_MINUTES = 60
BLOCK_RAW_SAMPLES = BLOCK_MINUTES * MINUTE_RAW_SAMPLES
ENROLL_START_SECONDS = 120
ENROLL_SECONDS = (10, 20, 30)
PROBE_START_MINUTES = 48
PROBE_END_MINUTES = 58
TOP_M_VALUES = (1, 3, 5, 10)
EPOCHS = 50
BATCH_SIZE = 25
LEARNING_RATE = 1e-3
RMSPROP_ALPHA = 0.9
RMSPROP_EPS = 1e-7
AAM_MARGIN = 0.2
AAM_SCALE = 30.0
BASE_FOLD_SEEDS = {1: 42, 2: 123, 3: 456, 4: 789}
SEED_OFFSET = int(os.environ.get("FINAL_SEED_OFFSET", "0"))
FOLD_SEEDS = {
    fold_id: seed + SEED_OFFSET for fold_id, seed in BASE_FOLD_SEEDS.items()
}
BASE_PROTOCOL_ID = "cornet_original_backbone_oob_fixed240540_v1"
PROTOCOL_ID = (
    BASE_PROTOCOL_ID
    if SEED_OFFSET == 0
    else f"{BASE_PROTOCOL_ID}_seed_offset_{SEED_OFFSET}"
)
PPG_SOS = signal.butter(
    4,
    [0.1, 18.0],
    btype="bandpass",
    fs=FS,
    output="sos",
)
MAX_INTERPOLATION_GAP_RAW_SAMPLES = RAW_FS - 1
NORMALIZATION_EPS = 1e-8


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
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".pid{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_clone_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def protocol_payload() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "variant": VARIANT,
        "seed_offset": SEED_OFFSET,
        "fold_config_source": FOLD_CONFIG_SOURCE,
        "folds": FOLD_CONFIG,
        "input": "single-channel green wrist PPG",
        "raw_sampling_rate_hz": RAW_FS,
        "model_sampling_rate_hz": FS,
        "resampling": "scipy.signal.resample_poly(up=125, down=128)",
        "filter": {
            "family": "Butterworth",
            "order": 4,
            "band_hz": [0.1, 18.0],
            "implementation": "SOS zero-phase sosfiltfilt on each full Temporal Block",
        },
        "normalization": {
            "scope": "independent 8-second window",
            "mean": "population mean",
            "std": "population std, ddof=0",
            "minimum_std": NORMALIZATION_EPS,
        },
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "overlap_seconds": WINDOW_SECONDS - STRIDE_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "stride_samples": STRIDE_SAMPLES,
        "activity_starts_minutes": list(ACTIVITY_START_MINUTES),
        "activity_names": list(ACTIVITY_NAMES),
        "background_training_relative_minutes": [0, BLOCK_MINUTES],
        "enrollment_start_seconds": ENROLL_START_SECONDS,
        "enrollment_seconds": list(ENROLL_SECONDS),
        "probe_relative_minutes": [PROBE_START_MINUTES, PROBE_END_MINUTES],
        "top_m_values": list(TOP_M_VALUES),
        "epochs": EPOCHS,
        "checkpoint_selection": "none; prespecified raw E50",
        "training_objective": {
            "name": "AAM-Softmax",
            "classes": 12,
            "margin": AAM_MARGIN,
            "scale": AAM_SCALE,
            "head_use": "training only; discarded for OOB scoring",
        },
        "optimizer": {
            "name": "RMSProp",
            "learning_rate": LEARNING_RATE,
            "alpha": RMSPROP_ALPHA,
            "epsilon": RMSPROP_EPS,
            "momentum": 0.0,
            "centered": False,
            "weight_decay": 0.0,
        },
        "batch_size": BATCH_SIZE,
        "sampling": "all windows with DataLoader shuffle=True; no identity balancing",
        "oob_separation": (
            "OOB enrollment and probe are excluded from training, loss, "
            "hyperparameter selection, and checkpoint selection; enrollment "
            "is used only to create frozen-encoder templates"
        ),
    }


def model_payload() -> dict:
    return {
        "name": "CorNET original backbone adapted to strict OOB verification",
        "input_shape": ["B", 1, WINDOW_SAMPLES],
        "conv_filters": 32,
        "kernel_size": 40,
        "conv_stride": 1,
        "conv_padding": 0,
        "pool_size": 4,
        "dropout": 0.1,
        "block_order": "Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d -> Dropout",
        "lstm_layers": 2,
        "lstm_hidden_size": 128,
        "lstm_direction": "unidirectional",
        "embedding": "second LSTM final hidden state h_T",
        "embedding_dim": 128,
        "dense_2": False,
        "temporal_mean": False,
        "projection_128_to_192": False,
        "training_head": "AAMSoftmax(128, 12), not used at evaluation",
    }


class RawPPGUser:
    """Load only the PPG column needed by this experiment."""

    def __init__(self, data_root: str | Path, user_id: int) -> None:
        self.user_id = int(user_id)
        self.path = Path(data_root) / f"user_{self.user_id}_final.csv"
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        frame = pd.read_csv(self.path, usecols=["PPG"])
        self.ppg = frame["PPG"].to_numpy(dtype=np.float64)
        self.n_samples = len(self.ppg)

    def block(self, block_index: int) -> tuple[np.ndarray, np.ndarray]:
        activity_start = ACTIVITY_START_MINUTES[int(block_index)]
        start = activity_start * MINUTE_RAW_SAMPLES
        end = min(self.n_samples, start + BLOCK_RAW_SAMPLES)
        if start >= end:
            raise RuntimeError(
                f"user {self.user_id}: missing Temporal Block {block_index + 1}"
            )
        values = self.ppg[start:end].copy()
        valid = np.isfinite(values)
        exclusion = EXCLUSIONS.get(self.user_id)
        if exclusion is not None:
            overlap_start = max(start, exclusion[0])
            overlap_end = min(end, exclusion[1])
            if overlap_start < overlap_end:
                relative_start = overlap_start - start
                relative_end = overlap_end - start
                valid[relative_start:relative_end] = False
                values[relative_start:relative_end] = np.nan
        return values, valid


def _missing_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


@dataclass
class ProcessedBlock:
    ppg: np.ndarray
    invalid_prefix_raw: np.ndarray
    raw_samples: int

    def window_is_valid(self, start: int, end: int) -> bool:
        if start < 0 or end > len(self.ppg) or end <= start:
            return False
        raw_start = max(0, int(math.floor(start * RAW_FS / FS)))
        raw_end = min(
            self.raw_samples, int(math.ceil(end * RAW_FS / FS))
        )
        if raw_start >= raw_end:
            return False
        invalid = (
            self.invalid_prefix_raw[raw_end]
            - self.invalid_prefix_raw[raw_start]
        )
        return bool(invalid == 0 and np.isfinite(self.ppg[start:end]).all())


def _filter_chunk(values: np.ndarray) -> np.ndarray | None:
    if len(values) < 64 or not np.isfinite(values).all():
        return None
    try:
        return signal.sosfiltfilt(PPG_SOS, values)
    except (ValueError, FloatingPointError):
        return None


def preprocess_block(raw_values: np.ndarray, raw_valid: np.ndarray) -> ProcessedBlock:
    values = np.asarray(raw_values, dtype=np.float64)
    valid = np.asarray(raw_valid, dtype=bool)
    if len(values) != len(valid):
        raise ValueError("raw values and validity mask must have equal length")
    valid_indexes = np.flatnonzero(valid)
    if len(valid_indexes) < 64:
        raise RuntimeError("Temporal Block has too few valid PPG samples")

    filled = values.copy()
    missing = ~valid
    if missing.any():
        indexes = np.arange(len(filled))
        filled[missing] = np.interp(
            indexes[missing], indexes[valid], filled[valid]
        )
    resampled = signal.resample_poly(filled, up=FS, down=RAW_FS)
    filtered = np.full(len(resampled), np.nan, dtype=np.float64)

    long_gaps = [
        (start, end)
        for start, end in _missing_runs(missing)
        if end - start > MAX_INTERPOLATION_GAP_RAW_SAMPLES
    ]
    output_gaps = []
    for start, end in long_gaps:
        output_start = max(0, int(math.floor(start * FS / RAW_FS)))
        output_end = min(
            len(resampled), int(math.ceil(end * FS / RAW_FS))
        )
        output_gaps.append((output_start, output_end))

    cursor = 0
    for gap_start, gap_end in output_gaps + [(len(resampled), len(resampled))]:
        chunk = _filter_chunk(resampled[cursor:gap_start])
        if chunk is not None:
            filtered[cursor:gap_start] = chunk
        cursor = max(cursor, gap_end)

    invalid_prefix = np.r_[0, np.cumsum(~valid, dtype=np.int64)]
    return ProcessedBlock(
        ppg=filtered.astype(np.float32),
        invalid_prefix_raw=invalid_prefix,
        raw_samples=len(values),
    )


def normalized_windows(
    block: ProcessedBlock,
    start: int,
    end: int,
) -> list[np.ndarray]:
    start = max(0, int(start))
    end = min(len(block.ppg), int(end))
    windows: list[np.ndarray] = []
    for offset in range(start, end - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        window_end = offset + WINDOW_SAMPLES
        if not block.window_is_valid(offset, window_end):
            continue
        window = block.ppg[offset:window_end]
        mean = float(np.mean(window, dtype=np.float64))
        std = float(np.std(window, dtype=np.float64, ddof=0))
        if not np.isfinite(mean) or not np.isfinite(std) or std < NORMALIZATION_EPS:
            continue
        normalized = ((window - mean) / std).astype(np.float32)
        if np.isfinite(normalized).all():
            windows.append(normalized)
    return windows


class CorNETTrainDataset(Dataset):
    def __init__(self, data_root: str | Path, user_ids: Sequence[int]) -> None:
        all_windows: list[np.ndarray] = []
        labels: list[int] = []
        per_user: dict[str, dict] = {}
        for label, user_id in enumerate(user_ids):
            raw_user = RawPPGUser(data_root, user_id)
            block_counts = []
            for block_index in range(len(ACTIVITY_START_MINUTES)):
                raw_values, raw_valid = raw_user.block(block_index)
                block = preprocess_block(raw_values, raw_valid)
                windows = normalized_windows(block, 0, len(block.ppg))
                all_windows.extend(windows)
                labels.extend([label] * len(windows))
                block_counts.append(len(windows))
            per_user[str(user_id)] = {
                "label": label,
                "block_window_counts": block_counts,
                "total_windows": int(sum(block_counts)),
            }
        if not labels:
            raise RuntimeError("No CorNET training windows were generated")
        self.ppg = torch.from_numpy(np.stack(all_windows)).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.metadata = {
            "user_ids": [int(user) for user in user_ids],
            "per_user": per_user,
            "total_windows": len(labels),
        }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ppg[index], self.labels[index]


@dataclass
class OOBUserWindows:
    user_id: int
    enrollment: Dict[int, torch.Tensor]
    probes: Dict[int, torch.Tensor]
    metadata: dict


def build_oob_windows(
    data_root: str | Path, user_ids: Sequence[int]
) -> list[OOBUserWindows]:
    users = []
    for user_id in user_ids:
        raw_user = RawPPGUser(data_root, user_id)
        enrollment_parts: dict[int, list[np.ndarray]] = {
            seconds: [] for seconds in ENROLL_SECONDS
        }
        probes: dict[int, torch.Tensor] = {}
        metadata = {"enrollment": {}, "probe": {}}
        for block_index in range(len(ACTIVITY_START_MINUTES)):
            raw_values, raw_valid = raw_user.block(block_index)
            block = preprocess_block(raw_values, raw_valid)
            enroll_start = ENROLL_START_SECONDS * FS
            for seconds in ENROLL_SECONDS:
                windows = normalized_windows(
                    block,
                    enroll_start,
                    enroll_start + seconds * FS,
                )
                enrollment_parts[seconds].extend(windows)
                metadata["enrollment"].setdefault(str(seconds), []).append(
                    len(windows)
                )
            probe_windows = normalized_windows(
                block,
                PROBE_START_MINUTES * 60 * FS,
                PROBE_END_MINUTES * 60 * FS,
            )
            # Match the established OOB protocol: an explicit exclusion may
            # remove every probe window from one participant × Temporal Block.
            # That block contributes no trials, but the OOB participant remains
            # in the evaluation when other blocks have valid probes.
            if probe_windows:
                probes[block_index] = torch.from_numpy(
                    np.stack(probe_windows)
                ).unsqueeze(1)
            metadata["probe"][str(block_index)] = len(probe_windows)

        enrollment = {}
        for seconds, windows in enrollment_parts.items():
            if not windows:
                raise RuntimeError(
                    f"user {user_id}: no {seconds}-second enrollment windows"
                )
            enrollment[seconds] = torch.from_numpy(np.stack(windows)).unsqueeze(1)
        users.append(
            OOBUserWindows(
                user_id=int(user_id),
                enrollment=enrollment,
                probes=probes,
                metadata=metadata,
            )
        )
        if not probes:
            raise RuntimeError(f"user {user_id}: no valid OOB probe windows")
    return users


class CorNETBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 32, kernel_size=40, stride=1, padding=0, bias=True
        )
        self.batch_norm = nn.BatchNorm1d(32)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.pool(self.activation(self.batch_norm(self.conv(x))))
        )


class CorNETOriginalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn1 = CorNETBlock(1)
        self.cnn2 = CorNETBlock(32)
        self.lstm1 = nn.LSTM(
            input_size=32,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        if x.ndim != 3 or x.shape[1:] != (1, WINDOW_SAMPLES):
            raise ValueError(f"expected [B,1,{WINDOW_SAMPLES}], got {tuple(x.shape)}")
        shapes = {"input": tuple(x.shape)}
        x = self.cnn1.conv(x)
        shapes["conv1"] = tuple(x.shape)
        x = self.cnn1.dropout(
            self.cnn1.pool(
                self.cnn1.activation(self.cnn1.batch_norm(x))
            )
        )
        shapes["pool1"] = tuple(x.shape)
        x = self.cnn2.conv(x)
        shapes["conv2"] = tuple(x.shape)
        x = self.cnn2.dropout(
            self.cnn2.pool(
                self.cnn2.activation(self.cnn2.batch_norm(x))
            )
        )
        shapes["pool2"] = tuple(x.shape)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        shapes["lstm1"] = tuple(x.shape)
        x, (hidden, _) = self.lstm2(x)
        shapes["lstm2"] = tuple(x.shape)
        embedding = hidden[-1]
        shapes["embedding"] = tuple(embedding.shape)
        return embedding, shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, WINDOW_SAMPLES):
            raise ValueError(f"expected [B,1,{WINDOW_SAMPLES}], got {tuple(x.shape)}")
        x = self.cnn2(self.cnn1(x)).transpose(1, 2)
        x, _ = self.lstm1(x)
        _, (hidden, _) = self.lstm2(x)
        return hidden[-1]


class AAMSoftmax(nn.Module):
    def __init__(
        self,
        in_features: int = 128,
        n_classes: int = 12,
        margin: float = AAM_MARGIN,
        scale: float = AAM_SCALE,
    ) -> None:
        super().__init__()
        self.margin = float(margin)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.square())
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        phi = torch.where(
            cosine > math.cos(math.pi - self.margin),
            phi,
            cosine - math.sin(self.margin) * self.margin,
        )
        one_hot = torch.zeros_like(cosine).scatter_(
            1, labels.view(-1, 1), 1.0
        )
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return F.cross_entropy(logits * self.scale, labels)


def get_embeddings(
    model: nn.Module,
    windows: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            embedding = model(windows[start : start + batch_size].to(device))
            outputs.append(F.normalize(embedding, p=2, dim=1).cpu().numpy())
    return np.concatenate(outputs)


def _score_top_m(
    probes: np.ndarray, enrollment: np.ndarray, top_m: int
) -> np.ndarray:
    similarities = probes @ enrollment.T
    effective_m = min(int(top_m), similarities.shape[1])
    selected = np.partition(similarities, -effective_m, axis=1)[
        :, -effective_m:
    ]
    return selected.mean(axis=1).astype(np.float64)


def _summarize(
    score_lists: Mapping[int, Mapping[str, Sequence[float]]]
) -> dict:
    activities = {}
    overall_genuine: list[float] = []
    overall_impostor: list[float] = []
    for block_index, activity_name in enumerate(ACTIVITY_NAMES):
        genuine = list(score_lists[block_index]["genuine"])
        impostor = list(score_lists[block_index]["impostor"])
        eer, threshold = compute_eer(genuine, impostor)
        activities[activity_name] = {
            "eer": eer,
            "threshold": threshold,
            "genuine_trials": len(genuine),
            "impostor_trials": len(impostor),
        }
        overall_genuine.extend(genuine)
        overall_impostor.extend(impostor)
    eer, threshold = compute_eer(overall_genuine, overall_impostor)
    return {
        "overall": {
            "eer": eer,
            "threshold": threshold,
            "genuine_trials": len(overall_genuine),
            "impostor_trials": len(overall_impostor),
        },
        "activity": activities,
    }


def evaluate_oob_users(
    model: nn.Module,
    users: Sequence[OOBUserWindows],
    device: torch.device,
) -> tuple[float, dict, dict]:
    embedded_enrollment: dict[int, dict[int, np.ndarray]] = {}
    embedded_probes: dict[int, dict[int, np.ndarray]] = {}
    window_metadata = {}
    for user in users:
        embedded_enrollment[user.user_id] = {
            seconds: get_embeddings(model, windows, device)
            for seconds, windows in user.enrollment.items()
        }
        embedded_probes[user.user_id] = {
            block_index: get_embeddings(model, windows, device)
            for block_index, windows in user.probes.items()
        }
        window_metadata[str(user.user_id)] = user.metadata

    cells = {}
    macro_eers = []
    for seconds in ENROLL_SECONDS:
        length_cells = {}
        for top_m in TOP_M_VALUES:
            score_lists = {
                block_index: {"genuine": [], "impostor": []}
                for block_index in range(len(ACTIVITY_NAMES))
            }
            for target in users:
                enrollment = embedded_enrollment[target.user_id][seconds]
                for owner in users:
                    key = "genuine" if owner.user_id == target.user_id else "impostor"
                    for block_index, probes in embedded_probes[owner.user_id].items():
                        scores = _score_top_m(probes, enrollment, top_m)
                        score_lists[block_index][key].extend(scores.tolist())
            summary = _summarize(score_lists)
            length_cells[str(top_m)] = summary
            macro_eers.append(summary["overall"]["eer"])
        cells[str(seconds)] = length_cells
    return float(np.mean(macro_eers)), cells, window_metadata


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))
