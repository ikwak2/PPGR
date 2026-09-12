"""Complete-block subject-disjoint training and claimed-target OOB protocol."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import signal
from torch.utils.data import Dataset

from models import EMBED_DIM, FS


WINDOW_SIZE = FS * 4
TRAIN_STRIDE = WINDOW_SIZE // 2
PROBE_STRIDE = WINDOW_SIZE
MINUTE_SAMPLES = FS * 60

ACTIVITY_START_MINUTES = (240, 300, 360, 420, 480)
ACTIVITY_NAMES = tuple(
    f"Temporal Block {index}" for index in range(1, 6)
)

# Every complete relative 0--60-minute Temporal Block from an in-fold user is
# training data.  OOB users remain entirely unseen during model fitting.
SEED_OFFSET = int(os.environ.get("FINAL_SEED_OFFSET", "0"))
BASE_PROTOCOL_ID = "final_260802_fixed_240_540_full60_claimed_target_v1"
DEFAULT_PROTOCOL_ID = (
    BASE_PROTOCOL_ID
    if SEED_OFFSET == 0
    else f"{BASE_PROTOCOL_ID}_seed_offset_{SEED_OFFSET}"
)
PROTOCOL_ID = os.environ.get("FINAL_PROTOCOL_ID", DEFAULT_PROTOCOL_ID)
STABILITY_DROP_SECONDS = 2 * 60
ENROLL_SECONDS = (10, 20, 30)
configured_enroll_seconds = tuple(
    int(value)
    for value in os.environ.get("FINAL_ENROLL_SECONDS", "10,20,30").split(",")
    if value.strip()
)
if configured_enroll_seconds != ENROLL_SECONDS:
    raise ValueError("FINAL_ENROLL_SECONDS must be exactly 10,20,30")
TOP_M_VALUES = (1, 3, 5, 10)

ENROLL_START_SECONDS = int(
    os.environ.get("FINAL_ENROLL_START_SECONDS", str(STABILITY_DROP_SECONDS))
)

TRAIN_START_SECONDS = 0
PROBE_START_MINUTES = int(
    os.environ.get("FINAL_PROBE_START_MINUTES", "48")
)
PROBE_END_MINUTES = int(
    os.environ.get("FINAL_PROBE_END_MINUTES", "58")
)
TRAIN_END_MINUTES = 60
# TRAIN_END_MINUTES intentionally overlaps the OOB probe clock range.  The
# overlap exists only across different people: in-fold users contribute all
# retained samples to training, while OOB users contribute no training sample.
if not (0 <= PROBE_START_MINUTES < PROBE_END_MINUTES <= 60):
    raise ValueError("Invalid OOB probe interval")
TEMP_OFFSET = 25.0
TEMP_SCALE = 15.0

EPOCHS = 50
HORIZONS = tuple(
    int(value)
    for value in os.environ.get(
        "FINAL_HORIZONS", ",".join(str(value) for value in range(5, EPOCHS + 1, 5))
    ).split(",")
    if value.strip()
)
if not HORIZONS or any(value < 1 or value > EPOCHS for value in HORIZONS):
    raise ValueError("FINAL_HORIZONS must be within 1..EPOCHS")
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

DEFAULT_FOLD_CONFIG: Dict[int, Dict[str, List[int]]] = {
    1: {
        "oob": [1, 2, 3, 4],
        "train": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
    2: {
        "oob": [5, 6, 7, 8],
        "train": [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16],
    },
    3: {
        "oob": [9, 10, 11, 12],
        "train": [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16],
    },
    4: {
        "oob": [13, 14, 15, 16],
        "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
}


def _load_fold_config() -> Tuple[Dict[int, Dict[str, List[int]]], str]:
    configured_path = os.environ.get("FINAL_FOLD_CONFIG_PATH")
    if not configured_path:
        return DEFAULT_FOLD_CONFIG, "protocol.py default"
    path = Path(configured_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "recommended" in payload:
        oob_folds = payload["recommended"]["folds"]
    elif "folds" in payload and isinstance(payload["folds"], list):
        oob_folds = payload["folds"]
    else:
        raise ValueError(
            f"{path} must contain recommended.folds or folds"
        )
    if len(oob_folds) != 4 or any(len(fold) != 4 for fold in oob_folds):
        raise ValueError(f"{path}: expected four OOB folds of four users")
    normalized = [[int(user) for user in fold] for fold in oob_folds]
    flat = [user for fold in normalized for user in fold]
    if sorted(flat) != list(range(1, 17)) or len(set(flat)) != 16:
        raise ValueError(
            f"{path}: folds must contain each user 1..16 exactly once"
        )
    all_users = set(flat)
    fold_config = {
        fold_id: {
            "oob": sorted(oob),
            "train": sorted(all_users - set(oob)),
        }
        for fold_id, oob in enumerate(normalized, 1)
    }
    return fold_config, str(path)


FOLD_CONFIG, FOLD_CONFIG_SOURCE = _load_fold_config()
FOLD_SEEDS = {
    fold_id: seed + SEED_OFFSET
    for fold_id, seed in {1: 42, 2: 123, 3: 456, 4: 789}.items()
}
EXCLUSIONS = {
    4: (3786939, 4194810),
    6: (4337572, 4545543),
}

PPG_SOS = signal.butter(
    4,
    [0.5 / (0.5 * FS), 8.0 / (0.5 * FS)],
    btype="bandpass",
    output="sos",
)


# Missing runs shorter than one second are linearly interpolated only to make
# fixed filtering possible; windows containing an originally missing sample
# are still rejected.  Longer runs (including explicit exclusions) split the
# filter so no synthetic bridge is passed through sosfiltfilt.
MAX_INTERPOLATION_GAP_SAMPLES = FS - 1


@dataclass
class PreprocessStats:
    ppg_mean: float
    ppg_std: float
    acc_mean: np.ndarray
    acc_std: np.ndarray

    def json_dict(self) -> dict:
        return {
            "ppg_mean": float(self.ppg_mean),
            "ppg_std": float(self.ppg_std),
            "acc_mean": self.acc_mean.reshape(-1).tolist(),
            "acc_std": self.acc_std.reshape(-1).tolist(),
        }


class RawUserData:
    def __init__(self, data_root: str | Path, user_id: int):
        self.user_id = int(user_id)
        path = Path(data_root) / f"user_{self.user_id}_final.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.columns = [column.strip() for column in frame.columns]
        self.ppg = frame["PPG"].to_numpy(dtype=np.float64)
        self.temp = frame["temperature"].to_numpy(dtype=np.float64)
        self.acc = frame[["acc_x", "acc_y", "acc_z"]].to_numpy(
            dtype=np.float64
        ).T
        self.n_samples = len(self.ppg)

    def exclusion_mask(self, start: int, end: int) -> np.ndarray:
        mask = np.zeros(max(0, end - start), dtype=bool)
        excluded = EXCLUSIONS.get(self.user_id)
        if excluded is None:
            return mask
        overlap_start = max(start, excluded[0])
        overlap_end = min(end, excluded[1])
        if overlap_start < overlap_end:
            mask[overlap_start - start : overlap_end - start] = True
        return mask


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def filter_ppg_segment(raw_segment: np.ndarray) -> np.ndarray:
    """Filter one role segment without crossing a long missing-data run."""
    values = np.asarray(raw_segment, dtype=np.float64).copy()
    if len(values) == 0:
        return values
    missing = ~np.isfinite(values)
    split_runs: List[Tuple[int, int]] = []
    run_start: int | None = None
    for index, is_missing in enumerate(missing):
        if is_missing and run_start is None:
            run_start = index
        if run_start is not None and (
            not is_missing or index == len(missing) - 1
        ):
            run_end = index if not is_missing else index + 1
            if run_end - run_start > MAX_INTERPOLATION_GAP_SAMPLES:
                split_runs.append((run_start, run_end))
            run_start = None

    filtered = np.full_like(values, np.nan)
    boundaries = [0]
    for start, end in split_runs:
        boundaries.extend((start, end))
    boundaries.append(len(values))
    for start, end in zip(boundaries[0::2], boundaries[1::2]):
        chunk = values[start:end].copy()
        chunk_missing = ~np.isfinite(chunk)
        chunk_valid = ~chunk_missing
        if int(chunk_valid.sum()) < 16:
            continue
        if chunk_missing.any():
            indexes = np.arange(len(chunk))
            chunk[chunk_missing] = np.interp(
                indexes[chunk_missing], indexes[chunk_valid], chunk[chunk_valid]
            )
        try:
            chunk_filtered = signal.sosfiltfilt(
                PPG_SOS, signal.detrend(chunk)
            )
        except (ValueError, FloatingPointError):
            continue
        chunk_filtered[chunk_missing] = np.nan
        filtered[start:end] = chunk_filtered
    return filtered


def extract_raw_segment(
    raw_user: RawUserData, start: int, end: int
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    start = max(0, int(start))
    end = min(raw_user.n_samples, int(end))
    if start >= end:
        return None, None, None
    ppg = raw_user.ppg[start:end].copy()
    temp = raw_user.temp[start:end].copy()
    acc = raw_user.acc[:, start:end].copy()
    excluded = raw_user.exclusion_mask(start, end)
    if excluded.any():
        ppg[excluded] = np.nan
        temp[excluded] = np.nan
        acc[:, excluded] = np.nan
    return filter_ppg_segment(ppg), temp, acc


def compute_stats_from_regions(
    raw_user: RawUserData, regions: Sequence[Tuple[int, int]]
) -> PreprocessStats:
    ppg_parts: List[np.ndarray] = []
    acc_parts: List[np.ndarray] = []
    for start, end in regions:
        ppg, _, acc = extract_raw_segment(raw_user, start, end)
        if ppg is None or acc is None:
            continue
        ppg_parts.append(ppg)
        acc_parts.append(acc)
    if not ppg_parts:
        raise RuntimeError(
            f"User {raw_user.user_id} has no valid normalization region"
        )
    ppg_all = np.concatenate(ppg_parts)
    acc_all = np.concatenate(acc_parts, axis=1)
    with np.errstate(invalid="ignore"):
        ppg_mean = float(np.nanmean(ppg_all))
        ppg_std = float(np.nanstd(ppg_all))
        acc_mean = np.nanmean(acc_all, axis=1, keepdims=True)
        acc_std = np.nanstd(acc_all, axis=1, keepdims=True)
    if not np.isfinite(ppg_mean):
        ppg_mean = 0.0
    if not np.isfinite(ppg_std) or ppg_std < 1e-6:
        ppg_std = 1.0
    acc_mean = np.where(np.isfinite(acc_mean), acc_mean, 0.0)
    acc_std = np.where(
        np.isfinite(acc_std) & (acc_std >= 1e-6), acc_std, 1.0
    )
    return PreprocessStats(
        ppg_mean=ppg_mean,
        ppg_std=ppg_std + 1e-6,
        acc_mean=acc_mean.astype(np.float64),
        acc_std=acc_std.astype(np.float64) + 1e-6,
    )


def preprocess_segment(
    raw_user: RawUserData,
    start: int,
    end: int,
    stats: PreprocessStats,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    ppg, temp, acc = extract_raw_segment(raw_user, start, end)
    if ppg is None or temp is None or acc is None:
        return None, None, None
    return (
        (ppg - stats.ppg_mean) / stats.ppg_std,
        (temp - TEMP_OFFSET) / TEMP_SCALE,
        (acc - stats.acc_mean) / stats.acc_std,
    )


def train_regions() -> List[Tuple[int, int]]:
    start_offset = TRAIN_START_SECONDS * FS
    end_offset = TRAIN_END_MINUTES * MINUTE_SAMPLES
    return [
        (
            activity_start * MINUTE_SAMPLES + start_offset,
            activity_start * MINUTE_SAMPLES + end_offset,
        )
        for activity_start in ACTIVITY_START_MINUTES
    ]


def enrollment_regions(enroll_seconds: int) -> List[Tuple[int, int]]:
    start_offset = ENROLL_START_SECONDS * FS
    duration = int(enroll_seconds) * FS
    return [
        (
            activity_start * MINUTE_SAMPLES + start_offset,
            activity_start * MINUTE_SAMPLES + start_offset + duration,
        )
        for activity_start in ACTIVITY_START_MINUTES
    ]


def probe_region(activity_index: int) -> Tuple[int, int]:
    activity_start = ACTIVITY_START_MINUTES[activity_index]
    return (
        (activity_start + PROBE_START_MINUTES) * MINUTE_SAMPLES,
        (activity_start + PROBE_END_MINUTES) * MINUTE_SAMPLES,
    )


def window_is_valid(
    ppg: np.ndarray,
    temp: np.ndarray,
    acc: np.ndarray,
    start: int,
    end: int,
) -> bool:
    return bool(
        np.isfinite(ppg[start:end]).all()
        and np.isfinite(temp[start:end]).all()
        and np.isfinite(acc[:, start:end]).all()
    )


def extract_windows(
    ppg: np.ndarray,
    temp: np.ndarray,
    acc: np.ndarray,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if len(ppg) < WINDOW_SIZE:
        return None
    ppg_windows: List[np.ndarray] = []
    temp_windows: List[np.ndarray] = []
    acc_windows: List[np.ndarray] = []
    for start in range(0, len(ppg) - WINDOW_SIZE + 1, int(stride)):
        end = start + WINDOW_SIZE
        if not window_is_valid(ppg, temp, acc, start, end):
            continue
        ppg_windows.append(ppg[start:end].astype(np.float32))
        temp_windows.append(temp[start:end].astype(np.float32))
        acc_windows.append(acc[:, start:end].astype(np.float32))
    if not ppg_windows:
        return None
    return (
        torch.from_numpy(np.stack(ppg_windows)).unsqueeze(1),
        torch.from_numpy(np.stack(temp_windows)).unsqueeze(1),
        torch.from_numpy(np.stack(acc_windows)),
    )


class TrainWindowDataset(Dataset):
    def __init__(self, data_root: str | Path, user_ids: Sequence[int]):
        ppg_windows: List[np.ndarray] = []
        temp_windows: List[np.ndarray] = []
        acc_windows: List[np.ndarray] = []
        labels: List[int] = []
        regions = train_regions()
        for label, user_id in enumerate(user_ids):
            raw_user = RawUserData(data_root, user_id)
            stats = compute_stats_from_regions(raw_user, regions)
            for start, end in regions:
                ppg, temp, acc = preprocess_segment(
                    raw_user, start, end, stats
                )
                if ppg is None or temp is None or acc is None:
                    continue
                windows = extract_windows(
                    ppg, temp, acc, stride=TRAIN_STRIDE
                )
                if windows is None:
                    continue
                ppg_tensor, temp_tensor, acc_tensor = windows
                count = len(ppg_tensor)
                ppg_windows.extend(ppg_tensor[:, 0, :].numpy())
                temp_windows.extend(temp_tensor[:, 0, :].numpy())
                acc_windows.extend(acc_tensor.numpy())
                labels.extend([label] * count)
        if not labels:
            raise RuntimeError("No FINAL training windows were generated")
        self.ppg = torch.from_numpy(np.stack(ppg_windows)).unsqueeze(1)
        self.temp = torch.from_numpy(np.stack(temp_windows)).unsqueeze(1)
        self.acc = torch.from_numpy(np.stack(acc_windows))
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.ppg[index],
            self.temp[index],
            self.acc[index],
        ), self.labels[index]


@dataclass
class VerificationUser:
    user_id: int
    enrollment: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    probes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    stats: dict


@dataclass
class ClaimedTargetCase:
    target_id: int
    enrollment: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    probes_by_owner: Dict[
        int,
        Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]
    stats: dict


def build_verification_users(
    data_root: str | Path,
    user_ids: Sequence[int],
    enroll_seconds: int,
) -> List[VerificationUser]:
    regions = enrollment_regions(enroll_seconds)
    users: List[VerificationUser] = []
    for user_id in user_ids:
        raw_user = RawUserData(data_root, user_id)
        stats = compute_stats_from_regions(raw_user, regions)
        enroll_parts: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        for start, end in regions:
            ppg, temp, acc = preprocess_segment(
                raw_user, start, end, stats
            )
            if ppg is None or temp is None or acc is None:
                continue
            windows = extract_windows(
                ppg, temp, acc, stride=TRAIN_STRIDE
            )
            if windows is not None:
                enroll_parts.append(windows)
        if not enroll_parts:
            continue
        enrollment = tuple(
            torch.cat([part[index] for part in enroll_parts], dim=0)
            for index in range(3)
        )
        probes: Dict[
            int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        for activity_index in range(len(ACTIVITY_NAMES)):
            start, end = probe_region(activity_index)
            ppg, temp, acc = preprocess_segment(
                raw_user, start, end, stats
            )
            if ppg is None or temp is None or acc is None:
                continue
            windows = extract_windows(
                ppg, temp, acc, stride=PROBE_STRIDE
            )
            if windows is not None:
                probes[activity_index] = windows
        if probes:
            users.append(
                VerificationUser(
                    user_id=user_id,
                    enrollment=enrollment,  # type: ignore[arg-type]
                    probes=probes,
                    stats=stats.json_dict(),
                )
            )
    if len(users) < 2:
        raise RuntimeError(
            f"Only {len(users)} verification users available for "
            f"{enroll_seconds}s enrollment"
        )
    return users


def _build_enrollment_tensors(
    raw_user: RawUserData,
    stats: PreprocessStats,
    enroll_seconds: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parts = []
    for start, end in enrollment_regions(enroll_seconds):
        ppg, temp, acc = preprocess_segment(
            raw_user, start, end, stats
        )
        if ppg is None or temp is None or acc is None:
            continue
        windows = extract_windows(
            ppg, temp, acc, stride=TRAIN_STRIDE
        )
        if windows is not None:
            parts.append(windows)
    if not parts:
        raise RuntimeError(
            f"User {raw_user.user_id}: no {enroll_seconds}s enrollment"
        )
    return tuple(
        torch.cat([part[index] for part in parts], dim=0)
        for index in range(3)
    )  # type: ignore[return-value]


def _build_probe_tensors(
    raw_user: RawUserData,
    stats: PreprocessStats,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    probes = {}
    for activity_index in range(len(ACTIVITY_NAMES)):
        start, end = probe_region(activity_index)
        ppg, temp, acc = preprocess_segment(
            raw_user, start, end, stats
        )
        if ppg is None or temp is None or acc is None:
            continue
        windows = extract_windows(
            ppg, temp, acc, stride=PROBE_STRIDE
        )
        if windows is not None:
            probes[activity_index] = windows
    if not probes:
        raise RuntimeError(
            f"User {raw_user.user_id}: no valid probe activities"
        )
    return probes


def build_claimed_target_cases(
    data_root: str | Path,
    user_ids: Sequence[int],
    enroll_seconds: int,
) -> List[ClaimedTargetCase]:
    """Build 1:1 verification cases without true-owner normalization.

    For every claimed target B, B's enrollment statistics are applied to B's
    enrollment and to every genuine/impostor probe presented to B.
    """
    raw_users = {
        user_id: RawUserData(data_root, user_id)
        for user_id in user_ids
    }
    cases = []
    regions = enrollment_regions(enroll_seconds)
    for target_id in user_ids:
        stats = compute_stats_from_regions(
            raw_users[target_id], regions
        )
        cases.append(
            ClaimedTargetCase(
                target_id=target_id,
                enrollment=_build_enrollment_tensors(
                    raw_users[target_id], stats, enroll_seconds
                ),
                probes_by_owner={
                    owner_id: _build_probe_tensors(
                        raw_users[owner_id], stats
                    )
                    for owner_id in user_ids
                },
                stats=stats.json_dict(),
            )
        )
    if len(cases) < 2:
        raise RuntimeError(
            f"Only {len(cases)} claimed targets available for "
            f"{enroll_seconds}s enrollment"
        )
    return cases


def get_embeddings(
    model: torch.nn.Module,
    tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    ppg, temp, acc = tensors
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ppg), batch_size):
            embedding = model(
                ppg[start : start + batch_size].to(device),
                temp[start : start + batch_size].to(device),
                acc[start : start + batch_size].to(device),
            )
            outputs.append(
                F.normalize(embedding, p=2, dim=1).cpu().numpy()
            )
    return np.concatenate(outputs)


@dataclass
class EmbeddedUser:
    user_id: int
    enrollment: np.ndarray
    probes: Dict[int, np.ndarray]


def embed_verification_users(
    model: torch.nn.Module,
    users: Sequence[VerificationUser],
    device: torch.device,
) -> List[EmbeddedUser]:
    return [
        EmbeddedUser(
            user_id=user.user_id,
            enrollment=get_embeddings(
                model, user.enrollment, device=device
            ),
            probes={
                activity: get_embeddings(model, tensors, device=device)
                for activity, tensors in user.probes.items()
            },
        )
        for user in users
    ]


def compute_eer(
    genuine_scores: Sequence[float] | np.ndarray,
    impostor_scores: Sequence[float] | np.ndarray,
) -> Tuple[float, float]:
    """Empirical ROC EER with linear interpolation at FAR=FRR."""
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("EER requires non-empty genuine and impostor scores")
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate(
        [
            np.ones(len(genuine), dtype=np.int8),
            np.zeros(len(impostor), dtype=np.int8),
        ]
    )
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    cumulative_tp = np.cumsum(labels)
    cumulative_fp = np.cumsum(1 - labels)
    group_ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    tpr = cumulative_tp[group_ends] / len(genuine)
    fpr = cumulative_fp[group_ends] / len(impostor)
    fnr = 1.0 - tpr
    thresholds = scores[group_ends]
    # Add the threshold above the maximum score.
    fpr = np.r_[0.0, fpr]
    fnr = np.r_[1.0, fnr]
    thresholds = np.r_[np.nextafter(scores[0], np.inf), thresholds]
    difference = fpr - fnr
    exact = np.flatnonzero(difference == 0)
    if len(exact):
        index = int(exact[0])
        return float(fpr[index]), float(thresholds[index])
    crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    if len(crossings):
        left = int(crossings[0])
        right = left + 1
        weight = -difference[left] / (
            difference[right] - difference[left]
        )
        eer = fpr[left] + weight * (fpr[right] - fpr[left])
        threshold = thresholds[left] + weight * (
            thresholds[right] - thresholds[left]
        )
        return float(eer), float(threshold)
    index = int(np.argmin(np.abs(difference)))
    return float((fpr[index] + fnr[index]) / 2.0), float(
        thresholds[index]
    )


def _score_top_m(
    probes: np.ndarray, enrollment: np.ndarray, top_m: int
) -> np.ndarray:
    similarities = probes @ enrollment.T
    effective_m = min(int(top_m), similarities.shape[1])
    selected = np.partition(
        similarities, -effective_m, axis=1
    )[:, -effective_m:]
    return selected.mean(axis=1).astype(np.float64)


def _summarize_score_lists(
    activity_scores: Mapping[int, Mapping[str, Sequence[float]]],
) -> dict:
    activity_output = {}
    overall_genuine: List[float] = []
    overall_impostor: List[float] = []
    for activity_index, activity_name in enumerate(ACTIVITY_NAMES):
        genuine = list(activity_scores[activity_index]["genuine"])
        impostor = list(activity_scores[activity_index]["impostor"])
        eer, threshold = compute_eer(genuine, impostor)
        activity_output[activity_name] = {
            "eer": eer,
            "threshold": threshold,
            "genuine_trials": len(genuine),
            "impostor_trials": len(impostor),
        }
        overall_genuine.extend(genuine)
        overall_impostor.extend(impostor)
    overall_eer, overall_threshold = compute_eer(
        overall_genuine, overall_impostor
    )
    return {
        "overall": {
            "eer": overall_eer,
            "threshold": overall_threshold,
            "genuine_trials": len(overall_genuine),
            "impostor_trials": len(overall_impostor),
        },
        "activity": activity_output,
    }


def score_claimed_target_cases(
    model: torch.nn.Module,
    cases: Sequence[ClaimedTargetCase],
    device: torch.device,
    top_m_values: Sequence[int] = TOP_M_VALUES,
) -> Dict[str, dict]:
    """Embed each claim once and accumulate every requested Top-M score."""
    score_lists = {
        top_m: {
            activity: {"genuine": [], "impostor": []}
            for activity in range(len(ACTIVITY_NAMES))
        }
        for top_m in top_m_values
    }
    for case in cases:
        # All tensors in a case use the same claimed-target normalization.
        # Embed them together so the model processes large batches instead of
        # paying Python and GPU-launch overhead for 61 tiny calls per target.
        tensor_groups = [case.enrollment]
        probe_keys = []
        for owner_id, probes_by_activity in case.probes_by_owner.items():
            for activity, tensors in probes_by_activity.items():
                tensor_groups.append(tensors)
                probe_keys.append((owner_id, activity))
        group_lengths = [len(group[0]) for group in tensor_groups]
        merged_tensors = tuple(
            torch.cat(
                [group[modality_index] for group in tensor_groups],
                dim=0,
            )
            for modality_index in range(3)
        )
        merged_embeddings = get_embeddings(
            model, merged_tensors, device=device
        )
        split_embeddings = np.split(
            merged_embeddings,
            np.cumsum(group_lengths)[:-1],
        )
        enrollment = split_embeddings[0]
        for (owner_id, activity), probes in zip(
            probe_keys, split_embeddings[1:]
        ):
            key = (
                "genuine"
                if owner_id == case.target_id
                else "impostor"
            )
            for top_m in top_m_values:
                scores = _score_top_m(
                    probes, enrollment, top_m=top_m
                )
                score_lists[top_m][activity][key].extend(
                    scores.tolist()
                )
    return {
        str(top_m): _summarize_score_lists(score_lists[top_m])
        for top_m in top_m_values
    }


def score_embedded_users(
    users: Sequence[EmbeddedUser], top_m: int
) -> dict:
    by_target: Dict[int, Dict[int, Dict[str, List[float]]]] = {
        target.user_id: {
            activity: {"genuine": [], "impostor": []}
            for activity in range(len(ACTIVITY_NAMES))
        }
        for target in users
    }
    for probe_owner in users:
        for activity, probes in probe_owner.probes.items():
            for target in users:
                scores = _score_top_m(
                    probes, target.enrollment, top_m=top_m
                )
                key = (
                    "genuine"
                    if target.user_id == probe_owner.user_id
                    else "impostor"
                )
                by_target[target.user_id][activity][key].extend(
                    scores.tolist()
                )

    overall_genuine: List[float] = []
    overall_impostor: List[float] = []
    activity_output: Dict[str, dict] = {}
    for activity_index, activity_name in enumerate(ACTIVITY_NAMES):
        genuine: List[float] = []
        impostor: List[float] = []
        for target_id in by_target:
            genuine.extend(
                by_target[target_id][activity_index]["genuine"]
            )
            impostor.extend(
                by_target[target_id][activity_index]["impostor"]
            )
        eer, threshold = compute_eer(genuine, impostor)
        activity_output[activity_name] = {
            "eer": eer,
            "threshold": threshold,
            "genuine_trials": len(genuine),
            "impostor_trials": len(impostor),
        }
        overall_genuine.extend(genuine)
        overall_impostor.extend(impostor)

    overall_eer, overall_threshold = compute_eer(
        overall_genuine, overall_impostor
    )
    return {
        "overall": {
            "eer": overall_eer,
            "threshold": overall_threshold,
            "genuine_trials": len(overall_genuine),
            "impostor_trials": len(overall_impostor),
        },
        "activity": activity_output,
    }


def evaluate_grid(
    model: torch.nn.Module,
    users_by_enrollment: Mapping[int, Sequence[VerificationUser]],
    device: torch.device,
) -> Tuple[float, dict]:
    cells: Dict[str, Dict[str, dict]] = {}
    eers: List[float] = []
    for enroll_seconds in ENROLL_SECONDS:
        embedded = embed_verification_users(
            model,
            users_by_enrollment[enroll_seconds],
            device=device,
        )
        length_cells: Dict[str, dict] = {}
        for top_m in TOP_M_VALUES:
            summary = score_embedded_users(embedded, top_m=top_m)
            length_cells[str(top_m)] = summary
            eers.append(summary["overall"]["eer"])
        cells[str(enroll_seconds)] = length_cells
    return float(np.mean(eers)), cells


def evaluate_claimed_target_grid(
    model: torch.nn.Module,
    cases_by_enrollment: Mapping[
        int, Sequence[ClaimedTargetCase]
    ],
    device: torch.device,
) -> Tuple[float, dict]:
    cells = {}
    eers = []
    for enroll_seconds in ENROLL_SECONDS:
        length_cells = score_claimed_target_cases(
            model,
            cases_by_enrollment[enroll_seconds],
            device=device,
        )
        cells[str(enroll_seconds)] = length_cells
        for top_m in TOP_M_VALUES:
            eers.append(
                length_cells[str(top_m)]["overall"]["eer"]
            )
    return float(np.mean(eers)), cells


def config_payload(
    verification_normalization: str = "true_owner",
) -> dict:
    if verification_normalization not in {
        "true_owner",
        "claimed_target",
    }:
        raise ValueError(
            "verification_normalization must be true_owner or "
            "claimed_target"
        )
    verification_description = (
        "per-user, per-enrollment-length, true source user's five "
        "enrollment segments"
        if verification_normalization == "true_owner"
        else "per-claimed-target, per-enrollment-length; claimed "
        "target's five enrollment segments are applied to enrollment "
        "and every probe presented to that target"
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "fs": FS,
        "window_size": WINDOW_SIZE,
        "train_stride": TRAIN_STRIDE,
        "probe_stride": PROBE_STRIDE,
        "activity_starts_min": list(ACTIVITY_START_MINUTES),
        "activity_names": list(ACTIVITY_NAMES),
        "stability_drop_seconds": STABILITY_DROP_SECONDS,
        "enrollment_start_seconds": ENROLL_START_SECONDS,
        "enrollment_seconds": list(ENROLL_SECONDS),
        "top_m_values": list(TOP_M_VALUES),
        "train_offset": {
            "start_seconds": TRAIN_START_SECONDS,
            "end_minutes": TRAIN_END_MINUTES,
        },
        "train_oob_separation": (
            "subject-disjoint; every retained training sample from relative "
            f"{TRAIN_START_SECONDS / 60:g}-{TRAIN_END_MINUTES:g} minutes is used "
            "only for in-fold users, and no OOB-user sample is used to train"
        ),
        "probe_offset_minutes": [
            PROBE_START_MINUTES,
            PROBE_END_MINUTES,
        ],
        "epochs": EPOCHS,
        "horizons": list(HORIZONS),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "fold_config": FOLD_CONFIG,
        "fold_config_source": FOLD_CONFIG_SOURCE,
        "fold_seeds": FOLD_SEEDS,
        "normalization": {
            "train": (
                "per-user retained training regions only: relative "
                f"{TRAIN_START_SECONDS / 60:g}-{TRAIN_END_MINUTES:g} minutes"
            ),
            "verification_mode": verification_normalization,
            "verification": verification_description,
            "probe_in_stats": False,
        },
        "ppg_filter": (
            "4th-order Butterworth 0.5-8Hz SOS; independent per activity/"
            "OOB role segment; missing runs >=1 second split filtering"
        ),
        "checkpoint_selection": (
            f"none; raw model state saved at prespecified horizons {list(HORIZONS)}, "
            "epoch 50 primary"
        ),
        "validation_used_for_selection": False,
        "oob_reporting": (
            f"prespecified raw horizons {list(HORIZONS)}; no OOB-best epoch selection"
        ),
    }
