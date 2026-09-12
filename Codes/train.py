"""Train one or more subject-disjoint folds and save raw epoch checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from models import (
    AAMSoftmax,
    CONTEXT_NORM_CHOICES,
    MODEL_SPECS,
    build_model,
    encoder_config_payload,
    model_result_subdir,
)
from protocol import (
    BATCH_SIZE,
    EPOCHS,
    FOLD_CONFIG,
    FOLD_SEEDS,
    HORIZONS,
    LEARNING_RATE,
    PROTOCOL_ID,
    TrainWindowDataset,
    atomic_json_dump,
    config_payload,
    set_seed,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/Data/CRS25/PPG_Certifiation/data/Final_Data")
DEFAULT_RESULT_ROOT = HERE / "results"
VARIANTS = (
    "baseline",
    "ppg",
    "ppg_acc",
    "cornet_ppg_acc",
    "cornet_ppg_acc_temp_mlp",
    "ndss_bilstm_attention_ppg_acc",
    "ndss_bilstm_attention_ppg_acc_temp_mlp",
    "ndss_bilstm_attention_ppg_acc_gate_temp_mlp",
    "ppg_acc_temp_mlp",
    "ppg_acc_gate_temp_mlp",
    "ppg_temp",
)


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


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def rng_payload(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng(payload: dict, loader_generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])
    loader_generator.set_state(payload["loader_generator"])


def model_spec_payload(variant: str) -> dict:
    spec = MODEL_SPECS[variant]
    return {
        "modalities": list(spec.modalities),
        "context_features": list(spec.context_features),
        "gated": spec.gated,
        "encoder_type": spec.encoder_type,
        "embedding_dim": spec.embedding_dim,
        "fusion_type": spec.fusion_type,
        "training_objective": spec.training_objective,
        "temperature_correction": spec.temperature_correction,
        "result_subdir": spec.result_subdir,
        "encoder_config": encoder_config_payload(spec.encoder_type),
    }


def train_fold(
    variant: str,
    fold_id: int,
    data_root: Path,
    result_root: Path,
    device: torch.device,
    resume: bool,
    context_norm: str,
) -> dict:
    variant_dir = result_root / model_result_subdir(variant)
    checkpoint_dir = variant_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(variant_dir / f"train_fold{fold_id}.log")
    fold = FOLD_CONFIG[fold_id]
    train_users = fold["train"]
    seed = FOLD_SEEDS[fold_id]

    log("")
    log("=" * 88)
    log(
        f"protocol={PROTOCOL_ID} variant={variant} fold={fold_id} "
        f"train={train_users} oob={fold['oob']} device={device}"
    )
    log(
        "Building training tensors from every in-fold user's configured "
        "Temporal Block training regions..."
    )
    train_dataset = TrainWindowDataset(data_root, train_users)
    log(f"train_windows={len(train_dataset)} validation=disabled")

    set_seed(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + 100_000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=loader_generator,
    )
    model = build_model(variant, context_norm=context_norm).to(device)
    embedding_dim = MODEL_SPECS[variant].embedding_dim
    if MODEL_SPECS[variant].training_objective != "aam_softmax":
        raise ValueError(
            f"Unsupported training objective: "
            f"{MODEL_SPECS[variant].training_objective}"
        )
    criterion: torch.nn.Module = AAMSoftmax(
        embedding_dim, len(train_users)
    ).to(device)
    optimizer = Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
    )

    latest_path = checkpoint_dir / f"fold{fold_id}_latest.pt"
    start_epoch = 1
    history = []
    epoch_records: Dict[str, dict] = {}
    if resume and latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        if (
            latest.get("variant") != variant
            or int(latest.get("fold_id", -1)) != fold_id
            or latest.get("protocol", {}).get("protocol_id") != PROTOCOL_ID
        ):
            raise RuntimeError(f"Resume metadata mismatch in {latest_path}")
        if latest.get("context_norm", "layernorm") != context_norm:
            raise RuntimeError(f"Context-normalization mismatch in {latest_path}")
        model.load_state_dict(latest["model_state"])
        criterion.load_state_dict(latest["criterion_state"])
        optimizer.load_state_dict(latest["optimizer_state"])
        start_epoch = int(latest["epoch"]) + 1
        history = list(latest["history"])
        epoch_records = dict(latest["epoch_records"])
        restore_rng(latest["rng"], loader_generator)
        log(f"Resuming after epoch {start_epoch - 1} from {latest_path.name}")

    started = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for (ppg, temp, acc), labels in train_loader:
            ppg = ppg.to(device)
            temp = temp.to(device)
            acc = acc.to(device)
            labels = labels.to(device)
            embeddings = model(ppg, temp, acc)
            loss = criterion(embeddings, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            with torch.no_grad():
                if isinstance(criterion, AAMSoftmax):
                    class_weights = F.normalize(
                        criterion.weight, p=2, dim=1
                    )
                    logits = (
                        F.normalize(embeddings, p=2, dim=1)
                        @ class_weights.T
                        * criterion.scale
                    )
                    predictions = logits.argmax(dim=1)
                    correct += int((predictions == labels).sum())
                    total += len(labels)
                else:
                    normalized = F.normalize(embeddings, p=2, dim=1)
                    similarities = normalized @ normalized.T
                    similarities.fill_diagonal_(-torch.inf)
                    neighbors = similarities.argmax(dim=1)
                    correct += int(
                        (labels[neighbors] == labels).sum().item()
                    )
                    total += len(labels)

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "train_accuracy": correct / max(1, total),
        }
        history.append(record)
        log(
            f"epoch={epoch:02d} loss={record['train_loss']:.6f} "
            f"train_acc={100 * record['train_accuracy']:.3f}%"
        )

        current_state = cpu_clone_state(model.state_dict())
        if epoch in HORIZONS:
            digest = state_digest(current_state)
            epoch_path = checkpoint_dir / f"fold{fold_id}_E{epoch:02d}.pt"
            atomic_torch_save(
                {
                    "format_version": 2,
                    "checkpoint_selection": "raw_epoch_no_validation",
                    "primary_checkpoint": epoch == EPOCHS,
                    "variant": variant,
                    "context_norm": context_norm,
                    "fold_id": fold_id,
                    "epoch": epoch,
                    "model_state_sha256": digest,
                    "model_state": current_state,
                    "model_spec": model_spec_payload(variant),
                    "protocol": config_payload("claimed_target"),
                },
                epoch_path,
            )
            epoch_records[str(epoch)] = {
                "epoch": epoch,
                "primary": epoch == EPOCHS,
                "model_state_sha256": digest,
                "checkpoint": str(epoch_path.resolve()),
            }
            log(f"saved raw E{epoch:02d} sha256={digest[:12]} path={epoch_path.name}")

        atomic_torch_save(
            {
                "format_version": 2,
                "variant": variant,
                "context_norm": context_norm,
                "fold_id": fold_id,
                "epoch": epoch,
                "embedding_dim": embedding_dim,
                "model_state": current_state,
                "criterion_state": cpu_clone_state(criterion.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "epoch_records": epoch_records,
                "rng": rng_payload(loader_generator),
                "protocol": config_payload("claimed_target"),
            },
            latest_path,
        )
        atomic_json_dump(
            {
                "status": "complete" if epoch == EPOCHS else "training",
                "variant": variant,
                "fold_id": fold_id,
                "completed_epoch": epoch,
                "primary_epoch": EPOCHS,
                "history": history,
                "raw_epoch_checkpoints": epoch_records,
                "protocol": config_payload("claimed_target"),
            },
            variant_dir / f"fold{fold_id}_training.json",
        )

    elapsed_hours = (time.time() - started) / 3600.0
    log(f"training complete variant={variant} fold={fold_id} elapsed={elapsed_hours:.3f}h")
    return {
        "variant": variant,
        "fold_id": fold_id,
        "completed_epoch": EPOCHS if start_epoch <= EPOCHS else start_epoch - 1,
        "raw_epoch_checkpoints": epoch_records,
        "elapsed_hours_this_launch": elapsed_hours,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--context-norm",
        choices=CONTEXT_NORM_CHOICES,
        default="layernorm",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    device = torch.device(args.device)
    for fold_id in args.folds:
        train_fold(
            variant=args.variant,
            fold_id=fold_id,
            data_root=args.data_root,
            result_root=args.result_root,
            device=device,
            resume=args.resume,
            context_norm=args.context_norm,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
