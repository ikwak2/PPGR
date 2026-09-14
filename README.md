# Context-Conditioned Modality Gating for In-the-Wild Wearable PPG Biometric Verification on Unseen Users

Official code for the paper *"Context-Conditioned Modality Gating for In-the-Wild Wearable PPG Biometric Verification on Unseen Users"*.

C. Song, N. Lee, W. Kang, M. Kim, S. Park, B. Oh, Y. Chang, J. Lee†, I.-Y. Kwak†  
Chung-Ang University / Hoseo University  
† Corresponding authors.

## Overview

Motion and changing sensing conditions make wearable photoplethysmography (PPG) biometric verification difficult, particularly for users **unseen during training**. Our framework combines separate PPG and three-axis accelerometer (ACC) ECAPA-TDNN encoders, a **context-conditioned gate**, and a compact device-temperature residual.

The gate rescales the PPG and ACC identity embeddings channel-wise before learned late fusion. Its four window-level descriptors are the PPG low-to-high log-power ratio (0.5–2.5 vs. 2.5–8 Hz), ACC low-band log power (<2 Hz), ACC high-band log power (2–5 Hz), and PPG–ACC Pearson correlation. Temperature is **not** a gate input: a separate compact MLP produces a bounded residual that is added after PPG–ACC fusion.

Training has two stages. Stage 1 learns identity representations with AAM-Softmax. Stage 2 freezes all Stage 1 components, including the encoders, fusion projector, and temperature MLP, and trains only the gate using a pairwise ranking loss and identity regularization. The same gate is jointly calibrated for the 10/20/30-second enrollment conditions.

Implementation: [models.py](Codes/models.py) and [Stage 2 training](Codes/train_residual_verification_gate.py).

## Main results

Under strict four-fold subject-disjoint evaluation on WildPPG, the proposed model achieves **21.23% Macro-12 EER**, compared with 31.94% for PPG-only and 25.48% for ungated PPG–ACC late fusion. Lower EER is better.

| Model | Params (M) | Macro-12 EER (%) |
|:---|---:|---:|
| PPG | 1.220 | 31.94 ± 0.47 |
| PPG+ACC | 2.441 | 25.48 ± 0.21 |
| PPG+ACC+Temp_MLP | 2.448 | 24.72 ± 0.36 |
| PPG+ACC+gate | 2.455 | 22.46 ± 0.55 |
| **Proposed (PPG+ACC+Temp_MLP+gate)** | **2.462** | **21.23 ± 0.26** |

`PPG+ACC+Temp_MLP` denotes PPG–ACC late fusion with a compact temperature residual MLP. Parameter counts exclude the training-only AAM-Softmax classification head.

**Macro-12 EER** is the equally weighted mean over three enrollment durations (10/20/30 seconds **per temporal block**) and four Top-M settings (`M = 1/3/5/10`), followed by equal averaging over the four subject-disjoint folds. Each condition's EER uses pooled trials from the five temporal blocks; the main metric does not average five block-specific EERs. Values above are **mean ± sample standard deviation across three complete four-fold runs**, using seed offsets `0`, `5000`, and `10000`. This SD is not the SD across folds or individual trials.

See [protocol.py](Codes/protocol.py) and [three-seed aggregation](Codes/summarize_fixed240540_three_seed_details.py) for the scoring and reporting implementation.

## Repository structure

```text
PPGR/
├── Codes/          # Models, protocol, training, evaluation, and aggregation
├── LICENSE         # MIT license for the code
└── README.md
```

The main entry points are:

| File in `Codes/` | Purpose |
|:---|:---|
| `models.py` | ECAPA-TDNN encoders, context-conditioned gate, temperature residual MLP, and AAM-Softmax |
| `protocol.py` | CSV loading, preprocessing, subject-disjoint folds, enrollment/probe construction, and scoring |
| `fixed240540_protocol.py` | Activates the final 240–540-minute protocol |
| `train_fixed240540.py` | Stage 1 training; uses `--folds` |
| `evaluate_fixed240540_oob.py` | Stage 1 evaluation; uses `--fold` |
| `train_fixed240540_ppg_acc_gate.py` | Stage 2 calibration and evaluation of PPG+ACC+gate |
| `train_fixed240540_temp_gate.py` | Stage 2 calibration and evaluation of PPG+ACC+Temp_MLP+gate |
| `summarize_fixed240540.py` | Combines six model results for one seed |
| `summarize_fixed240540_three_seed_details.py` | Combines the three complete seed runs |

Additional scripts are listed in [Codes/README.md](Codes/README.md). The data format and execution commands required for the main framework are provided below.

## Dataset and data preparation

We use [WildPPG](https://siplab.org/projects/WildPPG) (Meier, Demirel, and Holz, NeurIPS 2024), with recordings from 16 participants. Download the data from the dataset authors; it is not redistributed in this repository. The [official WildPPG repository](https://github.com/eth-siplab/WildPPG) documents the original data and loading utilities.

### Signals

| CSV column | Signal | Required representation |
|:---|:---|:---|
| `PPG` | Single-channel wrist green PPG | 128 Hz |
| `acc_x`, `acc_y`, `acc_z` | Wrist-device three-axis ACC | Public synchronized 128-Hz series |
| `temperature` | Internal device temperature | Original 0.5-Hz values, linearly interpolated onto the same 128-Hz timeline; values in °C |

Red and infrared PPG are not used. Device temperature is not a skin- or body-temperature measurement. Interpolation aligns the temperature stream with the other inputs; it does not create independent 128-Hz temperature measurements.

detailed explanation is writen on `/data`

## Evaluation protocol

The framework uses the interval **240 ≤ time < 540 minutes**, divided into five consecutive 60-minute **temporal blocks**. These blocks are not ground-truth activity classes.

| Setting | Value |
|:---|:---|
| Subject split | Four folds; 12 training users and 4 held-out users per fold |
| Held-out users | Fold 1: 1–4; Fold 2: 5–8; Fold 3: 9–12; Fold 4: 13–16 |
| Training interval | All retained data in the five blocks, from training users only |
| Window length | 4 seconds (512 samples) |
| Stride | 2 seconds for training/enrollment; 4 seconds for probes |
| Enrollment start | 2 minutes after the start of each temporal block |
| Enrollment duration | 10, 20, or 30 seconds per block; references pooled across the five blocks |
| Nominal pooled enrollment | 50, 100, or 150 seconds per user, before exclusions |
| Probe interval | Minutes 48–58 within each block |
| Verification decision | One 4-second probe window |
| Scoring | Mean of the largest `min(M, N)` cosine similarities to `N` enrollment embeddings |
| Top-M values | 1, 3, 5, 10 |
| Primary checkpoints | Stage 1: epoch 50 (`E50`); Stage 2: calibration epoch 10 (`C10`) |

PPG is filtered with a fourth-order Butterworth band-pass filter at 0.5–8 Hz. At verification time, PPG and ACC normalization statistics are computed **only from the claimed user's enrollment data** and applied to that enrollment and every genuine/impostor probe presented to that user. Statistics from probe data or the true probe owner's enrollment are not used to normalize a claim against another user.

Held-out users are excluded from identity training and gate calibration. The primary checkpoints are prespecified; they are not selected using held-out EER. Stage 1 uses AAM-Softmax (margin 0.2, scale 30), Adam (`1e-3`), batch size 128, and 50 epochs. Stage 2 uses Adam (`1e-3`) for 10 epochs, with equal numbers of updates for the three enrollment durations.

**Scope:** This is offline retrospective multi-context verification within the available recordings. Enrollment pools references from all five blocks, so some references occur after earlier probes. These results do not establish prospective, real-time, or cross-session authentication performance.

See [fixed240540_protocol.py](Codes/fixed240540_protocol.py) and [protocol.py](Codes/protocol.py) for the complete configuration.

## Quick start

The following commands use **Bash on Linux**. They train and evaluate the proposed model for **one fold and one seed configuration** after CSV preparation. This is not a pretrained inference demo or a reproduction of the complete three-seed table.

### 1. Clone and install dependencies

An example installation uses Python 3.11 and matching PyTorch/TorchAudio 2.8.0 packages. These commands are an installation example, not an exported lockfile of the original experiment.

```bash
git clone https://github.com/ikwak2/PPGR.git
cd PPGR

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# NVIDIA GPU: CUDA 12.8 wheels; requires a compatible NVIDIA driver.
python -m pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy pandas scipy

cd Codes
```

For a CPU-only environment, use the following PyTorch/TorchAudio installation command in place of the CUDA command above:

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

`torchaudio` is required for the ACC filters used by the gate. Keep `torch` and `torchaudio` on matching versions and builds. Other installation options are listed in the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).

### 2. Set paths and check the prepared inputs

Run the remaining commands from `PPGR/Codes/`. Replace the data path below with the directory containing the 16 prepared CSV files. Use a fresh shell without custom `FINAL_*` protocol overrides.

```bash
# Replace this value before running.
export DATA_ROOT="/absolute/path/to/prepared_csv"
export RESULT_ROOT="$PWD/results/FINAL_260802_FULL_240_540"
export FINAL_SEED_OFFSET=0
export DEVICE=cuda                  # Use cpu for a CPU-only installation.

# Retain the default subject-disjoint folds and fixed-protocol identifier.
unset FINAL_FOLD_CONFIG_PATH FINAL_PROTOCOL_ID
mkdir -p "$RESULT_ROOT"

python - <<'PY'
import os
from pathlib import Path
import pandas as pd
import torch
import torchaudio
from models import build_model

root = Path(os.environ["DATA_ROOT"])
required = {"PPG", "temperature", "acc_x", "acc_y", "acc_z"}
for user in range(1, 17):
    path = root / f"user_{user}_final.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = {str(c).strip() for c in pd.read_csv(path, nrows=0).columns}
    missing = required - columns
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
if os.environ["DEVICE"].startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; check the installation or set DEVICE=cpu.")
print(f"torch={torch.__version__}, torchaudio={torchaudio.__version__}")
print("Imports, file names, and CSV headers checked.")
PY
```

This check does not validate participant mapping, signal-channel selection, time alignment, or sufficient valid signal coverage.

### 3. Train Stage 1, evaluate its checkpoint, and calibrate the gate

```bash
# Stage 1: PPG+ACC+Temp_MLP, fold 1, through epoch 50.
python train_fixed240540.py \
  --variant ppg_acc_temp_mlp \
  --folds 1 \
  --data-root "$DATA_ROOT" \
  --result-root "$RESULT_ROOT" \
  --device "$DEVICE" \
  --resume

# Evaluate the ungated E50 source checkpoint.
python evaluate_fixed240540_oob.py \
  --variant ppg_acc_temp_mlp \
  --fold 1 \
  --data-root "$DATA_ROOT" \
  --result-root "$RESULT_ROOT" \
  --device "$DEVICE"

# Stage 2: initialize an identity gate, calibrate it, and run OOB evaluation.
python train_fixed240540_temp_gate.py \
  --fold 1 \
  --data-root "$DATA_ROOT" \
  --result-root "$RESULT_ROOT" \
  --device "$DEVICE"
```

The Stage 2 wrapper loads the Stage 1 `ppg_acc_temp_mlp` E50 checkpoint. It evaluates the prespecified calibration checkpoints `C00`, `C01`, `C02`, `C05`, and `C10`; **C10 is the primary result**. Do not replace this two-stage procedure with Stage 1 training of `--variant ppg_acc_gate_temp_mlp`.

Use the same data path, result root, and seed offset for all three steps. `--resume` resumes an existing Stage 1 run; it is not a Stage 2 option. Completed evaluation files may be reused, so do not reuse a populated result directory for changed data, code, or experimental settings.

### 4. Locate the outputs

Paths below are relative to `$RESULT_ROOT`:

| Output | Path |
|:---|:---|
| PPG+ACC+Temp_MLP E50 checkpoint | `ppg_acc_temp_mlp/checkpoints/fold1_E50.pt` |
| PPG+ACC+Temp_MLP fold evaluation | `ppg_acc_temp_mlp/oob_fold1.json` |
| Proposed C10 checkpoint | `ppg_acc_temp_mlp_residual_verification_gate/checkpoints/fold1_C10.pt` |
| Proposed fold evaluation | `ppg_acc_temp_mlp_residual_verification_gate/oob_fold1.json` |

For the proposed model, read the `calibration_epochs["10"]` entry. The legacy JSON field `oob_macro_eer_across_16_cells` stores **12-cell** Macro-12 EER under the fixed protocol; its field name has not been updated. Stored EER values are fractions: multiply by 100 for percentages.

Repeat the three steps for folds 2–4 to obtain one complete four-fold run. Cross-fold `oob_cross_fold.json` and `oob_cross_fold_summary.csv` files are generated after all four fold evaluations for a model are complete. A single-fold run does not produce the reported four-fold, three-seed result.

## Full ablation and three-seed aggregation

The existing one-seed summary script expects **six models**, including the additional `PPG+ACC+Temp_ECAPA` comparison (`--variant baseline`). That model is not included in the five-model main table above. The full commands below include it to satisfy the current aggregation scripts; it is not required to train or evaluate the proposed model alone.

### Model names and code identifiers

README display names do not change command-line variant names or output directories.

| Display name | Stage 1 variant | Stage 2 wrapper, when used |
|:---|:---|:---|
| PPG | `ppg` | — |
| PPG+ACC | `ppg_acc` | — |
| PPG+ACC+Temp_MLP | `ppg_acc_temp_mlp` | — |
| PPG+ACC+gate | `ppg_acc` | `train_fixed240540_ppg_acc_gate.py` |
| PPG+ACC+Temp_MLP+gate | `ppg_acc_temp_mlp` | `train_fixed240540_temp_gate.py` |
| PPG+ACC+Temp_ECAPA | `baseline` | — |

Some generated summaries retain historical display names. Use the code identifiers and result directories to identify models; the consistent display names for this README are those shown above.

### Run all folds and seeds

The three-seed aggregation script currently uses fixed input directories. **Keep the following directory names** unless its `RUNS` configuration is updated accordingly:

| Seed offset | Directory under `Codes/results/` |
|---:|:---|
| 0 | `FINAL_260802_FULL_240_540` |
| 5000 | `FINAL_260802_FULL_240_540_SEED1_OFFSET5000` |
| 10000 | `FINAL_260802_FULL_240_540_SEED1` |

The last directory corresponds to offset **10000**, despite its historical `SEED1` suffix. `FINAL_SEED_OFFSET` changes seeds and protocol metadata; it does **not** automatically select a separate result directory. Base fold seeds are `42`, `123`, `456`, and `789`; the selected offset is added to each.

With the environment and `DATA_ROOT` from Quick start configured, run this block from `PPGR/Codes/`:

```bash
(
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the prepared CSV directory first}"
DEVICE="${DEVICE:-cuda}"
unset FINAL_FOLD_CONFIG_PATH FINAL_PROTOCOL_ID

for OFFSET in 0 5000 10000; do
  case "$OFFSET" in
    0)     RUN_DIR="FINAL_260802_FULL_240_540" ;;
    5000)  RUN_DIR="FINAL_260802_FULL_240_540_SEED1_OFFSET5000" ;;
    10000) RUN_DIR="FINAL_260802_FULL_240_540_SEED1" ;;
  esac
  export FINAL_SEED_OFFSET="$OFFSET"
  RESULT_ROOT="$PWD/results/$RUN_DIR"
  mkdir -p "$RESULT_ROOT"

  # Four ungated models; baseline is the additional PPG+ACC+Temp_ECAPA model.
  for VARIANT in ppg ppg_acc ppg_acc_temp_mlp baseline; do
    python train_fixed240540.py \
      --variant "$VARIANT" --folds 1 2 3 4 \
      --data-root "$DATA_ROOT" --result-root "$RESULT_ROOT" \
      --device "$DEVICE" --resume

    for FOLD in 1 2 3 4; do
      python evaluate_fixed240540_oob.py \
        --variant "$VARIANT" --fold "$FOLD" \
        --data-root "$DATA_ROOT" --result-root "$RESULT_ROOT" \
        --device "$DEVICE"
    done
  done

  # Calibrate both gated models from their corresponding E50 sources.
  for FOLD in 1 2 3 4; do
    python train_fixed240540_ppg_acc_gate.py \
      --fold "$FOLD" --data-root "$DATA_ROOT" \
      --result-root "$RESULT_ROOT" --device "$DEVICE"

    python train_fixed240540_temp_gate.py \
      --fold "$FOLD" --data-root "$DATA_ROOT" \
      --result-root "$RESULT_ROOT" --device "$DEVICE"
  done

  python summarize_fixed240540.py --result-root "$RESULT_ROOT"
done

python summarize_fixed240540_three_seed_details.py
)
```

The scripts write one-seed summaries to each run directory as `six_model_main_summary.csv` and `six_model_main_summary.json`. Three-seed outputs are written to:

```text
Codes/results/FINAL_260802_THREE_SEED_SUMMARY/
```

`macro12_by_seed.csv` contains the run-level Macro-12 values and their three-run mean and sample SD. Additional files report enrollment-duration, Top-M, temporal-block, calibration-trajectory, and paired model-difference summaries. In these **three-seed report CSVs**, columns such as `mean_eer_percent` are already percentages, and SD/difference columns use percentage points. Do not multiply them by 100 again. Model-level OOB files and the one-seed `six_model_main_summary` EER fields use fractions.

These commands cover the framework ablations. They do not execute the separate CorNET/PulseID comparisons or the context-control experiments.

## Reproducibility notes

Exact reproduction requires the original CSV preparation and participant mapping, the fixed protocol, all four folds, and all three seed offsets. An installation example alone does not establish numerical equivalence to the original environment. Record the Git commit and installed package versions with each experiment:

```bash
git rev-parse HEAD
python -m pip freeze
```

The input checks above verify file presence and headers only. The code loads substantial signal tensors into host memory; serial execution does not eliminate per-fold memory requirements. No minimum host-RAM or GPU-memory requirement is claimed here.

The gate modulates embeddings; it does not reconstruct a clean PPG waveform. Its PPG power ratio is a spectral context descriptor, not an ECG-referenced SNR estimate. Observed improvements should be interpreted within the offline subject-disjoint protocol described above.

## Citation

The paper's venue and publication year are not specified here. The following entry cites the **code repository**, without asserting conference publication:

```bibtex
@misc{song_ppgr,
  title        = {{PPGR}: Context-Conditioned Modality Gating for In-the-Wild Wearable PPG Biometric Verification on Unseen Users},
  author       = {Song, Chaerin and Lee, Nahyun and Kang, Wonwoo and Kim, Minsu and Park, Sunghwan and Oh, Byunghoon and Chang, Yeseul and Lee, Jaewoo and Kwak, Il-Youp},
  howpublished = {GitHub repository},
  url          = {https://github.com/ikwak2/PPGR}
}
```

Please also cite WildPPG when using the dataset:

```bibtex
@inproceedings{meier2024wildppg,
  title     = {{WildPPG}: A Real-World PPG Dataset of Long Continuous Recordings},
  author    = {Meier, Manuel and Demirel, Berken Utku and Holz, Christian},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024}
}
```

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25477127).

## License

The code is distributed under the MIT License. See [LICENSE](LICENSE). WildPPG remains subject to its own dataset terms; the repository's code license does not relicense the dataset.
