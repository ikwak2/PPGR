# Code guide

Training, evaluation, and reporting utilities for PPGR. This guide focuses on code entry points, experiment dependencies, context controls, comparison models, and result files.

See the [main README](../README.md) for the method, results, installation, evaluation protocol, and complete framework-ablation commands. See [about_data.md](../about_data.md) for WildPPG resources, CSV preparation, recording alignment, and participant-ID requirements.

## 1. Execution conventions

The examples below use Bash and run from `PPGR/Codes/` in a clean experiment shell without unrelated `FINAL_*` overrides. Install the dependencies described in the [main Quick start](../README.md#quick-start), including **`torchaudio`**: `models.py` imports its filtering operation. The PulseID reporting path also imports `fcntl`, so its file-locking implementation requires a POSIX environment; it is not directly portable to native Windows.

```bash
# From the repository root:
cd Codes

# Replace with the directory containing the prepared participant CSVs.
DATA_ROOT="/absolute/path/to/prepared_csv"
DEVICE="cuda"

# Example: the offset-0 framework run.
export FINAL_SEED_OFFSET=0
RESULT_ROOT="$PWD/results/FINAL_260802_FULL_240_540"
unset FINAL_FOLD_CONFIG_PATH FINAL_PROTOCOL_ID
```

Always pass `--data-root` and `--result-root` explicitly. Some data-path defaults refer to internal storage, and the training/evaluation and aggregation scripts do not all have the same result-root default. Setting a shell variable named `RESULT_ROOT` does not configure a Python script unless it is passed as an argument.

Use the fixed-protocol training and evaluation wrappers listed below. Their call to `fixed240540_protocol.activate()` checks the fixed settings and rejects conflicting environment values. This does **not** mean that every file containing `fixed240540` in its name calls `activate()`; the reporting scripts have separate configuration paths.

`FINAL_SEED_OFFSET` changes random seeds and protocol metadata, not the output directory. Keep data, code, fold configuration, and seed offset consistent within a run. Use a new result directory after changing them; existing evaluations can be reused by the scripts.

In filenames, **OOB** denotes subject-disjoint held-out users, not a bootstrap resampling procedure. **E50** is the Stage 1 epoch-50 checkpoint; **C00** is the untrained identity gate; **C10** is the primary checkpoint after ten gate-calibration epochs.

## 2. Main framework: entry points and dependencies

### Entry points

| Task | Entry point | Relevant arguments |
|---|---|---|
| Stage 1 training | [train_fixed240540.py](train_fixed240540.py) | `--variant`, `--folds 1 2 3 4`, `--data-root`, `--result-root`, `--device`; optional `--resume` |
| Stage 1 evaluation | [evaluate_fixed240540_oob.py](evaluate_fixed240540_oob.py) | `--variant`, **`--fold`**, `--data-root`, `--result-root`, `--device`; optional `--skip-aggregate` |
| PPG+ACC+gate | [train_fixed240540_ppg_acc_gate.py](train_fixed240540_ppg_acc_gate.py) | `--fold`, `--data-root`, `--result-root`, `--device`; also supports `--aggregate-only` |
| PPG+ACC+Temp_MLP+gate | [train_fixed240540_temp_gate.py](train_fixed240540_temp_gate.py) | Same gate-runner arguments |
| One-seed ablation summary | [summarize_fixed240540.py](summarize_fixed240540.py) | `--result-root` |
| Three-seed ablation reports | [summarize_fixed240540_three_seed_details.py](summarize_fixed240540_three_seed_details.py) | Fixed input/output configuration in the script; no result-root CLI option |

**`--folds` and `--fold` are different options.** The Stage 1 trainer accepts multiple folds; evaluation and gate calibration take one fold per invocation. Aggregation commands do not take `--device`.

### Source-to-gate mapping

| Gated model | Stage 1 source (`--variant`) | Gate output directory under the result root |
|---|---|---|
| PPG+ACC+gate | `ppg_acc` | `ppg_acc_residual_verification_gate/` |
| PPG+ACC+Temp_MLP+gate | `ppg_acc_temp_mlp` | `ppg_acc_temp_mlp_residual_verification_gate/` |

For either source, follow this dependency order:

```text
Stage 1 source training (E50)
  -> source OOB evaluation
  -> identity-gate initialization (C00)
  -> gate-only calibration through C10
  -> OOB evaluation of C00, C01, C02, C05, and C10
  -> cross-fold aggregation
```

Stage 2 keeps the encoders, fusion projector, and any temperature residual module frozen. Training `--variant ppg_acc_gate_temp_mlp` directly in Stage 1 is a different, jointly trained configuration, not this two-stage procedure.

**C00 checks have two different timings.** Before calibration, the runner checks source/identity-gate output equality on synthetic tensors. After calibration and OOB evaluation, it compares C00 EER with the source evaluation **when the source OOB JSON exists**, allowing an absolute EER-fraction difference of `1e-4` (0.01 percentage points). The completed-result reuse branch also reads that source JSON. Evaluate the source first, as shown in the main README; do not describe the OOB EER check as a prerequisite executed before calibration.

`--resume` is available for Stage 1 training. The gate wrappers do not expose an optimizer-resume option: completed evaluations may be reused, but rerunning a partially completed gate experiment is not an instruction to resume from the latest calibration epoch.

### Aggregation requirements

The [complete ablation commands](../README.md#full-ablation-and-three-seed-aggregation) already cover all folds and seeds. The existing summary pipeline requires **six** model directories: `ppg`, `ppg_acc`, `ppg_acc_temp_mlp`, both gate directories above, and **`baseline`**, which is PPG+ACC+Temp_ECAPA. The five-model main results table does not change this six-model code dependency. PPG+ACC+Temp_ECAPA is unnecessary for training the proposed model alone.

The one-seed summarizer reads each model's `oob_cross_fold_summary.csv` and requires twelve primary-condition rows. Run this summarizer only after all six models have complete four-fold evaluations.

Gate cross-fold files can be regenerated from completed fold evaluations:

```bash
python train_fixed240540_ppg_acc_gate.py \
  --aggregate-only --result-root "$RESULT_ROOT"
python train_fixed240540_temp_gate.py \
  --aggregate-only --result-root "$RESULT_ROOT"

# Requires all six models, each with four completed folds.
python summarize_fixed240540.py --result-root "$RESULT_ROOT"
```

The three-seed reporter uses the fixed run directories documented in the [main README](../README.md#run-all-folds-and-seeds). In particular, the historical `FINAL_260802_FULL_240_540_SEED1` directory is the **offset-10000** run. Custom directory names require updating the reporter's `RUNS` configuration.

## 3. Context-control experiments

[train_fixed240540_gate_context_control.py](train_fixed240540_gate_context_control.py) reuses the gate-calibration runner while changing the context supplied to the gate. Selection is through **environment variables**, not `--source` or `--control` arguments.

| Environment variable | Accepted values and meaning |
|---|---|
| `GATE_CONTEXT_CONTROL_SOURCE` | `ppg_acc`: PPG+ACC source; `temp_mlp`: **PPG+ACC+Temp_MLP** source |
| `GATE_CONTEXT_CONTROL_MODE` | `context_free`, `context_shuffled`, or `real` |
| `GATE_CONTEXT_CONTROL_INITIALIZATION` | `seeded` (default), or `real_c00` to load the corresponding real-context C00 state |

**Context-free** supplies a single four-dimensional mean computed from training-fold contexts to every window. **Shuffled context** jointly permutes four-feature rows within each context-extraction batch, containing at most 512 rows. It does not shuffle each feature independently or perform a global cross-participant shuffle. The normal gate uses the aligned context from each window.

### Run matched controls

First finish the source and real-context gate runs for the same fold, seed offset, and result root. The example below explicitly selects `real_c00`, which requires the saved real-context `fold<F>_C00.pt`. It matches the untrained gate state; it does not initialize from the trained C10 gate.

With the variables in Section 1 set for one seed, run both controls for both sources:

```bash
(
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT first}"
: "${RESULT_ROOT:?Set RESULT_ROOT for the selected seed}"
: "${FINAL_SEED_OFFSET:?Set and export FINAL_SEED_OFFSET first}"
DEVICE="${DEVICE:-cuda}"
export FINAL_SEED_OFFSET
unset GATE_CONTEXT_VALIDATE_INITIALIZATION_ONLY

for SOURCE in ppg_acc temp_mlp; do
  for MODE in context_free context_shuffled; do
    for FOLD in 1 2 3 4; do
      GATE_CONTEXT_CONTROL_SOURCE="$SOURCE" \
      GATE_CONTEXT_CONTROL_MODE="$MODE" \
      GATE_CONTEXT_CONTROL_INITIALIZATION=real_c00 \
      python train_fixed240540_gate_context_control.py \
        --fold "$FOLD" --data-root "$DATA_ROOT" \
        --result-root "$RESULT_ROOT" --device "$DEVICE"
    done
  done
done
)
```

Repeat with the corresponding main-run result root and `FINAL_SEED_OFFSET` for offsets `5000` and `10000`. Control directories append `_context_free` or `_context_shuffled` to the source's gate directory. They contain the normal gate checkpoints/evaluations plus `fold<F>_context_control.json` metadata. The `real` mode uses the unsuffixed gate directory, so it should not be treated as a separate output location.


### Three-seed context-control reporting

[summarize_gate_context_controls_three_seed.py](summarize_gate_context_controls_three_seed.py)
is an optional, experiment-specific post-processing utility. It requires
the corresponding context-control outputs and associated run metadata.

Context-control experiments and this report are not required to reproduce
the main framework ablations. Their fold-level and cross-fold outputs can
be inspected independently of this reporting utility.

The standard reproducibility path is therefore:

```text
source Stage 1 / real-context gate
  -> matched context-free and shuffled-context runs
  -> fold and cross-fold outputs
  -> optional three-seed post-processing
```

For ordinary inspection or reproduction of the main framework, the fold-level and cross-fold control outputs are sufficient; no additional audit metadata needs to be supplied manually by the user.

## 4. Standalone PPG comparison models

The CorNET-style and PulseID-style runners below are separate from the earlier encoder-swap variants in `models.py`. Both use **single-channel wrist green PPG**, but they do not use the main framework's complete preprocessing/windowing pipeline.

| Implemented setting | CorNET-style comparison | PulseID-style comparison |
|---|---|---|
| Entry point | [run_cornet_original_oob.py](run_cornet_original_oob.py) | [pulseid_oob.py](pulseid_oob.py) |
| Model/preprocessing implementation | [cornet_original_oob.py](cornet_original_oob.py) | [pulseid_oob.py](pulseid_oob.py) |
| Window | 8 s; resampled to 125 Hz (1,000 samples) | 10 s at 128 Hz (1,280 samples) |
| Train/enrollment/probe stride | 2 s / 2 s / 2 s | 10 s / 10 s / 10 s |
| PPG preprocessing | 0.1–18 Hz Butterworth filtering and per-window z-normalization | No additional band-pass filter; per-window z-normalization |
| Training objective | Training-only 12-class AAM-Softmax | Cross-entropy for epochs 1–10, then cross-entropy plus triplet loss |
| Optimizer / batch size | RMSprop (`1e-3`) / 25 | Adam (`1e-3`) / 64 |

The default subject folds, temporal enrollment/probe regions, Top-M scoring conditions, and seed offsets are shared. **Window lengths, strides, normalization, and therefore trial construction are not identical.** These are subject-disjoint verification adaptations, not exact reproductions of the source papers' complete evaluation pipelines. In particular, `cornet_ppg_acc` is not an alias for this single-PPG CorNET comparison.

### Validation commands

These commands check implemented tensor shapes and a synthetic forward/backward pass. They do not validate prepared data, reproduce EER, or benchmark full-run memory requirements.

```bash
python run_cornet_original_oob.py validate --device cpu
python pulseid_oob.py validate --device cpu
```

### Training, evaluation, and aggregation

Use separate comparison roots. Each runner takes one fold per `train`/`evaluate` invocation, supports `train --resume`, and exposes separate `aggregate` and `summarize` subcommands. Pass `--device` to training/evaluation, not to aggregation.

The following block runs the three offsets and four folds serially using the `DATA_ROOT` and `DEVICE` configured above:

```bash
(
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT first}"
DEVICE="${DEVICE:-cuda}"
unset FINAL_FOLD_CONFIG_PATH FINAL_PROTOCOL_ID
CORNET_BASE="$PWD/results/FINAL_260904_CORNET_ORIGINAL_BACKBONE_OOB"
PULSEID_BASE="$PWD/results/FINAL_260906_PULSEID_GREEN_OOB"

for OFFSET in 0 5000 10000; do
  export FINAL_SEED_OFFSET="$OFFSET"
  CORNET_ROOT="$CORNET_BASE/offset_$OFFSET"
  PULSEID_ROOT="$PULSEID_BASE/offset_$OFFSET"

  for FOLD in 1 2 3 4; do
    python run_cornet_original_oob.py train \
      --fold "$FOLD" --data-root "$DATA_ROOT" \
      --result-root "$CORNET_ROOT" --device "$DEVICE" --resume
    python run_cornet_original_oob.py evaluate \
      --fold "$FOLD" --data-root "$DATA_ROOT" \
      --result-root "$CORNET_ROOT" --device "$DEVICE"
    python pulseid_oob.py train \
      --fold "$FOLD" --data-root "$DATA_ROOT" \
      --result-root "$PULSEID_ROOT" --device "$DEVICE" --resume
    python pulseid_oob.py evaluate \
      --fold "$FOLD" --data-root "$DATA_ROOT" \
      --result-root "$PULSEID_ROOT" --device "$DEVICE"
  done

  python run_cornet_original_oob.py aggregate --result-root "$CORNET_ROOT"
  python pulseid_oob.py aggregate --result-root "$PULSEID_ROOT"
done

python run_cornet_original_oob.py summarize --base-root "$CORNET_BASE"
python pulseid_oob.py summarize --base-root "$PULSEID_BASE"
)
```

The per-seed result root is the `offset_<N>` directory. The runners append their own subdirectories:

```text
<CORNET_BASE>/offset_<N>/comparison_model_redesign/cornet_original_backbone_oob/
<PULSEID_BASE>/offset_<N>/comparison_model_redesign/pulseid_green_oob/
```

Each comparison's three-seed summary is written under its own `<BASE>/three_seed_summary/`, including `three_seed_summary.json` and `three_seed_cells.csv`. These comparison outputs are not consumed by the main six-model ablation summarizer.

### PulseID input cache and locking

[oob_input_cache.py](oob_input_cache.py) caches deterministic PulseID evaluation inputs, not trained embeddings. The cache location is controlled by `FINAL_OOB_INPUT_CACHE`; its default is `Codes/results/macro12_migration_260908/input_cache`, independently of `--result-root`. An empty value disables this input cache.

[macro12_reporting.py](macro12_reporting.py) provides PulseID fold aggregation, three-seed summaries, and a POSIX per-fold evaluation lock. Avoid starting duplicate jobs against the same experiment outputs; the lock is not a repository-wide training lock.

## 5. Reading result files

For the main framework and its controls, each model/method directory contains `checkpoints/`, `oob_fold<F>.json`, and, after all four folds are available, `oob_cross_fold.json`, `oob_cross_fold_summary.csv`, and `oob_activity_summary.csv`. Stage 1 additionally writes `fold<F>_training.json` and `train_fold<F>.log`; gate runs write `fold<F>.log`.

| Main-framework fold JSON | Primary metric location |
|---|---|
| Stage 1 E50 | `epochs["50"]["oob_macro_eer_across_16_cells"]` |
| Gate C10, including controls | `calibration_epochs["10"]["oob_macro_eer_across_16_cells"]` |

The legacy `16_cells` key contains **twelve** enrollment-by-Top-M conditions under the fixed protocol. The `activity` keys and `oob_activity_summary.csv` refer to **temporal blocks**, not annotated activity classes. Keep these machine-readable keys and historical output labels unchanged when reading existing files; display names such as PPG+ACC+Temp_MLP do not rename them.

### Units and aggregation levels

| Output family | Stored EER convention |
|---|---|
| Main-framework/control OOB JSON and model-level CSV files | Fractions; multiply by 100 to display percent |
| One-seed `six_model_main_summary.csv` / `.json` | Fractions; `eer_fold_std_ddof1` is sample SD **across folds**, not across seeds |
| Main `FINAL_260802_THREE_SEED_SUMMARY/*_by_seed.csv` reports | EER fields ending in `percent` are already percentages; SD and difference fields use percentage points |
| CorNET/PulseID `three_seed_summary.json` and `three_seed_cells.csv` | Fractions, including `macro12_eer_mean` and `macro12_eer_seed_sd_ddof1`; multiply by 100 for percent/percentage-point display |

Do not apply a single blanket percentage conversion to every CSV. The main three-seed reports and comparison summaries use different storage conventions. Three-seed SD describes variability across complete runs on fixed subject folds, not uncertainty across independent datasets.

## 6. Implementation map

The entry-point tables above link the executable runners. The supporting modules are:

| Module | Responsibility |
|---|---|
| [models.py](models.py) | ECAPA-style encoders, fusion projector, context-conditioned gate, compact temperature residual, AAM-Softmax, and model construction |
| [protocol.py](protocol.py) | Shared CSV loader, preprocessing, folds/exclusions, claimed-target cases, Top-M scores, and EER |
| [fixed240540_protocol.py](fixed240540_protocol.py) | Fixed-protocol environment checks and activation |
| [train.py](train.py) | Generic Stage 1 runner used by the fixed-protocol wrapper |
| [evaluate_oob.py](evaluate_oob.py) | Generic source-model evaluation and cross-fold reporting |
| [train_residual_verification_gate.py](train_residual_verification_gate.py) | Shared gate warm start, frozen-feature extraction, calibration, evaluation, and aggregation |
| [macro12_reporting.py](macro12_reporting.py) | Reporting and evaluation-lock helpers used by PulseID |
| [oob_input_cache.py](oob_input_cache.py) | PulseID evaluation-input cache |

Use the entry points associated with the intended experiment. A variant appearing in a model registry is not evidence that it is interchangeable with a standalone comparison runner or the two-stage main model.
