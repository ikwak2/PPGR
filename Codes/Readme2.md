# Codes

Training, evaluation, and aggregation scripts for context-conditioned PPG–ACC biometric verification on
users unseen during training (WildPPG). The framework encodes PPG and ACC with separate ECAPA-TDNN encoders,
modulates the embeddings with a Gate conditioned on window-level signal-quality and motion descriptors, and adds
device temperature as a bounded residual from a compact MLP.

**Terminology used in file names and outputs**

| Term | Meaning |
|---|---|
| **OOB** | "Out-of-bag" users = the four held-out participants of a fold (never used for training or model selection). |
| **E50** | Stage 1 backbone checkpoint at epoch 50 (the only reported epoch). |
| **C10** | Stage 2 gate checkpoint after 10 calibration epochs (the reported gate). C00 = zero-initialized gate, identical to the backbone. |
| **Macro-12** | Mean EER over 3 enrollment durations × 4 Top-M values (12 conditions) and 4 folds. |
| **fixed240540** | Final protocol: 240–540 min interval, five 60-min temporal blocks. |

## Requirements

- Python ≥ 3.9, Linux/macOS (`fcntl` is used for result-file locking; Windows is not supported)
- PyTorch, NumPy, pandas, SciPy
- One GPU is sufficient; every script accepts `--device cpu` for smoke tests.

```bash
pip install torch numpy pandas scipy   # or: pip install -r ../requirements.txt
```

## Data preparation

The scripts read one CSV per WildPPG participant, sampled at **128 Hz** on a common timeline:

```
<data-root>/user_1_final.csv ... user_16_final.csv
columns: PPG, temperature, acc_x, acc_y, acc_z
```

- `PPG`: wrist green PPG (raw; band-pass filtering 0.5–8 Hz is applied inside `protocol.py`).
- `temperature`: internal device temperature, linearly interpolated from 0.5 Hz to the 128-Hz timeline.
- `acc_x/y/z`: wrist 3-axis accelerometer.
- Missing samples should be left as `NaN`; gaps shorter than 1 s are interpolated, longer gaps split filtering
  segments and are excluded from windowing.
- Two hard-coded exclusion ranges (`EXCLUSIONS` in `protocol.py`, participants 4 and 6) remove segments with
  [TODO: state reason, e.g. sensor dropout / corrupted data].

<!-- TODO: add and reference a `prepare_wildppg.py` script that converts the official WildPPG .mat files into this CSV format. -->

Pass the directory with `--data-root` (the default path in the scripts points to the authors' internal storage).

## Protocol summary (as implemented in `protocol.py` / `fixed240540_protocol.py`)

| Item | Value |
|---|---|
| Interval / blocks | 240–540 min → five 60-min temporal blocks |
| Folds | 4 subject-disjoint folds; held-out users {1–4}, {5–8}, {9–12}, {13–16} |
| Window | 4 s (512 samples); stride 2 s (training, enrollment), 4 s (probes) |
| Training data | all windows from the full 0–60 min of each block, in-fold users only |
| Enrollment | first 10 / 20 / 30 s of each block, taken after a 2-min stabilization offset (`FINAL_ENROLL_START_SECONDS=120`), pooled over the 5 blocks |
| Probes | minutes 48–58 of each block |
| Scoring | cosine, Top-M averaging with M ∈ {1, 3, 5, 10} |
| Normalization | verification statistics from the claimed identity's enrollment only (`claimed_target`) |
| Stage 1 | 50 epochs, AAM-Softmax (m = 0.2, s = 30), Adam 1e-3, batch 128 |
| Stage 2 | gate only, 10 epochs, pairwise softplus ranking loss + 0.01 identity regularization |
| Seeds | fold seeds 42 / 123 / 456 / 789 (+ `FINAL_SEED_OFFSET`); paper uses offsets 0, 5000, 10000 |

## Reproducing the paper

All `*_fixed240540*.py` entry points call `fixed240540_protocol.activate()` first, which pins the protocol
environment variables and verifies them before `protocol.py` is imported. Always launch these wrappers rather than
`train.py` / `evaluate_oob.py` / `train_residual_verification_gate.py` directly.

Set the result root once (the summary scripts expect this location):

```bash
cd Codes
DATA=/path/to/wildppg_csv
RES=results/FINAL_260802_FULL_240_540
```

### 1. Stage 1 — backbone training (E50)

```bash
for V in ppg ppg_acc ppg_acc_temp_mlp; do
  python train_fixed240540.py --variant $V --folds 1 2 3 4 --data-root $DATA --result-root $RES
done
```

Mapping between paper rows and `--variant`:

| Paper model | Stage 1 `--variant` | Stage 2 script |
|---|---|---|
| PPG only | `ppg` | – |
| PPG + ACC | `ppg_acc` | – |
| PPG + ACC + Temp | `ppg_acc_temp_mlp` | – |
| PPG + ACC + Gate | `ppg_acc` | `train_fixed240540_ppg_acc_gate.py` |
| **Proposed** | `ppg_acc_temp_mlp` | `train_fixed240540_temp_gate.py` |

Other variants listed in `train.py` (`baseline` = temperature encoded by a third ECAPA branch, `ppg_temp`,
`cornet_*`, `ndss_bilstm_attention_*`, `ppg_acc_gate_temp_mlp` = gate trained jointly in Stage 1) are
exploratory and **not reported in the paper**.

### 2. OOB evaluation of the backbones

```bash
for V in ppg ppg_acc ppg_acc_temp_mlp; do
  for F in 1 2 3 4; do
    python evaluate_fixed240540_oob.py --variant $V --fold $F --data-root $DATA --result-root $RES
  done
done
```

### 3. Stage 2 — gate calibration and evaluation (C10)

```bash
for F in 1 2 3 4; do
  python train_fixed240540_temp_gate.py    --fold $F --data-root $DATA --result-root $RES   # proposed
  python train_fixed240540_ppg_acc_gate.py --fold $F --data-root $DATA --result-root $RES   # PPG+ACC+Gate
done
# re-aggregate across folds if a run was interrupted
python train_fixed240540_temp_gate.py --aggregate-only --result-root $RES
```

Stage 2 requires the E50 checkpoint **and** the OOB evaluation JSON of the source variant from steps 1–2; the
script asserts that the zero-initialized gate (C00) reproduces the backbone EER before calibration.

### 4. Context controls (Sec. 5.1)

```bash
python train_fixed240540_gate_context_control.py --help   # context-free / shuffled-context gates
python summarize_gate_context_controls_three_seed.py
```

### 5. Additional seeds

Repeat steps 1–4 with a different seed offset and result root, e.g.

```bash
FINAL_SEED_OFFSET=5000  ... --result-root results/FINAL_260802_FULL_240_540_SEED1_OFFSET5000
FINAL_SEED_OFFSET=10000 ... --result-root results/FINAL_260802_FULL_240_540_SEED1
```

(The result-root names above are the ones hard-coded in `summarize_fixed240540_three_seed_details.py`.)

### 6. Aggregation

```bash
python summarize_fixed240540.py --result-root $RES      # one seed: six_model_main_summary.{csv,json}
python summarize_fixed240540_three_seed_details.py      # 3 seeds: Macro-12, per-enrollment, per-Top-M, per-block, gate trajectory
```

### 7. Comparison models (Table 3)

Both baselines use **green PPG only** with their original preprocessing: CorNET uses 8-s windows resampled to
125 Hz and a 0.1–18 Hz band-pass; PulseID uses 10-s windows at 128 Hz without additional filtering. Each window is
z-normalized. Folds, enrollment/probe intervals, scoring, and seeds are identical to the proposed model. CorNET uses
a training-only 12-class AAM-Softmax head; PulseID uses its multi-scale CNN with curriculum training.

```bash
python run_cornet_original_oob.py train    --fold 1 --data-root $DATA --result-root $RES   # folds 1..4
python run_cornet_original_oob.py evaluate --fold 1 --data-root $DATA --result-root $RES
python run_cornet_original_oob.py aggregate --result-root $RES
python pulseid_oob.py train    --fold 1 --data-root $DATA --result-root $RES
python pulseid_oob.py evaluate --fold 1 --data-root $DATA --result-root $RES
python pulseid_oob.py aggregate --result-root $RES
```

## File descriptions

| Category | File | Purpose |
|---|---|---|
| Models & protocol | `models.py` | ECAPA-TDNN encoders, PPG–ACC fusion, context-conditioned Gate, temperature residual MLP, AAM-Softmax. |
| Models & protocol | `protocol.py` | Data loading, filtering, subject-disjoint folds, enrollment/probe construction, claimed-target normalization, Top-M scoring, EER. |
| Models & protocol | `fixed240540_protocol.py` | Pins the final protocol (240–540 min, 10/20/30-s enrollment, E50) via environment variables; imported by every `*_fixed240540*` wrapper. |
| Training (Stage 1) | `train.py` | Generic Stage 1 training with AAM-Softmax; saves backbone checkpoints. |
| Training (Stage 1) | `train_fixed240540.py` | Stage 1 under the final protocol (`ppg`, `ppg_acc`, `ppg_acc_temp_mlp`, …). |
| Gate (Stage 2) | `train_residual_verification_gate.py` | Generic Stage 2 runner: freezes the backbone, trains only the Gate (ranking loss + identity regularization), evaluates C00–C10, aggregates folds. |
| Gate (Stage 2) | `train_fixed240540_temp_gate.py` | Proposed model: Gate on the `ppg_acc_temp_mlp` backbone. |
| Gate (Stage 2) | `train_fixed240540_ppg_acc_gate.py` | PPG+ACC+Gate ablation: Gate on the `ppg_acc` backbone. |
| Gate (Stage 2) | `train_fixed240540_gate_context_control.py` | Context-free and shuffled-context control gates. |
| Evaluation | `evaluate_oob.py` | Verification evaluation of a backbone on held-out users, with fold aggregation. |
| Evaluation | `evaluate_fixed240540_oob.py` | `evaluate_oob.py` under the final protocol (E50). |
| Evaluation | `oob_input_cache.py` | Caches PulseID evaluation inputs. |
| Comparison models | `cornet_original_oob.py`, `run_cornet_original_oob.py` | CorNET-style single-channel PPG baseline: model, preprocessing, training, evaluation, aggregation. |
| Comparison models | `pulseid_oob.py` | PulseID-style single-channel PPG baseline: multi-scale CNN, curriculum training, evaluation, aggregation. |
| Aggregation | `macro12_reporting.py` | Shared Macro-12 aggregation helpers (used by PulseID). |
| Aggregation | `summarize_fixed240540.py` | Per-seed cross-fold summary of the ablation models (also includes the unreported `baseline` Temp-ECAPA variant). |
| Aggregation | `summarize_fixed240540_three_seed_details.py` | Three-seed aggregation: Macro-12, per-enrollment, per-Top-M, per-block, gate trajectories, paired deltas. |
| Aggregation | `summarize_gate_context_controls_three_seed.py` | Real vs. context-free vs. shuffled-context gates over three seeds; checks matched initialization and frozen backbones. |

## Output layout

```
results/FINAL_260802_FULL_240_540/
├── ppg/, ppg_acc/, ppg_acc_temp_mlp/            # Stage 1: checkpoints/, train_fold*.log, oob_fold*.json, oob_cross_fold.json
├── ppg_acc_residual_verification_gate/          # Stage 2, PPG+ACC+Gate
├── ppg_acc_temp_mlp_residual_verification_gate/ # Stage 2, proposed
├── six_model_main_summary.{csv,json}
└── ...
```
