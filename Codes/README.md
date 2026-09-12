# Codes

This directory contains code for PPG biometric verification on users unseen during training using WildPPG. The framework encodes PPG and ACC separately and modulates their embeddings with a Gate conditioned on signal quality and motion information. Device temperature contributes an auxiliary residual through a compact MLP.

**File descriptions**

| Category | File | Purpose |
|---|---|---|
| Models and protocol | `models.py` | Defines the ECAPA-TDNN encoders, PPG–ACC fusion, context-conditioned Gate, temperature residual MLP, and AAM-Softmax. |
| Models and protocol | `protocol.py` | Implements data loading, preprocessing, subject-disjoint folds, enrollment/probe construction, claimed-target normalization, Top-M scoring, and equal error rate (EER) calculation. |
| Models and protocol | `fixed240540_protocol.py` | Activates the final experiment settings, including the 240–540-minute interval, 10/20/30-second enrollment durations, and evaluation at epoch 50 (E50). |
| Training | `train.py` | Implements Stage 1 identity classification training with AAM-Softmax and saves backbone checkpoints. |
| Training | `train_fixed240540.py` | Runs Stage 1 training under the final protocol for models such as PPG, PPG+ACC, and PPG+ACC+Temp MLP. |
| Gate | `train_residual_verification_gate.py` | Implements Stage 2 calibration, which freezes the trained backbone and optimizes only the Gate. Includes ranking loss, identity regularization, Gate evaluation, and fold aggregation. |
| Gate | `train_fixed240540_ppg_acc_gate.py` | Adds a Gate to the PPG+ACC backbone, calibrates the Gate, and evaluates the resulting model. |
| Gate | `train_fixed240540_temp_gate.py` | Adds a Gate to the PPG+ACC+Temp MLP backbone, calibrates the Gate, and evaluates the proposed model. |
| Gate | `train_fixed240540_gate_context_control.py` | Runs context-free and shuffled-context controls to examine how alignment between input windows and their context affects Gate performance. |
| Evaluation | `evaluate_oob.py` | Evaluates backbone verification performance using enrollment and probe data from participants excluded from training (OOB users), and aggregates results across folds. |
| Evaluation | `evaluate_fixed240540_oob.py` | Runs OOB evaluation of E50 backbone checkpoints under the final protocol. |
| Evaluation | `oob_input_cache.py` | Builds, caches, and reuses the inputs used for PulseID evaluation. |
| Comparison models | `cornet_original_oob.py` | Implements the single-channel green-PPG CorNET comparison model, including its architecture, preprocessing, data construction, and OOB scoring. |
| Comparison models | `run_cornet_original_oob.py` | Runs CorNET training, evaluation, architecture checks, fold aggregation, and aggregation across three seeds. |
| Comparison models | `pulseid_oob.py` | Implements the single-channel green-PPG PulseID comparison model, including its multi-scale CNN, curriculum training, evaluation, and aggregation. |
| Aggregation | `macro12_reporting.py` | Provides shared reporting functions used by PulseID. Aggregates EER over 12 conditions (3 enrollment durations × 4 Top-M settings), folds, and seeds. |
| Aggregation | `summarize_fixed240540.py` | Combines model-wise cross-fold results for one seed. The current aggregation includes the Temp-ECAPA model in addition to the five models in the paper's ablation study. |
| Aggregation | `summarize_fixed240540_three_seed_details.py` | Aggregates overall EER, results by enrollment duration, Top-M setting and temporal block, Gate calibration trajectories, and performance differences across three seeds. |
| Aggregation | `summarize_gate_context_controls_three_seed.py` | Compares real-context, context-free, and shuffled-context Gate results across three seeds, and verifies matched initialization and frozen backbone parameters. |




The CorNET and PulseID comparison models use **green-PPG only**, with model-specific preprocessing. CorNET uses 8-second windows resampled to 125 Hz and a 0.1–18 Hz band-pass filter. PulseID uses 10-second windows at 128 Hz without an additional signal filter. Both comparison models normalize each window to zero mean and unit variance.

