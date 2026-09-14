# WildPPG Data Preparation for PPGR

This document describes the WildPPG source dataset and the prepared CSV format expected by the PPGR training and evaluation pipeline.

The original WildPPG data are **not redistributed** with this repository. Users should obtain the dataset from the official WildPPG release and prepare one CSV file per participant before running the experiments.

---

## 1. Original WildPPG dataset

**WildPPG: A Real-World PPG Dataset of Long Continuous Recordings** was introduced by Manuel Meier, Berken Utku Demirel, and Christian Holz at **NeurIPS 2024, Datasets and Benchmarks Track**.

WildPPG contains long, synchronized multimodal recordings from **16 participants** collected under real-world indoor and outdoor conditions. The recordings were acquired during a day-long trip from Zurich to Jungfraujoch and include transportation, walking, hiking, stair climbing, eating, drinking, and resting under changing sensing and environmental conditions.

The original dataset contains wearable recordings from four body locations:

- head,
- sternum,
- wrist, and
- ankle.

Each wearable device records:

- reflective PPG at three wavelengths: **red, green, and infrared**,
- 3-axis accelerometer signals,
- device temperature, and
- barometric altitude.

A synchronized Lead-I ECG recorded at the sternum is also included in the original WildPPG dataset as a cardiac reference.

### Official WildPPG resources

- Project page: https://siplab.org/projects/WildPPG
- Official code: https://github.com/eth-siplab/WildPPG
- Official dataset download: https://polybox.ethz.ch/index.php/s/NWTuyNojU7aya1y
- Hugging Face dataset page: https://huggingface.co/datasets/eth-siplab/WildPPG
- Paper PDF: https://static.siplab.org/papers/neurips2024-wildppg.pdf
- NeurIPS proceedings: https://proceedings.neurips.cc/paper_files/paper/2024/hash/0433292a3eb101df8f4d72a63f5410d4-Abstract-Datasets_and_Benchmarks_Track.html

Please refer to the official WildPPG repository for the original release structure and loading utilities.

---

## 2. Signals used in PPGR

PPGR does **not** use every signal provided by WildPPG.

The biometric verification experiments in this repository use:

- **wrist green PPG**,
- **wrist 3-axis ACC**, and
- **wrist-device temperature**.

The red and infrared PPG channels and barometric altitude are not used.

The local preprocessing files may retain an `ECG` column, but **ECG is not used by the PPGR biometric model or verification protocol**. ECG is not read as a model input and is not used for enrollment, probe scoring, or EER computation.

Therefore, the `PPG` column in the prepared CSV files specifically denotes the **wrist green-PPG signal**.

---

## 3. Prepared CSV format

For the experiments in this repository, the WildPPG recordings were organized offline into one CSV file per participant.
We download dataset in official dataset download `https://polybox.ethz.ch/index.php/s/NWTuyNojU7aya1y` for _mat_ file and converted to _csv_ files for every 16 users.

### Required CSV files

The loader expects the following files:

```text
DATA_ROOT/
├── user_1_final.csv
├── user_2_final.csv
├── ...
└── user_16_final.csv
```

A prepared file may look like:

```text
 Index      PPG  temperature   ECG    acc_x    acc_y     acc_z
     0 0.745066    28.048786  4747 0.085561 0.613561  0.029288
     1 0.745453    28.048821  3393 0.133637 0.690485  0.002363
     2 0.745014    28.048856  1539 0.194561 0.758273  0.005439
     3 0.744970    28.048892  -578 0.314546 0.775788 -0.071363
     4 0.745228    28.048927 -2828 0.370303 0.782137 -0.015015
```

The shared loader requires the following named columns:

```text
PPG,temperature,acc_x,acc_y,acc_z
```

The current training and evaluation code reads only these five signal columns. `Index` and `ECG` may remain in the CSV but are ignored by PPGR.

### Column definitions

| Column | Description | Used by PPGR Verification |
|---|---|---|
| `Index` | Sample index in the locally prepared file | No |
| `PPG` | **Wrist green PPG** | Yes |
| `temperature` | Wrist-device temperature aligned to the common timeline | Yes |
| `ECG` | ECG retained in the local preprocessing output | **No** |
| `acc_x` | Wrist accelerometer x-axis | Yes |
| `acc_y` | Wrist accelerometer y-axis | Yes |
| `acc_z` | Wrist accelerometer z-axis | Yes |

All five required signal columns are used by the shared loader and window-validity check, including when running PPG-only or PPG+ACC model variants. For this reason, the prepared CSVs should retain `temperature` and all three ACC columns even when the selected model does not consume all modalities.

---

## 4. Sampling and signal alignment

The prepared files used in the reported experiments are aligned to a **128-Hz common timeline**.

- PPG: 128 Hz
- ACC: 128 Hz
- Temperature: the original low-rate temperature stream is **linearly interpolated onto the 128-Hz timeline**

Each data row therefore corresponds to one aligned 128-Hz sample.

The interpolated temperature values should not be interpreted as independent 128-Hz temperature measurements. Interpolation is used only to align the low-rate temperature stream with PPG/ACC window boundaries.

PPGR uses 4-s windows:

```text
4 s × 128 Hz = 512 samples
```

The compact temperature module subsequently summarizes the temperature trajectory using a small set of anchor-derived statistics instead of treating all 512 interpolated points as independent high-rate measurements.

Temperature interpolation must be completed **during CSV preparation**. The model loader does not resample temperature. It reads the aligned values from the CSV and applies:

```text
(temperature - 25) / 15
```

---

## 5. Recording time-axis requirements

**Preserve the original aligned recording time axis.**

The PPGR loader determines time from row positions rather than from a timestamp column. With the prepared 128-Hz timeline, zero-based row `i` corresponds to:

```text
time = i / 128 seconds
```

from the recording origin used by the protocol.

The main experimental protocol selects the interval:

```text
240 <= time < 540 minutes
```

using these sample positions.

Accordingly:

- **do not pre-crop** a CSV so that minute 240 becomes row 0,
- **do not delete rows** containing missing values,
- **do not concatenate** separated valid intervals,
- keep missing samples at their original positions, and
- do not apply the model's PPG filtering or normalization in advance.

PPG filtering, claimed-target normalization, invalid-window rejection, and protocol-specific exclusions are handled by `Codes/protocol.py`.

Deleting or shifting rows changes the sample-to-time correspondence and therefore changes the enrollment, probe, training, and exclusion intervals used by the reported protocol.

---

## 6. Participant-ID mapping

The prepared filenames `user_1_final.csv` through `user_16_final.csv` are not arbitrary labels.

**Participant IDs must match the experimental mapping used to create the reported results.**

The identifiers `user_1`–`user_16` determine:

- subject-disjoint fold membership, and
- participant-specific exclusion intervals used by the protocol.

An arbitrary renumbering of the original WildPPG participants will therefore not reproduce the reported split or protocol.

The fold structure used in the default protocol is:

```text
Fold 1 OOB: users 1–4
Fold 2 OOB: users 5–8
Fold 3 OOB: users 9–12
Fold 4 OOB: users 13–16
```

with the remaining 12 participants used for Stage-1 training in each fold.

---

## 7. Data-preparation prerequisite

The training and evaluation commands in the main repository README start from the **prepared CSV files described above**.

The official WildPPG download is therefore not a ready-to-run input to PPGR by itself.

In summary, reproduction from the original release requires:

1. obtaining the original WildPPG recordings,
2. selecting the wrist green PPG, wrist ACC x/y/z, and wrist-device temperature streams,
3. aligning the selected signals to the 128-Hz timeline,
4. linearly interpolating the low-rate temperature stream onto that timeline,
5. preserving the original protocol time origin and missing-sample positions,
6. preserving the participant-to-`user_1`–`user_16` mapping used by the experiment, and
7. saving one `user_<ID>_final.csv` file per participant.

The local preprocessing output may additionally retain `Index` and `ECG`, but neither is required by the model.

The input contract, missing-window rejection, signal preprocessing, fixed exclusions, and subject-disjoint fold configuration are implemented in:

```text
Codes/protocol.py
```

---

## 8. Relation to the original WildPPG benchmark

WildPPG was introduced primarily as a real-world dataset and benchmark for robust **heart-rate estimation** from PPG under challenging sensing conditions.

PPGR uses the same sensing data for a different task:

> **subject-disjoint biometric verification of users unseen during training**

Accordingly:

- ECG is not used as a biometric input,
- ECG-derived heart-rate labels are not used to train the PPGR identity representation,
- participant identity is used as the Stage-1 training target,
- the Stage-2 gate is calibrated for verification using training participants only, and
- final evaluation is performed on enrollment and probe embeddings from held-out participants.

The repository should therefore be understood as a biometric-verification use of WildPPG, not as a reproduction of the heart-rate estimation task from the original WildPPG paper.

---

## 9. Dataset license and citation

The WildPPG dataset is distributed under the license specified by the original authors. Please check the current license information in the official WildPPG repository before redistribution or reuse.

If you use WildPPG, please cite the original paper:

```bibtex
@inproceedings{meier2024wildppg,
  title     = {WildPPG: A Real-World PPG Dataset of Long Continuous Recordings},
  author    = {Meier, Manuel and Demirel, Berken Utku and Holz, Christian},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024}
}
```

---

## 10. Preparation summary

The local preparation used by PPGR can be summarized as:

```text
Original WildPPG participant recording
        |
        +-- wrist green PPG
        +-- wrist ACC x/y/z
        +-- wrist-device temperature
        |
        +-- align signals to the 128-Hz protocol timeline
        |     \-- linearly interpolate low-rate temperature
        |
        +-- preserve participant ID and recording time origin
        +-- preserve missing-sample positions
        |
        +-- optionally retain Index and ECG columns
        |     \-- neither is used by PPGR
        |
        `-- save as user_<ID>_final.csv
```

The current PPGR loader requires:

```text
PPG,temperature,acc_x,acc_y,acc_z
```

and interprets `PPG` as the **wrist green-PPG channel**.
