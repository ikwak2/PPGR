# Context-Conditioned Modality Gating for In-the-Wild Wearable PPG Biometric Verification on Unseen Users

Official code for the paper
*"Context-Conditioned Modality Gating for In-the-Wild Wearable PPG Biometric Verification on Unseen Users"*
(C. Song, N. Lee, W. Kang, M. Kim, S. Park, B. Oh, Y. Chang, J. Lee†, I.-Y. Kwak†; Chung-Ang University / Hoseo University).


## Overview

Photoplethysmography (PPG) is attractive for continuous wearable authentication, but motion and changing sensing
conditions make verification of users **unseen during training** difficult. We propose a multimodal framework that:

- encodes PPG and 3-axis accelerometer (ACC) signals with separate ECAPA-TDNN branches,
- recalibrates the two identity embeddings channel-wise with a **context-conditioned gate** driven by four
  window-level descriptors (PPG low/high log-power ratio, ACC low-/high-band power, PPG–ACC correlation),
- adds device temperature as a small bounded residual, and
- trains in **two stages**: identity representation learning (AAM-Softmax), then verification-aware gate calibration
  with a pairwise ranking loss while the backbone is frozen.

Under strict 4-fold subject-disjoint evaluation on WildPPG (3 seeds), the proposed model reaches a **Macro-12 EER of
21.23%**, versus 31.94% for PPG-only and 25.48% for fixed PPG–ACC fusion.

| Model | Params (M) | Macro-12 EER (%) |
|---|---|---|
| PPG only | 1.220 | 31.94 ± 0.47 |
| PPG + ACC | 2.441 | 25.48 ± 0.21 |
| PPG + ACC + Temp | 2.448 | 24.72 ± 0.36 |
| PPG + ACC + Gate | 2.455 | 22.46 ± 0.55 |
| **Proposed** (PPG + ACC + Gate + Temp) | 2.462 | **21.23 ± 0.26** |

## Repository structure

```
PPGR/
├── Codes/          # all training / evaluation / aggregation scripts (see Codes/README.md)
├── LICENSE         # MIT
└── README.md
```

## Dataset: WildPPG

We use [WildPPG](https://siplab.org/projects/WildPPG) (Meier, Demirel & Holz, NeurIPS 2024), which provides
long-duration synchronized recordings from 16 participants during unconstrained activities.

- **Signals used:** wrist **green PPG** (128 Hz), wrist **3-axis ACC** (128 Hz), and **internal device temperature**
  (0.5 Hz, linearly interpolated to 128 Hz).
- **Interval:** the common **240–540 min** segment of every participant, divided into five consecutive 60-min
  temporal blocks (not aligned to activities).
- **Split:** 4 subject-disjoint folds (12 participants for training, 4 held out for enrollment/verification).
  Fold assignment: {1–4}, {5–8}, {9–12}, {13–16} as held-out users.

The dataset must be downloaded from the WildPPG authors (it is not redistributed here). The code expects one CSV
per participant — see [Codes/README.md](Codes/README.md#data-preparation) for the exact format.

## Quick start

```bash
pip install -r requirements.txt   # torch, numpy, pandas, scipy
cd Codes
# Stage 1 (backbone), Stage 2 (gate), evaluation and aggregation:
# see Codes/README.md for the full pipeline and commands
```

## Citation

```bibtex
@inproceedings{song2026contextgating,
  title     = {Context-Conditioned Modality Gating for In-the-Wild Wearable PPG Biometric Verification on Unseen Users},
  author    = {Song, Chaerin and Lee, Nahyun and Kang, Wonwoo and Kim, Minsu and Park, Sunghwan and Oh, Byunghoon and Chang, Yeseul and Lee, Jaewoo and Kwak, Il-Youp},
  booktitle = {TODO},
  year      = {TODO}
}
```

Please also cite WildPPG if you use this code:

```bibtex
@inproceedings{meier2024wildppg,
  title     = {WildPPG: A Real-World PPG Dataset of Long Continuous Recordings},
  author    = {Meier, Manuel and Demirel, Berken Utku and Holz, Christian},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024}
}
```

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government
(MSIT) (RS-2026-25477127).

## License

MIT — see [LICENSE](LICENSE).
