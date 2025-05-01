# GDP-HMM-CWRU-UH-Cornell

> **Final submission to the AAPM 2025 GDP-HMM Challenge**  
> Built on top of the official baseline: [RiqiangGao/GDP-HMM_AAPMChallenge](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge/tree/main)

---

## Table of Contents

- [Overview](#overview)
- [Key Differences from Baseline](#key-differences-from-baseline)
- [Repository Structure](#repository-structure)
- [Model Configuration](#model-configuration)
- [DataLoader](#dataloader)
---

## Overview

This repository contains our team’s solution for the AAPM 2025 Grand Challenge on Dose Prediction using Hidden Markov Models (GDP-HMM). We extend the official challenge codebase with:

- A large **MedNeXt** backbone (kernel size = 5)  
- **Smooth L1 Loss** (𝛽 = 0.25) instead of plain L1  
- A streamlined `data_loader_efficient.py` for faster, leaner data loading

Everything else (evaluation scripts, inference pipeline) follows the baseline implementation at the linked challenge repo.

---

## Key Differences from Baseline

| Component            | Baseline                            | This Work                                         |
|----------------------|-------------------------------------|---------------------------------------------------|
| **Backbone**         | MedNeXt Base                        | MedNeXt Large (kernel=5)                          |
| **Loss Function**    | L1 loss                             | Smooth L1 loss (β=0.25)                           |
| **DataLoader**       | `data_loader.py`                    | `data_loader_efficient.py` with simplified logic   |

---

## Repository Structure

```plaintext
GDP-HMM-CWRU-UH-Cornell/
├── configs/                             # YAML configs (train & infer)
│   ├── config_infer.yaml                # Inference Configuration
│   └── config_train.yaml                # Training Configuration 
├── data_loader_efficient.py             # Simplified data loader
├── inference_mednext.py                 # Inference code
├── train_lightning_mednext_SmoothL1.py  # Training code
├── train_lightning.py                   # Source code from Challenge GitHub for reference 
├── toolkit.py                           # Auxillary functions
├── CWRU-UH-Cornell_tech_report.pdf      # Technical Report
└── README.md                 # This file
```

---

## Model Configuration

Training and inference code configurations can be found in `configs` folder.

Note that 'use_dist' argument is introduced but is not utilized. It was part of experiments.

- Setting for Inference code (in `configs/infer_val.yaml`)
- Setting for Training code (in `configs/train.yaml`)

---

## DataLoader

Our `data_loader_efficient.py` optimizes the baseline `data_loader.py` by:

- Does not save redundant keys and values into output dictionary 
- Minimizing memory overhead by loading only required channels on-the-fly

## Checkpoint

We will be sharing a link on trained weight of our model soon! 
