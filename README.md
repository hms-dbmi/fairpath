# Contrastive Learning Enhances Fairness in Pathology Artificial Intelligence Systems

*Shih-Yen Lin<sup>†</sup>, Pei-Chen Tsai<sup>†</sup>, Fang-Yi Su<sup>†</sup>, Chun-Yen Chen, Fuchen Li, Junhan Zhao, Yuk Yeung Ho, Tsung-Lu Michael Lee, Elizabeth Healey, Po-Jen Lin, Ting-Wan Kao, Dmytro Vremenko, Thomas Roetzer-Pejrimovsky, Lynette Sholl, Deborah Dillon, Nancy U. Lin, David Meredith, Keith L. Ligon, Ying-Chun Lo, Nipon Chaisuriya, David J. Cook, Adelheid Woehrer, Jeffrey Meyerhardt, Shuji Ogino, MacLean P. Nasrallah, Jeffrey A. Golden, Sabina Signoretti, Jung-Hsien Chiang<sup>‡</sup>, Kun-Hsing Yu<sup>‡</sup>*

<sup>†</sup> Equal contribution <sup>‡</sup> Corresponding authors

![Python](https://img.shields.io/badge/python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-red?logo=pytorch)
![Statistics](https://img.shields.io/badge/statistics-green)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)

> FAIR-Path (Fairness-aware Artificial Intelligence Review for Pathology) couples contrastive learning with weak supervision to mitigate demographic biases in AI pathology workflows across 20 cancer types.

![FAIR-Path overview](https://i.ibb.co/gL70h1Tp/2025-10-07-201353.jpg)

---

## Table of Contents
1. [Overview](#overview)
2. [Highlights](#highlights)
3. [Data & Resources](#data--resources)
4. [Requirements](#requirements)
5. [Whole Slide Image Tiling](#whole-slide-image-tiling)
6. [Usage](#usage)
7. [Tutorial Example](#tutorial-example)
8. [Citation](#citation)
9. [License](#license)

---

## Overview
AI-enhanced pathology systems improve cancer diagnostics but often underperform for under-represented populations due to limited training diversity. FAIR-Path provides an end-to-end framework that learns demographic-aware visual representations via contrastive objectives, fine-tunes for downstream classification, and quantifies bias mitigation. In a pan-cancer evaluation spanning race, sex, and age cohorts, FAIR-Path alleviated 88.5% of observed disparities and achieved 91.1% gap reduction on 15 independent validation cohorts.

## Highlights
- **Bias diagnosis at scale**: Benchmarks 20 cancer types and multiple demographic attributes to quantify performance gaps.
- **Contrastive pre-training**: Learns patch-level embeddings that preserve clinically relevant yet demographically balanced cues.
- **Two-stage pipeline**: Representation learning followed by fairness-aware fine-tuning for classification tasks.
- **External validation**: Demonstrates robustness across international cohorts while maintaining diagnostic accuracy.
- **Public artifacts**: Reproducible code, data schemas, and dataset layouts available via shared drives (pretrained checkpoints are not distributed).

## Data & Resources
- **FAIR-Path master folder**: https://drive.google.com/drive/u/1/folders/12og_0dCEj6ZJTvQ3oqhFJSRJ-GcbjpuE
- **`datasetpath` (PKL metadata)**: https://drive.google.com/drive/u/1/folders/1VGlj06b9UQgdVxcakhoTvYhy0zitQF1g  
  ![dataset schema](https://i.imgur.com/hMXp7HQ.png)
- **`patchesdirectory` (patch tiles)**: https://drive.google.com/drive/u/1/folders/1yFTQ8Vc9VqFXRy-oCDrBgafeYDoSQ-3m  
  ![patch layout](https://i.imgur.com/Qe9DGsU.png)
- **`patchesinformation` (tile metadata)**: https://drive.google.com/drive/u/1/folders/1XQF2zcTr5zwMvWza9w1C0rC_V_FEY1NY  
  ![patch info](https://i.imgur.com/SW13jlE.png)

Each folder follows the schema consumed by `representation_learning.py` and `mainFinetuneClassificationTask.py`; see `datasets.py` for field-level expectations.

## Requirements
- Docker image: `nvcr.io/nvidia/pytorch:22.03-py3`
- Python libs: `albumentations==1.3.0`, `mlxtend==0.20.0`, `numpy==1.22.4`, `opencv-python-headless==4.5.5.64`, `pandas==1.4.4`, `scikit-learn==1.1.2`, `wandb==0.13.2`, `warmup_scheduler`
- GPU with ≥12 GB memory recommended for pre-training
- WANDB account (optional) for experiment tracking

Install dependencies inside the provided Docker image or replicate the environment with your preferred package manager.

## Whole Slide Image Tiling
- Magnification: 20×
- Patch size: `512 × 512`
- Tiles are stored per sensitive attribute/class combination to ensure balanced sampling.
- Refer to `tile_extraction/README.md` for detailed tiling instructions and scripts.

## Usage

### Stage 1 — Representation Learning
Learn patch-level embeddings using contrastive objectives before downstream fine-tuning.

Key flags:
- `--datasetpath`: List of PKL files, one per sensitive attribute × class pairing.
- `--patchesdirectory`: Folder(s) with corresponding image tiles.
- `--patchesinformation`: PKL files containing tile-level metadata.
- `--model_save_directory`: Output directory for checkpoints.
- `--wandb`, `--wandb_projectname`: Optional experiment logging.
- `--pickType`, `--multiply`, `--specificInnerloop`: Sampling hyperparameters.

```bash
python representation_learning.py \
    --datasetpath 'sensitive_attribute0 Class0.pkl' \
                  'sensitive_attribute0 Class1.pkl' \
                  'sensitive_attribute1 Class0.pkl' \
                  'sensitive_attribute1 Class1.pkl' \
    --patchesdirectory 'path/to/attr0_class0_tiles' \
                       'path/to/attr0_class1_tiles' \
                       'path/to/attr1_class0_tiles' \
                       'path/to/attr1_class1_tiles' \
    --patchesinformation 'path/to/img_information_class0.pkl' \
                         'path/to/img_information_class1.pkl' \
    --model_save_directory 'path/to/checkpoints' \
    --epoch 200 --batch_size 12 --step 480 \
    --wandb --wandb_projectname project_name \
    --pickType k-step --multiply 24 --specificInnerloop 2 \
    --learning_rate 5e-3
```

### Stage 2 — Fairness-Aware Fine-Tuning
Leverage pretrained features for slide-level classification while preserving fairness constraints.

- `--folder`: Location where Stage 1 features are stored.
- `--pretraineddirectory`: Directory containing Stage 1 weights (if different from `--folder`).

```bash
python mainFinetuneClassificationTask.py --folder path/to/features
```

## Tutorial Example
Gender-conditioned tumor detection on cohort `33_CHOL` (frozen section):

```bash
# Stage 1
python representation_learning.py \
    --datasetpath 'female 33_CHOL 40 Frozen tumor0.pkl' \
                  'female 33_CHOL 40 Frozen tumor1.pkl' \
                  'male 33_CHOL 40 Frozen tumor0.pkl' \
                  'male 33_CHOL 40 Frozen tumor1.pkl' \
    --patchesdirectory '33_CHOL' '33_CHOL' '33_CHOL' '33_CHOL' \
    --patchesinformation 'img_information_20x.pkl' 'img_information_20x.pkl' \
    --model_save_directory 'path/to/checkpoints' \
    --epoch 200 --batch_size 12 --step 480 \
    --wandb --wandb_projectname project_name \
    --pickType k-step --multiply 24 --specificInnerloop 2 \
    --learning_rate 5e-3

# Stage 2
python mainFinetuneClassificationTask.py --folder path/to/features
```

## Citation
If you find FAIR-Path useful in your research, please cite:

```
@article{lin2025fairpath,
  title={Contrastive Learning Enhances Fairness in Pathology Artificial Intelligence Systems},
  author={Lin, Shih-Yen and Tsai, Pei-Chen and Su, Fang-Yi and et al.},
  journal={In preparation},
  year={2025}
}
```

## License
This project is licensed under the [AGPL-3.0](LICENSE). By using the code or models, you agree to comply with the license terms.
