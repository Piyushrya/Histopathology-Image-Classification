# Histopathology Image Classification using cSiT

A PyTorch-based self-supervised learning framework for histopathology image classification using a **Compact Convolutional Transformer (CCT)** backbone. The model combines three self-supervised learning objectives to learn robust visual representations from unlabeled medical images.

## Key Features

- **Self-Supervised Learning:** Reconstruction, rotation prediction, and contrastive learning.
- **Compact Architecture:** Modified CCT-14/7×2 with approximately 6M parameters.
- **Large-Scale Pretraining:** Approximately 600K unlabeled histopathology images.
- **Benchmark Evaluation:** NCT-CRC and BreakHis datasets.
- **Training Modes:** Pretraining, fine-tuning, linear evaluation, and semi-supervised learning.

## Methodology

The framework uses three pretext tasks:

1. **Image Reconstruction:** Reconstructs original images from corrupted inputs.
2. **Rotation Prediction:** Predicts rotations of 0°, 90°, 180°, and 270°.
3. **Contrastive Learning:** Learns discriminative representations from augmented image pairs.

The pretrained CCT backbone is fine-tuned for downstream histopathology classification tasks.

## Datasets

**Pretraining:** Approximately 600K unlabeled images collected from public histopathology datasets, including PatchCamelyon, ICIAR 2018, and TUPAC 2016.

**Classification:**
- [NCT-CRC-HE-100K](https://zenodo.org/record/1214456) — Colorectal tissue histopathology.
- [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) — Breast cancer histopathology.

## Results

### NCT-CRC Benchmark

| Evaluation Mode | Accuracy | Macro F1 | Macro AUC |
|---|---:|---:|---:|
| Fine-Tuning | 84% | 78% | 98.0% |
| SSL + Linear Evaluation | 91% | 86% | 98.5% |
| **SSL + Fine-Tuning** | **94%** | **91%** | **99.3%** |

### BreakHis Benchmark

*5-fold stratified cross-validation.*

| Evaluation Mode | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Fine-Tuning | 85.7% | 82.0% | 85.2% |
| SSL + Linear Evaluation | 84.8% | 81.2% | 84.5% |
| **SSL + Fine-Tuning** | **93.7%** | **92.8%** | **93.8%** |

### BreakHis Magnification Analysis

| Magnification | Accuracy |
|---|---:|
| 40× | 93.78% |
| 100× | 93.12% |
| 200× | 94.28% |
| 400× | 94.39% |
| **Mean** | **93.89%** |

*Results are based on the reported project experiments.*

## Installation

```bash
git clone https://github.com/alibalapour/Compact-SiT.git
cd Compact-SiT
pip install --upgrade -r requirements.txt
```

## Pretrained Model

Download the pretrained checkpoint:

```bash
gdown --id 1BqoJ_IJWjOwqueCZstch-XQERcUto3dt
```

## Training

### Self-Supervised Pretraining

```bash
python main.py --batch-size 72 --epochs 10 --min-lr 5e-6 --lr 1e-4 --training-mode SSL --dataset UH_main --output output --validate-every 1
```

### Fine-Tuning on NCT-CRC

```bash
python main.py --batch-size 120 --epochs 50 --min-lr 5e-6 --training-mode finetune --dataset NCT --finetune "path/to/pretrained_model" --output output --validate-every 1
```

### Model Evaluation

```bash
python evaluate.py --batch-size 180 --dataset Custom --custom_test_dataset_path "path/to/test_dataset" --model-path "path/to/finetuned_model"
```

**Note:** Update dataset paths and checkpoint locations according to your environment. See the repository's argument definitions for linear evaluation and custom dataset configuration.

## Evaluation Metrics

Accuracy, macro precision, macro recall, macro F1-score, weighted F1-score, ROC-AUC, Cohen's Kappa, and confusion matrix.
