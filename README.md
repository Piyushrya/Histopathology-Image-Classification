# Compact Self-Supervised Vision Transformer for Histopathology Classification

A self-supervised learning framework for histopathology image classification using a Compact Convolutional Transformer (CCT). The model learns visual representations from unlabeled medical images using reconstruction, rotation prediction, and contrastive learning, reducing reliance on labeled data.

## Key Features

- **Self-Supervised Learning:** Combines reconstruction, rotation prediction, and contrastive learning.
- **Compact Architecture:** Uses a modified CCT-14/7×2 model with approximately 6M parameters.
- **Pretraining:** Learns representations from approximately 600K unlabeled histopathology images.
- **Transfer Learning:** Supports linear evaluation and end-to-end fine-tuning.
- **Evaluation:** Includes classification metrics across datasets and magnification levels.

## Datasets

- **NCT-CRC-HE-100K:** Histopathology image classification.
- **BreakHis:** Breast cancer histopathology classification.
- **Additional pretraining datasets:** PatchCamelyon, ICIAR 2018, and TUPAC 2016 Mitosis.

## Methodology

The framework uses three self-supervised objectives:

1. **Image Reconstruction:** Learns representations by reconstructing input images.
2. **Rotation Prediction:** Captures spatial and structural features by predicting image rotations.
3. **Contrastive Learning:** Learns discriminative representations by bringing similar image features closer and separating dissimilar ones.

The pretrained model is evaluated using linear evaluation and supervised fine-tuning.

## Results

### NCT-CRC-HE-100K

| Method | Accuracy | Macro F1 | Macro AUC |
|---|---:|---:|---:|
| Fine-tuning baseline | 84% | 78% | 98.0% |
| Self-supervised + Linear Evaluation | 91% | 86% | 98.5% |
| Self-supervised + Fine-tuning | **94%** | **91%** | **99.3%** |

### BreakHis

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Fine-tuning baseline | 85.7% | 82.0% | 85.2% |
| Self-supervised + Linear Evaluation | 84.8% | 81.2% | 84.5% |
| Self-supervised + Fine-tuning | **93.7%** | **92.8%** | **93.8%** |

### BreakHis Accuracy by Magnification

| Magnification | Accuracy |
|---|---:|
| 40× | 93.78% |
| 100× | 93.12% |
| 200× | 94.28% |
| 400× | 94.39% |
| **Mean** | **93.89%** |

## Installation

```bash
git clone https://github.com/alibalapour/Compact-SiT.git
cd Compact-SiT
pip install --upgrade -r requirements.txt
```

Download the pretrained model:

```bash
gdown --id 1BqoJ_IJWjOwqueCZstch-XQERcUto3dt
```

## Usage

### Self-Supervised Pretraining

```bash
python pretrain.py
```

### Fine-Tuning on NCT-CRC

```bash
python finetune.py
```

### Linear Evaluation

```bash
python linear_eval.py
```

### Model Evaluation

```bash
python evaluate.py
```

*Note: Verify script names and available arguments against the repository before running the commands.*

## Technologies

- Python
- PyTorch
- Compact Convolutional Transformer (CCT)
- Self-Supervised Learning
- Contrastive Learning
- OpenCV
- Scikit-learn

## License

Refer to the repository for licensing details and conditions of use.
