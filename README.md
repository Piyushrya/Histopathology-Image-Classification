# Histopathology Image Classification
A PyTorch implementation of **self-supervised pretraining, fine-tuning, and evaluation** for histopathology image classification using a Compact Convolutional Transformer (CCT) backbone.

The model extends the Self-supervised Vision Transformer (SiT) framework with a more efficient transformer architecture to learn meaningful representations from unlabeled histopathology images.

## Key Features

- **Self-Supervised Learning:** Combines reconstruction, rotation prediction, and contrastive learning.
- **Compact Vision Transformer:** Uses a modified CCT-14/7×2 backbone with approximately 6 million parameters.
- **Large-Scale Pretraining:** Leverages approximately 600K unlabeled histopathology images.
- **Medical Image Classification:** Evaluated on NCT-CRC and BreakHis benchmarks.
- **Multiple Evaluation Modes:** Supports fine-tuning, linear evaluation, and semi-supervised learning.
- **Evaluation Pipeline:** Includes classification metrics and model evaluation utilities.

## Methodology

The framework consists of three self-supervised pretext tasks:

1. **Image Reconstruction:** Learns image representations by reconstructing original images from corrupted inputs.
2. **Rotation Prediction:** Predicts rotations of 0°, 90°, 180°, and 270° to learn spatial features.
3. **Contrastive Learning:** Learns discriminative image embeddings by comparing augmented views of the same image.

These objectives are integrated with a Compact Convolutional Transformer (CCT) backbone, which combines convolutional tokenization with transformer-based feature learning.

## Dataset

### Pretraining Dataset

Approximately **600,000 unlabeled histopathology images** collected from multiple public datasets, including:

- PatchCamelyon
- ICIAR 2018
- TUPAC 2016 Mitosis

### Downstream Classification Datasets

- **NCT-CRC-HE-100K:** Histopathology image classification for colorectal tissue.
- **BreakHis:** Breast cancer histopathology image classification.

## Model Architecture

The original SiT framework uses a Vision Transformer (ViT) backbone. This implementation replaces it with a modified **CCT-14/7×2** architecture to reduce model complexity while retaining effective visual representation learning.

| Component | Description |
|---|---|
| Backbone | Compact Convolutional Transformer (CCT) |
| Parameters | Approximately 6 million |
| Pretext tasks | Reconstruction, rotation prediction, contrastive learning |
| Framework | PyTorch |
| Application | Histopathology image classification |

## Results

### NCT-CRC Benchmark

| Evaluation Mode | Accuracy | Macro F1 | Weighted F1 | Macro AUC |
|---|---:|---:|---:|---:|
| Fine-tuning (FT) | 84% | 78% | 84% | 98.0% |
| SSL + Linear Evaluation | 91% | 86% | 90% | 98.5% |
| SSL + Fine-tuning | **94%** | **91%** | **94%** | **99.3%** |

### BreakHis Benchmark

Results using 5-fold stratified cross-validation.

| Evaluation Mode | Mean Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Fine-tuning (FT) | 85.7% | 82.0% | 85.2% |
| SSL + Linear Evaluation | 84.8% | 81.2% | 84.5% |
| SSL + Fine-tuning | **93.7%** | **92.8%** | **93.8%** |

### BreakHis Magnification Analysis

| Magnification | Accuracy |
|---|---:|
| 40× | 93.78% |
| 100× | 93.12% |
| 200× | 94.28% |
| 400× | 94.39% |
| **Mean** | **93.89%** |

*Results are reported from the project's experiments. Performance may vary depending on data splits, preprocessing, and training configuration.*

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/alibalapour/Compact-SiT.git
cd Compact-SiT

pip install --upgrade -r requirements.txt
```

## Download the Pretrained Model

Download the pretrained checkpoint using Google Drive:

```bash
gdown --id 1BqoJ_IJWjOwqueCZstch-XQERcUto3dt
```

Ensure the downloaded checkpoint is placed at the appropriate path before fine-tuning or evaluation.

## Training

### 1. Self-Supervised Pretraining

```bash
python main.py \
  --batch-size 72 \
  --epochs 10 \
  --min-lr 5e-6 \
  --lr 1e-4 \
  --training-mode SSL \
  --dataset UH_main \
  --output output \
  --validate-every 1
```

### 2. Fine-Tuning on NCT-CRC

```bash
python main.py \
  --batch-size 120 \
  --epochs 50 \
  --min-lr 5e-6 \
  --training-mode finetune \
  --dataset NCT \
  --finetune "path/to/pretrained_model" \
  --output output \
  --validate-every 1
```

### 3. Fine-Tuning on a Custom Dataset

```bash
python main.py \
  --batch-size 120 \
  --epochs 50 \
  --min-lr 5e-6 \
  --training-mode finetune \
  --dataset Custom \
  --custom_train_dataset_path "path/to/train_dataset" \
  --custom_val_dataset_path "path/to/val_dataset" \
  --finetune "output/checkpoint.pth" \
  --output output \
  --validate-every 1
```

### 4. Linear Evaluation

Use a linear projection head to evaluate the quality of pretrained representations.

```bash
python main.py \
  --batch-size 120 \
  --epochs 50 \
  --min-lr 5e-6 \
  --training-mode finetune \
  --dataset NCT \
  --finetune "path/to/pretrained_model" \
  --output output \
  --validate-every 1 \
  --SiT_LinearEvaluation 1
```

For a 2-layer MLP projection head, add:

```bash
--representation-size 1024
```

### 5. Model Evaluation

Evaluate a fine-tuned model on a custom test dataset.

```bash
python evaluate.py \
  --batch-size 180 \
  --dataset Custom \
  --custom_test_dataset_path "path/to/test_dataset" \
  --model-path "path/to/finetuned_model"
```

**Note:** Update dataset paths, checkpoint locations, and training parameters according to your environment. Refer to the repository's argument definitions for additional configuration options.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Macro Precision and Macro Recall
- Macro F1-score and Weighted F1-score
- ROC-AUC
- Cohen's Kappa Score
- Confusion Matrix
