# Compact-SiT (cSiT)

PyTorch code for self-supervised pretraining, fine-tuning, and evaluation of a **compact self-supervised vision transformer** for histopathology image classification. It adapts [SiT](https://github.com/Sara-Ahmed/SiT) with a **CCT-14/7×2** backbone (~6M parameters, vs. ~86M for ViT-Base).

## Method

Pretrained on ~600K unlabeled histopathology images (PatchCamelyon, ICIAR 2018, TUPAC 2016, and others) with three pretext tasks:

1. **Reconstruction:** restore the original image from a corrupted one (random drop/replace, color distortion, blur, grayscale)
2. **Rotation prediction:** classify 0°, 90°, 180°, 270°
3. **Contrastive learning:** learn embeddings invariant across augmented views

The pretrained model is then evaluated on [NCT-CRC-HE-100K](https://zenodo.org/record/1214456) and [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/), including a semi-supervised setting on NCT-CRC.

## Results

**NCT-CRC**

| Mode | Accuracy | Macro F1 | Macro AUC |
|---|---:|---:|---:|
| Fine-tuning (FT) | 0.84 | 0.78 | 0.980 |
| SSL + Linear Eval (LE) | 0.91 | 0.86 | 0.985 |
| **SSL + FT** | **0.94** | **0.91** | **0.993** |

**BreakHis** (5-fold stratified CV, mean)

| Mode | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| FT | 0.857 | 0.82 | 0.852 |
| SSL + LE | 0.848 | 0.812 | 0.845 |
| **SSL + FT** | **0.937** | **0.928** | **0.938** |

**BreakHis by magnification:** 40× 93.78 · 100× 93.12 · 200× 94.28 · 400× 94.39 · **mean 93.89**

**Comparison with other SSL methods (Macro F1)**

| Model | NCT-CRC | BreakHis | Backbone |
|---|---:|---:|---|
| Autoencoder | 37.0 | 36.0 | ResNet50 |
| Colorization | 80.2 | 72.4 | ResNet50 |
| CPCv2 | 80.1 | 71.1 | ResNet50 |
| SSL Contrastive | 86.2 | 78.2 | ResNet50 |
| SSL Contrastive (best) | 91.4 | 80.2 | ResNet34 |
| **cSiT** | **93.0** | **92.8** | CCT |

## Setup

```bash
git clone https://github.com/alibalapour/Compact-SiT.git
cd Compact-SiT
pip install --upgrade -r requirements.txt
gdown --id 1BqoJ_IJWjOwqueCZstch-XQERcUto3dt   # pretrained model (600k, 30 epochs)
```

## Usage

### Self-supervised pretraining

```bash
python main.py --batch-size 72 --epochs 10 --min-lr 5e-6 --lr 1e-4 --training-mode SSL --dataset UH_main --output output --validate-every 1
```

### Fine-tuning

Prepared dataset:

```bash
python main.py --batch-size 120 --epochs 50 --min-lr 5e-6 --training-mode finetune --dataset NCT --finetune <path/to/pretrained_model> --output output --validate-every 1
```

Custom dataset:

```bash
python main.py --batch-size 120 --epochs 50 --min-lr 5e-6 --training-mode finetune --dataset Custom --custom_train_dataset_path <path/to/train_dataset> --custom_val_dataset_path <path/to/val_dataset> --finetune output/checkpoint.pth --output output --validate-every 1
```

### Linear evaluation

Linear head:

```bash
python main.py --batch-size 120 --epochs 50 --min-lr 5e-6 --training-mode finetune --dataset NCT --finetune <path/to/pretrained_model> --output output --validate-every 1 --SiT_LinearEvaluation 1
```

2-layer MLP head:

```bash
python main.py --batch-size 120 --epochs 10 --lr 1e-3 --weight-decay 5e-4 --min-lr 5e-6 --training-mode finetune --dataset NCT --finetune <path/to/pretrained_model> --output output --validate-every 1 --SiT_LinearEvaluation 1 --representation-size 1024
```

> Set `--dataset_location` to the path of your downloaded dataset.

### Evaluation

```bash
python evaluate.py --batch-size 180 --dataset Custom --custom_test_dataset_path <path/to/test_dataset> --model-path <path/to/finetuned_model>
```

Reported metrics: accuracy, macro precision/recall/F1, weighted F1, ROC-AUC, Cohen's Kappa, and confusion matrix.
