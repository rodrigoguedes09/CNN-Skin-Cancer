# Skin Lesion Classification Using Convolutional Neural Networks

> Code for the paper **"Towards an Approach for Classifying Skin Lesions Using Convolutional Neural Networks"**  
> Rodrigo Guedes de Souza, Fabricio Ourique, Analúcia Schiaffino Morales, Antonio Carlos Sobieranski, Alison R. Panisson  
> Federal University of Santa Catarina (UFSC) — Brazil

---

## Overview

This project implements a deep learning pipeline for binary classification of skin lesions (benign vs. malignant) using a transfer-learning approach based on ResNet18. The system includes a preprocessing and data augmentation pipeline tailored for dermatoscopic images, Grad-CAM explainability, and a REST API for inference.

Trained on the **ISIC 2018** dataset, the model achieved:

| Metric | Value |
|---|---|
| Training Accuracy | 99.42% (epoch 6) |
| Validation Accuracy | **88.12%** |
| AUC-ROC | **0.8715** |
| Specificity | **94.94%** |
| Negative Predictive Value | **90.88%** |
| Sensitivity | 51.24% |

---

## Architecture

The classifier uses a **pre-trained ResNet18** (ImageNet weights) with its final fully connected layer replaced by a single output unit for binary classification:

```
ResNet18 (frozen backbone) → Linear(512 → 1) → Sigmoid
```

- Loss: `BCEWithLogitsLoss` with `pos_weight=5.0` to address class imbalance
- Optimizer: Adam (`lr=0.0001`, `weight_decay=1e-4`)
- Image size: 299×299 pixels
- Epochs: 20 (best checkpoint saved at epoch 6)
- Device: GPU (CUDA) if available, otherwise CPU

---

## Dataset

The **ISIC 2018** dataset contains 12,279 dermatoscopic images split into three classes:

| Class | Original Count |
|---|---|
| Benign | ~7,186 |
| Malignant | ~1,230 |
| Indeterminate | ~2,634 |

The dataset is organized under `data/processed/` with subdirectories for each split:

```
data/processed/
├── train/
│   ├── benign/
│   └── malignant/
└── validation/
    ├── benign/
    └── malignant/
```

> **Note:** The raw images are not included in this repository due to size. Download the ISIC 2018 dataset from https://www.isic-archive.com and organize it using `src/data_organization.py`.

---

## Data Augmentation

To address class imbalance, the custom `MalignantAugmentedSkinLesionDataset` applies aggressive augmentation **exclusively to malignant images**, generating multiple augmented copies of each sample. This expanded the malignant training set to ~6,200 images.

Augmentation pipeline (via [Albumentations](https://albumentations.ai/)):

- Random 90° rotations, horizontal and vertical flips
- Affine transformations (translation, scale, rotation up to 45°)
- Brightness and contrast adjustments
- Elastic deformations and grid distortions
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

Standard preprocessing is applied to benign and validation images (resize + normalize only).

---

## Project Structure

```
├── main.py                        # Training entry point
├── requirements.txt
├── docker-compose.yml             # API + Streamlit services
├── Makefile
├── data/
│   ├── balanced_dataset.py        # MalignantAugmentedSkinLesionDataset
│   ├── balanced_data_generator.py # DataLoader factory
│   ├── test_metadata.csv
│   └── validation_metadata.csv
├── src/
│   ├── model.py                   # ResNet18 definition + training parameters
│   ├── preprocessing.py           # SkinLesionDataset + data generators
│   ├── augmentation.py            # Augmentation transforms
│   ├── explainability.py          # Grad-CAM implementation (PyTorch)
│   ├── evaluate.py                # Metrics, ROC curve, confusion matrix
│   ├── config.py                  # Hyperparameters and paths
│   ├── train.py                   # Alternative training loop
│   ├── predict.py                 # Single-image inference
│   ├── ensemble.py                # Ensemble inference
│   ├── gradcam.py                 # Grad-CAM stub (legacy)
│   ├── monitoring.py              # Training monitoring utilities
│   ├── generate_report.py         # Evaluation report generation
│   ├── data_organization.py       # ISIC dataset organizer
│   ├── create_directories.py      # Directory structure setup
│   ├── api/
│   │   ├── app.py                 # FastAPI application
│   │   ├── schemas.py             # Pydantic request/response schemas
│   │   └── utils.py               # Model loading and inference utils
│   ├── evaluation/
│   │   ├── evaluation.py          # Full evaluation pipeline
│   │   └── metrics.py             # Metric computation (sensitivity, specificity, NPV, etc.)
│   └── web/
│       └── app.py                 # Streamlit web interface
├── models/
│   └── best_model.pth             # Best checkpoint (not tracked in git — too large)
├── logs/
│   ├── evaluation/
│   │   ├── evaluation_report.md   # Full evaluation report
│   │   ├── metrics.json           # All metrics in JSON
│   │   ├── roc_curve.png
│   │   ├── confusion_matrix.png
│   │   └── detailed_results.csv
│   └── evaluation_reports/
│       └── classification_report.txt
├── tests/
│   ├── test_model.py
│   ├── test_augmentation.py
│   ├── test_gradcam.py
│   ├── test_evaluation.py
│   ├── test_ensemble.py
│   └── test_api.py
└── docker/
    ├── Dockerfile
    ├── nginx.conf
    └── docker-entrypoint.sh
```

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/rodrigoguedes09/CNN-Skin-Cancer.git
cd CNN-Skin-Cancer

python -m venv venv
source venv/bin/activate       # Linux/macOS
# or: venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

### 2. Organize the dataset

Download the ISIC 2018 dataset and place the images under `data/ISIC-images-validation/` and `data/ISIC-images-test/`. Then run:

```bash
python -c "from src.create_directories import create_directory_structure; create_directory_structure()"
python -c "from src.data_organization import organize_isic_dataset; organize_isic_dataset()"
```

### 3. Train the model

```bash
python main.py
```

The best model checkpoint (by validation accuracy) is saved to `models/best_model.pth` at each improving epoch.

### 4. Evaluate

```bash
python -m src.evaluation.evaluation
```

Results are saved to `logs/evaluation/`.

### 5. Run the API

```bash
# With Docker
make build
make run

# Without Docker
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The FastAPI docs will be available at `http://localhost:8000/docs`.

### 6. Run tests

```bash
make test
# or
python -m pytest tests/
```

---

## Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to generate heatmaps that highlight the regions of a skin lesion image most responsible for the model's prediction. This is implemented in `src/explainability.py` using PyTorch forward/backward hooks on ResNet18's `layer4`.

The heatmaps allow clinicians to verify that the model focuses on the lesion itself rather than background artifacts, increasing interpretability and clinical trust.

---

## Training Results (Epoch Table)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.2200 | 95.30% | 0.8212 | 79.25% |
| 2 | 0.0871 | 98.16% | 1.1275 | 86.49% |
| 3 | 0.0611 | 98.75% | 1.2900 | 86.49% |
| 4 | 0.0523 | 98.97% | 2.0527 | 85.68% |
| 5 | 0.0448 | 99.12% | 1.2344 | 86.09% |
| **6** | **0.0338** | **99.42%** | **1.6744** | **88.12% ✓** |
| 7 | 0.0325 | 99.38% | 2.0619 | 85.68% |
| ... | ... | ... | ... | ... |
| 20 | 0.0123 | 99.81% | 2.9105 | 85.60% |

Best model saved at **epoch 6** (88.12% validation accuracy).

---

## Confusion Matrix (Validation Set)

```
              Predicted
              Benign   Malignant
Actual Benign   976       52
       Malignant  98      103
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{souza2026skin,
  title     = {Towards an Approach for Classifying Skin Lesions Using Convolutional Neural Networks},
  author    = {Souza, Rodrigo Guedes de and Ourique, Fabricio and Morales, Anal{\'u}cia Schiaffino and Sobieranski, Antonio Carlos and Panisson, Alison R.},
  booktitle = {Proceedings of the International Conference on Agents and Artificial Intelligence (ICAART)},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> **Important:** This system is designed as a supplementary tool for healthcare professionals and should **not** be used as the sole basis for clinical diagnosis. All interpretations must be validated by qualified medical specialists.
