# Membrane Segmentation · UNet vs UNet++ vs SimpleSegNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aceloolrd/membrane-segmentation/blob/main/segmentation.ipynb)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/aceloolrd/membrane-segmentation/blob/main/segmentation.ipynb)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Binary semantic segmentation of neuronal membranes in electron microscopy images (ISBI 2012). Three architectures are trained and compared under the same conditions.

---

## Features

- **Three architectures**: `SimpleUNet` · `SimpleSegNet` · `UNet++` (ResNeXt50 encoder, pretrained SSL)
- **BCEDice loss** — combined Binary Cross-Entropy + Dice
- **Albumentations** augmentations (HorizontalFlip, VerticalFlip, Normalize)
- **Metrics**: Accuracy · Jaccard Index (IoU) · F1-score
- **Custom callbacks**: `ModelCheckpoint` + `EarlyStopping`
- **Manual annotation comparison** — model output vs threshold-based CLAHE mask

---

## Results

| Model | Val Accuracy | Val Jaccard | Val F1 |
|-------|:-----------:|:-----------:|:------:|
| SimpleUNet | ~0.86 | — | — |
| SimpleSegNet | ~0.85 | ~0.88 | ~0.94 |
| **UNet++** | **~0.91** | **~0.88** | **~0.94** |

---

## Project Structure

```
membrane-segmentation/
├── segmentation.ipynb   # Main notebook: experiments + manual mask comparison
├── src/
│   ├── dataset.py       # MembraneDataset, MembraneDataModule
│   ├── losses.py        # BCEDiceLoss
│   ├── callbacks.py     # ModelCheckpoint, EarlyStopping
│   ├── models.py        # SimpleUNet, SimpleSegNet
│   └── visualization.py # show_first_batch, plot_metrics, visualize_predictions
├── data/
│   ├── membrane/        # ISBI 2012 dataset (train images + masks)
│   ├── sample.png       # Sample EM image for manual annotation demo
│   └── mask_manual.png  # Manually created threshold mask
├── requirements.txt
└── LICENSE
```

---

## How to Run

### Locally

```bash
git clone https://github.com/aceloolrd/membrane-segmentation.git
cd membrane-segmentation
pip install -r requirements.txt
jupyter notebook segmentation.ipynb
```

### Google Colab

Click the **Open in Colab** badge above, then run `download_data.py` inside the notebook to fetch the training dataset.

### nbviewer (read-only)

Click the **nbviewer** badge to browse a static render of the notebook.

---

## Data

**ISBI 2012 EM Segmentation Challenge** — 30 grayscale 512×512 images of *Drosophila* ventral nerve cord with binary membrane masks.

Training images + masks are **not included** in the repo. Download with:

```bash
pip install kaggle
# Configure credentials: https://www.kaggle.com/docs/api
python download_data.py
```

Or manually from Kaggle: [kmader/isbi-challenges-em-segmentation](https://www.kaggle.com/datasets/kmader/isbi-challenges-em-segmentation)

Expected structure:
```
data/membrane/
├── train/
│   ├── image/   ← 30 labeled EM images
│   └── mask/    ← 30 binary masks
└── test/
    └── image/   ← 30 unlabeled images (included)
```

### Google Colab

Update the **Open in Colab** link above — after cloning the repo in Colab, run `download_data.py` to fetch the dataset.

---

---

# Сегментация мембран · UNet vs UNet++ vs SimpleSegNet

Бинарная семантическая сегментация нейрональных мембран на изображениях электронной микроскопии (датасет ISBI 2012). Три архитектуры обучены и сравниваются в одинаковых условиях.

## Возможности

- **Три архитектуры**: `SimpleUNet` · `SimpleSegNet` · `UNet++` (энкодер ResNeXt50, предобученный SSL)
- **BCEDice loss** — комбинированный Binary Cross-Entropy + Dice
- **Albumentations** аугментации (горизонтальный/вертикальный flip, нормализация)
- **Метрики**: Accuracy · Jaccard Index (IoU) · F1-score
- **Кастомные callbacks**: `ModelCheckpoint` + `EarlyStopping`
- **Сравнение с ручной разметкой** — предсказание модели vs пороговая CLAHE-маска

## Результаты

| Модель | Val Accuracy | Val Jaccard | Val F1 |
|--------|:-----------:|:-----------:|:------:|
| SimpleUNet | ~0.86 | — | — |
| SimpleSegNet | ~0.85 | ~0.88 | ~0.94 |
| **UNet++** | **~0.91** | **~0.88** | **~0.94** |

## Запуск

```bash
git clone https://github.com/aceloolrd/membrane-segmentation.git
cd membrane-segmentation
pip install -r requirements.txt
jupyter notebook segmentation.ipynb
```
