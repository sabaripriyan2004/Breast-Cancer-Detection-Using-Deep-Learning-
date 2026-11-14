# Breast Cancer Detection Using Deep Learning (PyTorch)

A PyTorch implementation of a deep learning framework for breast cancer classification using convolutional neural networks, based on the research paper:

**[Deep Learning to Improve Breast Cancer Detection on Screening Mammography](https://www.nature.com/articles/s41598-019-48995-4)**

This repository implements a two-stage breast cancer classification pipeline using the **CBIS-DDSM mammography dataset**. The approach combines patch-level classification with whole-image classification using ResNet50.

The original implementation was developed using Keras by [Lishen et al.'s implementation](https://github.com/lishen/end2end-all-conv). This repository provides a PyTorch-based implementation intended to make the methodology easier to understand and reproduce.

> **Attribution:** This implementation is based on the work and methodology of the original paper and repository. Please refer to the original sources when using or adapting this project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Mammogram Preprocessing](#mammogram-preprocessing)
- [Stage 1: Patch-Level Classification](#stage-1-patch-level-classification)
- [Stage 2: Whole-Image Classification](#stage-2-whole-image-classification)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Execution](#execution)
- [Trained Models](#trained-models)
- [Technologies Used](#technologies-used)
- [References](#references)

---

## Project Overview

The project follows a two-stage deep learning architecture:

**Stage 1 – Patch-Level Classifier**

A ResNet50-based classifier is trained using image patches extracted from mammograms. Pixel-level annotations are used to identify regions of interest (ROIs) and distinguish them from background regions.

**Stage 2 – Whole-Image Classifier**

The trained patch classifier is converted into a whole-image classifier by replacing its fully connected layer with Global Average Pooling and adding two ResNet bottleneck blocks.

The final model performs binary classification:

- Benign
- Malignant

### Overall Workflow

```text
CBIS-DDSM Mammography Dataset
             |
             v
     Image Preprocessing
             |
             v
       ROI Extraction
             |
             v
      Patch Generation
             |
          +--+--+
          |     |
          v     v
       Dataset S  Dataset s10
          |         |
          +----+----+
               |
               v
       ResNet50 Patch Classifier
               |
               v
       Transfer Learned Features
               |
               v
      Global Average Pooling
               |
               v
       Two ResNet Bottleneck Blocks
               |
               v
       Fully Connected Layer
               |
               v
      Whole-Image Classification
               |
               v
        Benign / Malignant
```

---

## Dataset

This project uses the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** dataset, which contains mammographic images and associated annotations.

Dataset setup tutorial:

[CBIS-DDSM Dataset Repository](https://github.com/sposso/CBIS-DDSM-DATASET)

The original dataset split is retained. The training set is further divided into training and validation subsets using an 80/20 split.

The partitions are stratified to maintain the proportion of cancer cases across the datasets.

---

## Mammogram Preprocessing

The mammograms undergo the following preprocessing steps:

1. **Image resizing:** Images are resized to 1152 × 896 pixels while preserving their original aspect ratio.
2. **Aspect ratio preservation:** Images are scaled and symmetrically padded when necessary.
3. **Breast segmentation:** Segmentation is performed to remove the original watermark.
4. **Image storage:** Processed mammograms are saved as 16-bit PNG files.
5. **Dataset splitting:** The original dataset split is retained, with an additional 80/20 training-validation split.

The preprocessing implementation is available in the relevant image resizing scripts in the `Patches` directory.

---

## Stage 1: Patch-Level Classification

### 1. Patch Dataset Generation

Two patch datasets are generated from the mammograms: **S** and **s10**.

#### Dataset S

Contains two patches per image:

- One patch extracted from the center of the Region of Interest (ROI).
- One randomly sampled background patch from outside the ROI.

#### Dataset s10

Contains twenty patches per image:

- Ten patches randomly sampled from each ROI with at least 90% overlap.
- Ten randomly sampled patches from non-ROI regions.

### Patch Specifications

| Parameter | Description |
|---|---|
| Patch size | 224 × 224 pixels |
| Image format | 16-bit PNG |
| Pixel range | 0–255 |
| Normalization | Subtraction of the average grayscale value of the patch training set |
| Watermark removal | Performed before patch extraction |

### Patch Classification Labels

Each patch belongs to one of five classes:

| Label | Class |
|---|---|
| 0 | Background |
| 1 | Malignant Calcification |
| 2 | Benign Calcification |
| 3 | Malignant Mass |
| 4 | Benign Mass |

### Generate Patch Datasets

Navigate to the `Patches` directory and run:

```bash
cd Patches
python generating_patches.py
```

This script generates both S and s10 patch datasets.

### 2. ResNet50 Patch Classifier

ResNet50 is used as the backbone for patch classification.

The model uses ImageNet-pretrained weights and a three-stage fine-tuning strategy.

#### Three-Stage Training Strategy

| Stage | Trainable Layers | Learning Rate | Weight Decay | Epochs |
|---|---|---:|---:|---:|
| Stage 1 | Fully connected layer | 1e-3 | 1e-4 | 3 |
| Stage 2 | Last convolutional block and FC layer | 1e-4 | 1e-4 | 10 |
| Stage 3 | Entire ResNet50 | 1e-5 | Previously set | 37 |

For the S dataset, the third training stage is extended to 200 epochs because of its smaller size.

#### Data Augmentation

The following transformations are applied during training:

- Horizontal and vertical flips
- Random rotations between -25° and +25°
- Random zoom between 0.8 and 1.2
- Intensity shifts of ±20 pixel values

#### Training Configuration

| Parameter | Value |
|---|---|
| Backbone | ResNet50 |
| Pretrained weights | ImageNet |
| Optimizer | Adam |
| Batch size | 256 |
| Standard training duration | 50 epochs |
| Fine-tuning stages | 3 |

### Train the Patch Classifier

The training implementation is located in:

```text
patch_classifier/train_function.py
```

Trained patch classifier models are stored in the `trained_models` directory.

---

## Stage 2: Whole-Image Classification

The second stage converts the trained patch classifier into a whole-image classifier.

The backbone is based on ResNet50 trained on the s10 patch dataset.

### Model Architecture

The architecture consists of:

1. ResNet50 backbone
2. Global Average Pooling (GAP)
3. First ResNet bottleneck block
4. Second ResNet bottleneck block
5. Fully connected classification layer

The original fully connected layer of ResNet50 is replaced by Global Average Pooling.

The GAP layer produces a 2048-dimensional feature vector, which is passed through two additional ResNet bottleneck blocks.

Each bottleneck block uses three convolutional layers:

- 1 × 1 convolution
- 3 × 3 convolution
- 1 × 1 convolution

The block configuration is:

```text
[512 - 512 - 1024]
```

The final fully connected layer performs binary classification:

- Benign
- Malignant

### Two-Stage Training Strategy

| Stage | Training Configuration | Learning Rate | Weight Decay | Epochs |
|---|---|---:|---:|---:|
| Stage 1 | Train newly added layers | 1e-4 | 1e-3 | 30 |
| Stage 2 | Fine-tune the entire network | 1e-5 | Previously set | 20 |

### Training Configuration

| Parameter | Value |
|---|---|
| Backbone | ResNet50 trained on s10 |
| Optimizer | Adam |
| Batch size | 12 |
| Training strategy | Two-stage fine-tuning |
| Augmentation | Same transformations as patch classification |

The batch size is reduced to 12 due to GPU memory limitations.

### Train the Whole-Image Classifier

The implementation is located at:

```text
whole_image_classifier/main.py
```

The whole-image classifier architecture is defined in the same directory.

The trained whole-image classifier is stored in `trained_models`.

---
## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sabaripriyan2004/Breast-Cancer-Detection-Using-Deep-Learning-.git
```

### 2. Navigate to the Project

```bash
cd Breast-Cancer-Detection-Using-Deep-Learning-
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

Install PyTorch:

```bash
pip install torch torchvision torchaudio
```

Install common supporting libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow scipy opencv-python tqdm
```

Install additional project-specific dependencies as required by the source code.

---

## Execution

The main project components can be explored through the following scripts:

| Task | Script |
|---|---|
| Patch generation | `Patches/generating_patches.py` |
| Patch classifier training | `patch_classifier/train_function.py` |
| Whole-image classifier training | `whole_image_classifier/main.py` |
| Patch model evaluation | `evaluate_patch_model.py` |
| Heatmap generation | `Heatmaps/` |

**Important:** Training and inference require the appropriate CBIS-DDSM dataset files and trained model checkpoints. Update dataset paths and configuration settings according to your local environment before execution.

---

## Trained Models

The project uses trained model checkpoints for:

- S patch classifier
- s10 patch classifier
- Whole-image classifier

The model checkpoints are large binary files and may not be included in the GitHub repository because of file-size restrictions.

Keep the trained weights locally or distribute them separately using an appropriate large-file hosting solution such as Git LFS.

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Architecture | ResNet50 |
| Computer Vision | CNN, Image Preprocessing, Heatmap Generation |
| Image Processing | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Dataset | CBIS-DDSM |

---

## References

1. **Research Paper:** [Deep Learning to Improve Breast Cancer Detection on Screening Mammography](https://www.nature.com/articles/s41598-019-48995-4)
2. **Original Keras Implementation:** [lishen/end2end-all-conv](https://github.com/lishen/end2end-all-conv)
3. **PyTorch Documentation:** [https://pytorch.org/docs/](https://pytorch.org/docs/)

---

## Disclaimer

This repository is intended for educational and research purposes only. It is not a clinically validated diagnostic system and should not be used as a substitute for professional medical assessment.

## License and Attribution

Please refer to the original project's license and research paper when using, adapting, or redistributing this implementation. Retain appropriate attribution to the original authors and contributors.
