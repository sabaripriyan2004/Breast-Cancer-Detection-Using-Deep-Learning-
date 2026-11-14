# Deep Learning Model for Breast Cancer Classification Using PyTorch

In this repository, I implemented a deep learning classifier introduced in the research paper [**Deep Learning to Improve Breast Cancer Detection on Screening Mammography**](https://www.nature.com/articles/s41598-019-48995-4) using PyTorch.

The original code and model are available [here](https://github.com/lishen/end2end-all-conv). However, the original implementation was developed using Keras.

My main goal is to provide a comprehensible PyTorch implementation of this model that can be helpful for everyone, especially those beginning to work with deep learning and interested in medical applications.

The mammography dataset employed in this study is **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**. [Here](https://github.com/CBIIT/CBIS-DDSM) you can find information about the dataset.

---

# Summary of the Main Contribution of This Paper

The authors propose a breast cancer classifier based on a methodology composed of two stages.

The first stage consists of a **patch-level classifier** that uses pixel-level annotations from mammograms to discriminate regions of interest (ROIs) and train the model using those areas.

The second stage consists of a **whole-image classifier**. This image classifier uses the trained patch classifier as its backbone, replacing the top layers while incorporating two additional ResNet blocks.

The training of this whole-image classifier requires only image-level labels.

The patch-level and whole-image classifiers are described in more detail below.

---

# Mammogram Preprocessing

First, we resize the mammograms to **1152 × 896 pixels** while preserving their original aspect ratio to avoid image distortion.

To achieve this, each image is scaled based on the new height (1152 pixels), and the corresponding width is calculated using the original aspect ratio.

If the resulting width is smaller than the target width (896 pixels), symmetric padding is applied on both sides to reach the desired dimensions.

Next, breast segmentation is performed to remove the original watermark in the mammograms.

Finally, the processed images are saved as **16-bit PNG files**.

There is no reorienting of the mammograms.

We use the original split provided by the dataset authors and further split the training set to generate a validation set using an **80/20 split**. The partitions are stratified to maintain the same proportion of cancer cases across all sets.

---

# First Stage: Patch-Level Classifier

## Patch Dataset Generation from Mammograms

We generate two datasets from the mammograms for patch-based classification:

### 📁 Dataset S

Contains **2 patches per image**:

- 1 patch extracted from the **center of the Region of Interest (ROI)**.
- 1 **background patch** randomly sampled from the same image, outside the ROI.

### 📁 Dataset s10

Contains **20 patches per image**:

- 10 patches randomly sampled from each ROI with at least **90% overlap**.
- 10 patches randomly sampled from non-ROI regions.

### Patch Details

- **Size:** 224 × 224 pixels
- **Format:** 16-bit PNG
- **Pixel values:** Rescaled to the range [0.0, 255.0], followed by subtraction of the average grayscale value of the whole patch training set.
- **Watermarks:** Removed before patch extraction.

### 🏷️ Class Labels

Each patch is assigned to one of the following five classes:

| Label | Description |
|---|---|
| 0 | Background |
| 1 | Malignant Calcification |
| 2 | Benign Calcification |
| 3 | Malignant Mass |
| 4 | Benign Mass |

To generate both **S** and **s10** patch datasets, navigate to the `Patches` folder and execute the following script:

```bash
cd Patches
python generating_patches.py
```

---

# Patch Classification with ResNet50

The chosen model for patch classification is **ResNet50**, trained using a **3-stage fine-tuning strategy** on both the S and s10 datasets.

During training, layers are gradually unfrozen from top to bottom while reducing the learning rate at each stage.

## 🔁 3-Stage Training Procedure

### 🔹 Stage 1: Fine-Tune Fully Connected Layer

- **Learning rate:** `1e-3`
- **Weight decay:** `1e-4`
- **Layers trained:** Only the final fully connected (FC) layer
- **Epochs:** 3

### 🔸 Stage 2: Fine-Tune Top Layers

- **Learning rate:** `1e-4`
- **Weight decay:** `1e-4`
- **Layers trained:** Last convolutional layers (PyTorch: `layer4[2]`) and FC layer
- **Epochs:** 10

### 🔻 Stage 3: Fine-Tune Entire Network

- **Learning rate:** `1e-5`
- **Weight decay:** As previously set
- **Layers trained:** All layers in ResNet50
- **Epochs:** 37

During training, we augment mammograms to promote model generalizability by applying the following augmentations:

- **Horizontal and vertical flips**
- **Random rotations:** Between `-25°` and `+25°`
- **Zoom:** Within a scale of `[0.8, 1.2]`
- **Intensity shift:** ±20 of the pixel values

We train the ResNet50 model for a total of **50 epochs**. However, due to the smaller size of the S dataset compared to s10, we extend the third training stage to **200 epochs for S** to ensure sufficient learning.

We use a batch size of **256** and optimize the model using the **Adam optimizer**. All model parameters are initialized with ImageNet-pretrained weights.

### Train Patch Classifier

The training implementation is available in the `patch_classifier` folder:

```text
patch_classifier/train_function.py
```

### Trained Patch-Level Classifier Models

The trained patch-level classifier models are stored in the `trained_models` folder.

### Patch-Level Classification Results

| Dataset | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| S | 0.800 (80.0%) | 0.812 (81.2%) |
| s10 | 0.970 (97.0%) | 0.967 (96.7%) |

**Key Results:**

- Dataset S achieved **81.2% test accuracy**.
- Dataset s10 achieved **96.7% test accuracy**.
- Dataset s10 obtained the highest patch-level classification accuracy.

---

# Second Stage: Whole-Image Classifier

Based on the configurations evaluated in the [research paper](https://www.nature.com/articles/s41598-019-48995-4), the whole-image classifier uses a ResNet50 classifier followed by two identical ResNet blocks of `[512-512-1024]`.

Each ResNet block consists of repeated units of three convolutional layers with filter sizes:

- `1 × 1`
- `3 × 3`
- `1 × 1`

The values in the brackets indicate the depths of the three convolutional layers in each block.

Before assembling the ResNet blocks, the fully connected layer of the patch classifier is replaced with **Global Average Pooling (GAP)**, which outputs the average activation of each feature map.

There are 2048 feature maps in the last convolutional layer of ResNet50.

The output of the GAP layer is then passed through the two additional ResNet blocks and finally connected to a fully connected layer that performs binary classification:

- Benign
- Malignant

### Whole-Image Classifier Architecture

The image classifier model is defined in the `whole_image_classifier` folder.

```text
whole_image_classifier/whole_classifier_model.py
```

## 2-Stage Training Strategy

Similarly to the training method used for the patch classifier, we employ a two-stage training strategy for the whole-image classifier.

### 🔹 Stage 1: Train Newly Added Layers

- **Learning rate:** `1e-4`
- **Weight decay:** `1e-3`
- **Layers trained:** Only the newly added layers
- **Epochs:** 30

### 🔸 Stage 2: Fine-Tune Entire Network

- **Learning rate:** `1e-5`
- **Layers trained:** All layers in the network
- **Epochs:** 20

The script to train the whole-image classifier is available at:

```text
whole_image_classifier/main.py
```

Due to GPU memory limitations, we decreased the batch size to **12**.

We optimized the model using the **Adam optimizer** and used the same augmentations applied during patch classification.

**Note:** The backbone used in the whole-image classifier corresponds to the ResNet50 trained on the s10 patch dataset.

### Trained Whole-Image Classifier Model

The trained whole-image classifier model is stored in the `trained_models` folder.

### Whole-Image Classification Results

| Model | Test Accuracy | Test AUC |
|---|---:|---:|
| ResNet50 + 2 ResNet Blocks | **0.857 (85.7%)** | **0.856** |

**Key Results:**

- **Test Accuracy:** 85.7%
- **Test AUC:** 0.856

The whole-image classifier performs binary classification of complete mammograms into benign and malignant categories.

---

# Overall Classification Results

The following table summarizes the reported performance of both classification stages.

| Classification Stage | Dataset / Model | Validation Accuracy | Test Accuracy | Test AUC |
|---|---|---:|---:|---:|
| Patch-Level | Dataset S | 80.0% | 81.2% | — |
| Patch-Level | Dataset s10 | 97.0% | 96.7% | — |
| Whole-Image | ResNet50 + 2 ResNet Blocks | — | 85.7% | 0.856 |

**Summary:**

- The s10 patch dataset achieved the highest patch-level test accuracy of **96.7%**.
- The whole-image classifier achieved **85.7% test accuracy** and a **0.856 AUC**.

The patch-level and whole-image results represent different classification tasks and should be interpreted separately.

---

# Repository Structure

```text
Breast-Cancer-Detection-Using-Deep-Learning/
│
├── Heatmaps/
│
├── Patches/
│   ├── generating_patches.py
│   └── patches_utils.py
│
├── data/
│   ├── __init__.py
│   ├── split.py
│   └── test.csv
│
├── patch_classifier/
│   └── train_function.py
│
├── trained_models/
│   ├── best_s_patch_model.pt
│   ├── best_s10_patch_model.pt
│   └── best_whole_image_classifier.pt
│
├── utils/
│
├── whole_image_classifier/
│   ├── main.py
│   ├── tools.py
│   └── whole_classifier_model.py
│
├── evaluate_patch_model.py
├── main.py
├── README.md
└── .gitignore
```

**Note:** The trained model checkpoint files are large and may be maintained separately from the GitHub source repository.

---

# Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sabaripriyan2004/Breast-Cancer-Detection-Using-Deep-Learning-.git
```

### 2. Navigate to the Project Directory

```bash
cd Breast-Cancer-Detection-Using-Deep-Learning-
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 4. Install PyTorch

```bash
pip install torch torchvision torchaudio
```

### 5. Install Supporting Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow scipy opencv-python tqdm
```

---

# Execution

The main project components can be executed using the following scripts:

| Task | Script |
|---|---|
| Patch Dataset Generation | `Patches/generating_patches.py` |
| Patch Classifier Training | `patch_classifier/train_function.py` |
| Whole-Image Classifier Training | `whole_image_classifier/main.py` |
| Patch Model Evaluation | `evaluate_patch_model.py` |

### Execution Workflow

1. Download and configure the CBIS-DDSM dataset.
2. Perform mammogram preprocessing.
3. Generate the S and s10 patch datasets.
4. Train the ResNet50 patch classifier.
5. Evaluate the patch-level classification models.
6. Train the whole-image classifier using the s10-trained ResNet50 backbone.
7. Evaluate the whole-image classifier using test accuracy and AUC.

**Note:** Update the dataset paths and configuration settings according to your local environment before executing the scripts.

---

# Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Architecture | ResNet50 |
| Computer Vision | CNN, Transfer Learning |
| Image Processing | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn |
| Dataset | CBIS-DDSM |

---

# References

1. **Research Paper:** [Deep Learning to Improve Breast Cancer Detection on Screening Mammography](https://www.nature.com/articles/s41598-019-48995-4)
2. **Original Keras Implementation:** [lishen/end2end-all-conv](https://github.com/lishen/end2end-all-conv)
3. **CBIS-DDSM Dataset:** [CBIS-DDSM on GitHub](https://github.com/CBIIT/CBIS-DDSM)
4. **PyTorch Documentation:** [https://pytorch.org/docs/](https://pytorch.org/docs/)

---

# Disclaimer

This repository is intended for educational and research purposes only. It is not a clinically validated diagnostic system and should not be used as a substitute for professional medical assessment.

# License and Attribution

This implementation is based on the methodology described in the referenced research paper and related original implementation. Please retain appropriate attribution and comply with the original project's license and dataset terms when using, adapting, or redistributing this work.
