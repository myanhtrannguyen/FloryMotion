# Experimental Protocol for Task 1: Flower Segmentation

## 1. Objective

The objective of Task 1 is to train and evaluate multiple segmentation models for extracting the flower foreground from a single input image.

This task is formulated as a **binary segmentation problem**:

```text
Input  : RGB flower image
Output : binary mask
         1 = flower foreground
         0 = background
```

The segmentation output will be used by subsequent modules in the FloryMotion pipeline:

```text
flower segmentation → cartoonization → prompt-to-CIG → motion template generation
```

Therefore, the segmentation module should be evaluated not only by region-level overlap metrics such as Dice and IoU, but also by boundary-sensitive metrics such as HD95 and Boundary F-score.

---

## 2. Dataset

### 2.1 Dataset Description

The experiment uses the **Oxford Flowers 102** dataset.

The dataset is reorganized into the following structure:

```text
data/OxfordFlowers102/
├── oxford102_flower_segmentation.csv
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

The dataset is split using the following ratio:

```text
train : validation : test = 8 : 1 : 1
```

The split should be **stratified by flower category** to preserve the category distribution across the train, validation, and test subsets.

---

### 2.2 CSV Metadata

Each sample should be recorded in a CSV file with the following fields:

```text
image_id
split
class_label
name_cat
```

Example:

```text
6736,test,1,pink primrose
```

---

## 3. Mask Preprocessing

The original Oxford Flowers segmentation masks use a blue background. Therefore, the masks should be converted into binary masks by detecting and removing the blue background.

Recommended conversion:

```python
R = mask[:, :, 0]
G = mask[:, :, 1]
B = mask[:, :, 2]

background = (B > 100) & (B > R + 25) & (B > G + 25)
binary_mask = (~background).astype(np.uint8)
```

The final binary mask should satisfy:

```text
flower foreground = 1
background        = 0
```

Important note: masks must be resized using **nearest-neighbor interpolation**. Bilinear interpolation should not be used for masks because it can create invalid intermediate label values.

---

## 4. Image Preprocessing

All models should use the same preprocessing pipeline for fair comparison.

Recommended setting:

```text
Input size          : 256 × 256
Image normalization : ImageNet mean/std
Mask format         : binary mask 0/1
```

For RGB images:

```text
Resize → Normalize → Tensor
```

For masks:

```text
Resize with nearest-neighbor → Binary mask → Tensor
```

---

## 5. Data Augmentation

### 5.1 Training Augmentation

Recommended augmentations:

```text
Random horizontal flip
Random rotation within ±15 degrees
Random resized crop
Color jitter
Random brightness/contrast adjustment
```

These augmentations help the model generalize to variations in flower pose, lighting, and scale.

Avoid overly aggressive transformations such as:

```text
large rotation
strong perspective distortion
crop that removes most of the flower
heavy blur
```

because they may damage the flower structure and make the segmentation target ambiguous.

### 5.2 Validation and Test Preprocessing

For validation and test sets, use only deterministic preprocessing:

```text
Resize
Normalize
```

No random augmentation should be applied to validation or test data.

---

## 6. Models Compared

The segmentation models are grouped into four categories:

```text
CNN-based models
Salient CNN model
Transformer-based model
Hybrid / lightweight models
```

### 6.1 Experiment Table

| Group | Model | Backbone | Purpose |
|---|---|---|---|
| CNN | U-Net | ResNet34 | Basic encoder-decoder segmentation baseline |
| CNN | U-Net++ | ResNet34 | Tests the effect of nested skip connections |
| Salient CNN | U2-Net | U2-Net Small | Foreground/salient object segmentation baseline |
| Transformer | Swin-UNet | Swin-Tiny | Transformer-based encoder-decoder segmentation |
| Hybrid | LinkNet | EfficientNet-B0 | Lightweight segmentation baseline |
| Hybrid | EfficientUNet | EfficientNet-B0 | U-Net-style decoder with efficient CNN encoder |

---

## 7. Training Configuration

To ensure fair comparison, all models should be trained under the same configuration as much as possible.

Recommended configuration:

```text
Input size       : 256 × 256
Batch size       : 8 or 16
Epochs           : 50
Optimizer        : AdamW
Learning rate    : 1e-4
Weight decay     : 1e-4
Scheduler        : ReduceLROnPlateau or CosineAnnealingLR
Loss function    : BCE Loss + Dice Loss
Early stopping   : patience = 10
Random seed      : 42
```

If GPU memory is limited:

```text
Batch size       : 4
Input size       : 224 × 224 or 256 × 256
Epochs           : 30–50
```

---

## 8. Loss Function

The recommended loss is:

```text
L_seg = L_BCE + λ L_Dice
```

where:

```text
λ = 1.0
```

Explanation:

| Loss | Role |
|---|---|
| BCE Loss | Encourages correct pixel-level classification |
| Dice Loss | Handles foreground-background imbalance |

This loss is suitable because flower pixels may occupy only part of the image, while background can dominate the pixel distribution.

---

## 9. Evaluation Metrics

The following metrics are used:

| Metric | Direction | Meaning |
|---|---:|---|
| Dice | Higher is better | Measures region overlap between prediction and ground truth |
| IoU | Higher is better | Measures intersection-over-union of predicted and true masks |
| HD95 | Lower is better | Measures 95th percentile boundary distance |
| Boundary F-score | Higher is better | Measures boundary alignment quality |
| Precision | Higher is better | Measures how many predicted flower pixels are correct |

### 9.1 Dice Score

Dice measures the overlap between the predicted mask and ground-truth mask.

```text
Dice = 2TP / (2TP + FP + FN)
```

Higher Dice means better foreground segmentation.

### 9.2 IoU

IoU measures the ratio between intersection and union.

```text
IoU = TP / (TP + FP + FN)
```

IoU is usually stricter than Dice.

### 9.3 HD95

HD95 measures the 95th percentile Hausdorff Distance between predicted and ground-truth boundaries.

Lower HD95 means the predicted contour is closer to the ground-truth contour.

This is important because inaccurate boundaries may create visible artifacts in cartoonization and animation.

### 9.4 Boundary F-score

Boundary F-score measures how well the predicted object boundary matches the ground-truth boundary.

This metric is especially important for flower segmentation because flowers often have complex boundaries:

```text
thin petals
curved edges
small gaps between petals
overlapping structures
```

A model may achieve high Dice but still produce poor boundaries. Therefore, Boundary F-score is necessary for evaluating whether the mask is suitable for downstream animation.

### 9.5 Precision

Precision measures how many predicted foreground pixels are actually flower pixels.

```text
Precision = TP / (TP + FP)
```

High precision means the model does not include too much background in the flower mask.

---

## 10. Main Experiment

### Experiment 1: Model Comparison

All models are trained and evaluated on the same dataset split.

| Group | Model | Backbone | Dice ↑ | IoU ↑ | HD95 ↓ | Boundary F-Score ↑ | Precision ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| CNN | U-Net | ResNet34 | | | | | |
| CNN | U-Net++ | ResNet34 | | | | | |
| Salient CNN | U2-Net | U2-Net Small | | | | | |
| Transformer | Swin-UNet | Swin-Tiny | | | | | |
| Hybrid | LinkNet | EfficientNet-B0 | | | | | |
| Hybrid | EfficientUNet | EfficientNet-B0 | | | | | |

The best model should be selected based on both segmentation quality and downstream suitability.

---

## 11. Qualitative Evaluation

In addition to quantitative metrics, qualitative results should be reported.

For each model, visualize:

```text
Input image
Ground-truth mask
Predicted mask
Error map
```

Recommended qualitative cases:

```text
large flower with simple background
flower with complex background
thin-petal flower
flower with color similar to background
partially occluded flower
small flower region
```

The error map can be defined as:

```text
True positive  : correctly predicted flower region
False positive : background incorrectly predicted as flower
False negative : flower region missed by the model
```

This helps explain why certain models perform better or worse in Boundary F-score and HD95.

---

## 12. Per-category Analysis

Since the CSV contains `name_cat`, an additional analysis can be performed by flower category.

For each category, compute:

```text
mean Dice
mean IoU
mean HD95
mean Boundary F-score
mean Precision
```

Then report:

```text
Top-5 easiest flower categories
Top-5 hardest flower categories
```

This analysis helps identify which flower types are difficult to segment.

For example, categories with thin petals or complex shapes may have lower Boundary F-score.

---

## 13. Recommended Report Description

The experimental setup can be described as follows:

> For Task 1, we formulate flower extraction as a binary segmentation problem using the Oxford Flowers 102 dataset. The official segmentation masks are converted into binary foreground-background masks by removing the blue background. The dataset is split into train, validation, and test subsets using an 8:1:1 stratified split based on flower categories. We evaluate seven segmentation architectures from different families, including CNN-based models, a salient object segmentation model, a Transformer-based model, and hybrid lightweight models. All models are trained under the same preprocessing, augmentation, optimizer, loss function, and evaluation protocol. Dice and IoU are used as region-level metrics, while HD95 and Boundary F-score are used to evaluate boundary quality, which is important for downstream cartoonization and motion generation. Precision is additionally reported to measure over-segmentation.

---

## 14. Minimal Experimental Checklist

Before running experiments, verify the following:

```text
[ ] Dataset has been split into train/val/test = 8/1/1
[ ] Each image has a corresponding mask
[ ] Masks are correctly converted to binary 0/1
[ ] Train/val/test sets do not contain overlapping images
[ ] Data augmentation is applied only to the train set
[ ] All models use the same input size
[ ] All models use the same loss function
[ ] All models use the same evaluation metrics
[ ] Best checkpoint is saved according to validation Dice or validation IoU
[ ] Final evaluation is performed on the test set
[ ] Qualitative predictions are saved for each model
```

---

## 15. Output Folder Structure

Each experiment should create a separate folder:

```text
source/image_segmentation/
├── unet_resnet34/
│   ├── config.yaml
│   ├── best_model.pth
├── unetpp_resnet34/
├── u2net_small/
├── swinunet_swin_tiny/
├── linknet_efficientnet_b0/
└── efficientunet_b0/
```

Each experiment folder should contain:

| File / Folder | Description |
|---|---|
| `config.yaml` | Training configuration |
| `best_model.pth` | Best checkpoint |

This structure makes it easier to reproduce experiments and summarize results in the final comparison table.
