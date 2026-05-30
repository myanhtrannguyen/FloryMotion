# AI_Cinemagraph
The project focuses on building a deep learning system from scratch to transform a static image into a short video (3-5 seconds) featuring customizable artistic styles and localized motion effects.

## Task 1: LinkNet EfficientNet-B0 Flower Segmentation

Train the lightweight LinkNet baseline described in `docs/image_segmentation.md`:

```bash
python source/image_segmentation/linknet_efficientnet_b0/train.py
```

Evaluate the best checkpoint on the test split:

```bash
python source/image_segmentation/linknet_efficientnet_b0/evaluate.py --split test
```

Predict a binary mask for one image:

```bash
python source/image_segmentation/linknet_efficientnet_b0/predict.py data/OxfordFlowers102/test/images/image_06736.jpg
```

Default configuration and checkpoints are stored in `source/image_segmentation/linknet_efficientnet_b0/models/`.

## Task 1: U-Net EfficientNet-B0 Baseline

Train the U-Net baseline with the same EfficientNet-B0 encoder and experiment protocol:

```bash
python source/image_segmentation/unet_efficientnet_b0/train.py
```

Evaluate the best checkpoint:

```bash
python source/image_segmentation/unet_efficientnet_b0/evaluate.py --split test
```

Predict a binary mask:

```bash
python source/image_segmentation/unet_efficientnet_b0/predict.py data/OxfordFlowers102/test/images/image_06736.jpg
```

Default configuration and checkpoints are stored in `source/image_segmentation/unet_efficientnet_b0/models/`.
