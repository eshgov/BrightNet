# BrightNet

**Spatial attention for wildlife camera image classification under varying illumination.**

BrightNet is a deep learning project for classifying species in camera-trap imagery (WildCam). The main contribution is a **spatial attention** mechanism inside a ResNet-style backbone, designed to emphasize informative image regions and improve robustness across lighting conditions—especially relevant for nocturnal or low-light wildlife footage.

This repository contains the full implementation, baselines, data pipeline, and evaluation code. It accompanies the course final paper (`Final Paper.pdf`) and is intended as a self-contained example of applied computer vision and model design.

---

## Problem & Motivation

Camera traps produce huge volumes of images with **high variability in lighting** (day/night, weather, season). Standard CNNs can overfit to brightness and scene context. We focus on:

- **3-class species classification**: Rabbit, Bobcat, Cat  
- **Controlled evaluation on a “brightest” subset** (top 20% brightness per class) to study how models behave on well-lit vs. full distribution data

The goal is to build a compact, trainable-from-scratch architecture that uses **spatial attention** to focus on salient regions and generalize better across illumination.

---

## Method Overview

### BrightNet (proposed)

- **Backbone**: ResNet-style network (4 stages, [64, 128, 256, 512] channels, 2 blocks per stage) with a 9×9 initial conv and standard stem.
- **BrightFeatureBlock**: Each residual block adds a **spatial attention** branch:
  - 3×3 convs → BatchNorm → ReLU (main path).
  - Parallel branch: 7×7 conv mapping channels → 1 channel, then Sigmoid to get a **spatial attention map**.
  - Output: `features * attention_map` then added to the residual shortcut.
- **Training**: From scratch on WildCam (no ImageNet pretraining). Cross-entropy loss, Adam (1e-3), 5 epochs; images resized to 112×112, ImageNet normalization.

So compared to channel-wise attention (e.g., SE), BrightNet uses **spatial** attention to weight *where* to look in the image, which is well-suited to camera-trap frames where the animal may appear in a small region.

### Baselines

- **ResNet+**: Same backbone and training setup as BrightNet, but blocks are standard residual blocks with **Squeeze-and-Excitation (SE)** channel attention in the deeper layers (layer3, layer4). This isolates the effect of *spatial* vs *channel* attention.
- **ResNet-18 (pretrained)**: `torchvision` ResNet-18 pretrained on ImageNet, with the final FC layer replaced for 3-way classification and fine-tuned on WildCam. Represents a strong pretrained baseline.

---

## Repository Structure

| File / folder | Description |
|---------------|-------------|
| `BrightNet.ipynb` | BrightNet model definition, training loop, and evaluation (test/val/bright set). |
| `ResNet_Plus.ipynb` | ResNet+ (ResNet + SE) baseline: same pipeline, different block design. |
| `ResNet18.ipynb` | Pretrained ResNet-18 fine-tuning and evaluation. |
| `Brightness_subset_maker.ipynb` | Builds the “brightest” subset: per-class top 20% by mean pixel intensity; writes `WildCam_3classes/brightest/` and `brightest_labels.json`. Run this before evaluation on the bright set. |
| `Visualizations.ipynb` | Loads all three models and produces comparison plots (e.g., confusion matrices, metrics across splits). |
| `Final Paper.pdf` | Full write-up: motivation, method, experiments, and results. |
| `WildCam_3classes/` | Data directory (not in repo). Expected layout below. |

---

## Data: WildCam 3-Class

The code expects a **WildCam-derived** dataset with 3 classes (e.g., Rabbit, Bobcat, Cat) in this structure:

```
WildCam_3classes/
├── train/          # training images
├── val/            # validation images
├── test/           # test images
├── annotations.json
└── (after running Brightness_subset_maker.ipynb)
    ├── brightest/           # top 20% brightest per class (from test)
    └── brightest_labels.json
```

**`annotations.json`** should have:

- `labels`: map from image filename to class index (0/1/2).
- `locations`: map from image filename to location ID (optional but used by the dataset class).

`brightest_labels.json` should follow the same structure (e.g. `labels` and `locations` keyed by filename) so that `WildCamDataset` can load the brightest subset. Create it by running **Brightness_subset_maker.ipynb** after preparing `train/`, `val/`, `test/` and `annotations.json`.

---

## Setup & Usage

### Requirements

- Python 3
- PyTorch, torchvision
- NumPy, PIL, matplotlib, scikit-learn

No `requirements.txt` is committed; a minimal environment:

```bash
pip install torch torchvision numpy pillow matplotlib scikit-learn
```

(Use a CUDA-enabled PyTorch build if you want GPU training.)

### Running the pipeline

1. **Data**: Obtain or create WildCam 3-class splits and place them under `WildCam_3classes/` with `annotations.json` as above.
2. **Bright subset**: Open and run **Brightness_subset_maker.ipynb** to generate `WildCam_3classes/brightest/` and `brightest_labels.json`.
3. **Train / evaluate**:
   - **BrightNet**: Run **BrightNet.ipynb** (define model → train → save e.g. `bright3.0.pth` → evaluate on test/val/bright).
   - **ResNet+**: Run **ResNet_Plus.ipynb** (same flow).
   - **ResNet-18**: Run **ResNet18.ipynb** (load pretrained, replace FC, fine-tune, evaluate).
4. **Visualizations**: Run **Visualizations.ipynb** (load saved checkpoints and produce comparison figures).

Training defaults: batch size 256, 5 epochs, Adam 1e-3, 112×112 input. You can change hyperparameters and paths inside each notebook.

---

## Model Summary

| Model | Attention | Pretrained | Note |
|-------|-----------|------------|------|
| **BrightNet** | Spatial (7×7 → sigmoid per position) | No | Proposed; train from scratch. |
| **ResNet+** | Channel (SE in layer3/4) | No | Same backbone, SE instead of spatial. |
| **ResNet-18** | — | Yes (ImageNet) | Strong baseline. |

For quantitative results, ablations, and discussion (e.g., performance on full test set vs. brightest subset), see **Final Paper.pdf**.

---

## Citation & Context

This project was developed for **COS 429 (Computer Vision)** at Princeton. The repository includes:

- Designing a custom CNN with spatial attention for a real-world vision task.
- Rigorous comparison with pretrained and custom baselines.
- End-to-end pipeline: data subsetting, training, evaluation, and visualization.

For full methodology and experimental results, please refer to **Final Paper.pdf** in this repository.
