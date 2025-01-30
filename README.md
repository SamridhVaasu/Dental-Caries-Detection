# 🦷 YOLOv11n Dental Caries Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11n-green.svg)](https://github.com/ultralytics/ultralytics)

## Overview

This repository implements YOLOv11n for dental image analysis, featuring a lightweight architecture optimized for efficient inference while maintaining high detection accuracy. The model utilizes the Ultralytics framework and incorporates advanced features like C3K2 blocks and SPPF (Spatial Pyramid Pooling - Fast).

## Model Architecture

The network consists of 319 layers with 2,590,035 parameters (2,590,019 gradients) and operates at 6.4 GFLOPs. Key architectural components include:

- **Input Resolution:** 640x640
- **Backbone:** Progressive feature extraction using Conv and C3K2 blocks
- **Neck:** Feature pyramid with upsampling and concatenation
- **Head:** Multi-scale detection with 64x64, 128x128, and 256x256 feature maps

### Layer Configuration
- Initial Convolutions: `3→16→32` channels
- C3K2 blocks with varying channel depths: `64→128→256`
- SPPF module with 256 channels and 5x5 maximum pooling
- C2PSA attention module
- Detection head with three scales

## Training Configuration

The model was trained for 5 epochs with the following metrics:

### Training Progress
- **Initial (Epoch 1/5):**
  - Box Loss: `2.55`
  - Classification Loss: `1.526`
  - DFL Loss: `1.526`
  - mAP50: `0.238`
  - mAP50-95: `0.154`

- **Final (Epoch 5/5):**
  - Box Loss: `1.533`
  - Classification Loss: `1.513`
  - DFL Loss: `1.404`
  - mAP50: `0.665`
  - mAP50-95: `0.401`

## Quick Start

### Installation
```bash
pip install ultralytics
```

### Training
```bash
yolo detect train model=yolov11n.pt data=data.yaml epochs=5
```

### Inference
```bash
yolo detect predict model=path/to/best.pt source=path/to/images
```

## Model Performance

The model demonstrates progressive improvement across training epochs:
- **mAP50** increased from `0.238` to `0.665`
- **mAP50-95** improved from `0.154` to `0.401`
- **Processing speed:** ~4-6 seconds per iteration at 640x640 resolution

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics
- CUDA-capable GPU (recommended)

## License

This project uses the Ultralytics YOLO framework and follows its licensing terms.

## Acknowledgments

- **Ultralytics** for the YOLOv11n architecture
- **Original YOLO authors** for the detection methodology
