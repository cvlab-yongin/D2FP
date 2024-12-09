# D2FP

## Overview

The official PyTorch implementation of our paper:

> **D2FP: Learning Implicit Prior for Human Parsing** \
> Junyoung Hong, Hyeri Yang, Ye Ju Kim, Haerim Kim, Shinwoong Kim, Euna Shim, Kyungjae Lee

![overview-1](https://github.com/user-attachments/assets/ae493e56-e126-435a-8ceb-c751acec4741)

## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd m2fp/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Download pretrained weights
Download the `R-103.pkl` weights from the official website https://github.com/facebookresearch/MaskFormer/blob/main/MODEL_ZOO.md and place it in `D2FP/weights/R-103.pkl`

## Datasets

Please set the environment variable `DETECTRON2_DATASETS` and place the downloaded datasets in the following structure within the defined path. The official website where the dataset can be obtained is https://sysu-hcp.net/lip/overview.php.

### Expected dataset structure for LIP:

```
lip/
  Training/
    Images/             # all training images
    Category_ids/       # all training category labels
  Validation/
    Images/
    Category_ids/
```

### Expected dataset structure for CIHP:
```
cihp/
  Training/
    Images/             # all training images
    Category_ids/       # all training category labels
    Instance_ids/       # all training part instance labels
    Human_ids/          # all training human instance labels
  Validation/
    Images/
    Category_ids/
    Instance_ids/
    Human_ids/
```

## Training

Please set the number of GPUs and adjust the configuration as needed.

```
./train.sh
```

## Evaluation 

Please set the number of GPUs and adjust the configuration as needed.

```
./test.sh
```

## Acknowledgement

Code is largely based on M2FP (https://github.com/soeaver/M2FP).
