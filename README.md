# D2FP: Learning Implicit Prior for Human Parsing

This the official code for D2FP: Learning Implicit Prior for Human Parsing.

* [Paper](https://shamanneo.github.io/data/paper/2024384189.pdf)

This codebase is implemented using [Detectron2](https://github.com/facebookresearch/detectron2).

![Image](https://github.com/user-attachments/assets/6609de2b-604e-435d-84c0-847a198a73f6)

## Setup

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

Set the DETECTRON2_DATASETS environment variable and organize the downloaded datasets in the following directory structure within the specified path. You can download the dataset from the official website: https://sysu-hcp.net/lip/overview.php.

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

Set the number of GPUs and customize the configuration as needed.

```
./train.sh
```

## Evaluation 

Set the number of GPUs and customize the configuration as needed.

```
./test.sh
```
## Acknowledgement

Code is largely based on M2FP (https://github.com/soeaver/M2FP).
