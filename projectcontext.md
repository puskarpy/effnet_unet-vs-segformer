# Brain Tumor Segmentation Project Context

## Project Title

Comparative Study of EfficientNet-UNet and SegFormer for Brain Tumor Segmentation on MRI Images

---

# 1. Project Goal

The goal of this project is to implement and compare two deep learning architectures for binary brain tumor segmentation from MRI images:

1. EfficientNet-B0 + U-Net decoder
2. SegFormer (MiT-B0 Encoder + SegFormer Decoder)

The comparison should be fair by keeping:

- Same dataset
- Same preprocessing
- Same train/validation/test split
- Same loss function
- Same optimizer
- Same evaluation metrics
- Same input resolution
- Same training strategy


---

# 2. Dataset

Dataset:

BraTS MRI Dataset

Each patient contains:


patient/
│
├── patient_flair.nii
├── patient_t1.nii
├── patient_t1ce.nii
├── patient_t2.nii
└── patient_seg.nii


MRI modalities used:


Channel 1: FLAIR
Channel 2: T1
Channel 3: T1CE
Channel 4: T2


Input:


4-channel MRI slice
Shape:

(224,224,4)


Output:


Binary tumor mask

Shape:

(224,224,1)



---

# 3. Framework

Current implementation:


PyTorch


Environment:


Python 3.12
CUDA GPU support
WSL environment


Main libraries:


torch
torchvision
transformers
monai
nibabel
opencv
numpy
scikit-learn
matplotlib
timm



---

# 4. Project Structure


effnet_unet-vs-segformer/

├── configs/
│ └── config.py
│
├── data/
│ ├── dataset.py
│ ├── split.py
│ └── transform.py
│
├── models/
│ ├── effnet_unet.py
│ ├── segformer.py
│ └── losses.py
│
├── utils/
│ ├── metrics.py
│ ├── visualize.py
│ ├── augmentations.py
│ ├── train_utils.py
│ └── callbacks.py
│
├── checkpoints/
│
├── results/
│
├── notebooks/
│
├── train_effnet.py
├── train_segformer.py
├── evaluate.py
├── predict.py
│
└── README.md



---

# 5. Dataset Implementation

Dataset class:


BraTSDataset


Located:


data/dataset.py


Returns:

```python
image, mask

Image:

Tensor:

(4,224,224)

Mask:

Tensor:

(1,224,224)

Preprocessing:

Load NIfTI files using nibabel
Extract 2D slices
Normalize each MRI modality independently
Resize image to 224x224
Resize mask using nearest interpolation
Convert tumor mask to binary:
mask = mask > 0
6. Data Augmentation

Using MONAI transforms.

Current planned augmentations:

Random horizontal flip
Random vertical flip
Random rotation
Gaussian noise
Contrast adjustment
Intensity shift

IMPORTANT:

For segmentation tasks:

Image and mask must receive identical spatial augmentations.

Do NOT augment only the image.

7. EfficientNet-UNet Architecture

File:

models/effnet_unet.py

Architecture:

MRI Input
(224,224,4)

        |
        v

1x1 Conv Adapter

4 channels -> 3 channels


        |
        v


EfficientNet-B0 Encoder
(ImageNet pretrained)


Features:

skip1:
24 channels

skip2:
40 channels

skip3:
80 channels

skip4:
192 channels

bottleneck:
1280 channels


        |
        v


ASPP Module


        |
        v


Attention U-Net Decoder


        |
        v


Binary segmentation mask

(224,224,1)

Encoder feature sizes:

skip1:
[1,24,56,56]

skip2:
[1,40,28,28]

skip3:
[1,80,14,14]

skip4:
[1,192,7,7]

bottleneck:
[1,1280,7,7]
8. SegFormer Architecture

File:

models/segformer.py

Architecture:

MRI Input

(224,224,4)

        |
        v

1x1 Conv Adapter

4 -> 3 channels


        |
        v


MiT-B0 Encoder


        |
        |

C1:
32 channels
56x56


C2:
64 channels
28x28


C3:
160 channels
14x14


C4:
256 channels
7x7


        |
        v


SegFormer Decoder


        |
        v


Binary mask

224x224x1

The SegFormer implementation uses:

Overlap Patch Embedding
Efficient Self Attention
MixFFN
Transformer Blocks
Linear Projection Decoder

Feature outputs verified:

C1:
torch.Size([1,32,56,56])

C2:
torch.Size([1,64,28,28])

C3:
torch.Size([1,160,14,14])

C4:
torch.Size([1,256,7,7])
9. Loss Function

Planned loss:

Combined BCE + Dice Loss

Reason:

Brain tumor segmentation has strong class imbalance.

Background pixels are much larger than tumor pixels.

Dice loss improves overlap optimization.

10. Evaluation Metrics

Models will be compared using:

Dice Score

Measures overlap between prediction and ground truth.

IoU

Intersection over Union.

Precision

How many predicted tumor pixels are correct.

Recall

How many tumor pixels were detected.

Additional comparison:

Training:

Training loss curve
Validation loss curve

Performance:

Parameter count
Training time
Inference time
GPU memory usage
11. Training Pipeline

Flow:

train_effnet.py

        |
        v

BraTSDataset

        |
        v

DataLoader

        |
        v

EfficientNetUNet

        |
        v

train_model()

        |
        v

CheckpointManager

        |
        v

best_model.pth

SegFormer follows the exact same pipeline.

Only model changes:

build_effnet_unet()

vs

build_segformer()
12. Training Utilities
utils/train_utils.py

Contains:

train_one_epoch()

validate()

train_model()
utils/callbacks.py

Handles:

Save best model
Save last model
Early stopping
Reduce learning rate
Save training history
13. Important Design Decisions

DO NOT change:

Input size: 224x224
MRI channels: 4
Binary segmentation
Same metrics for both models
Same dataset split

The purpose is not to build two unrelated models.

The purpose is a controlled architecture comparison.

14. Current Progress

Completed:

✅ Dataset loader
✅ EfficientNet-UNet model
✅ SegFormer encoder
✅ SegFormer decoder
✅ Model shape verification
✅ Loss implementation
✅ Metrics implementation
✅ Callback system
✅ Training utilities

Currently working on:

⬜ Fixing MONAI image-mask synchronized augmentation

⬜ train_effnet.py

⬜ train_segformer.py

⬜ evaluate.py

⬜ predict.py

15. Important Rules for Future AI Assistance

When modifying this project:

Keep explanations beginner friendly.
Provide code directly when requested.
Do not redesign the architecture without discussion.
Maintain fair comparison between models.
Avoid unnecessary libraries.
Keep the project structure unchanged.
Explain why a change is needed before changing architecture.
Prioritize working code over theoretical complexity.

This should put another AI almost exactly where we are right now.