## Project Structure

brain-tumor-segmentation/
│
├── datasets/
│
├── configs/
│
├── models/
│   ├── effnet_unet.py
│   ├── segformer.py
│   └── losses.py
│
├── utils/
│   ├── dataset.py
│   ├── augmentations.py
│   ├── metrics.py
│   ├── visualize.py
│   └── train_utils.py
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
├── requirements.txt
└── README.md

## Build order

We'll implement everything in this order.

### Phase 1
Config
Dataset loader
Train/Validation/Test split
Data augmentation

### Phase 2
EfficientNet-UNet

pretrained EfficientNet
ASPP
Attention Gates
Decoder
Model

### Phase 3

SegFormer

MiT encoder
Patch Embedding
Efficient Self Attention
Mix FFN
Decoder
Prediction Head

### Phase 4

Training pipeline

Trainer
Validation
Saving checkpoints
Early stopping
Scheduler

### Phase 5

Evaluation

Dice

IoU

Precision

Recall

Accuracy

Confusion Matrix

Inference

Visualization


✅ config.py
✅ dataset.py
✅ transforms.py
✅ losses.py
✅ metrics.py
➜ EfficientNet model
➜ SegFormer model
➜ train.py
➜ evaluate.py
➜ inference.py


## EFFnet architecture

4-channel MRI
      │
      ▼
1×1 Conv (4 → 3)
      │
      ▼
ImageNet EfficientNet-B0
      │
      ▼
Skip Connections
      │
      ▼
ASPP
      │
      ▼
Attention U-Net Decoder
      │
      ▼
1×1 Conv
      │
      ▼
Sigmoid


## Segformer architecture

4-channel MRI
      │
1×1 Conv (4 → 3)
      │
MiT-B0 Encoder (ImageNet pretrained)
      │
Feature maps:
56×56 × 32
28×28 × 64
14×14 × 160
 7×7 × 256
      │
MLP Decoder
      │
Feature Fusion
      │
1×1 Conv
      │
Upsample → 224×224
      │
Sigmoid
