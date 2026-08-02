import numpy as np

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandShiftIntensityd,
)


train_transform = Compose([
    RandFlipd(
        keys=["image", "mask"],
        prob=0.5,
        spatial_axis=0,
    ),
    RandFlipd(
        keys=["image", "mask"],
        prob=0.5,
        spatial_axis=1,
    ),
    RandRotated(
        keys=["image", "mask"],
        range_x=np.pi / 12,   # ±15°
        prob=0.5,
        mode=("bilinear", "nearest"),
        padding_mode="zeros",
    ),
    RandGaussianNoised(
        keys=["image"],
        prob=0.2,
        mean=0.0,
        std=0.05,
    ),
    RandAdjustContrastd(
        keys=["image"],
        prob=0.3,
        gamma=(0.7, 1.5),
    ),
    RandShiftIntensityd(
        keys=["image"],
        offsets=0.1,
        prob=0.3,
    ),
])

val_transform = Compose([])