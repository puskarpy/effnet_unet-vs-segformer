from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate,
    RandGaussianNoise,
    RandAdjustContrast,
    RandShiftIntensity,
)


def get_train_transforms():

    return Compose([
        RandFlip(
            prob=0.5,
            spatial_axis=0,
        ),

        RandFlip(
            prob=0.5,
            spatial_axis=1,
        ),

        RandRotate(
            range_x=0.26,
            prob=0.5,
            keep_size=True,
        ),

        RandGaussianNoise(
            prob=0.2,
            mean=0.0,
            std=0.05,
        ),

        RandAdjustContrast(
            prob=0.3,
            gamma=(0.7, 1.5),
        ),

        RandShiftIntensity(
            offsets=0.10,
            prob=0.3,
        ),
    ])


def get_val_transforms():
    return None