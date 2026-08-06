import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


def tensor_to_numpy(image):

    """
    Convert torch tensor (C,H,W)
    to numpy (H,W,C)
    """

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    return image


def mask_to_numpy(mask):

    """
    Convert tensor mask
    (1,H,W) -> (H,W)
    """

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask)

    return mask


def get_flair(image):

    """
    Extract FLAIR channel
    for visualization.
    """

    image = tensor_to_numpy(image)

    flair = image[:, :, 0]

    return flair


def create_overlay(

    flair,

    mask,

    alpha=0.45,

):

    """
    Overlay predicted mask
    on grayscale MRI.
    """

    flair = cv2.normalize(

        flair,

        None,

        0,

        255,

        cv2.NORM_MINMAX,

    ).astype(np.uint8)

    flair = cv2.cvtColor(

        flair,

        cv2.COLOR_GRAY2RGB,

    )

    overlay = flair.copy()

    overlay[mask > 0] = [255, 0, 0]

    result = cv2.addWeighted(

        flair,

        1 - alpha,

        overlay,

        alpha,

        0,

    )

    return result


def show_prediction(

    image,

    prediction,

    ground_truth=None,

):

    flair = get_flair(image)

    pred = mask_to_numpy(prediction)

    pred = (pred > 0.5).astype(np.uint8)

    overlay = create_overlay(

        flair,

        pred,

    )

    if ground_truth is None:

        fig, ax = plt.subplots(

            1,

            3,

            figsize=(15, 5),

        )

        ax[0].imshow(

            flair,

            cmap="gray",

        )

        ax[0].set_title("FLAIR MRI")

        ax[1].imshow(

            pred,

            cmap="gray",

        )

        ax[1].set_title("Prediction")

        ax[2].imshow(

            overlay,

        )

        ax[2].set_title("Overlay")

    else:

        gt = mask_to_numpy(

            ground_truth,

        )

        fig, ax = plt.subplots(

            1,

            4,

            figsize=(20, 5),

        )

        ax[0].imshow(

            flair,

            cmap="gray",

        )

        ax[0].set_title("FLAIR MRI")

        ax[1].imshow(

            gt,

            cmap="gray",

        )

        ax[1].set_title("Ground Truth")

        ax[2].imshow(

            pred,

            cmap="gray",

        )

        ax[2].set_title("Prediction")

        ax[3].imshow(

            overlay,

        )

        ax[3].set_title("Overlay")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    plt.show()


def save_prediction(

    image,

    prediction,

    save_dir,

    filename,

):

    save_dir = Path(save_dir)

    save_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    flair = get_flair(image)

    pred = mask_to_numpy(prediction)

    pred = (pred > 0.5).astype(np.uint8)

    overlay = create_overlay(

        flair,

        pred,

    )

    plt.imsave(

        save_dir / f"{filename}_flair.png",

        flair,

        cmap="gray",

    )

    plt.imsave(

        save_dir / f"{filename}_prediction.png",

        pred,

        cmap="gray",

    )

    plt.imsave(

        save_dir / f"{filename}_overlay.png",

        overlay,

    )