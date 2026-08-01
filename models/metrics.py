import torch


SMOOTH = 1e-6


def dice_score(preds, targets):

    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    dice = (
        (2 * intersection + SMOOTH)
        /
        (preds.sum() + targets.sum() + SMOOTH)
    )

    return dice.item()


def iou_score(preds, targets):

    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum() - intersection

    iou = (
        (intersection + SMOOTH)
        /
        (union + SMOOTH)
    )

    return iou.item()


def precision(preds, targets):

    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()

    fp = (preds * (1 - targets)).sum()

    precision = (
        (tp + SMOOTH)
        /
        (tp + fp + SMOOTH)
    )

    return precision.item()


def recall(preds, targets):

    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()

    fn = ((1 - preds) * targets).sum()

    recall = (
        (tp + SMOOTH)
        /
        (tp + fn + SMOOTH)
    )

    return recall.item()


def accuracy(preds, targets):

    preds = (preds > 0.5).float()

    correct = (preds == targets).float().sum()

    total = targets.numel()

    acc = correct / total

    return acc.item()