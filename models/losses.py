import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (preds.sum() + targets.sum() + self.smooth)
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCELoss()

        self.dice = DiceLoss()

    def forward(self, preds, targets):

        bce_loss = self.bce(preds, targets)

        dice_loss = self.dice(preds, targets)

        return bce_loss + dice_loss