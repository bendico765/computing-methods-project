import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, target, pred):
        """
        target: one-hot encoded masks, shape (B, 2, W, H)
        pred: softmax probabilities, shape (B, 2, W, H)
        """
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()