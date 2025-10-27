import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCEWithLogitsLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction="mean", smooth_eps=None):
        super().__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        smooth_eps = self.smooth_eps or 0
        if smooth_eps > 0:
            target = target.float()
            target.add_(smooth_eps).div_(2.0)

        bce_loss = F.binary_cross_entropy_with_logits(
            input, target, reduction=self.reduction, weight=self.weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()
        return focal_loss
