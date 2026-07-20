import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        ignore_index=0,
        ce_weight=1.0,
        dice_weight=1.0,
        smooth=1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        """
        logits: [B, C, ...]
        target: [B, ...] with class indices
        """
        ce_loss = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)

        # valid mask for ignore_index
        valid_mask = (target != self.ignore_index)  # [B, ...]

        # replace ignore_index temporarily so one_hot works
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        # one-hot target -> [B, ..., C] -> [B, C, ...]
        target_1h = F.one_hot(target_safe, num_classes=self.num_classes)
        dims = list(range(target_1h.ndim))
        target_1h = target_1h.permute(0, -1, *dims[1:-1]).float()

        # expand valid mask to channel dim
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, ...]
        probs = probs * valid_mask
        target_1h = target_1h * valid_mask

        # optionally exclude ignore/background class from Dice
        probs = probs[:, 1:]
        target_1h = target_1h[:, 1:]

        # sum over batch + spatial dims, keep class dim
        reduce_dims = tuple(i for i in range(probs.ndim) if i != 1)

        intersection = (probs * target_1h).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + target_1h.sum(dim=reduce_dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        dice_loss = 1.0 - dice_per_class.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss