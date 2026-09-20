import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardDiceLoss(nn.Module):
    """Soft Dice loss restricted to the hazardous classes (water/puddle/mud/rubble
    by default), computed per-class and averaged.

    Dice penalizes fragmented, poorly-overlapping predictions harder than CE does
    for the same pixel-error count. This targets the "puddle broken into a
    patchwork of puddle/mud" failure documented in the README's worst-case example -
    a boundary/shape problem that per-class CE weighting alone cannot fix, since it
    only reweights individual pixel errors, not the overall shape of the prediction.

    hazard_class_ids: contiguous (post-remap) class indices to include, e.g. the
    indices of water/puddle/mud/rubble after build_class_remap().
    """

    def __init__(self, hazard_class_ids, ignore_index=0, eps=1e-6):
        super().__init__()
        self.hazard_class_ids = list(hazard_class_ids)
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        valid = (targets != self.ignore_index)

        dice_terms = []
        for class_id in self.hazard_class_ids:
            pred_c = probs[:, class_id, :, :]
            target_c = (targets == class_id).float()

            pred_c = pred_c * valid
            target_c = target_c * valid

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice_score = (2.0 * intersection + self.eps) / (union + self.eps)
            dice_terms.append(1.0 - dice_score.mean())

        return torch.stack(dice_terms).mean()


class CombinedCEDiceLoss(nn.Module):
    """Weighted CrossEntropyLoss (the repo's existing loss) plus a HazardDiceLoss term.

    total_loss = ce_loss + dice_weight * hazard_dice_loss

    CE (with the hazard-floor class weights) handles "don't neglect this class";
    Dice handles "don't fragment this class's predicted shape". Kept as separate,
    logged terms (see train.py) so it's visible whether the Dice term is actually
    moving during training, rather than folded into one opaque number.
    """

    def __init__(self, class_weights, hazard_class_ids, ignore_index=0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice = HazardDiceLoss(hazard_class_ids, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = ce_loss + self.dice_weight * dice_loss
        return total, ce_loss.detach(), dice_loss.detach()
