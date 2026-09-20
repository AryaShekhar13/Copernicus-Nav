import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardDiceLoss(nn.Module):
    """Soft Dice loss restricted to the hazardous classes (water/puddle/mud/rubble
    by default), computed per-class and averaged -- but only over classes that
    actually have ground-truth pixels in the current batch. Averaging in a
    near-1.0 "no support" term for an absent class dilutes/distorts the signal
    from the class(es) that do have real supervision this batch; skipping
    absent classes keeps the loss meaningful batch-to-batch.

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
            target_c = (targets == class_id).float() * valid

            if target_c.sum() == 0:
                # No ground-truth pixels of this hazard class anywhere in the
                # batch -- skip it rather than let a near-1.0 no-support term
                # dilute the classes that do have real supervision this batch.
                continue

            pred_c = probs[:, class_id, :, :] * valid

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice_score = (2.0 * intersection + self.eps) / (union + self.eps)
            dice_terms.append(1.0 - dice_score.mean())

        if not dice_terms:
            # None of the hazard classes are present in this batch at all --
            # return a zero loss (no hazard supervision available this batch),
            # not the old degenerate ~1.0 average.
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        return torch.stack(dice_terms).mean()


class CombinedCEDiceLoss(nn.Module):
    """Weighted CrossEntropyLoss (the repo's existing loss) plus a HazardDiceLoss term.

    total_loss = ce_loss + dice_weight * hazard_dice_loss

    CE (with the corrected multiplicative hazard weighting) handles "don't
    neglect this class"; Dice handles "don't fragment this class's predicted
    shape". Kept as separate, logged terms (see train.py) so it's visible
    whether the Dice term is actually moving during training.
    """

    def __init__(self, class_weights, hazard_class_ids, ignore_index=0, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice = HazardDiceLoss(hazard_class_ids, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total = ce_loss + self.dice_weight * dice_loss
        return total, ce_loss.detach(), dice_loss.detach()
