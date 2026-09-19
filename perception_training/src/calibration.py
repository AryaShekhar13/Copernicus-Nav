import torch

def compute_ece(probs, targets, num_bins=15, ignore_index=0):
    """
    Binned Expected Calibration Error for per-pixel classification.
    probs: [B, C, H, W] softmax/mean probabilities
    targets: [B, H, W] ground-truth class indices
    """
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == targets).float()

    valid_mask = targets != ignore_index
    confidences = confidences[valid_mask]
    correct = correct[valid_mask]
    total = confidences.numel()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = torch.zeros(1, device=probs.device)

    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        bin_count = in_bin.sum().item()
        if bin_count == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_count / total) * torch.abs(bin_acc - bin_conf)

    return ece.item()
