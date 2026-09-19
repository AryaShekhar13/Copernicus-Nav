import torch


def mc_dropout_predict(model, images, num_passes=10):
    """Run multiple stochastic forward passes with MC-Dropout enabled, returning:
      - mean_probs: [B, C, H, W] averaged softmax probabilities across passes
      - uncertainty: [B, H, W] per-pixel predictive entropy of the mean distribution
        (standard MC-Dropout uncertainty measure - high when the model is unsure
        which class a pixel belongs to, low when confident).

    model.enable_mc_dropout() must have been called beforehand (or is called here).
    """
    model.eval()
    model.enable_mc_dropout()

    all_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)

    stacked = torch.stack(all_probs, dim=0)  # [num_passes, B, C, H, W]
    mean_probs = stacked.mean(dim=0)  # [B, C, H, W]

    # Predictive entropy: -sum(p * log(p)) over the class dimension
    eps = 1e-8
    entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)  # [B, H, W]

    return mean_probs, entropy


def predicted_classes(mean_probs):
    """Argmax over the class dimension to get the final predicted class map."""
    return mean_probs.argmax(dim=1)
