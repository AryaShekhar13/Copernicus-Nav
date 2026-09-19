import os
import torch


def save_checkpoint(path, model, optimizer, epoch, step, best_val_loss, extra=None):
    """Save a resumable training checkpoint. Includes everything needed to exactly
    resume training after a Kaggle session interruption.
    """
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    """Load a checkpoint. Returns the checkpoint dict (epoch/step/best_val_loss/extra)
    after restoring model (and optimizer, if given) state in-place.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
