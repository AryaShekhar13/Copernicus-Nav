import os
import yaml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import RellisDataset
from model import TerrainSegModel
from checkpointing import save_checkpoint, load_checkpoint
from preprocessing import build_class_remap
from losses import CombinedCEDiceLoss


def train(dataset_config_path, classes_config_path, training_config_path, device=None,
          max_train_samples=None, max_val_samples=None, dice_weight=1.0):
    with open(training_config_path) as f:
        cfg = yaml.safe_load(f)
    with open(classes_config_path) as f:
        classes_cfg = yaml.safe_load(f)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = classes_cfg["num_classes"]

    train_ds = RellisDataset(dataset_config_path, split="train",
                              classes_config_path=classes_config_path,
                              training_config_path=training_config_path)
    val_ds = RellisDataset(dataset_config_path, split="val",
                            classes_config_path=classes_config_path,
                            training_config_path=training_config_path)

    if max_train_samples is not None:
        train_ds = Subset(train_ds, range(min(max_train_samples, len(train_ds))))
    if max_val_samples is not None:
        val_ds = Subset(val_ds, range(min(max_val_samples, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"])

    model = TerrainSegModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    class_weights = None
    if cfg.get("class_weights_path"):
        weights_path = os.path.join(os.path.dirname(training_config_path), "..",
                                     cfg["class_weights_path"]) \
            if not os.path.isabs(cfg["class_weights_path"]) else cfg["class_weights_path"]
        weights_path = os.path.normpath(weights_path)
        if os.path.exists(weights_path):
            class_weights = torch.tensor(np.load(weights_path), dtype=torch.float32).to(device)
            print(f"Loaded class weights from {weights_path}")
        else:
            print(f"WARNING: class_weights_path set but file not found at {weights_path}; "
                  f"falling back to unweighted loss")

    # Hazard classes (water/puddle/mud/rubble), mapped to their contiguous (post-remap)
    # indices, since that's what the model's output channels and mask tensors use.
    remap, _ = build_class_remap(classes_config_path)
    hazard_class_ids = [
        remap[c["index"]] for c in classes_cfg["classes"] if c.get("hazardous")
    ]
    print(f"Hazard class ids (contiguous): {hazard_class_ids}")

    criterion = CombinedCEDiceLoss(
        class_weights=class_weights,
        hazard_class_ids=hazard_class_ids,
        ignore_index=0,
        dice_weight=dice_weight,
    )

    ckpt_path = os.path.join(cfg["checkpoint_dir"], cfg["checkpoint_filename"])
    best_ckpt_path = os.path.join(cfg["checkpoint_dir"], cfg["best_checkpoint_filename"])
    patience = cfg.get("early_stopping_patience")
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = load_checkpoint(ckpt_path, model, optimizer, map_location=device)
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed at epoch {start_epoch}, step {global_step}, best_val_loss {best_val_loss}")
    else:
        print("No checkpoint found, starting fresh")

    for epoch in range(start_epoch, cfg["num_epochs"]):
        model.train()
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, ce_loss, dice_loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % cfg["log_every_n_steps"] == 0:
                print(f"Epoch {epoch} step {global_step} loss {loss.item():.4f} "
                      f"(ce {ce_loss.item():.4f}, hazard-dice {dice_loss.item():.4f})")

            if global_step % cfg["save_every_n_steps"] == 0:
                save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, best_val_loss)
                print(f"Checkpoint saved at step {global_step}")

        model.eval()
        val_losses, val_ce_losses, val_dice_losses = [], [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss, ce_loss, dice_loss = criterion(outputs, masks)
                val_losses.append(loss.item())
                val_ce_losses.append(ce_loss.item())
                val_dice_losses.append(dice_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        avg_val_ce = sum(val_ce_losses) / len(val_ce_losses) if val_ce_losses else float("nan")
        avg_val_dice = sum(val_dice_losses) / len(val_dice_losses) if val_dice_losses else float("nan")
        print(f"Epoch {epoch} complete. Avg val loss: {avg_val_loss:.4f} "
              f"(ce {avg_val_ce:.4f}, hazard-dice {avg_val_dice:.4f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            save_checkpoint(best_ckpt_path, model, optimizer, epoch + 1, global_step, best_val_loss)
            print(f"New best val loss {best_val_loss:.4f} - best checkpoint saved (epoch {epoch + 1})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s) "
                  f"(best remains {best_val_loss:.4f} from an earlier epoch)")

        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, global_step, best_val_loss)
        print(f"End-of-epoch checkpoint saved (epoch {epoch + 1})")

        if patience is not None and epochs_since_improvement >= patience:
            print(f"Early stopping: no improvement for {epochs_since_improvement} epochs "
                  f"(patience={patience}). Stopping at epoch {epoch + 1}.")
            break

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train TerrainSegModel with weighted CE + hazard Dice loss."
    )
    parser.add_argument("--dataset-config", default="../configs/dataset.yaml")
    parser.add_argument("--classes-config", default="../configs/classes.yaml")
    parser.add_argument("--training-config", default="../configs/training.yaml")
    parser.add_argument("--dice-weight", type=float, default=1.0,
                         help="Weight on the hazard Dice term (0.0 disables it, "
                              "matching the original CE-only baseline).")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    train(
        dataset_config_path=args.dataset_config,
        classes_config_path=args.classes_config,
        training_config_path=args.training_config,
        dice_weight=args.dice_weight,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
