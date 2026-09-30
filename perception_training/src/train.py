import json
import os
import random
import time
import yaml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dataset import RellisDataset
from models_zoo import build_model
from checkpointing import save_checkpoint, load_checkpoint


def build_criterion(cfg, training_config_path, device):
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

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=class_weights)
    return criterion


def train(dataset_config_path, classes_config_path, training_config_path, device=None,
          max_train_samples=None, max_val_samples=None,
          model_name="terrainseg", checkpoint_dir=None, seed=None, hf_id=None, dl_encoder=None,
          num_epochs=None, extra_train_ds=None):
    with open(training_config_path) as f:
        cfg = yaml.safe_load(f)
    with open(classes_config_path) as f:
        classes_cfg = yaml.safe_load(f)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = classes_cfg["num_classes"]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

    if extra_train_ds is not None:
        train_ds = ConcatDataset([train_ds, extra_train_ds])
        print(f"Training on {len(train_ds)} samples ({len(extra_train_ds)} extra)")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"])

    model = build_model(model_name, num_classes, hf_id=hf_id, dl_encoder=dl_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    criterion = build_criterion(cfg, training_config_path, device)

    ckpt_dir = checkpoint_dir or cfg["checkpoint_dir"]
    ckpt_path = os.path.join(ckpt_dir, cfg["checkpoint_filename"])
    best_ckpt_path = os.path.join(ckpt_dir, cfg["best_checkpoint_filename"])
    history_path = os.path.join(ckpt_dir, "history.json")
    extra = {"model_name": model_name, "hf_id": hf_id, "dl_encoder": dl_encoder, "seed": seed}
    history = json.load(open(history_path)) if os.path.exists(history_path) and os.path.exists(ckpt_path) else []
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

    for epoch in range(start_epoch, num_epochs or cfg["num_epochs"]):
        epoch_t0 = time.time()
        train_loss_sum, n_train_batches = 0.0, 0
        model.train()
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            global_step += 1
            train_loss_sum += loss.item()
            n_train_batches += 1

            if global_step % cfg["log_every_n_steps"] == 0:
                print(f"Epoch {epoch} step {global_step} loss {loss.item():.4f}")

            if global_step % cfg["save_every_n_steps"] == 0:
                save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, best_val_loss, extra=extra)
                print(f"Checkpoint saved at step {global_step}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_losses.append(criterion(outputs, masks).item())
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        print(f"Epoch {epoch} complete. Avg val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            save_checkpoint(best_ckpt_path, model, optimizer, epoch + 1, global_step, best_val_loss, extra=extra)
            print(f"New best val loss {best_val_loss:.4f} - best checkpoint saved (epoch {epoch + 1})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s) "
                  f"(best remains {best_val_loss:.4f} from an earlier epoch)")

        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, global_step, best_val_loss, extra=extra)
        print(f"End-of-epoch checkpoint saved (epoch {epoch + 1})")

        history.append({"epoch": epoch + 1, "train_loss": train_loss_sum / max(n_train_batches, 1),
                        "val_loss": avg_val_loss, "epoch_seconds": time.time() - epoch_t0})
        with open(history_path, "w") as f:
            json.dump(history, f, indent=1)

        if patience is not None and epochs_since_improvement >= patience:
            print(f"Early stopping: no improvement for {epochs_since_improvement} epochs "
                  f"(patience={patience}). Stopping at epoch {epoch + 1}.")
            break

    return model


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train a segmentation model with the project's RELLIS-3D recipe.")
    ap.add_argument("--model", default="terrainseg")
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--classes-config", default="configs/classes.yaml")
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument("--checkpoint-dir", default=None, help="per-model dir; default = training.yaml's")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hf-id", default=None, help="override SegFormer init, e.g. nvidia/segformer-b0-finetuned-ade-512-512")
    ap.add_argument("--dl-encoder", default=None, help="DeepLabV3+ encoder, default tu-mobilenetv3_large_100")
    ap.add_argument("--epochs", type=int, default=None, help="override num_epochs (smoke tests)")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    a = ap.parse_args()
    train(a.dataset_config, a.classes_config, a.training_config,
          max_train_samples=a.max_train_samples, max_val_samples=a.max_val_samples,
          model_name=a.model, checkpoint_dir=a.checkpoint_dir, seed=a.seed,
          hf_id=a.hf_id, dl_encoder=a.dl_encoder, num_epochs=a.epochs)
