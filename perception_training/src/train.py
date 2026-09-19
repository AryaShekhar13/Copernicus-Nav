import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import RellisDataset
from model import TerrainSegModel
from checkpointing import save_checkpoint, load_checkpoint


def train(dataset_config_path, classes_config_path, training_config_path, device=None,
          max_train_samples=None, max_val_samples=None):
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
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    ckpt_path = os.path.join(cfg["checkpoint_dir"], cfg["checkpoint_filename"])
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

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
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % cfg["log_every_n_steps"] == 0:
                print(f"Epoch {epoch} step {global_step} loss {loss.item():.4f}")

            if global_step % cfg["save_every_n_steps"] == 0:
                save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, best_val_loss)
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

        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, global_step, best_val_loss)
        print(f"End-of-epoch checkpoint saved (epoch {epoch + 1})")

    return model
