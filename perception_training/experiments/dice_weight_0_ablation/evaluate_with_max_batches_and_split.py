"""
Full validation evaluation for a TerrainSegModel checkpoint: overall mIoU,
hazard vs non-hazard mIoU, per-class IoU/recall, ECE, and a reliability
diagram -- the same metric set the handover's baseline/weighted tables used,
so a new checkpoint's numbers drop straight into that comparison table.

Streaming/accumulator-based throughout (confusion counts + binned ECE sums
updated per-batch) rather than concatenating full per-pixel probability
tensors across the whole val set, which would be far too much memory at
512x512 x 20 classes over ~1000 val images.
"""
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import RellisDataset
from model import TerrainSegModel
from checkpointing import load_checkpoint
from preprocessing import build_class_remap


def evaluate(checkpoint_path, dataset_config_path, classes_config_path, training_config_path,
             device=None, reliability_diagram_path=None, num_ece_bins=15, max_batches=None, split="val"):
    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)
    with open(classes_config_path) as f:
        classes_cfg = yaml.safe_load(f)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = classes_cfg["num_classes"]
    ignore_index = classes_cfg.get("ignore_index", 0)

    remap, _ = build_class_remap(classes_config_path)
    names_by_contig = {remap[c["index"]]: c["name"] for c in classes_cfg["classes"]}
    hazard_contig_ids = {remap[c["index"]] for c in classes_cfg["classes"] if c.get("hazardous")}

    val_ds = RellisDataset(dataset_config_path, split=split,
                            classes_config_path=classes_config_path,
                            training_config_path=training_config_path)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                             num_workers=train_cfg["num_workers"])

    model = TerrainSegModel(num_classes=num_classes).to(device)
    load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)
    model.eval()

    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    gt_pixel_count = torch.zeros(num_classes, dtype=torch.float64)
    correct_pixel_count = torch.zeros(num_classes, dtype=torch.float64)

    bin_edges = torch.linspace(0, 1, num_ece_bins + 1)
    bin_count = torch.zeros(num_ece_bins, dtype=torch.float64)
    bin_correct_sum = torch.zeros(num_ece_bins, dtype=torch.float64)
    bin_conf_sum = torch.zeros(num_ece_bins, dtype=torch.float64)

    num_batches = 0
    with torch.no_grad():
        for images, masks in val_loader:
            if max_batches is not None and num_batches >= max_batches:
                break
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confidences, preds = probs.max(dim=1)

            valid = masks != ignore_index

            for c in range(num_classes):
                pred_c = (preds == c) & valid
                gt_c = (masks == c) & valid
                inter = (pred_c & gt_c).sum().item()
                intersection[c] += inter
                union[c] += (pred_c | gt_c).sum().item()
                gt_pixel_count[c] += gt_c.sum().item()
                correct_pixel_count[c] += inter  # correct predictions of class c = intersection

            correct = (preds == masks).float()
            conf_valid = confidences[valid].cpu()
            correct_valid = correct[valid].cpu()
            for i in range(num_ece_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                in_bin = (conf_valid > lo) & (conf_valid <= hi)
                cnt = in_bin.sum().item()
                if cnt == 0:
                    continue
                bin_count[i] += cnt
                bin_correct_sum[i] += correct_valid[in_bin].sum().item()
                bin_conf_sum[i] += conf_valid[in_bin].sum().item()

            num_batches += 1

    iou_per_class = torch.where(union > 0, intersection / union.clamp(min=1),
                                 torch.full_like(union, float("nan")))
    recall_per_class = torch.where(gt_pixel_count > 0, correct_pixel_count / gt_pixel_count.clamp(min=1),
                                    torch.full_like(gt_pixel_count, float("nan")))

    present_ids = [c for c in range(num_classes) if gt_pixel_count[c] > 0 and c != ignore_index]
    hazard_ids_present = [c for c in present_ids if c in hazard_contig_ids]
    nonhazard_ids_present = [c for c in present_ids if c not in hazard_contig_ids]

    overall_miou = float(np.nanmean([iou_per_class[c].item() for c in present_ids])) if present_ids else float("nan")
    hazard_miou = float(np.nanmean([iou_per_class[c].item() for c in hazard_ids_present])) if hazard_ids_present else float("nan")
    nonhazard_miou = float(np.nanmean([iou_per_class[c].item() for c in nonhazard_ids_present])) if nonhazard_ids_present else float("nan")

    total_binned = bin_count.sum().item()
    ece = 0.0
    bin_acc, bin_conf = [], []
    for i in range(num_ece_bins):
        if bin_count[i] == 0:
            bin_acc.append(0.0)
            bin_conf.append(((bin_edges[i] + bin_edges[i + 1]) / 2).item())
            continue
        acc = (bin_correct_sum[i] / bin_count[i]).item()
        conf = (bin_conf_sum[i] / bin_count[i]).item()
        ece += (bin_count[i].item() / total_binned) * abs(acc - conf)
        bin_acc.append(acc)
        bin_conf.append(conf)

    print(f"Evaluated {num_batches} val batches")
    print(f"Overall mIoU: {overall_miou:.4f} (n={len(present_ids)})")
    print(f"Hazardous mIoU: {hazard_miou:.4f} (n={len(hazard_ids_present)})")
    print(f"Non-hazardous mIoU: {nonhazard_miou:.4f} (n={len(nonhazard_ids_present)})")
    print(f"ECE: {ece:.4f}")
    print()
    print(f"{'class':10s} {'IoU':>8s} {'recall':>8s} {'GT px':>12s} {'hazard':>7s}")
    for c in range(num_classes):
        if c == ignore_index:
            continue
        name = names_by_contig.get(c, f"idx{c}")
        haz = "yes" if c in hazard_contig_ids else ""
        print(f"{name:10s} {iou_per_class[c].item():8.4f} {recall_per_class[c].item():8.4f} "
              f"{int(gt_pixel_count[c].item()):12d} {haz:>7s}")

    dead = [names_by_contig.get(c, f"idx{c}") for c in present_ids
            if iou_per_class[c].item() == 0.0 and recall_per_class[c].item() == 0.0]
    print(f"\nDead classes (IoU=0, recall=0, GT support>0): {dead}")

    if reliability_diagram_path:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.bar(bin_edges[:-1].numpy(), bin_acc, width=1.0 / num_ece_bins, align="edge",
               edgecolor="black", alpha=0.7, label="Accuracy")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability diagram (ECE={ece:.4f})")
        ax.legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(reliability_diagram_path) or ".", exist_ok=True)
        fig.savefig(reliability_diagram_path, dpi=150)
        plt.close(fig)
        print(f"\nReliability diagram saved to {reliability_diagram_path}")

    return {
        "overall_miou": overall_miou,
        "hazard_miou": hazard_miou,
        "nonhazard_miou": nonhazard_miou,
        "ece": ece,
        "iou_per_class": {names_by_contig.get(c, f"idx{c}"): iou_per_class[c].item()
                           for c in range(num_classes) if c != ignore_index},
        "recall_per_class": {names_by_contig.get(c, f"idx{c}"): recall_per_class[c].item()
                              for c in range(num_classes) if c != ignore_index},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a TerrainSegModel checkpoint: mIoU, hazard mIoU, ECE, per-class IoU."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-config", default="../configs/dataset.yaml")
    parser.add_argument("--classes-config", default="../configs/classes.yaml")
    parser.add_argument("--training-config", default="../configs/training.yaml")
    parser.add_argument("--reliability-diagram", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        classes_config_path=args.classes_config,
        training_config_path=args.training_config,
        reliability_diagram_path=args.reliability_diagram,
        max_batches=args.max_batches,
        split=args.split,
    )
