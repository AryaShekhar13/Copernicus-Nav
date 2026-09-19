"""
Full raw-image segmentation + uncertainty pipeline for Copernicus-Nav.

This is the piece that was missing: inference.py operates on already-
preprocessed batched tensors. This module takes a raw image on disk,
does the exact preprocessing used in training, runs the model, and
produces a matched segmentation + uncertainty output pair.

Uncertainty method: MC-Dropout (via src.uncertainty.mc_dropout_predict).
Per architecture.md section 9, MC-Dropout/ensemble uncertainty is scoped
to run on Kaggle (GPU) -- this module is that offline/training-side path,
NOT the live ONNX/CPU deployment path. The deployed ONNX model is a
single frozen forward pass and does not support MC-Dropout; see the
project handover notes for that tradeoff.

Depends only on existing repo modules -- no logic is duplicated:
    src/preprocessing.py -> resize + normalize (same as training)
    src/inference.py     -> load_model_for_inference
    src/uncertainty.py   -> mc_dropout_predict (via inference.predict)
    src/export.py        -> save_segmentation_with_uncertainty
"""

import cv2
import yaml
import numpy as np
import torch

from src.preprocessing import normalize_image, build_class_remap
from src.inference import load_model_for_inference, predict
from src.export import save_segmentation_with_uncertainty


def load_raw_image(image_path, target_size):
    """
    Loads an image from disk and resizes it to the model's expected input
    size. target_size is (H, W), matching training.yaml's input_size.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = target_size
    image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return image_resized


def preprocess_image(image_rgb, mean, std):
    """
    Normalizes an already-resized RGB image (H, W, 3) uint8 array and
    converts it to a model-ready tensor [1, 3, H, W].
    """
    normalized = normalize_image(image_rgb, mean=mean, std=std)  # (H, W, 3) float32
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
    return tensor


def run_segmentation_pipeline(image_path, checkpoint_path, training_config_path,
                               classes_yaml_path, output_dir, sample_id,
                               device="cpu", num_passes=10):
    """
    End-to-end: raw image path -> preprocess -> model -> MC-Dropout
    segmentation + uncertainty -> saved output pair.

    Returns (seg_path, uncertainty_path, pred_mask, entropy_map).
    pred_mask/entropy_map are returned as numpy arrays for any immediate
    in-notebook use (e.g. visualization) without re-reading the saved files.
    """
    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)

    input_size = tuple(train_cfg["input_size"])       # (H, W)
    mean = tuple(train_cfg["normalize_mean"])
    std = tuple(train_cfg["normalize_std"])

    with open(classes_yaml_path) as f:
        classes_cfg = yaml.safe_load(f)
    num_classes = classes_cfg["num_classes"]

    # 1. Load + preprocess the raw image exactly as training did
    image_resized = load_raw_image(image_path, target_size=input_size)
    image_tensor = preprocess_image(image_resized, mean=mean, std=std)

    # 2. Load model (raw checkpoint, model_state_dict only, CPU-safe)
    model = load_model_for_inference(checkpoint_path, num_classes=num_classes, device=device)

    # 3. Run MC-Dropout: segmentation + uncertainty in one call
    pred_masks, entropy = predict(
        model, image_tensor, device=device,
        with_uncertainty=True, num_passes=num_passes,
    )
    pred_mask = pred_masks[0]   # [H, W], contiguous class indices (0..num_classes-1)
    entropy_map = entropy[0]    # [H, W], float

    # 4. Save the matched pair via the existing export contract
    seg_path, unc_path = save_segmentation_with_uncertainty(
        seg_mask=pred_mask,
        entropy_map=entropy_map,
        output_dir=output_dir,
        sample_id=sample_id,
    )

    return seg_path, unc_path, pred_mask.cpu().numpy(), entropy_map.cpu().numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full segmentation+uncertainty pipeline on one image.")
    parser.add_argument("--image", required=True, help="Path to a raw input image")
    parser.add_argument("--checkpoint", default="/kaggle/working/checkpoints/best.pt")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--classes", default="configs/classes.yaml")
    parser.add_argument("--output-dir", default="exported_models/pipeline_outputs")
    parser.add_argument("--sample-id", default="pipeline_sample")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-passes", type=int, default=10)
    args = parser.parse_args()

    seg_path, unc_path, _, _ = run_segmentation_pipeline(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        training_config_path=args.training_config,
        classes_yaml_path=args.classes,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        device=args.device,
        num_passes=args.num_passes,
    )
    print(f"Segmentation saved to: {seg_path}")
    print(f"Uncertainty saved to: {unc_path}")
