"""
ONNX/CPU counterpart of pipeline.py (the live/deployed path).

raw image -> same preprocessing as training -> ONNX Runtime (single pass)
-> segmentation + softmax-entropy uncertainty -> saved pair via export.py.

Uncertainty is single-pass softmax entropy (risk #13, option d), not MC-Dropout.
"""

import numpy as np
import onnxruntime as ort
import yaml

from src.pipeline import load_raw_image, preprocess_image
from src.export import save_segmentation_with_uncertainty

def load_onnx_session(onnx_path):
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def softmax_entropy(logits):
    """logits: [C, H, W] -> (pred [H, W] uint8, entropy [H, W] float32, in nats)."""
    z = logits - logits.max(axis=0, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=0, keepdims=True)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0)
    return p.argmax(axis=0).astype(np.uint8), entropy.astype(np.float32)

def run_onnx_pipeline(image_path, onnx_path, training_config_path, output_dir,
                      sample_id, session=None):
    """Returns (seg_path, uncertainty_path, pred_mask, entropy_map).
    Pass a preloaded `session` when processing many frames."""
    with open(training_config_path) as f:
        cfg = yaml.safe_load(f)
    size = tuple(cfg["input_size"])
    mean, std = tuple(cfg["normalize_mean"]), tuple(cfg["normalize_std"])

    x = preprocess_image(load_raw_image(image_path, target_size=size), mean=mean, std=std)
    x = (x.numpy() if hasattr(x, "numpy") else np.asarray(x)).astype(np.float32)
    if x.ndim == 3:
        x = x[None]

    session = session or load_onnx_session(onnx_path)
    logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    pred_mask, entropy_map = softmax_entropy(logits)

    seg_path, unc_path = save_segmentation_with_uncertainty(
        seg_mask=pred_mask,
        entropy_map=entropy_map,
        output_dir=output_dir,
        sample_id=sample_id,
    )
    return seg_path, unc_path, pred_mask, entropy_map

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ONNX/CPU segmentation + softmax-entropy pipeline on one image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--onnx", default="/kaggle/working/segmodel_v1_2026-09-19.onnx")
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument("--output-dir", default="/kaggle/working/pipeline_outputs")
    ap.add_argument("--sample-id", default="onnx_sample")
    a = ap.parse_args()

    seg, unc, _, _ = run_onnx_pipeline(a.image, a.onnx, a.training_config, a.output_dir, a.sample_id)
    print("Segmentation saved to:", seg)
    print("Uncertainty saved to:", unc)
