import yaml
import numpy as np
import cv2


def build_class_remap(classes_yaml_path):
    """Build sparse-index -> contiguous-index mapping (0..num_classes-1), derived from
    classes.yaml's class list order. Returns (remap_dict, inverse_remap_dict).
    """
    with open(classes_yaml_path) as f:
        cfg = yaml.safe_load(f)

    sparse_indices = [c["index"] for c in cfg["classes"]]
    remap = {sparse_idx: contiguous_idx for contiguous_idx, sparse_idx in enumerate(sparse_indices)}
    inverse_remap = {v: k for k, v in remap.items()}
    return remap, inverse_remap


def remap_mask(mask, remap_dict):
    """Vectorized remap of a sparse-index mask to contiguous indices."""
    out = np.zeros_like(mask, dtype=np.uint8)
    for sparse_idx, contiguous_idx in remap_dict.items():
        out[mask == sparse_idx] = contiguous_idx
    return out


def resize_image_and_mask(image, mask, size):
    """Resize image (bilinear) and mask (nearest-neighbor, to avoid inventing class boundaries)."""
    h, w = size
    image_resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized


def normalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Standard ImageNet normalization, since we'll use an ImageNet-pretrained backbone."""
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = (image - mean) / std
    return image
