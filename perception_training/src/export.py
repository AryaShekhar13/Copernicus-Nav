import os
import numpy as np
from PIL import Image


def save_segmentation_with_uncertainty(seg_mask, entropy_map, output_dir, sample_id):
    """
    Saves segmentation output + uncertainty map as a matched pair, per the
    team_division.md interface contract:
      - Segmentation output: per-pixel class index array (.png or numpy array)
      - Uncertainty output: per-pixel float map, same resolution, saved alongside it

    seg_mask: [H, W] numpy array or tensor of class indices (0-19)
    entropy_map: [H, W] numpy array or tensor of float entropy values
    output_dir: directory to save into
    sample_id: string/int identifier used for both filenames (keeps the pair linked)

    Returns (seg_path, uncertainty_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(seg_mask, "cpu"):
        seg_mask = seg_mask.cpu().numpy()
    if entropy_map is not None and hasattr(entropy_map, "cpu"):
        entropy_map = entropy_map.cpu().numpy()

    seg_path = os.path.join(output_dir, f"{sample_id}_seg.png")
    Image.fromarray(seg_mask.astype(np.uint8)).save(seg_path)

    # entropy is float, not PNG-representable directly -- save as .npy alongside
    unc_path = None
    if entropy_map is not None:
        unc_path = os.path.join(output_dir, f"{sample_id}_uncertainty.npy")
        np.save(unc_path, entropy_map.astype(np.float32))

    return seg_path, unc_path
