"""
Frozen inference module for Copernicus-Nav terrain segmentation.
Depends only on existing repo modules: model.py, uncertainty.py, export.py.
"""

import torch

from src.model import TerrainSegModel
from src.uncertainty import mc_dropout_predict
from src.export import save_segmentation_with_uncertainty


def load_model_for_inference(checkpoint_path, num_classes=20, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = TerrainSegModel(num_classes=num_classes)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict(
    model,
    images,
    device="cpu",
    with_uncertainty=False,
    num_passes=10,
):
    images = images.to(device)

    if not with_uncertainty:
        logits = model(images)
        return logits.argmax(dim=1)

    mean_probs, entropy = mc_dropout_predict(
        model,
        images,
        num_passes=num_passes,
    )

    pred_masks = mean_probs.argmax(dim=1)

    return pred_masks, entropy


def run_inference_and_export(
    model,
    images,
    sample_ids,
    output_dir,
    device="cpu",
    with_uncertainty=True,
    num_passes=10,
):
    if with_uncertainty:
        pred_masks, entropy = predict(
            model,
            images,
            device=device,
            with_uncertainty=True,
            num_passes=num_passes,
        )
    else:
        pred_masks = predict(
            model,
            images,
            device=device,
            with_uncertainty=False,
        )
        entropy = None

    assert len(sample_ids) == pred_masks.shape[0], (
        f"{len(sample_ids)} sample_ids for batch of {pred_masks.shape[0]}"
    )
    results = []

    for i, sample_id in enumerate(sample_ids):
        seg_path, unc_path = save_segmentation_with_uncertainty(
            seg_mask=pred_masks[i],
            entropy_map=entropy[i] if entropy is not None else None,
            output_dir=output_dir,
            sample_id=sample_id,
        )

        results.append(
            {
                "sample_id": sample_id,
                "segmentation_path": seg_path,
                "uncertainty_path": unc_path,
            }
        )

    return results
