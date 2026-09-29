"""Model registry for the pretrained-vs-baseline comparison.

Contract: every model maps [B,3,H,W] -> [B,C,H,W] logits at INPUT resolution, so the
existing loss / evaluation code is used unchanged for all of them.

  terrainseg          current model (U-Net + EfficientNet-lite0). NOTE: its encoder is
                      ImageNet-pretrained by default, so it is NOT trained from scratch.
  terrainseg_scratch  same architecture, random init (encoder_weights=None) - optional arm
                      that shows what ImageNet pretraining is worth.
  segformer_b0/b1     HF SegFormer, ImageNet-1k MiT encoder (nvidia/mit-b0 / mit-b1),
                      freshly initialised decode head sized to num_classes.
  deeplabv3plus       smp DeepLabV3+ with a lightweight ImageNet encoder.

Pretrained weights are used for initialisation only; nothing is frozen.
"""
import torch.nn as nn
import torch.nn.functional as F

from model import TerrainSegModel

SEGFORMER_DEFAULT_IDS = {
    "segformer_b0": "nvidia/mit-b0",
    "segformer_b1": "nvidia/mit-b1",
}
DEFAULT_DL_ENCODER = "tu-mobilenetv3_large_100"
MODEL_NAMES = ["terrainseg", "terrainseg_scratch", "segformer_b0", "segformer_b1", "deeplabv3plus"]


class SegFormerSeg(nn.Module):
    def __init__(self, num_classes, hf_id, pretrained=True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            # ignore_mismatched_sizes lets segmentation checkpoints (ADE/Cityscapes) be used
            # too: their 150/19-class classifier is dropped and re-initialised.
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                hf_id, num_labels=num_classes, ignore_mismatched_sizes=True)
        else:  # architecture only (used when a fine-tuned state_dict is loaded afterwards)
            self.net = SegformerForSemanticSegmentation(
                SegformerConfig.from_pretrained(hf_id, num_labels=num_classes))
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.net(pixel_values=x).logits           # [B,C,H/4,W/4]
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def enable_mc_dropout(self):  # API parity with TerrainSegModel (offline MC-dropout path)
        self.net.decode_head.dropout.train()


class DeepLabV3PlusSeg(nn.Module):
    def __init__(self, num_classes, encoder_name, pretrained=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.DeepLabV3Plus(encoder_name=encoder_name,
                                     encoder_weights="imagenet" if pretrained else None,
                                     in_channels=3, classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.net(x)

    def enable_mc_dropout(self):  # no dropout layer in this variant
        pass


def build_model(name, num_classes, pretrained=True, hf_id=None, dl_encoder=None):
    if name == "terrainseg":
        return TerrainSegModel(num_classes=num_classes,
                               encoder_weights="imagenet" if pretrained else None)
    if name == "terrainseg_scratch":
        return TerrainSegModel(num_classes=num_classes, encoder_weights=None)
    if name in SEGFORMER_DEFAULT_IDS:
        return SegFormerSeg(num_classes, hf_id or SEGFORMER_DEFAULT_IDS[name], pretrained)
    if name == "deeplabv3plus":
        return DeepLabV3PlusSeg(num_classes, dl_encoder or DEFAULT_DL_ENCODER, pretrained)
    raise ValueError(f"unknown model '{name}'. choices: {MODEL_NAMES}")
