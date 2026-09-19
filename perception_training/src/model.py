import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class TerrainSegModel(nn.Module):
    """Lightweight segmentation model for RELLIS-3D terrain classes, with MC-Dropout
    for per-pixel uncertainty estimation (per team_division.md Workstream 1 spec).

    Backbone: EfficientNet-lite0 (via smp's timm passthrough), ImageNet-pretrained.
    Decoder: U-Net (smp does not ship PIDNet/BiSeNet specifically; U-Net is a standard,
    well-tested lightweight alternative serving the same role).

    Uncertainty: MC-Dropout - a dropout layer is inserted before the segmentation head
    and kept active at inference time (multiple stochastic forward passes) to estimate
    per-pixel predictive uncertainty, per the architecture doc's uncertainty contract.
    """

    def __init__(self, num_classes, encoder_name="timm-tf_efficientnet_lite0",
                 encoder_weights="imagenet", dropout_p=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.backbone = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

        self.mc_dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_output = self.backbone.decoder(features)
        decoder_output = self.mc_dropout(decoder_output)
        logits = self.backbone.segmentation_head(decoder_output)
        return logits

    def enable_mc_dropout(self):
        """Keep dropout active during eval() for MC-Dropout inference passes."""
        for module in self.modules():
            if isinstance(module, nn.Dropout2d):
                module.train()
