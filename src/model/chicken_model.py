"""
Chicken counting model wrapper using a timm PVT backbone, PFA and MDC head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .modules import PyramidFeatureAggregation, MultiScaleDilatedConv


class ChickenCountingModel(nn.Module):
    """
    Full chicken counting model.
    """

    def __init__(self, backbone_name: str = "pvt_v2_b2", pretrained: bool = False):
        super(ChickenCountingModel, self).__init__()

        # create backbone; use features_only to get pyramid features
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)

        # get channel sizes for each feature stage
        feature_info = self.backbone.feature_info.channels()  # list[int]

        self.pfa = PyramidFeatureAggregation(in_channels_list=feature_info, out_channels=256)
        self.mdc = MultiScaleDilatedConv(in_channels=256, mid_channels=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract pyramid features
        features = self.backbone(x)
        aggregated = self.pfa(features)
        density = self.mdc(aggregated)
        # upsample to input size
        density = F.interpolate(density, size=x.shape[2:], mode="bilinear", align_corners=False)
        return density

    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        density = self.forward(x)
        count = density.view(density.size(0), -1).sum(dim=1)
        return count


__all__ = ["ChickenCountingModel"]
