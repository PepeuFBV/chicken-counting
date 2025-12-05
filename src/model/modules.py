import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class PyramidFeatureAggregation(nn.Module):
    """
    Simple top-down FPN-like aggregation for pyramid features.

    Expects a list of feature maps [f1, f2, f3, f4] where f1 is highest resolution.
    Produces a single aggregated feature map with `out_channels`.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        # lateral convolutions to unify channels
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        # smooth conv after merge
        self.smooth = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features: [f1, f2, f3, f4] where f1 is smallest? timm features_only returns from low to high resolution
        # we will treat the last entry as highest-level and upsample downwards
        # convert all to same channels
        laterals = [l(f) for l, f in zip(self.laterals, features)]

        # build top-down
        x = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(x, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            x = up + laterals[i]

        out = self.smooth(x)
        return out


class MultiScaleDilatedConv(nn.Module):
    """
    MDC head: applies parallel dilated convs and fuses them to produce a single-channel density map.
    """

    def __init__(self, in_channels: int = 256, mid_channels: int = 128):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1)
        self.d1 = ConvBNReLU(mid_channels, mid_channels, kernel_size=3, padding=1, dilation=1)
        self.d2 = ConvBNReLU(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.d3 = ConvBNReLU(mid_channels, mid_channels, kernel_size=3, padding=3, dilation=3)

        self.fuse = ConvBNReLU(mid_channels * 3, mid_channels, kernel_size=3, padding=1)
        self.reg = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        a = self.d1(x)
        b = self.d2(x)
        c = self.d3(x)
        cat = torch.cat([a, b, c], dim=1)
        f = self.fuse(cat)
        out = self.reg(f)
        # ensure non-negative densities (ReLU)
        out = F.relu(out)
        return out


__all__ = ["PyramidFeatureAggregation", "MultiScaleDilatedConv"]
