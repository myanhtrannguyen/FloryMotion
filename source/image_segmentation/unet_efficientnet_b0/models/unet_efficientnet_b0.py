from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(channels, squeeze_channels, 1)
        self.expand = nn.Conv2d(squeeze_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.reduce(scale), inplace=True)
        scale = torch.sigmoid(self.expand(scale))
        return x * scale


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_channels, kernel_size=1))

        layers.extend(
            [
                ConvBNAct(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=hidden_channels,
                ),
                SqueezeExcite(hidden_channels, _make_divisible(in_channels * se_ratio)),
                ConvBNAct(hidden_channels, out_channels, kernel_size=1, activation=False),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class EfficientNetB0Encoder(nn.Module):
    """EfficientNet-B0 encoder stages used as U-Net skip features."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1))
        self.stage2 = nn.Sequential(
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
        )
        self.stage3 = nn.Sequential(
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
        )
        self.stage4 = nn.Sequential(
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
        )
        self.stage5 = nn.Sequential(
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x1, x2, x3, x4, x5


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNAct(in_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetEfficientNetB0(nn.Module):
    """
    U-Net baseline with an EfficientNet-B0 encoder.

    This follows the EfficientUnet-PyTorch idea: EfficientNet features form the
    encoder, and the decoder upsamples while concatenating encoder skips.
    """

    def __init__(self, num_classes: int = 1, concat_input: bool = True) -> None:
        super().__init__()
        self.concat_input = concat_input
        self.encoder = EfficientNetB0Encoder()
        self.center = DoubleConv(320, 256)
        self.up4 = UNetDecoderBlock(256, 112, 128)
        self.up3 = UNetDecoderBlock(128, 40, 64)
        self.up2 = UNetDecoderBlock(64, 24, 48)
        self.up1 = UNetDecoderBlock(48, 16, 32)
        self.input_refine = DoubleConv(32 + 3, 32) if concat_input else DoubleConv(32, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_image = x
        input_size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.encoder(x)
        x = self.center(f5)
        x = self.up4(x, f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        if self.concat_input:
            x = torch.cat([x, input_image], dim=1)
        x = self.input_refine(x)
        return self.final(x)

    def load_encoder_weights(self, checkpoint_path: str | Path, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key.removeprefix("encoder.")] = value
            else:
                encoder_state[key] = value
        self.encoder.load_state_dict(encoder_state, strict=strict)
