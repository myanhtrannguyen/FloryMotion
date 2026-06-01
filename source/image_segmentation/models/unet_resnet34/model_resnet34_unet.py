import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """
    Basic convolution block used in the decoder.
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Decoder block:
    Upsample -> Concatenate skip connection -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels
        )

    def forward(self, x, skip=None):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class ResNet34UNet(nn.Module):
    """
    U-Net with ResNet34 encoder.

    Encoder:
        ResNet34 pretrained on ImageNet

    Decoder:
        U-Net style decoder with skip connections

    Output:
        Segmentation mask with num_classes channels
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = models.resnet34(weights=weights)

        # Encoder
        self.input_block = nn.Sequential(
            resnet.conv1,   # output: 64 channels, H/2, W/2
            resnet.bn1,
            resnet.relu
        )

        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1  # 64 channels, H/4
        self.encoder2 = resnet.layer2  # 128 channels, H/8
        self.encoder3 = resnet.layer3  # 256 channels, H/16
        self.encoder4 = resnet.layer4  # 512 channels, H/32

        # Decoder
        self.decoder4 = DecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256
        )

        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128
        )

        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64
        )

        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=64
        )

        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(64, 32)
        )

        self.segmentation_head = nn.Conv2d(
            in_channels=32,
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # Encoder
        x0 = self.input_block(x)      # 64, H/2
        x1 = self.maxpool(x0)         # 64, H/4

        e1 = self.encoder1(x1)        # 64, H/4
        e2 = self.encoder2(e1)        # 128, H/8
        e3 = self.encoder3(e2)        # 256, H/16
        e4 = self.encoder4(e3)        # 512, H/32

        # Decoder
        d4 = self.decoder4(e4, e3)    # 256, H/16
        d3 = self.decoder3(d4, e2)    # 128, H/8
        d2 = self.decoder2(d3, e1)    # 64, H/4
        d1 = self.decoder1(d2, x0)    # 64, H/2

        out = self.final_upsample(d1) # 32, H
        out = self.segmentation_head(out)

        return out