import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, up_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            up_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            up_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x


class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        try:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.encoder = models.resnet34(weights=weights)
        except AttributeError:
            self.encoder = models.resnet34(pretrained=pretrained)

        # Quan trọng: checkpoint không có encoder.fc.weight/bias
        self.encoder.fc = nn.Identity()

        self.up4 = UpBlock(
            in_channels=512,
            skip_channels=256,
            up_channels=256,
            out_channels=256,
        )

        self.up3 = UpBlock(
            in_channels=256,
            skip_channels=128,
            up_channels=128,
            out_channels=128,
        )

        self.up2 = UpBlock(
            in_channels=128,
            skip_channels=64,
            up_channels=64,
            out_channels=64,
        )

        # Quan trọng: checkpoint có up1.up.weight shape [64, 32, 2, 2]
        # nghĩa là ConvTranspose2d(64 -> 32)
        # sau đó concat skip 64 channel => 32 + 64 = 96
        self.up1 = UpBlock(
            in_channels=64,
            skip_channels=64,
            up_channels=32,
            out_channels=64,
        )

        self.up_final = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.final_conv = nn.Sequential(
            DoubleConv(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)

        x1 = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x1)

        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)

        out = self.up_final(d1)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(
                out,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        out = self.final_conv(out)

        return out