from __future__ import annotations
from pathlib import Path
from typing import List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """BasicBlock cho ResNet-34."""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet34Encoder(nn.Module):
    """ResNet-34 Encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        f1 = self.relu(x)        # C/64, H/2, W/2
        
        x = self.maxpool(f1)
        f2 = self.layer1(x)      # C/64, H/4, W/4
        f3 = self.layer2(f2)     # C/128, H/8, W/8
        f4 = self.layer3(f3)     # C/256, H/16, W/16
        f5 = self.layer4(f4)     # C/512, H/32, W/32
        
        return f1, f2, f3, f4, f5


class DoubleConv(nn.Module):
    """Block 2 lớp Conv2d liên tiếp."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNetPlusPlusBlock(nn.Module):
    """
    Block (Node) đặc thù cho kiến trúc U-Net++ với Dense Skip Connections.
    """
    def __init__(self, prev_channels: int, skip_channels_list: List[int], out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=2, stride=2)
        total_in_channels = out_channels + sum(skip_channels_list)
        self.conv = DoubleConv(total_in_channels, out_channels)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skips[0].shape[-2:]:
            x = F.interpolate(x, size=skips[0].shape[-2:], mode="bilinear", align_corners=False)
        
        # Kết hợp Tensor được upsample (x) với TẤT CẢ các Tensors cùng cấp từ trước (skips)
        inputs = skips + [x]
        return self.conv(torch.cat(inputs, dim=1))


class UNetPlusPlusResNet34(nn.Module):
    """
    U-Net++ với Backbone ResNet-34
    """
    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.encoder = ResNet34Encoder()
        
        # === DECODER U-NET++ ===
        # ResNet-34 Channels: f1=64, f2=64, f3=128, f4=256, f5=512

        # Level 3 (Kích thước H/16, W/16)
        self.x3_1 = UNetPlusPlusBlock(prev_channels=512, skip_channels_list=[256], out_channels=256)

        # Level 2 (Kích thước H/8, W/8)
        self.x2_1 = UNetPlusPlusBlock(prev_channels=256, skip_channels_list=[128], out_channels=128)
        self.x2_2 = UNetPlusPlusBlock(prev_channels=256, skip_channels_list=[128, 128], out_channels=128)

        # Level 1 (Kích thước H/4, W/4)
        self.x1_1 = UNetPlusPlusBlock(prev_channels=128, skip_channels_list=[64], out_channels=64)
        self.x1_2 = UNetPlusPlusBlock(prev_channels=128, skip_channels_list=[64, 64], out_channels=64)
        self.x1_3 = UNetPlusPlusBlock(prev_channels=128, skip_channels_list=[64, 64, 64], out_channels=64)

        # Level 0 (Kích thước H/2, W/2)
        self.x0_1 = UNetPlusPlusBlock(prev_channels=64, skip_channels_list=[64], out_channels=64)
        self.x0_2 = UNetPlusPlusBlock(prev_channels=64, skip_channels_list=[64, 64], out_channels=64)
        self.x0_3 = UNetPlusPlusBlock(prev_channels=64, skip_channels_list=[64, 64, 64], out_channels=64)
        self.x0_4 = UNetPlusPlusBlock(prev_channels=64, skip_channels_list=[64, 64, 64, 64], out_channels=64)

        # Upsample cuối cùng đưa về kích thước ảnh gốc (H, W)
        self.up_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            DoubleConv(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        f1, f2, f3, f4, f5 = self.encoder(x)

        # Decoder (Tính toán các Node theo từng cấp độ)
        x3_1 = self.x3_1(f5, [f4])

        x2_1 = self.x2_1(f4, [f3])
        x2_2 = self.x2_2(x3_1, [f3, x2_1])

        x1_1 = self.x1_1(f3, [f2])
        x1_2 = self.x1_2(x2_1, [f2, x1_1])
        x1_3 = self.x1_3(x2_2, [f2, x1_1, x1_2])

        x0_1 = self.x0_1(f2, [f1])
        x0_2 = self.x0_2(x1_1, [f1, x0_1])
        x0_3 = self.x0_3(x1_2, [f1, x0_1, x0_2])
        x0_4 = self.x0_4(x1_3, [f1, x0_1, x0_2, x0_3])

        # Node ngoài cùng x0_4 chứa thông tin tổng hợp tốt nhất
        out = self.up_final(x0_4)
        
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
        return self.final_conv(out)

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