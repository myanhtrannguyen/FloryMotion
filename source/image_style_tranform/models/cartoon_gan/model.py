from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CartoonGenerator(nn.Module):
    def __init__(self, num_residual_blocks: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        channels = base_channels
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, channels, kernel_size=7, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
        ]

        in_channels = channels
        for _ in range(2):
            out_channels = in_channels * 2
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(in_channels))

        for _ in range(2):
            out_channels = in_channels // 2
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                    nn.InstanceNorm2d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, 3, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int, stride: int, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(3, base_channels, stride=2, norm=False),
            *block(base_channels, base_channels * 2, stride=2),
            *block(base_channels * 2, base_channels * 4, stride=2),
            *block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("InstanceNorm") != -1 and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def build_cartoon_gan(
    generator_blocks: int = 4,
    generator_channels: int = 64,
    discriminator_channels: int = 64,
) -> tuple[CartoonGenerator, PatchDiscriminator]:
    generator = CartoonGenerator(
        num_residual_blocks=generator_blocks,
        base_channels=generator_channels,
    )
    discriminator = PatchDiscriminator(base_channels=discriminator_channels)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    return generator, discriminator
