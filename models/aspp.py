import torch
import torch.nn as nn


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels=256):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=6,
                dilation=6,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=12,
                dilation=12,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv18 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=18,
                dilation=18,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(

            nn.Conv2d(
                out_channels * 4,
                out_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x1 = self.conv1(x)

        x2 = self.conv6(x)

        x3 = self.conv12(x)

        x4 = self.conv18(x)

        x = torch.cat(
            [
                x1,
                x2,
                x3,
                x4,
            ],
            dim=1,
        )

        x = self.project(x)

        return x