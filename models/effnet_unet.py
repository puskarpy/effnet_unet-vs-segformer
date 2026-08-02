import torch
import torch.nn as nn
import torchvision.models as models

from .aspp import ASPP


# --------------------------------------------------
# Conv Block
# --------------------------------------------------

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

        )

    def forward(self, x):

        return self.block(x)


# --------------------------------------------------
# Attention Gate
# --------------------------------------------------

class AttentionGate(nn.Module):

    def __init__(

        self,

        F_g,

        F_l,

        F_int,

    ):

        super().__init__()

        self.W_g = nn.Sequential(

            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(F_int),

        )

        self.W_x = nn.Sequential(

            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(F_int),

        )

        self.psi = nn.Sequential(

            nn.Conv2d(
                F_int,
                1,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(1),

            nn.Sigmoid(),

        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)

        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)

        psi = self.psi(psi)

        return x * psi


# --------------------------------------------------
# Decoder Block
# --------------------------------------------------

class AttentionDecoderBlock(nn.Module):

    def __init__(

        self,

        in_channels,

        skip_channels,

        out_channels,

    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(

            in_channels,

            out_channels,

            kernel_size=2,

            stride=2,

        )

        self.attention = AttentionGate(

            F_g=out_channels,

            F_l=skip_channels,

            F_int=out_channels,

        )

        self.conv = ConvBlock(

            out_channels + skip_channels,

            out_channels,

        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:

            x = nn.functional.interpolate(

                x,

                size=skip.shape[2:],

                mode="bilinear",

                align_corners=False,

            )

        skip = self.attention(x, skip)

        x = torch.cat(

            [

                x,

                skip,

            ],

            dim=1,

        )

        x = self.conv(x)

        return x


# --------------------------------------------------
# Encoder
# --------------------------------------------------

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Learnable adapter: 4 MRI channels -> 3 RGB channels
        self.input_adapter = nn.Conv2d(
            in_channels=4,
            out_channels=3,
            kernel_size=1,
            bias=False,
        )

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        self.features = backbone.features

    def forward(self, x):

        x = self.input_adapter(x)

        skips = []

        for idx, layer in enumerate(self.features):

            x = layer(x)

            if idx == 2:
                skip1 = x          # 24 x 56 x 56

            elif idx == 3:
                skip2 = x          # 40 x 28 x 28

            elif idx == 4:
                skip3 = x          # 80 x 14 x 14

            elif idx == 6:
                skip4 = x          # 192 x 7 x 7

        bottleneck = x             # 1280 x 7 x 7

        skips = [

            skip1,

            skip2,

            skip3,

            skip4,

        ]

        return skips, bottleneck


# --------------------------------------------------
# EfficientNet Attention U-Net
# --------------------------------------------------

class EfficientNetUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()

        self.aspp = ASPP(
            in_channels=1280,
            out_channels=256,
        )

        # Fuse ASPP output with Skip4 (7x7)
        self.skip4_fuse = nn.Sequential(

            nn.Conv2d(
                256 + 192,
                256,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

        )

        # Decoder

        self.decoder3 = AttentionDecoderBlock(
            in_channels=256,
            skip_channels=80,
            out_channels=128,
        )

        self.decoder2 = AttentionDecoderBlock(
            in_channels=128,
            skip_channels=40,
            out_channels=64,
        )

        self.decoder1 = AttentionDecoderBlock(
            in_channels=64,
            skip_channels=24,
            out_channels=32,
        )

        # Final Upsampling

        self.final_up = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2,
        )

        self.final_conv = ConvBlock(
            16,
            16,
        )

        self.output = nn.Conv2d(
            16,
            1,
            kernel_size=1,
        )

    def forward(self, x):

        skips, bottleneck = self.encoder(x)

        skip1, skip2, skip3, skip4 = skips

        # ASPP
        x = self.aspp(bottleneck)

        # Fuse Skip4 (7×7) with ASPP output
        x = torch.cat(
            [
                x,
                skip4,
            ],
            dim=1,
        )

        x = self.skip4_fuse(x)

        # Decoder
        x = self.decoder3(
            x,
            skip3,
        )

        x = self.decoder2(
            x,
            skip2,
        )

        x = self.decoder1(
            x,
            skip1,
        )

        # Final Upsampling (56 -> 112)
        x = self.final_up(x)

        x = self.final_conv(x)

        # Final Upsampling (112 -> 224)
        x = nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.output(x)

        x = torch.sigmoid(x)

        return x

# --------------------------------------------------
# Builder
# --------------------------------------------------

def build_effnet_unet():

    return EfficientNetUNet()


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    model = build_effnet_unet()

    x = torch.randn(
        2,
        4,
        224,
        224,
    )

    y = model(x)

    print(model)

    print("\nInput :", x.shape)

    print("Output:", y.shape)