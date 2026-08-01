import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SegformerModel


# ------------------------------- #
# Input Adapter
# ------------------------------- #

class InputAdapter(nn.Module):

    def __init__(self):

        super().__init__()

        self.adapter = nn.Conv2d(
            in_channels=4,
            out_channels=3,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):

        return self.adapter(x)

# ------------------------------- #
# MLP
# ------------------------------- #

class MLP(nn.Module):

    def __init__(

        self,

        input_dim,

        embed_dim,

    ):

        super().__init__()

        self.proj = nn.Linear(
            input_dim,
            embed_dim,
        )

    def forward(self, x):

        B, C, H, W = x.shape

        x = x.flatten(2)

        x = x.transpose(1, 2)

        x = self.proj(x)

        x = x.transpose(1, 2)

        x = x.reshape(
            B,
            -1,
            H,
            W,
        )

        return x

# ------------------------------- #
# Encoder
# ------------------------------- #

class MiTEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.adapter = InputAdapter()

        self.encoder = SegformerModel.from_pretrained(
            "nvidia/mit-b0"
        )

    def forward(self, x):

        x = self.adapter(x)

        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
        )

        return outputs.hidden_states


# ------------------------------- #
# Decoder
# ------------------------------- #

class SegFormerDecoder(nn.Module):

    def __init__(

        self,

        embed_dim=256,

        num_classes=1,

    ):

        super().__init__()

        self.linear_c1 = MLP(32, embed_dim)

        self.linear_c2 = MLP(64, embed_dim)

        self.linear_c3 = MLP(160, embed_dim)

        self.linear_c4 = MLP(256, embed_dim)

        self.fuse = nn.Sequential(

            nn.Conv2d(

                embed_dim * 4,

                embed_dim,

                kernel_size=1,

                bias=False,

            ),

            nn.BatchNorm2d(embed_dim),

            nn.ReLU(inplace=True),

        )

        self.dropout = nn.Dropout(0.1)

        self.pred = nn.Conv2d(

            embed_dim,

            num_classes,

            kernel_size=1,

        )

    def forward(

        self,

        c1,

        c2,

        c3,

        c4,

    ):

        c1 = self.linear_c1(c1)

        c2 = self.linear_c2(c2)

        c3 = self.linear_c3(c3)

        c4 = self.linear_c4(c4)

        c2 = F.interpolate(

            c2,

            size=c1.shape[2:],

            mode="bilinear",

            align_corners=False,

        )

        c3 = F.interpolate(

            c3,

            size=c1.shape[2:],

            mode="bilinear",

            align_corners=False,

        )

        c4 = F.interpolate(

            c4,

            size=c1.shape[2:],

            mode="bilinear",

            align_corners=False,

        )

        x = torch.cat(

            [

                c1,

                c2,

                c3,

                c4,

            ],

            dim=1,

        )

        x = self.fuse(x)

        x = self.dropout(x)

        x = self.pred(x)

        return x 


# ------------------------------- #
# Segformer
# ------------------------------- #

class SegFormer(nn.Module):

    def __init__(

        self,

        num_classes=1,

        embed_dim=256,

    ):

        super().__init__()

        self.encoder = MiTEncoder()

        self.decoder = SegFormerDecoder(

            embed_dim=embed_dim,

            num_classes=num_classes,

        )

    def forward(self, x):

        features = self.encoder(x)

        c1, c2, c3, c4 = features

        x = self.decoder(

            c1,

            c2,

            c3,

            c4,

        )

        x = F.interpolate(

            x,

            size=(224, 224),

            mode="bilinear",

            align_corners=False,

        )

        x = torch.sigmoid(x)

        return x   

# ------------------------------- #
# Builder
# ------------------------------- #

def build_segformer():

    return SegFormer()


# test

if __name__ == "__main__":

    model = build_segformer()

    x = torch.randn(

        2,

        4,

        224,

        224,

    )

    y = model(x)

    print(model)

    print()

    print("Input :", x.shape)

    print("Output:", y.shape)