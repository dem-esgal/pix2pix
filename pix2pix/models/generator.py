import torch
import torch.nn as nn
from torch.nn import ModuleList


class Pix2PixBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False, dropout_rate=0.2):
        super(Pix2PixBlock, self).__init__()
        conv2d = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") \
            if down \
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

        self.conv = nn.Sequential(
            conv2d,
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, kernel_size=4, stride=2):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size, stride, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        down_layers = []
        down_in_multipliers = [1, 2, 4, 8, 8, 8]
        down_out_multipliers = [2, 4, 8, 8, 8, 8]
        for i in range(6):
            down_layers.append(Pix2PixBlock(
                features * down_in_multipliers[i],
                features * down_out_multipliers[i],
                down=True,
                act="leaky",
                use_dropout=False))

        self.down_layers = ModuleList(down_layers)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        up_layers = []
        up_dropouts = [True, True, True, False, False, False, False]
        up_in_multipliers = [8, 16, 16, 16, 16, 8, 4]
        up_out_multipliers = [8, 8, 8, 8, 4, 2, 1]
        for i in range(7):
            up_layers.append(Pix2PixBlock(
                features * up_in_multipliers[i],
                features * up_out_multipliers[i],
                down=False, act="relu",
                use_dropout=up_dropouts[i]
            ))

        self.up_layers = ModuleList(up_layers)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        down_x = [self.initial_down(x)]
        for layer in self.down_layers:
            down_x.append(layer(down_x[-1]))

        bottleneck = self.bottleneck(down_x[-1])

        up_x = [self.up_layers[0](bottleneck)]
        down_x = list(reversed(down_x))
        for i, layer in enumerate(self.up_layers[1:]):
            up_x.append(layer(torch.cat([up_x[-1], down_x[i]], 1)))

        return self.final_up(torch.cat([up_x[-1], down_x[-1]], 1))
