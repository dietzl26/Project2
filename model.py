"""Configurable U-Net for segmentation."""

import torch.nn as nn
from unet_simplistic import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_channels=64, depth=4, bilinear=True, dropout=0.0):
        super().__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Encoder: inc + depth Down blocks
        self.inc = DoubleConv(n_channels, base_channels)
        self.downs = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            out_ch = ch * 2 if i < depth - 1 else ch * 2 // factor
            self.downs.append(Down(ch, out_ch))
            ch = out_ch

        # Skip-connection channel sizes: one per encoder level (inc + depth-1 downs)
        skip_chs = [base_channels]
        c = base_channels
        for _ in range(depth - 1):
            skip_chs.append(c * 2)
            c = c * 2

        # Decoder: depth Up blocks
        self.ups = nn.ModuleList()
        for i in range(depth):
            skip_ch = skip_chs[depth - 1 - i]
            if bilinear:
                # prev_ch == skip_ch, so cat input = 2 * skip_ch
                in_ch = ch + skip_ch
                out_ch = skip_ch // 2 if i < depth - 1 else skip_ch
            else:
                # ConvTranspose halves ch to skip_ch, cat gives back ch
                in_ch = ch
                out_ch = skip_ch
            self.ups.append(Up(in_ch, out_ch, bilinear))
            ch = out_ch

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.outc = OutConv(ch, n_classes)

    def forward(self, x):
        xs = [self.inc(x)]
        for down in self.downs:
            xs.append(down(xs[-1]))
        x = xs.pop()  # bottleneck
        if self.dropout is not None:
            x = self.dropout(x)
        for up in self.ups:
            x = up(x, xs.pop())
        return self.outc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
