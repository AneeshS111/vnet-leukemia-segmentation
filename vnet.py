import torch
import torch.nn as nn

class VNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(VNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class VNet2D(nn.Module):
    def __init__(self):
        super(VNet2D, self).__init__()
        self.enc1 = VNetBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = VNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = VNetBlock(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = VNetBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = VNetBlock(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))
