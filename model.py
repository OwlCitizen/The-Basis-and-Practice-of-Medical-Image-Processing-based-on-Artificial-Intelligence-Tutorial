#model.py

import torch.nn.functional as F

from modules import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, emb_dim = 64, num_layer = 4, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.emb_dim = emb_dim
        self.num_layer = num_layer

        self.inc = DoubleConv(n_channels, emb_dim)
        self.down = nn.ModuleList()
        self.up  = nn.ModuleList() 
        for i in range(num_layer):
            self.down.append(Down(emb_dim*(2**i), emb_dim*(2**(i+1))))
        for i in range(num_layer):
            self.up.append(Up(emb_dim*(2**(4-i)), emb_dim*(2**(4-i-1))))
        self.outc = OutConv(emb_dim, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down[0](x1)
        x3 = self.down[1](x2)
        x4 = self.down[2](x3)
        x5 = self.down[3](x4)
        x = self.up[0](x5, x4)
        x = self.up[1](x, x3)
        x = self.up[2](x, x2)
        x = self.up[3](x, x1)
        logits = self.outc(x)
        return logits
