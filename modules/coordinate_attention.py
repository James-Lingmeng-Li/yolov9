import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateAttention(nn.Module):
    """Coordinate Attention Module for enhancing spatial feature representations."""
    def __init__(self, channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.channels = channels
        mid_channels = channels // reduction
        self.conv1 = nn.Conv2d(2 * channels, mid_channels, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=True)

    def forward(self, x):
        """Forward pass of the Coordinate Attention."""
        n, _, h, w = x.size()
        x_h = F.adaptive_avg_pool2d(x, (1, w))
        x_w = F.adaptive_avg_pool2d(x, (h, 1)).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_h * a_w