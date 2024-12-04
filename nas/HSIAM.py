import torch
import torch.nn as nn


class MultiScaleSpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(MultiScaleSpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.conv1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.conv3 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=5, padding=2, bias=False)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction_ratio * 3, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x1.size:", x.size())
        avg = self.avg_pool(x)
        y1 = self.conv1(avg)
        y3 = self.conv3(avg)
        y5 = self.conv5(avg)
        y = torch.cat([y1, y3, y5], dim=1)
        y = self.fc(y)
        return self.sigmoid(y) * x


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # print("x2.size:", x.size())
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiScaleSpatialAttention(nn.Module):
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        self.conv1 = DepthwiseSeparableConv3d(2, 1, 3, 1)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(1, 1, kernel_size=5, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, spectral_attention):
        # print("x4.size:", x.size())
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y = y1 + y2 + y3
        return self.sigmoid(y) * spectral_attention


class EnhancedChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out * x


class HSIAM(nn.Module):
    def __init__(self, a, b, c, in_channels):
        super(HSIAM, self).__init__()
        self.spectral_attention = MultiScaleSpectralAttention(in_channels)
        self.spatial_attention = MultiScaleSpatialAttention()
        self.channel_attention = EnhancedChannelAttention(in_channels)
        self.fc_spectral = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        self.fc_spatial = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        self.fc_channel = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        spectral_attention = self.spectral_attention(x)
        spatial_attention = self.spatial_attention(x, spectral_attention)
        channel_attention = self.channel_attention(x)
        spectral_out = self.fc_spectral(spectral_attention)
        spatial_out = self.fc_spatial(spatial_attention)
        channel_out = self.fc_channel(channel_attention)
        output = self.a * spectral_out + self.b * spatial_out + self.c * channel_out
        return output

