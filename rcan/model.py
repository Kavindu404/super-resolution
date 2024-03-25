import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class UNetSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64, upscale_factor=4):
        super(UNetSR, self).__init__()
        self.upscale_factor = upscale_factor

        self.encoder_blocks = nn.ModuleList([
            ConvBlock(in_channels, num_filters),
            ConvBlock(num_filters, num_filters * 2),
            ConvBlock(num_filters * 2, num_filters * 4),
            ConvBlock(num_filters * 4, num_filters * 8)
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_bottle = ConvBlock(num_filters * 8, num_filters * 16)
        self.res_block = ResidualBlock(num_filters * 16)

        self.upconv_blocks = nn.ModuleList([
            PixelShuffleBlock(num_filters * 16, num_filters * 8, upscale_factor=2),
            PixelShuffleBlock(num_filters * 8, num_filters * 4, upscale_factor=2),
            PixelShuffleBlock(num_filters * 4, num_filters * 2, upscale_factor=2),
            PixelShuffleBlock(num_filters * 2, num_filters, upscale_factor=2)
        ])
        self.att_blocks = nn.ModuleList([
            SpatialAttention(kernel_size=7),
            SpatialAttention(kernel_size=7),
            SpatialAttention(kernel_size=7),
            SpatialAttention(kernel_size=7)
        ])
        self.conv_blocks = nn.ModuleList([
            ConvBlock(num_filters * 16, num_filters * 8),
            ConvBlock(num_filters * 8, num_filters * 4),
            ConvBlock(num_filters * 4, num_filters * 2),
            ConvBlock(num_filters * 2, num_filters)
        ])
        self.upscale_layer = PixelShuffleBlock(num_filters, num_filters, upscale_factor=2)
        self.conv_output = nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        encoder_features = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_features.append(x)
            x = self.maxpool(x)

        x = self.conv_bottle(x)
        x = self.res_block(x)

        for upconv_block, att_block, conv_block in zip(self.upconv_blocks, self.att_blocks, self.conv_blocks):
            x = upconv_block(x)
            encoder_feature = encoder_features.pop()
            encoder_feature = att_block(encoder_feature) * encoder_feature
            x = torch.cat([x, encoder_feature], dim=1)
            x = conv_block(x)
        x = self.upscale_layer(x)
        output = self.conv_output(x)

        return output