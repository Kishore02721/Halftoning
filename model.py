import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (Contracting path)
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck (Bridge)
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (Expanding path)
        self.decoder4 = self.upconv_block(1024 + 512, 512)  # Adjust channels to match concatenation
        self.decoder3 = self.upconv_block(512 + 256, 256)
        self.decoder2 = self.upconv_block(256 + 128, 128)
        self.decoder1 = self.upconv_block(128 + 64, 64)

        # Output layer (1x1 convolution)
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Convolution block consisting of two convolutions followed by ReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Max pooling after every block to down-sample
        )

    def upconv_block(self, in_channels, out_channels):
        """Up-sampling block consisting of a transposed convolution followed by ReLU activation."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # Upsample with stride=2
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder with Skip Connections
        # Resize bottleneck to match enc4's spatial dimensions before concatenation
        bottleneck_resized = F.interpolate(bottleneck, size=enc4.size()[2:], mode='bilinear', align_corners=True)
        dec4 = self.decoder4(torch.cat((bottleneck_resized, enc4), dim=1))  # Concatenate with encoder4

        # Resize dec4 to match enc3's spatial dimensions before concatenation
        dec4_resized = F.interpolate(dec4, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        dec3 = self.decoder3(torch.cat((dec4_resized, enc3), dim=1))  # Concatenate with encoder3

        # Resize dec3 to match enc2's spatial dimensions before concatenation
        dec3_resized = F.interpolate(dec3, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        dec2 = self.decoder2(torch.cat((dec3_resized, enc2), dim=1))  # Concatenate with encoder2

        # Resize dec2 to match enc1's spatial dimensions before concatenation
        dec2_resized = F.interpolate(dec2, size=enc1.size()[2:], mode='bilinear', align_corners=True)
        dec1 = self.decoder1(torch.cat((dec2_resized, enc1), dim=1))  # Concatenate with encoder1

        # Output layer
        out = self.output_layer(dec1)

        return out

