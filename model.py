# model.py
import torch
import torch.nn as nn

class VDSR(nn.Module):
    def __init__(self, num_channels=3, num_filters=64, num_layers=20):
        super(VDSR, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Add more convolution layers
        for _ in range(num_layers - 2):  # excluding first and last layers
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        # Last convolution layer to predict residual
        layers.append(nn.Conv2d(num_filters, num_channels, kernel_size=3, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

