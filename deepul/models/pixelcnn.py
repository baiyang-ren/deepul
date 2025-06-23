import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert mask_type in {'A', 'B'}, "mask_type must be 'A' or 'B'"
        self.register_buffer('mask', self.weight.data.clone())
        self.mask_type = mask_type
        self.build_mask()

    def build_mask(self):
        kH, kW = self.kernel_size
        self.mask.fill_(1)
        yc, xc = kH // 2, kW // 2
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc+1:] = 0
        if self.mask_type == 'A':
            self.mask[:, :, yc, xc] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class PixelCNN(nn.Module):
    def __init__(self, input_channels=1, n_filters=64, n_classes=1):
        super().__init__()
        self.input_channels = input_channels
        self.n_filters = n_filters
        self.n_classes = n_classes
        # First layer: 7x7 masked conv type A
        self.conv1 = MaskedConv2d(input_channels, n_filters, 7, mask_type='A', padding=3)
        # 5 layers: 7x7 masked conv type B
        self.conv2 = nn.ModuleList([
            MaskedConv2d(n_filters, n_filters, 7, mask_type='B', padding=3)
            for _ in range(5)
        ])
        # 2 layers: 1x1 masked conv type B
        self.conv3 = nn.ModuleList([
            MaskedConv2d(n_filters, n_filters, 1, mask_type='B')
            for _ in range(2)
        ])
        # Output layer
        self.output = nn.Conv2d(n_filters, n_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        for conv in self.conv2:
            x = conv(x)
            x = F.relu(x)
        for conv in self.conv3:
            x = conv(x)
            x = F.relu(x)
        x = self.output(x)
        return x 