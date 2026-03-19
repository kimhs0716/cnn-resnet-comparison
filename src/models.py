import torch.nn as nn
import torch.nn.functional as F


class PlainCNN(nn.Module):
    def __init__(self, num_classes=10, num_blocks=1):
        super().__init__()

        def make_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
            )

        def make_stage(in_ch, out_ch, pool=True):
            layers = [make_block(in_ch, out_ch)]
            for _ in range(num_blocks - 1):
                layers.append(make_block(out_ch, out_ch))
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        # (B, 3, 32, 32) -> (B, 32, 16, 16)
        self.stage1 = make_stage(3, 32, pool=True)
        # (B, 32, 16, 16) -> (B, 64, 8, 8)
        self.stage2 = make_stage(32, 64, pool=True)
        # (B, 64, 8, 8) -> (B, 128, 8, 8)
        self.stage3 = make_stage(64, 128, pool=False)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10, num_blocks=1):
        super().__init__()

        def make_stage(in_ch, out_ch, stride):
            layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
            for _ in range(num_blocks - 1):
                layers.append(ResidualBlock(out_ch, out_ch, stride=1))
            return nn.Sequential(*layers)

        # (B, 3, 32, 32) -> (B, 32, 16, 16)
        self.stage1 = make_stage(3, 32, stride=2)
        # (B, 32, 16, 16) -> (B, 64, 8, 8)
        self.stage2 = make_stage(32, 64, stride=2)
        # (B, 64, 8, 8) -> (B, 128, 8, 8)
        self.stage3 = make_stage(64, 128, stride=1)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
