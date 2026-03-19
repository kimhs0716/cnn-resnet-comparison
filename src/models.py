import torch.nn as nn
import torch.nn.functional as F


class PlainCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # (B, 3, 32, 32) -> (B, 32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # (B, 32, 16, 16) -> (B, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # (B, 64, 8, 8) -> (B, 128, 8, 8)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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

        if stride==1 and in_channels==out_channels:
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
    def __init__(self, num_classes=10):
        super().__init__()
        
        # (B, 3, 32, 32) -> (B, 32, 16, 16)
        self.block1 = ResidualBlock(3, 32, stride=2)

        # (B, 32, 16, 16) -> (B, 64, 8, 8)
        self.block2 = ResidualBlock(32, 64, stride=2)

        # (B, 64, 8, 8) -> (B, 128, 8, 8)
        self.block3 = ResidualBlock(64, 128, stride=1)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)