import torch.nn as nn
from torchvision import models

class ResNet18_Dropout(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x):
        out = self.model(x)
        return out, 0  # 无KL项
