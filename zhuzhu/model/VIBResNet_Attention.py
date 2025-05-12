import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn

class VIBResNet_Attention(nn.Module):
    def __init__(self, latent_dim=256, num_classes=10, pretrained=False):
        super(VIBResNet_Attention, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.attention = Attention(in_channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = backbone.fc.in_features
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        h = self.attention(h)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        out = self.classifier(z)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return out, kld
