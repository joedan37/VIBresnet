import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class VIBResNet(nn.Module):
    """
    Variational Information Bottleneck (VIB) + ResNet18 backbone.
    """
    def __init__(self, latent_dim=256, num_classes=10, pretrained=False):
        super().__init__()
        # torchvision ≥0.13 用 weights 参数代替 pretrained
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        # 去掉最后的 fc 层，只保留特征提取部分
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features

        # 编码网络：输出 μ 与 logσ²
        self.fc_mu     = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        # 分类头
        self.classifier = nn.Linear(latent_dim, num_classes)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # 1) 特征提取
        h = self.features(x).view(x.size(0), -1)
        # 2) 编码
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        # 3) 分类
        out = self.classifier(z)
        # 4) KL 正则项
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return out, kld
