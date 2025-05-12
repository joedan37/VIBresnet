import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 for classification.
    引入 `weights` 参数来控制是否加载 ImageNet 预训练权重。
    """
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        # 如果 pretrained=True，则加载默认的 ImageNet 权重；否则随机初始化
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None  # :contentReference[oaicite:2]{index=2}
        backbone = efficientnet_b0(weights=weights, progress=True)        # :contentReference[oaicite:3]{index=3}

        # 修改分类头，将 1000 类替换为指定 num_classes
        in_features = backbone.classifier[1].in_features                  # :contentReference[oaicite:4]{index=4}
        backbone.classifier[1] = nn.Linear(in_features, num_classes)

        self.model = backbone

    def forward(self, x):
        """
        返回 logits 与 0 作为 KL 整数项占位（与其他模型接口保持一致）。
        """
        out = self.model(x)
        return out, 0
