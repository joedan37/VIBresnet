import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# 从 data_loader.py 导入
from data_loader import get_dataloaders

# 从 model 目录导入不同模型
from model.VIBResNet       import VIBResNet
from model.ResNet18        import ResNet18
from model.ResNet18_Dropout import ResNet18_Dropout
from model.EfficientNetB0 import EfficientNetB0
from model.VIBResNet_Attention import VIBResNet_Attention
from model.SimpleCNN       import SimpleCNN

#实验候选池就包含了从轻量级（SimpleCNN）到中等（ResNet18、ResNet18_Dropout）再到高效能（EfficientNet‑B0）的多样化架构,更全面地评估各方法在自然与对抗样本上的性能差异。

# ========== 0. 参数 & 环境 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)

# 超参数
batch_size    = 128
epochs        = 200
lr            = 0.1
momentum      = 0.9
weight_decay  = 5e-4
latent_dim    = 256
lambda_ib     = 1e-3
eps           = 8/255
alpha         = 2/255
pgd_steps     = 10

# ========== 1. 选择数据集 & 加载 ==========
# 可替换为 'mnist' 或 'cifar10'
dataset_name = 'cifar10'
train_loader, test_loader, input_channels = get_dataloaders(
    dataset_name,
    batch_size=batch_size,
    num_workers=4,
    data_root='./data'
)

# ========== 2. 选择模型 ==========
# model = VIBResNet(latent_dim=latent_dim, num_classes=10).to(device)
# model = ResNet18(num_classes=10).to(device)
# model = ResNet18_Dropout(num_classes=10).to(device)
# model = EfficientNetB0(num_classes=10, pretrained=true).to(device)
model = VIBResNet_Attention(latent_dim=256, num_classes=10, pretrained=False).to(device)
# model = SimpleCNN(num_classes=10).to(device)


# ========== 3. PGD 对抗攻击函数 ==========
def pgd_attack(model, X, y,
               eps=8/255, alpha=2/255, iters=10):
    X_adv = X.detach() + 0.001 * torch.randn_like(X)
    X_adv = torch.clamp(X_adv, 0, 1)
    for _ in range(iters):
        X_adv.requires_grad_(True)
        logits, _ = model(X_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        X_adv = X_adv + alpha * torch.sign(grad)
        X_adv = torch.min(torch.max(X_adv, X - eps), X + eps)
        X_adv = torch.clamp(X_adv, 0, 1).detach()
    return X_adv

# ========== 4. 优化器和学习率调度 ==========
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[100, 150],
                                           gamma=0.1)

# ========== 5. 训练 & 验证 ==========
def train_one_epoch(epoch: int):
    model.train()
    total_loss = total_nat = total_adv = 0

    loader = tqdm(enumerate(train_loader, 1),
                  total=len(train_loader),
                  desc=f"Epoch {epoch} ▶ Init", ncols=100)
    for i, (X, y) in loader:
        X, y = X.to(device), y.to(device)

        # 1) 生成对抗样本
        loader.set_description(f"Epoch {epoch} ▶ PGD ({i}/{len(train_loader)})")
        X_adv = pgd_attack(model, X, y, eps, alpha, pgd_steps)

        # 2) NAT forward
        loader.set_description(f"Epoch {epoch} ▶ NAT fwd ({i}/{len(train_loader)})")
        logits_nat, kld_nat = model(X)
        loss_nat = F.cross_entropy(logits_nat, y)

        # 3) ADV forward
        loader.set_description(f"Epoch {epoch} ▶ ADV fwd ({i}/{len(train_loader)})")
        logits_adv, kld_adv = model(X_adv)
        loss_adv = F.cross_entropy(logits_adv, y)

        # 4) Backprop
        loader.set_description(f"Epoch {epoch} ▶ Backprop ({i}/{len(train_loader)})")
        loss = loss_nat + loss_adv + lambda_ib * (kld_nat + kld_adv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_nat  += (logits_nat.argmax(1) == y).sum().item()
        total_adv  += (logits_adv.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    nat_acc = total_nat / len(train_loader.dataset) * 100
    adv_acc = total_adv / len(train_loader.dataset) * 100
    print(f"Epoch {epoch} Train ▶ Loss: {avg_loss:.4f} | Nat Acc: {nat_acc:.2f}% | Adv Acc: {adv_acc:.2f}%")

def test(epoch: int):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    acc = correct / total * 100
    print(f"Epoch {epoch} Test ▶ Clean Acc: {acc:.2f}%")

# ========== 6. 主循环 ==========
if __name__ == "__main__":
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_one_epoch(epoch)
        test(epoch)
        scheduler.step()
        # 保存最优模型
        torch.save(model.state_dict(),
                   f"checkpoints/vib_adv_epoch{epoch}.pth")

