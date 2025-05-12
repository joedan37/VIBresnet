import os
import argparse
import torch
import torch.nn.functional as F
from data_loader import get_dataloaders
from model.VIBResNet       import VIBResNet
from model.ResNet18        import ResNet18
from model.ResNet18_Dropout import ResNet18_Dropout
from model.SimpleCNN       import SimpleCNN
from model.EfficientNetB0  import EfficientNetB0

def pgd_attack(model, X, y, eps=8/255, alpha=2/255, iters=10):
    """简单 PGD 生成函数，与 train.py 中一致"""
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

def test(model, device, test_loader, eps, alpha, pgd_steps):
    model.eval()
    correct_nat = correct_adv = total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            # 自然样本
            logits_nat, _ = model(X)
            pred_nat = logits_nat.argmax(dim=1)
            correct_nat += (pred_nat == y).sum().item()

            # 对抗样本
            X_adv = pgd_attack(model, X, y, eps, alpha, pgd_steps)
            logits_adv, _ = model(X_adv)
            pred_adv = logits_adv.argmax(dim=1)
            correct_adv += (pred_adv == y).sum().item()

            total += y.size(0)

    acc_nat = 100. * correct_nat / total
    acc_adv = 100. * correct_adv / total
    return acc_nat, acc_adv

def main():
    parser = argparse.ArgumentParser(description="Test adversarially trained models")
    parser.add_argument('--dataset',   choices=['cifar10','mnist'], default='cifar10')
    parser.add_argument('--model',     choices=['VIB','ResNet','ResNetDrop','SimpleCNN','EffNetB0'], default='VIB')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model .pth checkpoint")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eps',        type=float, default=8/255)
    parser.add_argument('--alpha',      type=float, default=2/255)
    parser.add_argument('--pgd-steps',  type=int, default=10)
    parser.add_argument('--no-cuda',    action='store_true', default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # 加载数据
    test_loader, _, _ = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=4,
        data_root='./data'
    )

    # 选择模型
    if args.model == 'VIB':
        model = VIBResNet(latent_dim=256, num_classes=10)
    elif args.model == 'ResNet':
        model = ResNet18(num_classes=10)
    elif args.model == 'ResNetDrop':
        model = ResNet18_Dropout(num_classes=10)
    elif args.model == 'SimpleCNN':
        model = SimpleCNN(num_classes=10)
    elif args.model == 'EffNetB0':
        model = EfficientNetB0(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)

    # 加载 checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint '{args.checkpoint}'")

    # 测试
    acc_nat, acc_adv = test(
        model, device, test_loader,
        eps=args.eps, alpha=args.alpha, pgd_steps=args.pgd_steps
    )

    print(f"Test on {args.dataset} with model {args.model}:")
    print(f"  Natural Accuracy     : {acc_nat:.2f}%")
    print(f"  Adversarial Accuracy : {acc_adv:.2f}%")

if __name__ == "__main__":
    main()
