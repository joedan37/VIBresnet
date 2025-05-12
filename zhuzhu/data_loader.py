import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(dataset_name: str,
                    batch_size: int = 128,
                    num_workers: int = 4,
                    data_root: str = './data'):
    name = dataset_name.lower()

    if name == 'cifar10':
        # CIFAR-10 图像是 RGB 3 通道，大小 32×32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10(root=data_root, train=True,
                                     download=True, transform=transform_train)

        test_set = datasets.CIFAR10(root=data_root, train=False,
                                    download=True, transform=transform_test)
        input_channels = 3

    elif name == 'mnist':
        # MNIST 是灰度图 1 通道，大小 28×28
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root=data_root, train=True,
                                   download=True, transform=transform)

        test_set = datasets.MNIST(root=data_root, train=False,
                                  download=True, transform=transform)
        input_channels = 1

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return train_loader, test_loader, input_channels


