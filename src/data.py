import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_loaders(data_dir: str, batch_size: int, num_workers: int, augment: bool, val_split_seed: int = 42):
    """returns: train / val / test dataloaders"""

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if not augment:
        train_transforms = test_transforms

    train_data = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True,
        transform=train_transforms
    )
    val_data = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=test_transforms
    )
    test_data = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transforms
    )

    generator = torch.Generator()
    generator.manual_seed(val_split_seed)
    indices = torch.randperm(50000, generator=generator)
    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_data = Subset(train_data, train_indices)
    val_data = Subset(val_data, val_indices)

    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

