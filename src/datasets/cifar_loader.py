from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(
    root="data/raw",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data/raw",
    train=False,
    download=True,
    transform=transform
)