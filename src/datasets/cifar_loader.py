import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from config.config import Config

transform = transforms.Compose([
    transforms.ToTensor(),  # Transform images to Tensors
    # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5): mean for each channel and std for each channel (RGB)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Standardized version (pixel - mean) / std
])

# Load full training dataset
full_train_dataset = datasets.CIFAR10(
    root="data/raw",  # Dataset location
    train=True,  # Training dataset
    download=True,  # Download it, if not downloaded yet
    transform=transform  # Apply the transform function
)

# Load test dataset
test_dataset = datasets.CIFAR10(
    root="data/raw",
    train=False,
    download=True,
    transform=transform
)

# Split training into train (80%) and validation (20%)
train_size = int(0.8 * len(full_train_dataset))  # 80% for training
val_size = len(full_train_dataset) - train_size   # 20% for validation

train_dataset, val_dataset = random_split(
    full_train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=Config.batch_size, 
    shuffle=True  # Shuffle training data
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=Config.batch_size, 
    shuffle=False  # Don't shuffle validation data
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=Config.batch_size, 
    shuffle=False  # Don't shuffle test data
)