from config.config import Config

from src.models.mlp3 import MLP3
from src.models.mlp5 import MLP5
from src.models.cnn3 import CNN3
from src.models.cnn5 import CNN5
from src.models.cnn7 import CNN7
from src.models.resnet18 import ResNet18
from src.models.resnet50 import ResNet50

from datasets.cifar_loader import train_loader
from datasets.cifar_loader import test_loader

model = Config.model_type

def get_model(model_name):
    if model_name == 'mlp3':
        return MLP3()
    elif model_name == 'mlp5':
        return MLP5()
    elif model_name == 'cnn3':
        return CNN3()
    elif model_name == 'cnn5':
        return CNN5()
    elif model_name == 'cnn7':
        return CNN7()
    elif model_name == 'resnet18':
        return ResNet18()
    elif model_name == 'resnet50':
        return ResNet50()
    else:
        raise ValueError(f"Unknown model {model_name}")


import torch
import os
from training.trainer import train_model
from training.trainer import evaluate_model
from datasets.corruption_engine import CORRUPTION_TYPES

def run_experiment(config, train_loader, test_loader):

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = get_model(config.model_type).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # ===== TRAINING =====
    for epoch in range(config.epochs):
        loss = train_model(model, train_loader, optimizer, criterion, device, isMLP=config.model_type in ['mlp3', 'mlp5'])
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    # ===== SAVE MODEL =====
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, f"{config.model_type}.pt")
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to {save_path}")

    # ===== EVALUATION =====
    results = {}

    # Clean accuracy
    clean_acc = evaluate_model(model, test_loader, device, isMLP=config.model_type in ['mlp3', 'mlp5'])
    results["clean"] = clean_acc

    # Corrupted accuracy
    # for corruption in CORRUPTION_TYPES:
    #     results[corruption] = []

    #     for severity in range(1, 5):
    #         acc = evaluate_model(
    #             model,
    #             test_loader,
    #             device,
    #             corruption_type=corruption,
    #             severity=severity
    #         )
    #         results[corruption].append(acc)

    return results


if __name__ == "__main__":
    print(run_experiment(Config,train_loader=train_loader, test_loader=test_loader))