import numpy as np
import os
import time
import torch

from sklearn.metrics import precision_score, recall_score, f1_score
from src.datasets.corruption_engine import apply_corruption_batch, CORRUPTION_TYPES
from config.config import Config
from utils.start import save_results
from utils.start import get_model

from src.datasets.cifar_loader import train_loader, test_loader, val_loader

# ======================================================
# PREPROCESSING
# ======================================================

def preprocess_dl(images, corruption_type=None, severity=0):
    """Preprocess images for deep learning models (no flattening)"""
    if corruption_type is not None:
        images = apply_corruption_batch(images, corruption_type, severity)
    return images


# ======================================================
# TRAINING (UNIFIED - DL ONLY)
# ======================================================

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def train_model_unified(model, train_loader, val_loader, device, config):
    """Train PyTorch model with validation and learning rate scheduling"""
    start_time = time.time()
    train_losses = []
    val_losses = []
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Add scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,  # Reduce LR every 10 epochs
        gamma=0.1  # Multiply LR by 0.1
    )
    
    for epoch in range(config.epochs):
        # -------- Training --------
        model.train()
        train_epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # -------- Validation --------
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_epoch_loss += loss.item()
        
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Step the scheduler
        scheduler.step()
        
        # Print epoch info
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR: {current_lr:.6f}")
    
    # -------- Visualize Losses --------
    plt.figure(figsize=(12, 6))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, marker='o', linestyle='-', linewidth=2, label='Training Loss', color='#2E86AB')
    plt.plot(epochs_range, val_losses, marker='s', linestyle='-', linewidth=2, label='Validation Loss', color='#A23B72')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(f'Training vs Validation Loss Over Epochs - {config.model_type}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_dir = os.path.join(os.getcwd(), "results", "loss")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"training_validation_loss_{config.model_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot at: {save_path}")
    plt.close()  # Close to free memory
    
    end_time = time.time()
    training_time_minutes = (end_time - start_time) / 60
    return training_time_minutes


# ======================================================
# EVALUATION (UNIFIED - DL ONLY)
# ======================================================

def evaluate_model_unified(model, loader, device, corruption_type=None, severity=0):
    """Evaluate PyTorch model on a dataset with optional corruptions"""
    predictions = []
    targets = []

    model.eval()

    for images, labels in loader:
        images = preprocess_dl(images, corruption_type, severity)
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(dim=1)

        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    accuracy = (predictions == targets).mean()
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ======================================================
# ROBUSTNESS EVALUATION
# ======================================================

def evaluate_full_robustness(model, test_loader, device):
    """Evaluate model robustness across multiple corruption types and severities"""
    results = {}

    results["model_name"] = Config.model_type
    
    clean = evaluate_model_unified(model, test_loader, device)
    results["clean"] = clean

    clean_acc = clean["accuracy"]

    corruption_summary = {}
    sensitivity_scores = {}
    all_crs = []

    for corruption in CORRUPTION_TYPES:

        accs, precs, recs, f1s = [], [], [], []
        severity_list = []

        for severity in range(1, 5):

            metrics = evaluate_model_unified(
                model,
                test_loader,
                device,
                corruption_type=corruption,
                severity=severity
            )

            severity_list.append(metrics)

            accs.append(metrics["accuracy"])
            precs.append(metrics["precision"])
            recs.append(metrics["recall"])
            f1s.append(metrics["f1"])

        avg_acc = np.mean(accs)

        avg = {
            "accuracy": avg_acc,
            "precision": np.mean(precs),
            "recall": np.mean(recs),
            "f1": np.mean(f1s)
        }

        # RQ1: Sensitivity to corruption
        sensitivity = clean_acc - avg_acc
        sensitivity_scores[corruption] = sensitivity

        # RQ5: Severity degradation
        severity_degradation = accs[0] - accs[-1]

        # Stability across severity levels
        stability = 1 - np.std(accs)

        # CRS: Corruption Robustness Score
        crs = avg_acc / clean_acc if clean_acc > 0 else 0

        all_crs.append(crs)

        corruption_summary[corruption] = {
            "severities": severity_list,
            "average": avg,
            "crs": crs,
            "stability": stability,
            "worst_case": min(accs),
            "severity_degradation": severity_degradation,
            "sensitivity_score": sensitivity
        }

    corrupted_avg = np.mean([c["average"]["accuracy"] for c in corruption_summary.values()])
    generalization_gap = clean_acc - corrupted_avg

    worst_corruption = max(sensitivity_scores.items(), key=lambda x: x[1])[0]

    results.update({
        "clean": clean,
        "corruptions": corruption_summary,

        "generalization_gap": generalization_gap,
        "corruption_sensitivity_scores": sensitivity_scores,
        "worst_corruption": worst_corruption,
        "corruption_ranking": sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True),

        "overall_crs": np.mean(all_crs),
        "robustness_stability": 1 - np.std(all_crs)
    })

    return results


# ======================================================
# MAIN RUN
# ======================================================

if __name__ == '__main__':

    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")

    model = get_model(Config.model_type)
    model = model.to(device)

    training_time = train_model_unified(model, train_loader, val_loader, device, Config)

    os.makedirs(Config.save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(Config.save_dir, f"{Config.model_type}.pt"))

    results = evaluate_full_robustness(model, test_loader, device)
    
    results["training_time_minutes"] = training_time

    save_results(results=results, model_name=Config.model_type)