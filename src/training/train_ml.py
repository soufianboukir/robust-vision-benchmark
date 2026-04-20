import numpy as np
import os
import time
import joblib
import torch

from sklearn.metrics import precision_score, recall_score, f1_score
from src.datasets.corruption_engine import apply_corruption_batch, CORRUPTION_TYPES
from config.config import Config
from utils.start import save_results
from utils.start import get_model

from src.datasets.cifar_loader import train_loader 
from src.datasets.cifar_loader import test_loader

def is_torch_model(model):
    return isinstance(model, torch.nn.Module)

# ======================================================
# PREPROCESSING
# ======================================================

def preprocess_ml(images, corruption_type=None, severity=0):
    if corruption_type is not None:
        images = apply_corruption_batch(images, corruption_type, severity)
    return images.view(images.size(0), -1)


# ======================================================
# TRAINING (UNIFIED)
# ======================================================

def train_model_unified(model, train_loader, device, is_torch):
    start_time = time.time()

    # ---------------- ML MODE ----------------
    if not is_torch:
        X_list, y_list = [], []

        for images, labels in train_loader:
            images = preprocess_ml(images)
            X_list.append(images.detach().cpu().numpy())
            y_list.append(labels.detach().cpu().numpy())

        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        X = (X - 0.5) / 0.5
        model.fit(X, y)

    # ---------------- DL MODE ----------------
    else:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    end_time = time.time()

    return (end_time - start_time) / 60


# ======================================================
# EVALUATION (UNIFIED)
# ======================================================

def evaluate_model_unified(model, loader, device, corruption_type=None, severity=0):
    predictions = []
    targets = []

    torch_model = is_torch_model(model)

    if torch_model:
        model.eval()

    for images, labels in loader:

        images = preprocess_ml(images, corruption_type, severity)

        # ---------------- DL ----------------
        if torch_model:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

        # ---------------- ML ----------------
        else:
            preds = model.predict(images.detach().cpu().numpy())

            predictions.extend(preds)
            targets.extend(labels.detach().cpu().numpy())

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
    results = {}

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

        # RQ1
        sensitivity = clean_acc - avg_acc
        sensitivity_scores[corruption] = sensitivity

        # RQ5
        severity_degradation = accs[0] - accs[-1]

        # Stability
        stability = 1 - np.std(accs)

        # CRS
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

    is_torch = is_torch_model(model)

    if is_torch:
        model = model.to(device)

    training_time = train_model_unified(model, train_loader, device, is_torch)

    os.makedirs(Config.save_dir, exist_ok=True)

    if is_torch:
        torch.save(model.state_dict(), os.path.join(Config.save_dir, f"{Config.model_type}.pt"))
    else:
        joblib.dump(model, os.path.join(Config.save_dir, f"{Config.model_type}.joblib"))

    results = evaluate_full_robustness(model, test_loader, device)

    results["training_time_minutes"] = training_time

    save_results(results=results, model_name=Config.model_type)