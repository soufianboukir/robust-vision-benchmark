from src.datasets.corruption_engine import apply_corruption_batch

import numpy as np

from src.models.logistic_regression import LogisticModel
from src.models.knn import KNNModel
from src.models.random_forest import RandomForestModel

from src.datasets.cifar_loader import train_loader
from src.datasets.cifar_loader import test_loader

import joblib
import os

from config.config import Config
from utils.start import save_results

def save_ml_model(model, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.joblib")
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def get_model(model_name):
    if model_name == 'lr':
        return LogisticModel()
    elif model_name == 'knn':
        return KNNModel()
    elif model_name == 'rf':
        return RandomForestModel()
    else:
        raise ValueError(f"Unknown model {model_name}")


def preprocess_ml(images, corruption_type=None, severity=0):
    if corruption_type is not None:
        images = apply_corruption_batch(images, corruption_type, severity)

    return images.view(images.size(0), -1) # flatten the input


def collect_data(loader, corruption_type=None, severity=0):
    X_list, y_list = [], []

    for images, labels in loader:
        images = preprocess_ml(images, corruption_type, severity)

        X_list.append(images.detach().cpu().numpy())
        y_list.append(labels.detach().cpu().numpy())

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    X = (X - 0.5) / 0.5

    return X, y


def train_ml_model(model, train_loader):
    X, y = collect_data(train_loader)
    model.fit(X, y)

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_ml_model(model, loader, corruption_type=None, severity=0):
    predictions = []
    targets = []

    for images, labels in loader:
        images = preprocess_ml(images, corruption_type, severity)

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


from src.datasets.corruption_engine import CORRUPTION_TYPES

from src.datasets.corruption_engine import CORRUPTION_TYPES

def evaluate_all_ml(model, test_loader):
    results = {}

    # Clean evaluation
    clean_metrics = evaluate_ml_model(model, test_loader)
    results['model_name'] = Config.model_type
    results["clean"] = clean_metrics

    # Corrupted evaluation
    for corruption in CORRUPTION_TYPES:
        results[corruption] = []

        for severity in range(1, 5):
            metrics = evaluate_ml_model(
                model,
                test_loader,
                corruption_type=corruption,
                severity=severity
            )

            results[corruption].append(metrics)

    return results


if __name__ == '__main__':
    model = get_model('lr')

    train_ml_model(model, train_loader)
    save_ml_model(model, save_dir="saved_models", model_name=Config.model_type)
    results = evaluate_all_ml(model, test_loader)
    save_results(results=results, model_name=Config.model_type)


