import os
import json
import torch

def is_torch_model(model):
    return isinstance(model, torch.nn.Module)


def save_results(results, model_name):
    save_dir = os.path.join(os.getcwd(), "results/logs")
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{model_name}_results.json")

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    results_clean = convert(results)

    with open(path, "w") as f:
        json.dump(results_clean, f, indent=4)

    print(f"Results saved to {path}")



def get_model(model_name):
    # ML models
    if model_name == 'logistic_regression':
        from src.models.logistic_regression import LogisticModel
        return LogisticModel()
    elif model_name == 'knn':
        from src.models.knn import KNNModel
        return KNNModel()
    elif model_name == 'random_forest':
        from src.models.random_forest import RandomForestModel
        return RandomForestModel()

    # DL models
    elif model_name == 'mlp3':
        from src.models.mlp3 import MLP3
        return MLP3()
    elif model_name == 'mlp5':
        from src.models.mlp5 import MLP5
        return MLP5()
    elif model_name == 'cnn3':
        from src.models.cnn3 import CNN3
        return CNN3()
    elif model_name == 'cnn5':
        from src.models.cnn5 import CNN5
        return CNN5()
    elif model_name == 'cnn7':
        from src.models.cnn7 import CNN7
        return CNN7()
    elif model_name == 'resnet18':
        from src.models.resnet18 import ResNet18
        return ResNet18()
    elif model_name == 'resnet50':
        from src.models.resnet50 import ResNet50
        return ResNet50()

    else:
        raise ValueError(f"Unknown model {model_name}")



def count_parameters_torch(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def count_parameters_ml(model):
    if hasattr(model, "coef_"):  # Logistic Regression
        return model.coef_.size + model.intercept_.size

    elif hasattr(model, "n_features_in_"):  # KNN
        return model.n_features_in_

    elif hasattr(model, "estimators_"):  # Random Forest
        return sum(tree.tree_.node_count for tree in model.estimators_)

    else:
        return None


def get_model_parameters(model):
    if is_torch_model(model):
        return count_parameters_torch(model)
    else:
        return count_parameters_ml(model)