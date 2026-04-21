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
    # Logistic Regression
    if hasattr(model, "coef_"):
        coef_params = model.coef_.size
        intercept_params = model.intercept_.size if hasattr(model, "intercept_") else 0
        return coef_params + intercept_params
 
    # Random Forest
    elif hasattr(model, "estimators_"):
        return sum(tree.tree_.node_count for tree in model.estimators_)
 
    # KNN (store entire training data)
    elif hasattr(model, "n_features_in_"):
        if hasattr(model, "_fit_X"):  # Training data stored
            return model._fit_X.size
        else:
            return model.n_features_in_
 
    # Support Vector Machine
    elif hasattr(model, "support_vectors_"):
        return model.support_vectors_.size + (model.n_support_.size if hasattr(model, "n_support_") else 0)
 
 
    # MLP/Neural Network
    elif hasattr(model, "coefs_"):
        return sum(coef.size for coef in model.coefs_) + sum(intercept.size for intercept in model.intercepts_)
 
    else:
        # Fallback: try to get model size in memory
        try:
            import sys
            return sys.getsizeof(model)
        except:
            return 0  # Return 0 instead of None
 
 
def get_model_parameters(model):
    if is_torch_model(model):
        return count_parameters_torch(model)
    else:
        return count_parameters_ml(model)