import os
import json


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

    if model_name == 'mlp3':
        from src.models.mlp3 import MLP3
        return MLP3()
    elif model_name == 'mlp5':
        from src.models.mlp5 import MLP5
        return MLP5()
    elif model_name == 'leNet5':
        from src.models.LeNet5 import LeNet5
        return LeNet5()
    elif model_name == 'alexNet':
        from src.models.AlexNet import AlexNet
        return AlexNet()
    elif model_name == 'resnet18':
        from src.models.resnet18 import ResNet18
        return ResNet18()
    else:
        raise ValueError(f"Unknown model {model_name}")
