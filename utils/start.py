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