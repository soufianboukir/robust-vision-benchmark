import matplotlib.pyplot as plt
import numpy as np
import os

# ===== Updated data from your LaTeX table =====
models = ['ResNet-18', 'AlexNet', 'LeNet-5', 'MLP-5', 'MLP-3']

metrics_data = {
    'Accuracy':  [0.8591, 0.7986, 0.5921, 0.5543, 0.5411],
    'Precision': [0.8590, 0.7986, 0.5889, 0.5537, 0.5374],
    'Recall':    [0.8591, 0.7986, 0.5921, 0.5543, 0.5411],
    'F1-Score':  [0.8589, 0.7985, 0.5900, 0.5528, 0.5381]
}

# ===== Create figure =====
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.2

metrics = list(metrics_data.keys())
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

# ===== Plot grouped bars =====
for i, (metric, color) in enumerate(zip(metrics, colors)):
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, metrics_data[metric], width,
                  label=metric, color=color, edgecolor='black', linewidth=0.7)
    
    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# ===== Formatting =====
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Clean Metrics Comparison Across Models', fontsize=14, fontweight='bold', pad=15)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right')

ax.set_ylim(0.5, 0.9)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

# Save
output_path = 'results/plots/metrics_comparison_clean.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"✓ Saved to {output_path}")
plt.show()