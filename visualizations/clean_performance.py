

# Visualization 1: Horizontal Bar Chart — Clean Accuracy Ranking
# Clean Accuracy Benchmark — CIFAR-10 Test Set
# ─────────────────────────────────────────────
# ResNet50        ██████████████████████████████░░ 0.9247 [Baseline]
# ResNet18        █████████████████████████░░░░░░░ 0.9156
# CNN7            ████████████████████░░░░░░░░░░░░ 0.8934
# CNN5            ███████████████████░░░░░░░░░░░░░ 0.8756
# CNN3            ████████████████░░░░░░░░░░░░░░░░ 0.8412
# MLP5            ███████████████░░░░░░░░░░░░░░░░░ 0.7834
# MLP3            ██████████████░░░░░░░░░░░░░░░░░░ 0.7456
# Random Forest   ██████████████░░░░░░░░░░░░░░░░░░ 0.7189
# KNN (k=5)       █████████████░░░░░░░░░░░░░░░░░░░ 0.6912
# Log. Reg. (PCA) ████████████░░░░░░░░░░░░░░░░░░░░ 0.6234
#                 └─────────────────────────────
#                 0.6    0.7    0.8    0.9    1.0



# Visualization 2: Grouped Bar Chart — Metrics Comparison (Top 5 Models)
# Clean Metrics Comparison (Top 5 Models)
# ───────────────────────────────────────
#        Accuracy  Precision  Recall   F1-Score
# ResNet50   ███░░░    ███░░░    ███░░░   ███░░░
# ResNet18   ███░░░    ███░░░    ███░░░   ███░░░
# CNN7       ██░░░░    ██░░░░    ██░░░░   ██░░░░
# CNN5       ██░░░░    ██░░░░    ██░░░░   ██░░░░
# CNN3       ██░░░░    ██░░░░    ██░░░░   ██░░░░


# Visualization 3: Scatter Plot — Accuracy vs. Training Time
# Clean Accuracy vs. Computational Cost
# ────────────────────────────────────────────────────────
# Accuracy │
# 1.0  ┤                                     ResNet50 ●
#      │
# 0.95 ┤                    ResNet18 ●
#      │
# 0.90 ┤              CNN7 ●
#      │
# 0.85 ┤         CNN5 ●
#      │
# 0.80 ┤    MLP5 ●
#      │
# 0.75 ┤ MLP3 ●
#      │
# 0.70 ┤ RF ●
#      │
#      └────┬────┬────┬────┬────┬───┬────┬────┬────┬────┬────
#           5   10   15   20   30   45  55   60   65   75
#             Training Time (minutes)















import matplotlib.pyplot as plt
import numpy as np
import os

# Fake data for accuracy comparison
models = ['ResNet50', 'ResNet18', 'CNN7', 'CNN5', 'CNN3', 'MLP5', 'MLP3', 'Random Forest', 'KNN (k=5)', 'Log. Reg. (PCA)']
accuracy = [0.9247, 0.9156, 0.8934, 0.8756, 0.8412, 0.7834, 0.7456, 0.7189, 0.6912, 0.6234]

# Metrics comparison data (top 5 models)
top_models = ['ResNet50', 'ResNet18', 'CNN7', 'CNN5', 'CNN3']
metrics_data = {
    'Accuracy': [0.9247, 0.9156, 0.8934, 0.8756, 0.8412],
    'Precision': [0.9231, 0.9124, 0.8892, 0.8701, 0.8331],
    'Recall': [0.9256, 0.9178, 0.8967, 0.8812, 0.8489],
    'F1-Score': [0.9243, 0.9151, 0.8929, 0.8756, 0.8409]
}

# Training time vs accuracy data
training_models = ['ResNet50', 'ResNet18', 'CNN7', 'CNN5', 'CNN3', 'MLP5', 'MLP3', 'Random Forest', 'KNN (k=5)', 'Log. Reg. (PCA)']
training_time = [75, 45, 35, 28, 18, 12, 8, 22, 5, 3]
accuracy_scatter = [0.9247, 0.9156, 0.8934, 0.8756, 0.8412, 0.7834, 0.7456, 0.7189, 0.6912, 0.6234]

# Create figure with 3 subplots
fig = plt.figure(figsize=(16, 14))

# ===== CHART 1: Horizontal Bar Chart - Model Accuracy Comparison =====
ax1 = plt.subplot(3, 1, 1)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))
bars = ax1.barh(models, accuracy, color=colors, edgecolor='black', linewidth=0.7)

# Add value labels on bars
for i, (model, acc) in enumerate(zip(models, accuracy)):
    ax1.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Accuracy Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance: Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlim(0.6, 1.0)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.invert_yaxis()

# ===== CHART 2: Grouped Bar Chart - Metrics Comparison =====
ax2 = plt.subplot(3, 1, 2)
x = np.arange(len(top_models))
width = 0.2
metrics = list(metrics_data.keys())
colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
    offset = width * (i - 1.5)
    bars = ax2.bar(x + offset, metrics_data[metric], width, label=metric, color=color, edgecolor='black', linewidth=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Metrics Comparison: Top 5 Models', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(top_models, rotation=45, ha='right')
ax2.set_ylim(0.8, 0.95)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# ===== CHART 3: Scatter Plot - Accuracy vs Training Time =====
ax3 = plt.subplot(3, 1, 3)
scatter = ax3.scatter(training_time, accuracy_scatter, s=200, c=np.linspace(0.2, 0.9, len(training_models)), 
                      cmap='viridis', edgecolor='black', linewidth=1.5, alpha=0.7)

# Add model labels to scatter points
for model, x, y in zip(training_models, training_time, accuracy_scatter):
    ax3.annotate(model, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax3.set_xlabel('Training Time (minutes)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Accuracy vs Computational Cost', fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0.6, 0.95)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax3, pad=0.02)
cbar.set_label('Performance Tier', fontsize=10)

# Overall title
fig.suptitle('ML Model Performance Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Define your output path
output_path = 'results/plots/clean_performance.png'

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Now save
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved to {output_path}")

plt.show()