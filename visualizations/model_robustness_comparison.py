import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# ============================================================================
# 1. LOAD JSON FILES
# ============================================================================

json_files = [
    'results/logs/resnet18_results.json',
    'results/logs/alexNet_results.json',
    'results/logs/leNet5_results.json',
    'results/logs/mlp5_results.json',
    'results/logs/mlp3_results.json',
]

# Load all results
all_results = {}
for file_path in json_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        model_name = data['model_name']
        all_results[model_name] = data

# ============================================================================
# 2. EXTRACT MODEL STATISTICS
# ============================================================================

model_stats = []
for model_name, model_data in all_results.items():
    clean_acc = model_data['clean']['accuracy']
    
    # Get all worst-case accuracies
    worst_accs = []
    for corruption_type, corruption_data in model_data['corruptions'].items():
        worst_accs.append(corruption_data['worst_case'])
    
    avg_worst_acc = np.mean(worst_accs)
    avg_drop_pct = ((clean_acc - avg_worst_acc) / clean_acc) * 100
    
    model_stats.append({
        'Model': model_name,
        'Clean Accuracy': clean_acc,
        'Avg Worst Accuracy': avg_worst_acc,
        'Avg Drop %': avg_drop_pct,
        'Min Accuracy': min(worst_accs),
        'Max Accuracy': max(worst_accs)
    })

df_stats = pd.DataFrame(model_stats).sort_values('Avg Drop %', ascending=False)

print("=" * 80)
print("MODEL ROBUSTNESS STATISTICS")
print("=" * 80)
print(df_stats.to_string(index=False))

# ============================================================================
# 3. CREATE BAR CHARTS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Average Drop %
ax1 = axes[0]
colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_stats)))
bars1 = ax1.bar(df_stats['Model'], df_stats['Avg Drop %'], color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Average Drop %', fontweight='bold', fontsize=12)
ax1.set_xlabel('Model', fontweight='bold', fontsize=12)
ax1.set_title('Average Accuracy Drop Across All Corruptions\n(Lower = More Robust)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, max(df_stats['Avg Drop %']) * 1.1])

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
    
# Chart 2: Average Worst Accuracy
ax2 = axes[1]
df_stats_acc = df_stats.sort_values('Avg Worst Accuracy', ascending=False)
colors2 = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(df_stats_acc)))
bars2 = ax2.bar(df_stats_acc['Model'], df_stats_acc['Avg Worst Accuracy'], 
                color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Average Worst Accuracy', fontweight='bold', fontsize=12)
ax2.set_xlabel('Model', fontweight='bold', fontsize=12)
ax2.set_title('Average Worst-Case Accuracy Across All Corruptions\n(Higher = More Robust)', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0, 1.0])

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.suptitle('Model Robustness Comparison: Average Performance Under Corruption', 
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
output_path = 'results/plots/model_robustness_comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\n✅ Saved: model_robustness_comparison.png")
plt.close()