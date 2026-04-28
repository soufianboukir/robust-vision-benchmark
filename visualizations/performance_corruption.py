import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path

# ============================================================================
# 1. LOAD ALL JSON FILES
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

print("=" * 80)
print("LOADED MODELS")
print("=" * 80)
for model_name, data in all_results.items():
    print(f"  ✓ {model_name:12} - Clean Accuracy: {data['clean']['accuracy']:.4f}")

# ============================================================================
# 2. BUILD COMPREHENSIVE DATAFRAME
# ============================================================================

data_rows = []

for model_name, model_data in all_results.items():
    clean_acc = model_data['clean']['accuracy']
    
    # Get all corruptions and their severities
    for corruption_type, corruption_data in model_data['corruptions'].items():
        
        # Extract severity levels (0-3 correspond to 4 severity levels in the data)
        for severity_idx, severity_data in enumerate(corruption_data['severities']):
            data_rows.append({
                'Model': model_name,
                'Corruption': corruption_type,
                'Severity': severity_idx,
                'Accuracy': severity_data['accuracy'],
                'F1': severity_data['f1'],
                'Precision': severity_data['precision'],
                'Recall': severity_data['recall']
            })
        
        # Also get the worst-case accuracy (at highest severity)
        worst_case_acc = corruption_data['worst_case']
        data_rows.append({
            'Model': model_name,
            'Corruption': corruption_type,
            'Severity': 'worst',
            'Accuracy': worst_case_acc,
            'F1': corruption_data['average']['f1'],
            'Precision': corruption_data['average']['precision'],
            'Recall': corruption_data['average']['recall']
        })

df = pd.DataFrame(data_rows)

print("\n" + "=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total records: {len(df)}")
print(f"Models: {df['Model'].nunique()} - {sorted(df['Model'].unique())}")
print(f"Corruption types: {df['Corruption'].nunique()} - {sorted(df['Corruption'].unique())}")
print(f"Severity levels: {sorted([x for x in df['Severity'].unique() if x != 'worst'])}")

# ============================================================================
# 3. PREPARE DATA FOR VISUALIZATIONS
# ============================================================================

# Get clean accuracy for each model
clean_acc_dict = {}
for model_name, model_data in all_results.items():
    clean_acc_dict[model_name] = model_data['clean']['accuracy']

# Get worst-case accuracy per model-corruption
worst_case_data = []
for model_name, model_data in all_results.items():
    for corruption_type, corruption_data in model_data['corruptions'].items():
        worst_case_data.append({
            'Model': model_name,
            'Corruption': corruption_type,
            'Worst Accuracy': corruption_data['worst_case'],
            'Sensitivity Score': model_data['corruption_sensitivity_scores'][corruption_type],
            'Clean Accuracy': clean_acc_dict[model_name]
        })

worst_df = pd.DataFrame(worst_case_data)

# Compute drop percentage
worst_df['Drop %'] = ((worst_df['Clean Accuracy'] - worst_df['Worst Accuracy']) / 
                       worst_df['Clean Accuracy'] * 100)

# Create heatmap data
heatmap_data = worst_df.pivot(index='Model', columns='Corruption', values='Worst Accuracy')

# Get corruption ranking by average sensitivity
corruption_ranking = worst_df.groupby('Corruption')['Sensitivity Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)

print("\n" + "=" * 80)
print("WORST-CASE ACCURACY (Per Model-Corruption)")
print("=" * 80)
print(heatmap_data.round(4))

print("\n" + "=" * 80)
print("CORRUPTION SENSITIVITY RANKING")
print("=" * 80)
print(corruption_ranking.round(4))

# ============================================================================
# 4. CREATE VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# ─────────────────────────────────────────────────────────────────────────
# VISUALIZATION 1: HEATMAP (Global view)
# ─────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    vmin=0.1,
    vmax=0.95,
    cbar_kws={'label': 'Worst-Case Accuracy'},
    ax=ax1,
    linewidths=0.5,
    cbar=True
)
ax1.set_title('1. HEATMAP: Model × Corruption Type\n(Worst-Case Accuracy at Highest Severity)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Corruption Type', fontweight='bold', fontsize=11)
ax1.set_ylabel('Model', fontweight='bold', fontsize=11)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# ─────────────────────────────────────────────────────────────────────────
# VISUALIZATION 2: CORRUPTION SENSITIVITY BAR CHART
# ─────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
corruption_sorted = corruption_ranking.sort_values('mean', ascending=True)
colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(corruption_sorted)))
bars = ax2.barh(range(len(corruption_sorted)), corruption_sorted['mean'], 
                 xerr=corruption_sorted['std'], color=colors, capsize=6, alpha=0.85)
ax2.set_yticks(range(len(corruption_sorted)))
ax2.set_yticklabels(corruption_sorted.index, fontsize=11)
ax2.set_xlabel('Average Sensitivity Score', fontweight='bold', fontsize=11)
ax2.set_title('2. CORRUPTION RANKING: Which Corruptions Are Most Harmful?\n(Higher = More Destructive)', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (idx, row) in enumerate(corruption_sorted.iterrows()):
    ax2.text(row['mean'] + row['std'] + 0.01, i, f"{row['mean']:.3f}", 
             va='center', fontweight='bold', fontsize=9)

# ─────────────────────────────────────────────────────────────────────────
# VISUALIZATION 3: LINE PLOTS (Degradation curves)
# ─────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :])

models_list = sorted(df['Model'].unique())
corruptions_list = sorted([c for c in df['Corruption'].unique()])

colors_models = plt.cm.tab10(np.linspace(0, 1, len(models_list)))
colors_corruptions = plt.cm.Set3(np.linspace(0, 1, len(corruptions_list)))

# Create line plots for each corruption type
for corruption_idx, corruption in enumerate(corruptions_list):
    corruption_data = df[df['Corruption'] == corruption].copy()
    
    for model_idx, model in enumerate(models_list):
        model_data = corruption_data[corruption_data['Model'] == model].copy()
        # Filter only numeric severity levels (0, 1, 2, 3), exclude 'worst'
        model_data = model_data[model_data['Severity'] != 'worst'].sort_values('Severity')
        
        if len(model_data) > 0:
            ax3.plot(
                model_data['Severity'],
                model_data['Accuracy'],
                marker='o',
                linestyle='-' if model_idx % 2 == 0 else '--',
                linewidth=2.2,
                alpha=0.7,
                color=colors_models[model_idx],
                markersize=5
            )

ax3.set_xlabel('Corruption Severity Level (0=None, 3=Worst)', fontweight='bold', fontsize=12)
ax3.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
ax3.set_title('3. DEGRADATION CURVES: How Accuracy Drops with Corruption Severity\n(All Models × All Corruption Types)', 
              fontsize=13, fontweight='bold', pad=15)
ax3.set_xticks([0, 1, 2, 3])
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_ylim([0.1, 0.95])

# Create custom legend with model names
handles_models = [plt.Line2D([0], [0], color=colors_models[i], marker='o', linestyle='-', 
                             linewidth=2, markersize=6) for i in range(len(models_list))]
legend1 = ax3.legend(handles_models, models_list, loc='upper left', fontsize=10, 
                     title='Models', title_fontsize=11, ncol=1)
ax3.add_artist(legend1)

# Add corruption type legend as colored boxes on the right
for idx, corruption in enumerate(corruptions_list):
    ax3.text(0.98, 0.95 - idx*0.065, f'■ {corruption}', 
             transform=ax3.transAxes, fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor=colors_corruptions[idx], alpha=0.4),
             ha='right', va='top')

plt.suptitle('Real Model Robustness Evaluation Dashboard\n' + 
             f'{len(models_list)} Models × {len(corruptions_list)} Corruption Types', 
             fontsize=16, fontweight='bold', y=0.998)

output_path = 'results/plots/robustness_dashboard.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print("\n✅ Saved: real_robustness_dashboard.png")
plt.close()
plt.show()

 
# ============================================================================
# 5. CREATE DETAILED SUMMARY TABLES
# ============================================================================
 
print("\n" + "=" * 80)
print("TABLE 1: WORST-CASE ACCURACY PER MODEL-CORRUPTION")
print("=" * 80)
table1 = worst_df.pivot_table(index='Model', columns='Corruption', values='Worst Accuracy')
print(table1.round(4).to_string())
 
print("\n" + "=" * 80)
print("TABLE 2: ACCURACY DROP (%) - CLEAN vs WORST CASE")
print("=" * 80)
table2 = worst_df.pivot_table(index='Model', columns='Corruption', values='Drop %')
print(table2.round(2).to_string())
 
print("\n" + "=" * 80)
print("TABLE 3: OVERALL MODEL ROBUSTNESS RANKING")
print("=" * 80)
model_stats = []
for model_name in models_list:
    model_rows = worst_df[worst_df['Model'] == model_name]
    model_stats.append({
        'Model': model_name,
        'Clean Accuracy': clean_acc_dict[model_name],
        'Avg Worst Accuracy': model_rows['Worst Accuracy'].mean(),
        'Avg Drop %': model_rows['Drop %'].mean(),
        'Min Accuracy': model_rows['Worst Accuracy'].min(),
        'Max Accuracy': model_rows['Worst Accuracy'].max(),
        'Robustness Std': model_rows['Worst Accuracy'].std()
    })
 
table3 = pd.DataFrame(model_stats).sort_values('Avg Worst Accuracy', ascending=False)
print(table3.round(4).to_string(index=False))
 
print("\n" + "=" * 80)
print("TABLE 4: CORRUPTION IMPACT ANALYSIS")
print("=" * 80)
corruption_stats = []
for corruption in corruptions_list:
    corruption_rows = worst_df[worst_df['Corruption'] == corruption]
    corruption_stats.append({
        'Corruption': corruption,
        'Avg Worst Accuracy': corruption_rows['Worst Accuracy'].mean(),
        'Worst Model': corruption_rows.loc[corruption_rows['Worst Accuracy'].idxmin(), 'Model'],
        'Best Model': corruption_rows.loc[corruption_rows['Worst Accuracy'].idxmax(), 'Model'],
        'Avg Drop %': corruption_rows['Drop %'].mean(),
        'Sensitivity': corruption_rows['Sensitivity Score'].mean()
    })
 
table4 = pd.DataFrame(corruption_stats).sort_values('Sensitivity', ascending=False)
print(table4.round(4).to_string(index=False))
 
print("\n" + "=" * 80)
print("TABLE 5: PER-MODEL CORRUPTION RANKING")
print("=" * 80)
for model_name in models_list:
    print(f"\n{model_name}:")
    model_data = all_results[model_name]
    ranking = model_data['corruption_ranking']
    for rank, (corruption, score) in enumerate(ranking, 1):
        worst_acc = model_data['corruptions'][corruption]['worst_case']
        print(f"  {rank}. {corruption:20} - Sensitivity: {score:.4f}, Worst Accuracy: {worst_acc:.4f}")
 
print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS AND ANALYSES COMPLETE!")
print("=" * 80)
