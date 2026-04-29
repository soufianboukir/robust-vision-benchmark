import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from PIL import Image
import os
from src.datasets.corruption_engine import (
    add_gaussian_noise,
    apply_blur,
    apply_occlusion,
    apply_jpeg_compression,
    apply_brightness_contrast,
    apply_rotation,
)
import torch
import torchvision.transforms as T

# Set page config (no emoji)
st.set_page_config(
    page_title="Robust Vision Benchmark",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

def tensor_to_pil(tensor):
    np_img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

@st.cache_data
def load_results():
    results_files = {
        "alexNet": "results/logs/alexNet_results.json",
        "leNet5": "results/logs/leNet5_results.json",
        "mlp3": "results/logs/mlp3_results.json",
        "mlp5": "results/logs/mlp5_results.json",
        "resnet18": "results/logs/resnet18_results.json",
    }
    all_results = {}
    for model_name, file_path in results_files.items():
        try:
            with open(file_path, "r") as f:
                all_results[model_name] = json.load(f)
        except FileNotFoundError:
            st.warning(f"Could not load {model_name} results")
    return all_results

all_results = load_results()
models_list = list(all_results.keys())
corruptions_list = [
    "blur",
    "gaussian_noise",
    "jpeg_compression",
    "contrast",
    "brightness",
    "rotation",
    "occlusion",
]

st.sidebar.title("Control Panel")
st.sidebar.markdown("---")
selected_model = st.sidebar.selectbox("Select Model", options=models_list, format_func=lambda x: x.upper())
selected_corruption = st.sidebar.selectbox(
    "Corruption Type",
    options=corruptions_list,
    format_func=lambda x: x.replace("_", " ").title(),
)
severity_level = st.sidebar.slider(
    "Severity Level",
    min_value=0,
    max_value=4,
    value=2,
    help="0 = Clean, 4 = Strongest corruption",
)
show_comparison = st.sidebar.checkbox("Compare All Models", value=True)
show_insights = st.sidebar.checkbox("Show Insights", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    Robust Vision Benchmark
    Evaluate model robustness under corruptions:
    - 5 Models (ResNet, AlexNet, LeNet, MLPs)
    - 7 Corruptions (blur, noise, occlusion, etc.)
    - 4 Severity Levels
    """
)

st.title("Robust Vision Benchmark")
st.markdown(
    """
This interactive dashboard evaluates the robustness of deep learning models
against image corruptions. Explore how different architectures degrade under
adversarial conditions.
"""
)
st.markdown("---")

st.header("Section 1: Clean Performance Dashboard")
col1, col2, col3, col4 = st.columns(4)
model_data = all_results[selected_model]
clean_metrics = model_data["clean"]
with col1:
    st.metric("Accuracy", f"{clean_metrics['accuracy']:.4f}", f"{clean_metrics['accuracy']*100:.2f}%")
with col2:
    st.metric("Precision", f"{clean_metrics['precision']:.4f}", f"{clean_metrics['precision']*100:.2f}%")
with col3:
    st.metric("Recall", f"{clean_metrics['recall']:.4f}", f"{clean_metrics['recall']*100:.2f}%")
with col4:
    st.metric("F1-Score", f"{clean_metrics['f1']:.4f}", f"{clean_metrics['f1']*100:.2f}%")

st.subheader("Clean Accuracy Comparison (All Models)")
clean_accs = {model: all_results[model]["clean"]["accuracy"] for model in models_list}
df_clean = pd.DataFrame({"Model": list(clean_accs.keys()), "Accuracy": list(clean_accs.values())}).sort_values("Accuracy", ascending=False)
fig, ax = plt.subplots(figsize=(12, 5))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_clean)))
ax.barh(df_clean["Model"], df_clean["Accuracy"], color=colors, edgecolor="black", linewidth=1.5)
ax.set_xlabel("Accuracy", fontweight="bold", fontsize=11)
ax.set_title("Clean Accuracy Comparison (Baseline Performance)", fontweight="bold", fontsize=13)
ax.set_xlim([0, 1.0])
ax.grid(axis="x", alpha=0.3)
for i, (model, acc) in enumerate(zip(df_clean["Model"], df_clean["Accuracy"])):
    ax.text(acc + 0.02, i, f"{acc:.4f}", va="center", fontweight="bold")
st.pyplot(fig)
plt.close()
st.markdown("---")

st.header("Section 2: Corruption Simulator")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sample Image")
    st.info("CIFAR-10 sample from dataset")
    image_path = "sample_image.png"
    if os.path.exists(image_path):
        pil_img = Image.open(image_path).convert("RGB")
        transform = T.Compose([T.ToTensor()])
        img = transform(pil_img)
        st.image(pil_img, caption="Original Image", use_container_width=True)
    else:
        st.error(f"Image not found: {image_path}")
        img = torch.rand(3, 32, 32)
        dummy_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        st.image(dummy_np, caption="Original Image (Random)", use_container_width=True)

with col2:
    st.subheader("Corrupted Image")
    st.info(f"Applied: {selected_corruption.replace('_', ' ').title()}")
    if selected_corruption == "blur":
        corrupted_tensor = apply_blur(img, severity_level)
    elif selected_corruption == "gaussian_noise":
        corrupted_tensor = add_gaussian_noise(img, severity_level)
    elif selected_corruption == "jpeg_compression":
        corrupted_tensor = apply_jpeg_compression(img, severity_level)
    elif selected_corruption == "contrast":
        corrupted_tensor = apply_brightness_contrast(img, severity_level, mode="contrast")
    elif selected_corruption == "brightness":
        corrupted_tensor = apply_brightness_contrast(img, severity_level, mode="brightness")
    elif selected_corruption == "rotation":
        corrupted_tensor = apply_rotation(img, severity_level)
    elif selected_corruption == "occlusion":
        corrupted_tensor = apply_occlusion(img, severity_level)
    else:
        corrupted_tensor = img.clone()
    corrupted_pil = tensor_to_pil(corrupted_tensor)
    st.image(corrupted_pil, caption="Corrupted Image", use_container_width=True)

corruption_data = model_data["corruptions"][selected_corruption]

# Handle clean case (severity_level == 0)
if severity_level == 0:
    st.subheader(f"{selected_corruption.title()} - Clean Image (Severity 0)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy (Clean)", f"{clean_metrics['accuracy']:.4f}")
    with col2:
        st.metric("Worst-Case Accuracy", f"{corruption_data['worst_case']:.4f}")
    with col3:
        drop_pct = ((clean_metrics['accuracy'] - corruption_data['worst_case']) / clean_metrics['accuracy']) * 100
        st.metric("Max Drop %", f"{drop_pct:.1f}%", delta=f"-{drop_pct:.1f}%")
else:
    idx = severity_level - 1  # map severity 1-4 to list indices 0-3
    st.subheader(f"{selected_corruption.title()} - Severity {severity_level}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy (This Severity)", f"{corruption_data['severities'][idx]['accuracy']:.4f}")
    with col2:
        st.metric("Worst-Case Accuracy", f"{corruption_data['worst_case']:.4f}")
    with col3:
        drop_pct = ((clean_metrics['accuracy'] - corruption_data['worst_case']) / clean_metrics['accuracy']) * 100
        st.metric("Max Drop %", f"{drop_pct:.1f}%", delta=f"-{drop_pct:.1f}%")

st.markdown("---")

st.header("Section 3: Performance Under Corruption")
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Degradation: {selected_corruption.title()}")
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models_list if show_comparison else [selected_model]:
        model_corruption_data = all_results[model]["corruptions"][selected_corruption]
        accuracies = [sev["accuracy"] for sev in model_corruption_data["severities"]]
        severity_levels_plot = list(range(len(accuracies)))
        ax.plot(severity_levels_plot, accuracies, marker="o", linewidth=2.5, label=model.upper(), markersize=8)
    ax.set_xlabel("Corruption Severity", fontweight="bold", fontsize=11)
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=11)
    ax.set_title(f"{selected_corruption.title()} - Degradation Curve", fontweight="bold", fontsize=12)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Clean", "L1", "L2", "L3", "L4"])
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1.0])
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("All Corruptions - Current Model")
    worst_accs = {c: model_data["corruptions"][c]["worst_case"] for c in corruptions_list}
    df_corruptions = pd.DataFrame({"Corruption": list(worst_accs.keys()), "Accuracy": list(worst_accs.values())}).sort_values("Accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_corruptions)))
    ax.barh(df_corruptions["Corruption"], df_corruptions["Accuracy"], color=colors, edgecolor="black", linewidth=1.5)
    ax.set_xlabel("Worst-Case Accuracy", fontweight="bold", fontsize=11)
    ax.set_title(f"{selected_model.upper()} - Worst Accuracy per Corruption", fontweight="bold", fontsize=12)
    ax.set_xlim([0, 1.0])
    ax.grid(axis="x", alpha=0.3)
    for i, (corruption, acc) in enumerate(zip(df_corruptions["Corruption"], df_corruptions["Accuracy"])):
        ax.text(acc + 0.02, i, f"{acc:.3f}", va="center", fontsize=9)
    st.pyplot(fig)
    plt.close()
st.markdown("---")

st.header("Section 4: Corruption Impact Ranking")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Sensitivity Score (Global Ranking)")
    sensitivity_scores = {}
    for model in models_list:
        for corruption in corruptions_list:
            sensitivity_scores.setdefault(corruption, []).append(
                all_results[model]["corruption_sensitivity_scores"][corruption]
            )
    avg_sensitivity = {k: np.mean(v) for k, v in sensitivity_scores.items()}
    df_sensitivity = pd.DataFrame({"Corruption": list(avg_sensitivity.keys()), "Sensitivity": list(avg_sensitivity.values())}).sort_values("Sensitivity", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(df_sensitivity)))
    ax.bar(df_sensitivity["Corruption"], df_sensitivity["Sensitivity"], color=colors, edgecolor="black", linewidth=1.5, alpha=0.85)
    ax.set_ylabel("Sensitivity Score", fontweight="bold", fontsize=11)
    ax.set_title("Corruption Severity Ranking (Higher = More Destructive)", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for i, (corruption, score) in enumerate(zip(df_sensitivity["Corruption"], df_sensitivity["Sensitivity"])):
        ax.text(i, score + 0.01, f"{score:.3f}", ha="center", fontweight="bold", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close()
with col2:
    st.subheader("Ranking")
    for idx, row in df_sensitivity.iterrows():
        st.write(f"**{row['Corruption'].title()}** -> {row['Sensitivity']:.3f}")
st.markdown("---")

st.header("Section 5: Model x Corruption Heatmap")
st.subheader("Worst-Case Accuracy Matrix")
heatmap_data = []
for model in models_list:
    row = [all_results[model]["corruptions"][corruption]["worst_case"] for corruption in corruptions_list]
    heatmap_data.append(row)
df_heatmap = pd.DataFrame(heatmap_data, columns=corruptions_list, index=models_list)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.15, vmax=0.95,
            cbar_kws={"label": "Worst-Case Accuracy"}, ax=ax, linewidths=0.5)
ax.set_title("Model x Corruption Heatmap (Worst-Case Accuracy)", fontweight="bold", fontsize=13, pad=15)
ax.set_xlabel("Corruption Type", fontweight="bold", fontsize=11)
ax.set_ylabel("Model", fontweight="bold", fontsize=11)
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)
plt.close()
st.markdown("---")

st.header("Section 6: Model Robustness Comparison")
model_stats = []
for model in models_list:
    clean_acc = all_results[model]["clean"]["accuracy"]
    worst_accs = [all_results[model]["corruptions"][c]["worst_case"] for c in corruptions_list]
    avg_worst_acc = np.mean(worst_accs)
    avg_drop = ((clean_acc - avg_worst_acc) / clean_acc) * 100
    model_stats.append({
        "Model": model.upper(),
        "Clean Accuracy": clean_acc,
        "Avg Worst Accuracy": avg_worst_acc,
        "Avg Drop %": avg_drop,
        "Min Accuracy": min(worst_accs),
        "Max Accuracy": max(worst_accs)
    })
df_stats = pd.DataFrame(model_stats).sort_values("Avg Drop %", ascending=True)

st.subheader("Average Accuracy Drop (%)")
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(df_stats)))
bars = ax.bar(df_stats["Model"], df_stats["Avg Drop %"], color=colors, edgecolor="black", linewidth=1.5, alpha=0.8)
ax.set_ylabel("Average Drop %", fontweight="bold", fontsize=11)
ax.set_title("Robustness: Lower Drop = More Robust", fontweight="bold", fontsize=12)
ax.grid(axis="y", alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f"{height:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
st.pyplot(fig)
plt.close()


st.subheader("Detailed Comparison Table")
st.dataframe(df_stats.style.format({
    "Clean Accuracy": "{:.4f}",
    "Avg Worst Accuracy": "{:.4f}",
    "Avg Drop %": "{:.2f}",
    "Min Accuracy": "{:.4f}",
    "Max Accuracy": "{:.4f}"
}), use_container_width=True)
st.markdown("---")

st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 12px;'>
<p>Robust Vision Benchmark</p>
<p>Models: ResNet-18, AlexNet, LeNet-5, MLP-3, MLP-5</p>
<p>Dataset: CIFAR-10</p>
<p>Corruptions: 7 types x 4 severity levels</p>
</div>
""",
    unsafe_allow_html=True,
)