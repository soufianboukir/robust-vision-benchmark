# Robust Vision Benchmark - Streamlit App

A comprehensive interactive dashboard for evaluating the robustness of deep learning models against image corruptions.

## Features

### Control Panel (Sidebar)
- Select from 5 pre-trained models
- Choose from 7 corruption types
- Adjust corruption severity (0-4)
- Toggle model comparison mode

### 6 Interactive Sections

**Section 1: Clean Performance Dashboard**
- View baseline accuracy metrics (Accuracy, Precision, Recall, F1)
- Compare clean accuracy across all models
- Bar chart visualization

**Section 2: Corruption Simulator**
- Display original CIFAR-10 sample image
- Show corrupted version with applied corruption
- Display per-corruption statistics
- Worst-case accuracy metrics

**Section 3: Performance Under Corruption**
- Degradation curves showing accuracy vs severity
- Compare multiple models on selected corruption
- Per-model worst-accuracy ranking

**Section 4: Corruption Impact Ranking**
- Sensitivity score ranking (which corruptions are most harmful)
- Global comparison across all models
- Visual ranking with color coding

**Section 5: Model × Corruption Heatmap**
- Matrix visualization of all model-corruption combinations
- Worst-case accuracy color-coded
- Easy identification of weak spots

**Section 6: Model Robustness Comparison**
- Average accuracy drop (% loss under corruption)
- Worst-case accuracy retention
- Detailed comparison table
- Identifies accuracy-robustness trade-offs

---

## Installation & Setup

### Step 1: Clone and Navigate to Project
```bash
git clone https://github.com/soufianboukir/robust-vision-benchmark.git
cd robust-vision-benchmark
```

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Train models using this command 
- For each model, modify the `config/config.py`, set the variable value to the model name,
- names must be like this: (mlp3,mlp5,leNet5,alexNet,resnet18),
- You can adjust also epochs, base lr, ...
- 
```
python -m src.training.train
```
- do this for all models

### Step 4: Ensure Results Data Exists
Make sure you have these JSON files in the `results/` directory:
```
results/
├─ logs
   ├── alexNet_results.json
   ├── leNet5_results.json
   ├── mlp3_results.json
   ├── mlp5_results.json
   └── resnet18_results.json
```

### Step 4: Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Folder Structure Expected

```
robust-vision-benchmark/
├── app.py                          # Main Streamlit app
├── requirements.txt      # Dependencies
├── results/
├──── logs/
│     ├── alexNet_results.json
│     ├── leNet5_results.json
│     ├── mlp3_results.json
│     ├── mlp5_results.json
│     └── resnet18_results.json
├── saved_models/
│   ├── alexNet.pt
│   ├── leNet5.pt
│   ├── mlp3.pt
│   ├── mlp5.pt
│   └── resnet18.pt
├── src/
│   ├── datasets/
│   ├── models/
│   └── corruption_engine.py
└── data/
    └── raw/
```

---

## 🎮 How to Use

### Basic Workflow

1. **Select a Model** (sidebar)
   - Choose from: ResNet-18, AlexNet, LeNet-5, MLP-3, MLP-5

2. **Choose a Corruption** (sidebar)
   - Options: Blur, Gaussian Noise, JPEG Compression, Contrast, Brightness, Rotation, Occlusion

3. **Adjust Severity** (slider, 0-4)
   - 0 = No corruption (clean)
   - 4 = Strongest corruption

4. **Toggle Comparison Mode** (checkbox)
   - Shows all models vs just selected model

5. **View Results**
   - Scroll through all 7 sections
   - Each section provides different insights
   - Color-coded visualizations (red=bad, green=good)

### Advanced Usage

**Identify Model Weaknesses:**
- Use the heatmap (Section 5) to find which model-corruption pairs perform worst
- Red cells indicate vulnerabilities

**Compare Trade-offs:**
- Section 6 shows accuracy vs robustness
- ResNet-18 has high clean accuracy but poor robustness
- MLPs have lower clean accuracy but better robustness

**Export for Paper:**
- Download CSV files for results tables
- All plots are publication-ready
- Insights panel provides discussion points

---

## 🔧 Customization

### Change Colors
In `app.py`, modify colormap:
```python
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(df)))  # Red-Yellow-Green
colors = plt.cm.cool(np.linspace(0.2, 0.8, len(df)))    # Blue-Cyan
colors = plt.cm.viridis(np.linspace(0, 1, len(df)))     # Viridis
```

### Add New Models
1. Generate results JSON for new model
2. Update `models_list` in app.py
3. Update results loading function

### Add New Corruptions
1. Generate results with new corruption type
2. Add to `corruptions_list`
3. Results automatically appear in all sections

### Modify Thresholds
Change sensitivity score thresholds:
```python
st.sidebar.slider(..., min_value=0, max_value=4, value=2)
```

---

## 📊 Key Insights from Dashboard

### Model Rankings
1. **Most Robust**: MLP-3 (27% avg drop)
2. **Best Clean**: ResNet-18 (85.91% accuracy)
3. **Most Balanced**: MLP-5
4. **Least Robust**: ResNet-18 (50.9% avg drop)

### Corruption Rankings
1. **Most Harmful**: Blur (56% avg drop)
2. **Moderately Harmful**: JPEG, Gaussian Noise
3. **Least Harmful**: Occlusion (16% avg drop)

### Trade-offs
- Deep architectures (ResNet) → High accuracy, low robustness
- Shallow architectures (MLP) → Low accuracy, high robustness
- CNNs vulnerable to blur/occlusion affecting spatial features

---

## 🐛 Troubleshooting

**"ModuleNotFoundError: No module named 'streamlit'"**
```bash
pip install streamlit
```

**"FileNotFoundError: results/alexNet_results.json"**
- Make sure you've generated results JSON files
- Update file paths if results are in different location

**"Port 8501 already in use"**
```bash
streamlit run app.py --server.port 8502
```

**Charts not displaying**
- Ensure matplotlib is installed: `pip install matplotlib`
- Try clearing browser cache

---

## 📈 Example Analysis Workflow

1. **Identify Problem**: "Which model should I use?"
   - Go to Section 1, compare clean accuracy
   - Check Section 6 for robustness metrics

2. **Diagnose Weakness**: "Why is Model X bad?"
   - Go to Section 5 (Heatmap), find red cells
   - Select that corruption in sidebar
   - View degradation curve in Section 3

3. **Benchmark Against Paper**: "How do these models compare?"
   - Export CSV from Export section
   - Use in manuscript
   - Reference insights from Section 7

4. **Find Best For Application**: "I need robustness"
   - Sort Section 6 by "Avg Drop %"
   - Choose lowest drop percentage
   - Check if clean accuracy is acceptable

---

## 🤝 Contributing

To extend this app:

1. Add new visualizations by creating new sections
2. Add new analysis metrics
3. Improve existing plots
4. Add interactive model selection for side-by-side comparison

---

## 📝 Citation

If you use this benchmark in your research:

```
@app{RobustVisionBenchmark2024
  title={Robust Vision Benchmark: Interactive Dashboard},
  year={2024}
}
```

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all JSON files exist in results/
3. Ensure all dependencies are installed
4. Check Streamlit documentation: https://docs.streamlit.io/

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Production Ready ✅
