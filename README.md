# Robust Vision Benchmark - Streamlit App

A comprehensive interactive dashboard for evaluating the robustness of deep learning models against image corruptions.

<img width="2045" height="1036" alt="Screenshot from 2026-04-30 12-39-09" src="https://github.com/user-attachments/assets/73ce6d07-f6c7-4ef3-99c5-c8d909f5d590" />


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

## Folder Structure Expected

```
robust-vision-benchmark/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── .gitignore               
├── LICENSE               
├── sample_iamage.png               
├── config/
│   └── config.py
├── data/
├── notebooks/
│   ├── cifar-10.ipynb
│   └── corruption.ipynb
├── previous-work/
├── reports/
├── results/
├──── imgs/
├──── plots/
├──── loss/
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
│   └── training/
├── utils/
└── visualizations/
```

---

## How to Use

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
   - Scroll through all 6 sections
   - Each section provides different insights

---

## Key Insights from Dashboard

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

built with ❤️ by **soufian**.
