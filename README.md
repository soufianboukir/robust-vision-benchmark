


### Project structure

```
vision-robustness-project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── corrupted/
│
├── src/
│   ├── datasets/
│   │   ├── cifar_loader.py
│   │   ├── corruption_engine.py
│   │
│   ├── models/
│   │   ├── simple_cnn.py
│   │   ├── medium_cnn.py
│   │   ├── resnet_model.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │
│   ├── experiments/
│   │   ├── run_clean.py
│   │   ├── run_corrupted.py
│   │
│   ├── config.py
│
├── notebooks/
│   ├── analysis.ipynb
│   ├── corruption_visualization.ipynb
│
├── results/
│   ├── plots/
│   ├── confusion_matrices/
│   ├── logs/
│
├── reports/
│   ├── final_report.pdf
│
├── requirements.txt
├── README.md
└── main.py
```