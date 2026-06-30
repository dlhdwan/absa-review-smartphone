# ABSA Review Smartphone

A deep learning project for **Vietnamese Aspect-Based Sentiment Analysis (ABSA)** on smartphone reviews.

This project aims to automatically identify aspects mentioned in Vietnamese smartphone reviews and predict the corresponding sentiment polarity (**Positive**, **Negative**, or **Neutral**) for each aspect. It investigates and compares multiple PhoBERT-based architectures under different ABSA formulations, including both end-to-end and two-stage approaches.

## Features

This repository includes:

- One-stage ABSA using PhoBERT
- Two-stage ABSA (Aspect Term Extraction + Aspect Category Sentiment Classification)
- PhoBERT + BiLSTM + TC-LSTM architecture
- Training and evaluation notebooks
- Streamlit demonstration application
- Evaluation metrics and visualization results
- Pretrained checkpoints hosted on Hugging Face

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/dlhdwan/absa-review-smartphone.git
cd absa-review-smartphone
```

---

## 2. Install uv (Recommended)

```bash
pip install uv
```

Verify installation:

```bash
uv --version
```

---

## 3. Install dependencies

```bash
uv sync
```

This command automatically

- creates a virtual environment (`.venv`)
- installs all dependencies from `pyproject.toml`
- restores exact package versions from `uv.lock`

---

## 4. Activate the virtual environment

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### Windows (CMD)

```cmd
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 5. Download pretrained checkpoints

Model checkpoints are hosted on **Hugging Face** instead of GitHub to keep this repository lightweight.

```bash
python scripts/download_checkpoints.py
```

The script automatically downloads all pretrained models into

```text
checkpoints/
```

Only the first execution requires downloading.

---

## 6. Prepare datasets

Download the datasets below and place them into

```text
data/
├── UIT-ViSD4SA/
└── UIT-ViSFD/
```

---

# Datasets

The experiments are conducted using two publicly available Vietnamese datasets.

## UIT-ViSD4SA

Repository:

https://github.com/kimkim00/UIT-ViSD4SA

UIT-ViSD4SA is a benchmark dataset for Vietnamese Aspect-Based Sentiment Analysis in the smartphone domain. Each review is annotated with predefined aspect categories and corresponding sentiment labels.

Dataset structure

```text
data/
└── UIT-ViSD4SA/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

Used for

- One-stage ABSA
- Two-stage ABSA (ATE + ACSC)
- PhoBERT + BiLSTM + TC-LSTM
- End-to-end evaluation

---

## UIT-ViSFD

Repository:

https://github.com/LuongPhan/UIT-ViSFD

UIT-ViSFD (Vietnamese Smartphone Feedback Dataset) contains Vietnamese smartphone reviews collected from e-commerce platforms and is widely used for Vietnamese sentiment analysis research.

Dataset structure

```text
data/
└── UIT-ViSFD/
    ├── Train.csv
    ├── Dev.csv
    └── Test.csv
```

---
# Training Environment

All models in this repository were trained and evaluated using **Google Colaboratory** with **NVIDIA Tesla T4 GPU**.

The training pipelines are organized into three independent Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `pipeline1.ipynb` | One-stage ABSA using PhoBERT |
| `ate_acsc_base.ipynb` | Two-stage ABSA (ATE + ACSC) |
| `absa_phobert_bilstm_tclstm.ipynb` | PhoBERT + BiLSTM + TC-LSTM |

Each notebook is self-contained and includes:

- Data preprocessing
- Model training
- Validation
- Testing
- Performance evaluation
- Visualization of experimental results

No local GPU is required for reproducing the training experiments if Google Colab with a **Tesla T4 GPU** is available.

---

# Running the Project

## Streamlit Demo

```bash
streamlit run demo/app.py
```

---

## Jupyter Notebook

```bash
jupyter notebook
```

Open any notebook under

```text
notebooks/
```

---

# Project Structure

```text
absa-review-smartphone/
│
├── checkpoints/                 # Downloaded automatically from Hugging Face
├── data/
│   ├── UIT-ViSD4SA/
│   └── UIT-ViSFD/
├── demo/
│   └── app.py
├── notebooks/
│   ├── ate_acsc_base.ipynb
│   ├── absa_phobert_bilstm_tclstm.ipynb
│   ├── pipeline1.ipynb
│   └── test.ipynb
├── outputs/
├── scripts/
│   └── download_checkpoints.py
├── src/
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Outputs

The `outputs/` directory contains the generated experimental results, including

- Training curves
- Evaluation metrics
- Confusion matrices
- Classification reports
- Performance visualizations

---

# Notebooks

The `notebooks/` directory contains the complete experimental pipelines, including

- One-stage ABSA
- Two-stage ABSA (ATE + ACSC)
- PhoBERT + BiLSTM + TC-LSTM
- Training
- Evaluation
- Visualization

---

# Model Checkpoints

To keep the GitHub repository lightweight, pretrained checkpoints are **not stored on GitHub**.

They are hosted on **Hugging Face** and can be downloaded automatically using

```bash
python scripts/download_checkpoints.py
```

---

# Requirements

- Python 3.11+
- uv
- PyTorch
- Transformers
- Streamlit

---

# License

This repository is intended for academic research and educational purposes.