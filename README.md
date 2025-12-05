Intro to ML Group 1 Final Project

# Intro to ML – Group 1 Final Project (MNIST)

This repo contains our **MNIST handwritten digit classification** project for  
**AIGC-5102 – Intro to Machine Learning** (Group 1).

The main course deliverable is implemented in **Azure Machine Learning Designer**, and we
also include an additional notebook-based study in `gen_ai_study/` for extra experiments.

---

## Folder structure

- `data/` – Original MNIST `.gz` files (train/test images + labels).
- `azure/`
  - `reports/report.docx` – Final project report (editable).
  - `reports/report.pdf` – Final project report (submitted version).
  - (Optional) extra screenshots/exports from Azure ML (pipelines, metrics, etc.).
- `gen_ai_study/`
  - Separate mini-project with Jupyter notebooks (baseline, kNN, MLP, CNN, etc.).
  - Has its **own** `README.md` describing the notebook workflow and results.
- `data_preparation.py`
  - Helper script to turn the raw MNIST `.gz` files into a single cleaned table
    (e.g., Parquet/CSV) suitable for upload to Azure ML Designer.


---

## 1. Azure ML Designer pipeline (main deliverable)

The core of the project is an **Azure ML Designer pipeline** that:

1. **Ingests data**

   - Uploads the cleaned MNIST table (generated from `data_preparation.py`) as an Azure dataset.
   - The table contains 784 pixel columns, a `label` column (0–9), and an optional `usage` flag.

2. **Preprocesses data**

   - Uses a Python script module to:
     - Split rows into train/test based on `usage`.
     - Drop helper columns.
     - Convert pixel values to `float` and normalize them to `[0, 1]`.

3. **Trains and tunes models**

   We implement and compare the following Designer components:

   - **Multiclass Logistic Regression** – linear baseline on flattened pixels.
   - **Multiclass Decision Forest** – bagged ensemble of decision trees.
   - **Multiclass Boosted Decision Tree** – boosted tree ensemble (our best model).
   - **Multiclass Neural Network (MLP)** – fully connected network on 784-dimensional input.

   Each model is connected to **Tune Model Hyperparameters** in “Entire grid” mode, using
   **Accuracy** as the optimization metric and a validation split created by `Split Data`.

4. **Evaluation**

   - Uses `Score Model` + `Evaluate Model` to compute accuracy, micro/macro precision & recall.
   - Exports confusion matrices for the final report.
   - Results and screenshots are saved under `azure/reports/`.

---

## 2. Additional notebook-based study – `gen_ai_study/`

Besides the Azure pipeline, we also explored MNIST models in plain Python / Jupyter,
stored under `gen_ai_study/`. That mini-project includes:

- A multinomial **Logistic Regression** baseline.
- **k-Nearest Neighbours (kNN)** on flattened pixels.
- A 1-hidden-layer **MLP** (dense neural network).
- A small **CNN** that operates on 28×28 images.

Those experiments have their own folder layout (`data/`, `notebooks/`, `reports/`) and a
separate `README.md` with more details, usage instructions and metrics.

---

## 3. How to run / reproduce

### 3.1 Data preparation (local)

1. Download the four MNIST `.gz` files and place them under `data/`:

   - `train-images.idx3-ubyte.gz`
   - `train-labels.idx1-ubyte.gz`
   - `t10k-images.idx3-ubyte.gz`
   - `t10k-labels.idx1-ubyte.gz`

2. (Optional but recommended) Run:

   ```bash
   python data_preparation.py
