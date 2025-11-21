Intro to ML Group 1 Final Project

# Intro to ML – Group 1 Final Project (MNIST)

This repo contains our MNIST classification project for AIGC-5102.

## Folder structure

- `data/` – MNIST `.gz` files (train/test images + labels).
- `notebooks/`
  - `01_data_exploration.ipynb` – load MNIST from `data/`, basic EDA and sample digits.
  - `02_baseline_model.ipynb` – multinomial Logistic Regression baseline on flattened pixels.
  - `03_mlp_model.ipynb` – 1-hidden-layer MLP (256 ReLU units) on flattened pixels, better than baseline.
  - `04_cnn_model.ipynb` – small CNN on 28×28 images, best performance.
- `reports/` – exported plots (confusion matrices etc.) for the final report.
- `src/` – (optional) helper code if we refactor functions.
- `azure/` – Azure ML / AutoML screenshots later.

## How to run (Colab)

1. Open any notebook from the `notebooks/` folder in GitHub.
2. Click the **"Open in Colab"** button (if available) or copy the GitHub URL into Colab (`File → Open notebook → GitHub`).
3. Run the cells from top to bottom.
   - Each notebook automatically looks for the MNIST `.gz` files in `data/`.

## Models and results (test set)

| Model                     | Features           | Test Accuracy | Test Macro F1 |
|--------------------------|--------------------|--------------:|--------------:|
| Logistic Regression      | 784 raw pixels     | ~0.916        | ~0.915        |
| MLP (1×256 ReLU)         | 784 raw pixels     | ~0.974        | ~0.974        |
| CNN (2 conv + 1 dense)   | 28×28 images (1ch) | ~0.989        | ~0.989        |

The CNN clearly outperforms the linear baseline and the MLP, and its confusion matrix is almost perfectly diagonal.
