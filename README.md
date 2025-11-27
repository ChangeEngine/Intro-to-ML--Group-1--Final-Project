Intro to ML Group 1 Final Project

# Intro to ML – Group 1 Final Project (MNIST)

This repo contains our MNIST classification project for AIGC-5102 (Intro to Machine Learning).  
We compare several models on the MNIST handwritten digit dataset and follow the course
requirement of exploring multiple algorithms with hyperparameter tuning and proper evaluation.

## Folder structure

- `data/` – MNIST `.gz` files (train/test images + labels).
- `notebooks/`
  - `01_data_exploration.ipynb` – load MNIST from `data/`, basic EDA and sample digits.
  - `02_baseline_model.ipynb` – multinomial Logistic Regression baseline on flattened pixels,
    with a light GridSearchCV over the regularization strength `C` (on a training subset).
  - `03_mlp_model.ipynb` – 1-hidden-layer MLP (dense neural network) on flattened pixels,
    tuned with GridSearchCV over hidden layer size and L2 regularization (`alpha`).
  - `04_cnn_model.ipynb` – small CNN on 28×28 images. Architecture and training settings were
    tuned manually (no full GridSearchCV because of training time), but we report validation
    and test metrics plus a confusion matrix.
  - `05_knn_model.ipynb` – k-Nearest Neighbors on flattened pixels, tuned with GridSearchCV
    over `k` (number of neighbors) and the distance weighting scheme.
- `reports/` – exported plots (confusion matrices, misclassified examples, etc.) for the final report.
- `azure/` – Azure ML Designer and AutoML screenshots

## How to run (Colab)

1. Open any notebook from the `notebooks/` folder in GitHub.
2. In Colab: `File → Open notebook → GitHub` and paste the repo URL, then pick the notebook.
3. Run the cells from top to bottom.
   - Each notebook automatically looks for the MNIST `.gz` files in `data/`.
   - All models use `random_state=42` where relevant for reproducibility.

## Models and results (test set)

All metrics are on the **10k MNIST test set**. Accuracy and F1 are macro-averaged across digits.

| Model                                 | Features           | Test Accuracy | Test Macro F1 |
|---------------------------------------|--------------------|--------------:|--------------:|
| Logistic Regression (tuned)           | 784 raw pixels     | ~0.905        | ~0.904        |
| k-NN (tuned)                          | 784 raw pixels     | ~0.944        | ~0.943        |
| MLP (1 hidden layer, tuned)           | 784 raw pixels     | ~0.964        | ~0.964        |
| CNN (2 conv + 1 dense, manually tuned)| 28×28 images (1ch) | ~0.989        | ~0.989        |

### High-level story

We start from a multinomial logistic regression baseline at about **90–91%** test accuracy and macro F1.  
Switching to **k-Nearest Neighbors** on the same flattened pixels increases performance to around **94%**.  
A 1-hidden-layer **MLP** pushes test accuracy and macro F1 to about **96–97%**, showing that non-linear features already help a lot on MNIST.  
Finally, a small **CNN** that works directly on the 28×28 images reaches roughly **98.9%** accuracy and macro F1, with a confusion matrix that is almost perfectly diagonal.
