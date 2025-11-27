from pathlib import Path
import gzip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Try to locate the data/ folder.
DATA_DIR = Path("data")
if not DATA_DIR.exists():
    DATA_DIR = Path("..") / "data"

# print("Using data folder:", DATA_DIR.resolve())
# print("Files in data/:")
# for p in DATA_DIR.iterdir():
#     print(" -", p.name)


plt.rcParams["figure.figsize"] = (4, 4)

train_images_path = DATA_DIR / "train-images.idx3-ubyte.gz"
train_labels_path = DATA_DIR / "train-labels.idx1-ubyte.gz"
test_images_path  = DATA_DIR / "t10k-images.idx3-ubyte.gz"
test_labels_path  = DATA_DIR / "t10k-labels.idx1-ubyte.gz"

# print(train_images_path)
# print(train_labels_path)
# print(test_images_path)
# print(test_labels_path)

def load_mnist_images(path):
    # idx3-ubyte.gz: first 16 bytes = header
    with gzip.open(path, "rb") as f:
        data = f.read()

    header = np.frombuffer(data[:16], dtype=">i4")
    magic, num_images, rows, cols = header
    assert magic == 2051, "Not an images file"

    images = np.frombuffer(data[16:], dtype=np.uint8)
    images = images.reshape(num_images, rows, cols)
    return images


def load_mnist_labels(path):
    # idx1-ubyte.gz: first 8 bytes = header
    with gzip.open(path, "rb") as f:
        data = f.read()

    header = np.frombuffer(data[:8], dtype=">i4")
    magic, num_labels = header
    assert magic == 2049, "Not a labels file"

    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels


def load_mnist_dataset(images_path, labels_path, normalize=True):
    X = load_mnist_images(images_path)
    y = load_mnist_labels(labels_path)

    if normalize:
        X = X.astype("float32") / 255.0

    return X, y


X_train, y_train = load_mnist_dataset(train_images_path, train_labels_path)
X_test, y_test = load_mnist_dataset(test_images_path, test_labels_path)

# print("X_train:", X_train.shape, X_train.dtype)
# print("y_train:", y_train.shape, y_train.dtype)
# print("X_test :", X_test.shape, X_test.dtype)
# print("y_test :", y_test.shape, y_test.dtype)


# def show_examples(images, labels, n=10):
#     plt.figure(figsize=(n, 1))
#     for i in range(n):
#         plt.subplot(1, n, i + 1)
#         plt.imshow(images[i], cmap="gray")
#         plt.axis("off")
#         plt.title(int(labels[i]))
#     plt.tight_layout()
#     plt.show()
#
#
# show_examples(X_train, y_train, n=10)

X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

features = [f"Pixel_{x}" for x in range(784)]


def create_df(x, y):
    y_df = pd.DataFrame(y, columns=["label"])

    X_df = pd.DataFrame(x, columns=features)

    df = pd.merge(y_df, X_df, left_index=True, right_index=True, how="left")

    return df


train_df = create_df(X_train_flat, y_train)
test_df = create_df(X_test_flat, y_test)

train_df.to_csv("data/output/train_data.csv")
test_df.to_csv("data/output/test_data.csv")



