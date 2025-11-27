### Model comparison

| Model                                 | Features           | Test Accuracy | Test Macro F1 |
|--------------------------------------|--------------------|--------------:|--------------:|
| Logistic Regression (tuned)          | 784 raw pixels     | 0.9051        | 0.9037        |
| k-NN (tuned)                         | 784 raw pixels     | 0.9435        | 0.9429        |
| MLP (1 hidden layer, tuned)          | 784 raw pixels     | 0.9642        | 0.9638        |
| CNN (2 conv + 1 dense, manually tuned)| 28×28 images (1ch) | 0.9893        | 0.9892        |

We start from a multinomial logistic regression baseline which reaches about 90–91% test accuracy
and macro F1 on the MNIST test set. Replacing this linear model with a k-nearest-neighbours
classifier on the same flattened 784-pixel features improves performance to around 94%.

Adding a single hidden layer (ReLU MLP) pushes test accuracy and macro F1 into the mid-96% range,
showing that even a simple dense neural network can extract non-linear patterns that the linear
model and k-NN miss.

Finally, using a small convolutional neural network that operates directly on the 28×28 images
further boosts performance to roughly 98.9% accuracy and macro F1. The CNN’s confusion matrix
is almost perfectly diagonal, which means it makes very few mistakes and handles all digit
classes more consistently than the other models.

In terms of tuning, Logistic Regression, k-NN and the MLP are all trained via small
GridSearchCV runs (on a subset of the training data) over key hyperparameters such as C,
number of neighbours, hidden layer size and L2 regularisation. The CNN is tuned manually
by trying a few reasonable architectures and training settings and selecting the one
with the best validation performance, mainly due to computational cost.
