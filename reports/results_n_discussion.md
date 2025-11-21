Baseline: Logistic Regression

The multinomial logistic regression baseline on flattened pixels achieved about 91–92% accuracy and macro F1 on the test set (test accuracy ≈ 0.916, macro F1 ≈ 0.915). The confusion matrix was mostly diagonal, but some digits such as 5 and 8 had noticeably more errors than others. This confirms that even a simple linear model can learn useful decision boundaries for MNIST, but it struggles with more ambiguous handwriting.

Improved model: MLP

When we switched from a linear model to a small MLP with one hidden layer of 256 ReLU units, performance improved clearly. Using the same train/validation split and flattened pixels, the MLP reached about 97.2% validation accuracy (macro F1 ≈ 0.97) and 97.4% test accuracy (macro F1 ≈ 0.974). The test confusion matrix for the MLP is almost perfectly diagonal, and all digit classes have F1 scores around 0.96–0.99. This shows that adding non-linear hidden layers already makes a big difference, even without changing the input representation.

Final model: CNN

The best performance came from the CNN that works directly on the 28×28 image grids. With two convolutional layers, max-pooling, and a 128-unit dense layer, the CNN reached about 98.9% accuracy and macro F1 on both validation and test sets (test accuracy ≈ 0.9893, macro F1 ≈ 0.9892). Compared to the logistic regression baseline, this is a gain of roughly +7 percentage points, and even compared to the MLP it is about +1.5 points. The CNN confusion matrix is very close to a perfect diagonal, meaning the model is extremely consistent across all ten digits.

Error analysis:

Even with the CNN, there are still a small number of mistakes. By inspecting misclassified test images, we observed that most of them are visually ambiguous or very messy digits, such as a 5 that looks like a 3 or 8, or a 9 that is written almost like a 4. In other words, the remaining errors seem to be more about how the digits are written rather than a systematic problem with the model. This suggests that further gains would probably require either more complex architectures or data augmentation to handle unusual handwriting styles, rather than simple hyper-parameter tuning.
