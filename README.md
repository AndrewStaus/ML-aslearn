# AS-Learn
**aslearn** is the name of my toy ML library.  It is named aslearn in homage to scikit-learn (sklearn) replacing “sk” with my initials “as”-- very creative I know. As the name might suggest, aslearn been developed to function with a similar API to scikit-learn, with a few extra features that I thought would be interesting.  I have not looked at the scikit-learn source code or documentation for reference and have reversed engineered its functionality solely from memory, so some things may be a bit different.  Because aslearn is implemented completely in python using only the numpy library.  I have vectorized the math as much as I could with numpy to keep performance reasonable, but the main goal of the project is not bleeding edge performance.

# Overview:
The goal of this project is to create a fully featured production quality machine learning library that has support for multiple machine learning models.
While there are multiple libraries already available, building one from the ground up will help me learn how the models work under the hood, and put into practice object orientated methodologies.

Please review the workbooks to see the library in use, as well as the aslearn.py for the library code.
 - <a href = 'https://github.com/AndrewStaus/ML-aslearn/blob/main/aslearn.py'> aslearn Library</a>
 - <a href = 'https://github.com/AndrewStaus/ML-aslearn/blob/main/Notebook%20-%20Neural%20Network%20on%20MNIST.ipynb'> Neural Network AND PCA</a>
 - <a href = 'https://github.com/AndrewStaus/ML-aslearn/blob/main/Notebook%20-%20PCA%20and%20K%20Means%20on%20MNIST.ipynb'> K-Means and PCA</a>
 - <a href = 'https://github.com/AndrewStaus/ML-aslearn/blob/main/Notebook%20-%20Logistic%20Regression%20on%20MNIST.ipynb'> Logistic Regression</a>


# Supported Models:

**1. Neural Network:**
  - Fully Connected Deep Neural Network.
  - Backpropagation with a SGD optimizer
  - Reinforcement learning through a genetic algorithm
  - Multiple Activation Functions including leaky ReLU, Softmax, and Linear

**2. Stochastic Gradient Descent:**
  - Two child classes: Linear Regression and Logistic Regression
  - Logistic Regression supports multi-class data detection and automatically trains required classifiers

**3. Principal Comonent Analysis:**
  - Dimensionality reduction through eigenvectors
  - Supports specified number of features
  - Supports variance percentage to automatically select number of feature to retain variance

**4. K-Means:**
  -  Cluster assignment through finding the mean value of centroids

# Utilities:
  **1.Scaler:**
  - Fits to the mean and standard deviation of the data and transforms it to be centered around 0
  
  **2. Label Encoder:**
  - Replaces Labels with integer values, retains mapping so data can be transformed back to original
  
  **3. One Hot Encoder**
  - Replaces Labels with one hot encoded array, retains mapping so data can be transformed back to original
  
  **4. Helper Functions:**
  - Confusion Matrix: Creates confusion matrix data for classification validation
  - Under Sample: Rebalances dataset so that all classes have equal representation by pruning excess classes
  - Prepend Ones: Adds a column vector of ones to the start of a matrix.  Useful for many machine learning algorithms that use biases
  - Prepend Zeros: Adds a column vector of zeros to the start of a matrix.  Useful for regularized back propagation
  - Shuffle:  Shuffles two datasets such that the indexes of the data are aligned.  Useful for shuffling X and y matrices between epoch iterations while keeping samples and targets aligned
  - Train Test Split:  Splits data into two sets.  Useful for separating Training, Testing, and Validation sets for machine learning.
