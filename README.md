# Speech Classification from 0 to 9 with ANN Model


## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Preprocessing](#project-overview)
- [Model Architecture](#project-overview)
- [Training](#project-overview)
- [Evaluation](#project-overview)
- [Results](#project-overview)
- [License](#project-overview)

## Project Overview

This project focuses on building a neural network to classify spoken digits (0 to 9) based on audio data. Using a Sequential Artificial Neural Network (ANN), the model takes pre-processed audio features and learns to classify each digit. The project includes essential visualizations such as learning curves and a confusion matrix to evaluate model performance.

## Dataset

The dataset used in this project consists of pre-labeled audio files where each recording corresponds to a spoken digit between 0 and 9. Common datasets for this task include Google’s Speech Commands dataset or Free Spoken Digit Dataset (FSDD). Each audio file is converted into a suitable feature format (e.g., MFCC, Mel spectrogram) for input into the neural network.

## Installation
To set up the project environment, first clone the repository and install the required packages:

```bash
Copy code
git clone https://github.com/your-username/speech-classification-ann.git
cd speech-classification-ann
pip install -r requirements.txt
```
Preprocessing
Audio Feature Extraction: Each audio file is pre-processed into a feature vector. Common feature extraction techniques used are:

MFCC (Mel Frequency Cepstral Coefficients)
Mel Spectrogram
Zero-Crossing Rate, Spectral Roll-off, etc.
Normalization: The feature vectors are normalized to improve training efficiency and convergence.

Splitting the Data: The data is split into training, validation, and testing sets for unbiased evaluation of the model.

Model Architecture
This project uses a Sequential ANN with Dense layers for classification. The architecture may look like this:

Input Layer: Accepts feature vectors.
Hidden Layers: Multiple Dense layers with ReLU activation.
Output Layer: Softmax activation with 10 neurons for the 10 classes (digits 0–9).
python
Copy code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Training
To train the model, run:

python
Copy code
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
Learning rate, batch size, and the number of epochs can be adjusted for optimal performance. Early stopping is also implemented to avoid overfitting.

Evaluation
After training, the model is evaluated on the test dataset. The evaluation metrics include:

Accuracy: Overall classification accuracy.
Confusion Matrix: Visual representation of true vs. predicted labels.
Learning Curves: Plots of training and validation accuracy/loss over epochs.
Results
Learning Curves: Learning curves plot training and validation accuracy/loss, helping to assess model performance and diagnose potential overfitting or underfitting.

python
Copy code
import matplotlib.pyplot as plt

# Plot accuracy curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
Confusion Matrix: The confusion matrix provides detailed insights into model performance by visualizing the distribution of correct and incorrect predictions across all classes.

python
Copy code
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
License
This project is licensed under the MIT License.