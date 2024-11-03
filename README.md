# Fruit and Vegetable Classification with Recipe Generation

This project classifies images of fruits and vegetables into 36 distinct categories and generates recipes based on the identified items. Using both traditional machine learning and deep learning methods, the project incorporates the Edamam API to provide relevant recipes, adding a practical layer of functionality to the classification task.

## Project Overview
- **Objective**: Classify images of fruits and vegetables and generate recipes based on the classified items.
- **Dataset**: Sourced from Kaggle, containing images of 36 distinct fruit and vegetable classes.

## Pipeline

### 1. Data Preprocessing
   - **Normalization**: Pixel values scaled to [0, 1] for stable model performance.
   - **Flattening** (for traditional ML): Converted images into 1D arrays for models like SVM and KNN.

### 2. Traditional Machine Learning Models
   - **K-Nearest Neighbors (KNN)**: Achieved a Validation Accuracy of 96.5% and Testing Accuracy of 96.6%.
   - **Random Forest (RF)**: Validation Accuracy of 96.0% and Testing Accuracy of 96.1%.
   - **Naive Bayes**: Performed poorly with an accuracy of 26.7%.
   - **Logistic Regression**: Validation and Testing Accuracy of 96.1%.

   **Ensemble Model**: A Voting Classifier combining KNN, RF, and Logistic Regression achieved 97.2% accuracy.

   **Evaluation Metrics for Voting Classifier**:
   - **Precision**: 0.973
   - **Recall**: 0.972
   - **F1-Score**: 0.972

### 3. Deep Learning Approaches
   - **Artificial Neural Network (ANN)**:
     - Architecture: Two hidden layers (3000 and 1000 neurons), output layer with 36 classes.
     - Final accuracy: 95.2% on the validation set.
   - **Convolutional Neural Networks (CNN)**:
     - Architecture: Two Conv2D layers with MaxPooling, and a Dense output layer.
     - Improved with Data Augmentation (rescaling, flipping, shearing, rotation, zooming) and BatchNormalization/Dropout.
   - **Transfer Learning with ResNet50**:
     - Fine-tuned on the dataset with an additional dense layer of 512 neurons.
     - Achieved the best results without overfitting.

### 4. Recipe Generation
   - After classification, the identified fruit/vegetable is passed to the Edamam API, which generates a relevant recipe based on the detected items.

## Requirements
- Python
- TensorFlow, Keras
- Scikit-learn
- NLTK
- Edamam API access

Install dependencies:
```bash
pip install -r requirements.txt
