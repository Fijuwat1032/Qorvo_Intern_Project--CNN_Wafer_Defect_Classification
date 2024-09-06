# CNN Wafer Defect Classification

## Overview

This project focuses on the use of **Convolutional Neural Networks (CNN)** for the classification of wafer defects in semiconductor manufacturing. Wafer defect classification is a crucial step in ensuring the quality of semiconductor wafers during the manufacturing process. By automating defect detection and classification, companies can improve manufacturing yield, reduce costs, and enhance product reliability.

The goal of this project is to develop a CNN model that can accurately classify different types of wafer defects using image data. The project leverages deep learning techniques, specifically CNNs, which are well-suited for image classification tasks due to their ability to automatically extract and learn features from images.

## Table of Contents
- [Project Description](#project-description)
- [Dataset Information](#dataset-information)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Project Description

The **"CNN Wafer Defect Classification"** project is centered on the development of a deep learning model that uses Convolutional Neural Networks to classify defects in semiconductor wafers. Semiconductor wafer images are analyzed, and the model identifies and classifies defects based on image features. The classification process helps improve quality control in semiconductor manufacturing.

This notebook guides through the complete process, including:
- Data preprocessing and augmentation
- CNN model design and training
- Evaluation of the model's performance using metrics such as accuracy and confusion matrices
- Hyperparameter tuning and model optimization

## Dataset Information

The dataset used in this project consists of wafer images, where each image corresponds to a wafer map labeled with specific defect classes. The dataset may include various defect types such as:
- **Center defect**
- **Edge defect**
- **Random defect**
- **Scratch defect**
- **No defect (normal wafer)**

Each image is labeled with its corresponding defect class, and the CNN model is trained to classify the images into these defect categories.

### Dataset Preprocessing:
- Images are normalized and resized to a fixed input size for the CNN.
- Data augmentation techniques such as rotation, flipping, and zooming are used to increase the diversity of training samples.

## Requirements

This project is implemented in Python and utilizes deep learning libraries such as TensorFlow or PyTorch for building and training the CNN model. The following libraries are required:

- Python 3.x
- Jupyter Notebook
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow or pytorch (for CNN model)
- keras (if using TensorFlow backend)
- OpenCV (for image processing)

### Install the required libraries:
You can install the necessary dependencies using `pip`:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras opencv-python
```

## Installation

1. **Clone the Repository**:
   Clone the GitHub repository to your local machine using the following command:
   ```bash
   git clone https://github.com/Fijuwat1032/ML_Project_CNN_Wafer_Defect_Classification.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd ML_Project_CNN_Wafer_Defect_Classification
   ```

3. **Install the Dependencies**:
   Install all the required libraries using `pip` as mentioned above.

4. **Open the Jupyter Notebook**:
   Launch Jupyter Notebook by running the following command:
   ```bash
   jupyter notebook
   ```
   Open the notebook titled **`CNN_Wafer_Defect_Classification.ipynb`** to explore the code and analysis.

## Usage

1. **Data Loading**:
   The notebook begins with loading and visualizing the wafer image dataset. Images are preprocessed and normalized for use with the CNN.

2. **Model Building**:
   The Convolutional Neural Network (CNN) is built using TensorFlow/Keras. The model consists of multiple convolutional layers, pooling layers, and fully connected layers designed to extract meaningful features from the wafer images.

3. **Training**:
   The CNN model is trained using the preprocessed wafer images, and various hyperparameters such as learning rate, batch size, and number of epochs are tuned to optimize the model's performance.

4. **Evaluation**:
   The model's performance is evaluated using test data. The evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrices. Visualizations of the model’s predictions are provided.

5. **Prediction**:
   Once trained, the model can be used to classify new wafer images and predict the type of defect present in the images.

## Model Architecture

The CNN architecture is designed to handle image classification tasks efficiently. The architecture may include:

- **Input Layer**: Accepts images of a fixed size (e.g., 128x128).
- **Convolutional Layers**: Extracts features from the images using filters and feature maps.
- **Pooling Layers**: Reduces the dimensionality of feature maps to focus on important features.
- **Fully Connected Layers**: Combines the extracted features and outputs predictions for the defect classes.
- **Output Layer**: Uses a softmax activation function to classify the wafer into one of the predefined defect classes.

The model is compiled using a loss function like `categorical_crossentropy` and optimized using an optimizer like `Adam`.

## Results

The CNN model achieved high accuracy in classifying wafer defects. The key results include:
- **Model Accuracy**: The final accuracy of the model on the test dataset.
- **Confusion Matrix**: A matrix displaying the true positives, false positives, true negatives, and false negatives for each defect class.
- **Classification Report**: A detailed report showing precision, recall, F1-score for each defect class.

The model was able to successfully differentiate between different types of defects with high precision and recall.

## Acknowledgments

I would like to extend my gratitude to **Ganesh** for his guidance and support throughout the project. 


## Conclusion

This project demonstrates the effectiveness of Convolutional Neural Networks in detecting and classifying defects in wafer images. By automating defect classification, the model can help semiconductor manufacturers identify faulty wafers early in the production process, potentially saving costs and improving product yield.
