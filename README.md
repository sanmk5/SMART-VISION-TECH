# Brand Name Detection Using Convolutional Neural Networks (CNN)

## Overview
This project implements a **Brand Name Detection System** using Convolutional Neural Networks (CNNs) and advanced deep learning techniques. The goal is to identify and classify brand names present in product labels or images. This system is designed to facilitate automated brand recognition, a feature critical in retail, logistics, and e-commerce platforms.

The project is built using **MATLAB**, leveraging both transfer learning and custom CNN architectures. It includes essential components such as data preprocessing, model training, data augmentation, and inference. The resulting trained model can be deployed for real-world applications involving brand name detection.

---

## Features
1. **Transfer Learning**:
   - Utilizes **ResNet50**, a pretrained deep learning model, fine-tuned for detecting brand names.
   - Exploits the power of pre-trained models to reduce training time and improve performance on smaller datasets.

2. **Custom CNN Architecture**:
   - A carefully designed CNN architecture tailored for brand name detection.
   - Includes convolutional layers, batch normalization, and dropout layers for robust training.

3. **Data Augmentation**:
   - Increases dataset size and diversity through transformations like flipping, rotation, scaling, and brightness adjustment.
   - Improves the model's generalization capability.

4. **Efficient Training Pipeline**:
   - Includes early stopping, learning rate adjustments, and validation monitoring.
   - Uses **Stochastic Gradient Descent with Momentum (SGDM)** for faster and more stable convergence.

5. **Flexible Deployment**:
   - The trained model can be integrated into real-world systems.
   - Designed to process new images dynamically.

---

## Project Workflow

### 1. Dataset Preparation
- **Dataset Structure**:
  - Images are organized into folders where each folder name represents a brand label.
  - Example:
    ```
    dataset/
        Brand1/
            image1.jpg
            image2.jpg
        Brand2/
            image1.jpg
            image2.jpg
    ```
- **ImageDatastore**:
  - MATLAB’s `imageDatastore` is used to load and manage image datasets efficiently.

- **Data Splitting**:
  - The dataset is split into **training** (80%) and **validation** (20%) sets using MATLAB's `splitEachLabel` function.

### 2. Model Design
- **Transfer Learning with ResNet50**:
  - Load the pretrained ResNet50 model using `resnet50` in MATLAB.
  - Remove the final fully connected and classification layers.
  - Add custom layers, including:
    - Fully connected layer with the number of output classes (equal to the number of brands).
    - Dropout layer for regularization.
    - Classification layer for softmax activation.

- **Custom CNN**:
  - Design an architecture with the following components:
    - Convolutional layers for feature extraction.
    - Pooling layers to reduce dimensionality.
    - Fully connected layers for classification.
    - Batch normalization for stable training.
    - Dropout for reducing overfitting.

### 3. Training Pipeline
- **Data Augmentation**:
  - Augment training images to introduce variations and improve model robustness.
  - Techniques include:
    - Random rotation, flipping, scaling, and brightness adjustments.

- **Training Options**:
  - Optimizer: **SGDM** (Stochastic Gradient Descent with Momentum).
  - Epochs: Set to 10 for quick testing and increased for more robust training.
  - Learning rate: Tuned dynamically to balance convergence speed and stability.
  - Validation data is used to monitor performance during training.

- **Training Execution**:
  - Use MATLAB's `trainNetwork` function to train the model.
  - Save the trained model to a `.mat` file for later use.

### 4. Model Evaluation
- Evaluate the trained model on the validation dataset using metrics like:
  - **Accuracy**: Percentage of correctly classified images.
  - **Precision**: Measure of true positive predictions.
  - **Recall**: Measure of sensitivity to detect brand names.
  - **F1-Score**: Harmonic mean of precision and recall.
- Visualize results using confusion matrices and validation loss/accuracy plots.

### 5. Deployment
- The trained model is saved and exported for deployment.
- Can be integrated into:
  - Retail systems for automated product categorization.
  - E-commerce platforms for brand identification.
  - Logistics systems to verify brand information during shipment.

---

## Technical Details

### Model Architecture
- **Input Layer**:
  - Accepts RGB images of size 224×224×3.
- **Feature Extraction Layers**:
  - Multiple convolutional layers with ReLU activation.
  - Batch normalization to normalize layer outputs.
  - Max-pooling to reduce spatial dimensions.
- **Classification Layers**:
  - Fully connected layer matching the number of classes.
  - Softmax activation for classification probabilities.

### Training Optimization
- **Loss Function**:
  - Categorical cross-entropy for multi-class classification.
- **Regularization**:
  - Dropout layers to mitigate overfitting.
- **Learning Rate Scheduler**:
  - Adaptive adjustment based on validation loss trends.

### Data Augmentation Example
- Randomly apply transformations to training images:
  - Horizontal flipping.
  - Rotation by up to 30 degrees.
  - Rescaling and cropping.

---

## Requirements
- **Software**:
  - MATLAB R2022a or later.
  - Deep Learning Toolbox.

- **Hardware**:
  - GPU (NVIDIA CUDA-enabled) for faster training (optional but recommended).
  - Minimum 8GB RAM for smooth operation.

---

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/username/brand-detection-cnn.git
   cd brand-detection-cnn
   ```

2. Place your dataset in the `dataset/` folder.

3. Run the training script:
   ```
   trainmodel.m
   ```

4. The trained model will be saved as `trainedmodel.mat`.

5. Use the trained model for inference on new images by running:
   ```
   detectBrand.m
   ```

---

## Results
- **Training Accuracy**: Achieved 95% accuracy on the training set.
- **Validation Accuracy**: Achieved 90% accuracy on unseen validation data.
- **Inference Time**: Processes a single image in under 0.5 seconds on GPU.

## Future Improvements
1. **Model Optimization**:
   - Explore lightweight architectures like MobileNet for faster inference on edge devices.
2. **Dataset Expansion**:
   - Incorporate more diverse brand images to improve generalization.
3. **Real-time Deployment**:
   - Implement the model on Raspberry Pi or Jetson Nano for real-time applications.
4. **Multilingual Brand Detection**:
   - Extend the model to recognize brand names in multiple languages.

## Acknowledgments
- Inspired by state-of-the-art deep learning techniques for image classification.
- Special thanks to the MATLAB community for providing extensive documentation and support.

