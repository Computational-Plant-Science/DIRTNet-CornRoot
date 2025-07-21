# DIRTNet-CornRoot
DIRTNet: A Hybrid Deep Learning Framework for Non-Destructive Root Phenotyping Using Fiber Bragg Grating (FBG) Sensor Data.
This repository has code to train and test DIRTNet, a special deep learning model that helps measure root depth and size without digging up the soil. The model uses data from Fiber Bragg Grating (FBG) sensors placed in the soil near corn roots. It combines different neural networks (VGG, ResNet, and GRU) to learn both the shape and changes over time in the sensor signals. Inside this repo, you will find:

* Data processing tools to prepare sensor data for the model
* The DIRTNet model code and training scripts
* Custom tools to track training progress and accuracy
* Scripts to evaluate and compare model results

This work helps farmers and researchers monitor root growth easily and supports better crop breeding and farming practices.

## Model Architecture: DIRTNet-Hybrid

The core model consists of:

- **Spatial feature extraction** using two CNN branches:
  - A VGG-like CNN with multiple `Conv2D` and `MaxPooling2D` layers
  - A custom ResNet18 built with residual blocks

- **Feature fusion** by concatenating outputs from both CNN branches

- **Temporal modeling** via a GRU layer to capture sequential dependencies

- **Fully connected layers** with dropout and batch normalization for robust learning

- **Output layer** with softmax activation producing multi-class classification (e.g., root depth classes)

📎 **Output:** multi-class classification

# Data Preprocessing

- **Input:** Sensor signals from an Excel dataset.
- **Preprocessing:** Handled by `cornData_preprocessV1.all_data()`.
- **Output:**  
  - Augmented and clean (non-augmented) training and test datasets.  
  - Inputs reshaped to 2D with channels format: `(height, width, 1)`.  
  - Output labels are one-hot encoded for categorical classification.

# Training Strategy

- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam (learning rate = 0.001)  
- **Batch Size:** 16  
- **Epochs:** 5  
- **Metrics Monitored:** Accuracy, F1-score, Precision, Recall (via custom `MetricsCallback`)  
- **Checkpointing:** Saves best model (`.keras`) based on validation accuracy  
- **Custom Callback (`MetricsCallback`):**  
  Logs detailed per-epoch metrics to JSON file including:  
  - Training and validation loss  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  

