# DIRTNet-CornRoot

<img width="783" height="632" alt="image" src="https://github.com/user-attachments/assets/1bb7b5f9-f0a0-4037-9de2-8bc9604c9043" /> 

Example of corn root data collection. 

DIRTNet: A Hybrid Deep Learning Framework for Non-Destructive Root Phenotyping Using Fiber Bragg Grating (FBG) Sensor Data.
This repository has code to train and test DIRTNet, a special deep learning model that helps measure root depth and size without digging up the soil. The model uses data from Fiber Bragg Grating (FBG) sensors placed in the soil near corn roots. It combines different neural networks (VGG, ResNet, and GRU) to learn both the shape and changes over time in the sensor signals. Inside this repo, you will find:

* Data processing tools to prepare sensor data for the model
* The DIRTNet model code and training scripts
* Custom tools to track training progress and accuracy
* Scripts to evaluate and compare model results

This work helps farmers and researchers monitor root growth easily and supports better crop breeding and farming practices.

# DIRTNet: Hybrid Neural Network for Root Phenotyping

This repository contains the implementation of DIRTNet, a hybrid deep learning model combining ResNet, VGG, and GRU architectures for classification of root depth and diameter using Fiber Bragg Grating (FBG) sensor data.

---

## Model Architecture

- **ResNet18**: Extracts spatial features through residual convolutional blocks.
- **VGG-like model**: Additional spatial feature extraction using stacked convolution and max pooling layers.
- **GRU layer**: Captures temporal dependencies by processing concatenated ResNet and VGG features.
- **Fully Connected layers**: Dense layers with batch normalization and dropout for robust classification.
- **Output layer**: Softmax activation for multi-class classification of root traits.

---

## Training Setup

- **Data preprocessing**  
  Sensor signals are loaded and preprocessed from an Excel dataset using `cornData_preprocessV1.all_data()`.  
- **Input shape**: `(height, width, 1)` representing 2D sensor data with a single channel.  
- **Loss function**: Categorical cross-entropy.  
- **Optimizer**: Adam with learning rate 0.001.  
- **Batch size**: 16.  
- **Epochs**: 5.  
- **Metrics monitored**: Accuracy, precision, recall, and F1-score via a custom callback `MetricsCallback`.  
- **Checkpointing**: Saves the best model based on validation accuracy.

---

## Training Code Snippet

```python
days=25
sample_length=6
noise_label=0.1

## Augmented trained data: X_train and y_train. Before augmented trained data: X_train1 and y_train1, Test data (un-augmented): X_test and y_test
X_train1,  y_train1, X_train,  y_train, X_test, y_test = cornData_preprocessV1.all_data(days,sample_length,noise_label,file_path)


model = build_hybrid_model(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

metrics_callback = MetricsCallback(X_train, y_train, X_test, y_test)
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, mode='max')

checkpoint = ModelCheckpoint(
    filepath=model_path,
    monitor='val_accuracy',  # You can also monitor 'val_loss' or 'val_f1' if custom
    save_best_only=True,
    save_weights_only=False,  # Set to True if you only want weights
    mode='max',
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, metrics_callback]
)


# Data Preprocessing

- Input: Sensor signals from an Excel dataset.
- Preprocessing: Handled by `cornData_preprocessV1.all_data()`.
- Output:  
  - Augmented and clean (non-augmented) training and test datasets.  
  - Inputs reshaped to 2D with channels format: `(height, width, 1)`.  
  - Output labels are one-hot encoded for categorical classification (multi-class).

# Training Strategy

- Loss Function: Categorical Crossentropy  
- Optimizer: Adam (learning rate = 0.001)  
- Batch Size: 16  
- Epochs: 5  
- Metrics Monitored: Accuracy, F1-score, Precision, Recall (via custom `MetricsCallback`)  
- Checkpointing: Saves best model (`.keras`) based on validation accuracy  
- Custom Callback (`MetricsCallback`):  
  Logs detailed per-epoch metrics to JSON file including:  
  - Training and validation loss  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score
```
# How to Run the Code
Step 1: Prepare Your Folder Structure 

<img width="506" height="174" alt="image" src="https://github.com/user-attachments/assets/ea8ca2b0-557e-451a-9cfb-051c77dfa81f" /> 

Step 2: Place Dataset 

- Copy your codes (DIRTNETv2.py, cornData_preprocessv1.py and processingV1.py) into root folder
- Copy your .xlsx dataset into the data folder.
- Example dataset filename: cornDepthDiameterData.xlsx

Step 3: Run the Main Training Script
- Run the following from your terminal or IDE: python DIRTNetV2.py

# Confusion Matrix Results

```python
y_pred_classes = np.argmax(model.predict(X_test), axis=1)
y_true_classes = np.argmax(y_test, axis=1)
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)

print("Confusion matrix:\n", confusion_mtx)

```

The confusion matrix evaluates the classification performance of the model by showing how many samples were correctly and incorrectly classified per class.

Example output for the test set:


<img width="648" height="445" alt="image" src="https://github.com/user-attachments/assets/19f6dc7d-4e3f-4549-aed6-2a5c8829a56c" /> 

# Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- pandas
- scikit-learn 
