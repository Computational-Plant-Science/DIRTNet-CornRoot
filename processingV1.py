import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pre_process(X):
    X=X[2:len(X)]
    return X
def data_preperation_diameter(file_path,sheet_name):
    # Load data from Excel

    # sheet_name = "P1"
    data = pd.read_excel(file_path, sheet_name)
    # sample_length=3
    # Select data from column K
    # left_i_data = data.iloc[:, 8]
    # right_j_data = data.iloc[:, 9]  # Column K is the 11th column (index 10)
    # bottom_k_data = data.iloc[:, 10]
    # depth_m_label = data.iloc[:, 12]
    # diameter_n_label = data.iloc[:, 13]

    left_i_data1 = data.iloc[:, 4]
    right_j_data1 = data.iloc[:, 2]  # Column K is the 11th column (index 10)
    bottom_k_data1 = data.iloc[:, 3]

    left_i_data2 = data.iloc[:, 9]
    right_j_data2 = data.iloc[:, 7]  # Column K is the 11th column (index 10)
    bottom_k_data2 = data.iloc[:, 8]

    left_i_data3 = data.iloc[:, 15]
    right_j_data3 = data.iloc[:, 13]  # Column K is the 11th column (index 10)
    bottom_k_data3 = data.iloc[:, 14]

    left_i_data4 = data.iloc[:, 19]
    right_j_data4 = data.iloc[:, 17]  # Column K is the 11th column (index 10)
    bottom_k_data4 = data.iloc[:, 18]

    left_i_data1=pre_process(left_i_data1)
    right_j_data1=pre_process(right_j_data1)
    bottom_k_data1=pre_process(bottom_k_data1)

    left_i_data2=pre_process(left_i_data2)
    right_j_data2=pre_process(right_j_data2)
    bottom_k_data2=pre_process(bottom_k_data2)

    left_i_data3=pre_process(left_i_data3)
    right_j_data3=pre_process(right_j_data3)
    bottom_k_data3=pre_process(bottom_k_data3)

    left_i_data4=pre_process(left_i_data4)
    right_j_data4=pre_process(right_j_data4)
    bottom_k_data4=pre_process(bottom_k_data4)



    return left_i_data1, right_j_data1, bottom_k_data1, left_i_data2, right_j_data2, bottom_k_data2, left_i_data3, right_j_data3, bottom_k_data3, left_i_data4, right_j_data4, bottom_k_data4


def plot_three_signals(left_sensor, right_sensor, middle_sensor, title="Plant Sensor Signals", xlabel="Time",
                       ylabel="Sensor Value"):
    """
    Plots three signals: left, right, and middle sensor data.

    Parameters:
    - left_sensor (array-like): Data from the left sensor.
    - right_sensor (array-like): Data from the right sensor.
    - middle_sensor (array-like): Data from the middle sensor.
    - title (str): Plot title.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    # Create time axis
    x = range(len(left_sensor))

    plt.figure(figsize=(12, 6))
    plt.plot(x, left_sensor, label='Left Sensor', color='blue')
    plt.plot(x, right_sensor, label='Right Sensor', color='red')
    plt.plot(x, middle_sensor, label='Middle Sensor', color='green')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def trainData(trainX, num_samples):
    sample_length = 3
    # Reshape trainX into (num_samples, 7, 1)
    num_samples = len(trainX) // sample_length
    trainX = trainX[:num_samples * sample_length]  # Trim excess rows if not divisible by 7
    trainX = trainX.reshape(num_samples, sample_length, 1)

    # Generate labels for every "sample_length" rows
    # trainY = np.arange(1, num_samples + 1)
    trainY = np.arange(0, num_samples)

    return trainX, trainY

def trainData(trainX, days = 25, sample_length = 3):
    trainX = trainX[0:days]
    # Reshape trainX into (num_samples, 7, 1)
    num_samples = len(trainX) // sample_length
    trainX = trainX[:num_samples * sample_length]  # Trim excess rows if not divisible by 7
    trainX = trainX.reshape(num_samples, sample_length, 1)

    # Generate labels for every "sample_length" rows
    # trainY = np.arange(1, num_samples + 1)
    trainY = np.arange(0, num_samples)

    return trainX, trainY



def data_extract(file_path,sheet_name="P1", days=25, sample_length=3):

    left_i_data1, right_j_data1, bottom_k_data1, left_i_data2, right_j_data2, bottom_k_data2, left_i_data3, right_j_data3, bottom_k_data3, left_i_data4, right_j_data4, bottom_k_data4 =data_preperation_diameter(file_path,sheet_name)

    # plot_three_signals(left_i_data3, right_j_data3, bottom_k_data3, title="Plant Sensor Signals", xlabel="Time",ylabel="Sensor Value")
    # plot_three_signals(left_i_data4, right_j_data4, bottom_k_data4, title="Plant Sensor Signals", xlabel="Time",ylabel="Sensor Value")

    # Assuming the data contains only features (no labels yet)
    trainX1 = left_i_data3.values.reshape(-1, 1)  # Convert to NumPy array
    trainX2 = right_j_data3.values.reshape(-1, 1)  # Convert to NumPy array
    trainX3 = bottom_k_data3.values.reshape(-1, 1)  # Convert to NumPy array

    trainX11 = left_i_data4.values.reshape(-1, 1)  # Convert to NumPy array
    trainX22 = right_j_data4.values.reshape(-1, 1)  # Convert to NumPy array
    trainX33 = bottom_k_data4.values.reshape(-1, 1)  # Convert to NumPy array


    # Sensor 1 & 2 of a plant x1
    trainX1, trainY1 = trainData(trainX1, days, sample_length)
    trainX2, trainY2 = trainData(trainX2, days, sample_length)
    trainX3, trainY3 = trainData(trainX3, days, sample_length)

    trainX11, trainY11 = trainData(trainX11, days, sample_length)
    trainX22, trainY22 = trainData(trainX22, days, sample_length)
    trainX33, trainY33 = trainData(trainX33, days, sample_length)

    trainX = np.concatenate([trainX1, trainX2, trainX3,trainX11, trainX22, trainX33], axis=0)
    trainY = np.concatenate([trainY1, trainY2, trainY3,trainY11, trainY22, trainY33], axis=0)

    return trainX,trainY

def augment_data_by_class_updated(X, y, augmentation_factor=5, noise_level=0.2, scaling_range=(0.8, 1.2), shift_max=2):
    augmented_X = []
    augmented_y = []

    for class_label in np.unique(y):
        # Extract samples of the current class
        X_class = X[y == class_label]

        for _ in range(augmentation_factor):
            for sample in X_class:
                # Original sample
                augmented_X.append(sample)
                augmented_y.append(class_label)

                # Apply noise
                noisy_sample = sample + np.random.normal(0, noise_level, sample.shape)
                augmented_X.append(noisy_sample)
                augmented_y.append(class_label)

                # Apply scaling
                scaling_factor = np.random.uniform(*scaling_range)
                scaled_sample = sample * scaling_factor
                augmented_X.append(scaled_sample)
                augmented_y.append(class_label)

                # Apply time shifting
                shift = np.random.randint(-shift_max, shift_max + 1)
                shifted_sample = np.roll(sample, shift, axis=0)
                augmented_X.append(shifted_sample)
                augmented_y.append(class_label)

    # Combine all augmented samples and labels
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    return augmented_X, augmented_y

def augment_data_by_class_updated2(X, y, augmentation_factor=5, noise_level=0.2, scaling_range=(0.8, 1.2), shift_max=2):
    augmented_X = []
    augmented_y = []

    for class_label in np.unique(y):
        # Extract samples of the current class
        X_class = X[y == class_label]

        for _ in range(augmentation_factor):
            for sample in X_class:
                # Original sample
                augmented_X.append(sample)
                augmented_y.append(class_label)

                # Apply noise
                # noisy_sample = sample + np.random.normal(0, noise_level, sample.shape)
                # augmented_X.append(noisy_sample)
                # augmented_y.append(class_label)

                # Apply scaling
                scaling_factor = np.random.uniform(*scaling_range)
                scaled_sample = sample * scaling_factor
                augmented_X.append(scaled_sample)
                augmented_y.append(class_label)

                # Apply time shifting
                # shift = np.random.randint(-shift_max, shift_max + 1)
                # shifted_sample = np.roll(sample, shift, axis=0)
                # augmented_X.append(shifted_sample)
                # augmented_y.append(class_label)

    # Combine all augmented samples and labels
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    return augmented_X, augmented_y