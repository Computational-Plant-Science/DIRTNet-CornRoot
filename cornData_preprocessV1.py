import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import processingV1
from sklearn.model_selection import train_test_split, cross_val_score

def test_data_aug(X_test, y_test,noise_label ):
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    X_augmented, y_test_aug = processingV1.augment_data_by_class_updated(X_test_flattened, y_test, augmentation_factor=20,
                                                                      noise_level=noise_label)

    X_test_aug = np.expand_dims(X_augmented, axis=-1)

    return X_test_aug,y_test_aug


def all_data(days,sample_length,noise_label,file_path):
    sheet_name="P1"
    trainX_plant1,trainY_plant1 = processingV1.data_extract(file_path,sheet_name, days, sample_length)

    sheet_name="P2"
    trainX_plant2,trainY_plant2 = processingV1.data_extract(file_path,sheet_name, days, sample_length)

    sheet_name="P3"
    trainX_plant3,trainY_plant3 = processingV1.data_extract(file_path,sheet_name, days, sample_length)

    sheet_name="P4"
    trainX_plant4,trainY_plant4 = processingV1.data_extract(file_path,sheet_name, days, sample_length)

    sheet_name="P5"
    trainX_plant5,trainY_plant5 = processingV1.data_extract(file_path,sheet_name, days, sample_length)

    trainX = np.concatenate([trainX_plant1, trainX_plant2, trainX_plant3, trainX_plant4, trainX_plant5], axis=0)
    trainY = np.concatenate([trainY_plant1, trainY_plant2, trainY_plant3, trainY_plant4, trainY_plant5], axis=0)
    # trainX = np.concatenate([trainX_plant1, trainX_plant3, trainX_plant4, trainX_plant5], axis=0)
    # trainY = np.concatenate([trainY_plant1, trainY_plant3, trainY_plant4, trainY_plant5], axis=0)

    print("trainX shape before augmentation:", trainX.shape)
    print("trainY shape before augmentation:", trainY.shape)

    # Split the data into train and test sets
    X_train1, X_test, y_train1, y_test = train_test_split(trainX, trainY, test_size=0.3, random_state=42,
                                                        stratify=trainY)

    X_flattened = X_train1.reshape(X_train1.shape[0], -1)

    X_augmented, y_train = processingV1.augment_data_by_class_updated(X_flattened, y_train1, augmentation_factor=20,noise_level=noise_label)

    X_train = np.expand_dims(X_augmented, axis=-1)

    # Check the processed data
    print("trainX shape after augmentation:", X_train.shape)
    print("trainY shape after augmentation:", y_train.shape)

    print("Un-augmented test Data:", X_test.shape)
    print("Un-augmentated test Label:", y_test.shape)
    return X_train1,  y_train1, X_train,  y_train, X_test, y_test

## Augmented trained data: X_train and y_train. Before augmented trained data: X_train1 and y_train1, Test data (un-augmented): X_test and y_test

#left_i_data1, right_j_data1, bottom_k_data1, left_i_data2, right_j_data2, bottom_k_data2, left_i_data3, right_j_data3, bottom_k_data3, left_i_data4, right_j_data4, bottom_k_data4 =processingV1.data_preperation_diameter(sheet_name)
# file_path = "Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/cornData.xlsx"
# days=25
# sample_length=3
# noise_label=0.2
#
# X_train1,  y_train1, X_train,  y_train, X_test, y_test = all_data(days,sample_length,noise_label,file_path)
#
# # Particular senso data ploting, for particular sheep, e.g., "P1"
# # left_i_data1, right_j_data1, bottom_k_data1, left_i_data2, right_j_data2, bottom_k_data2, left_i_data3, right_j_data3, bottom_k_data3, left_i_data4, right_j_data4, bottom_k_data4 =processingV1.data_preperation_diameter(file_path,"P1")
# # processingV1.plot_three_signals(left_i_data1, right_j_data1, bottom_k_data1, title="Plant Sensor Signals", xlabel="Time", ylabel="Sensor Value")
#
# print(X_train1.shape)
