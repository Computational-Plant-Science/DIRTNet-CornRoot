# this verson of code is using for new data
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GRU, BatchNormalization, Dropout, \
    concatenate, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from tensorflow.keras.callbacks import Callback
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, \
    accuracy_score
from sklearn.model_selection import train_test_split
import json
import cornData_preprocessV1
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
# Define custom callback
class MetricsCallback(Callback):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.x_train = X_train
        self.y_train = y_train
        self.history = {
            'loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_f1': [],
            'val_f1': []
        }

    def on_epoch_end(self, epoch, logs=None):
        # Training metrics

        y_pred_train = np.argmax(self.model.predict(X_train), axis=1)
        y_true_train = self.y_train
        y_true_train = np.argmax(y_true_train, axis=1)
        y_pred_val = np.argmax(self.model.predict(X_test), axis=1)
        y_true_val = self.y_test
        y_true_val = np.argmax(y_true_val, axis=1)

        # y_pred_train = np.argmax(self.model.predict(self.x_train), axis=1)
        # y_pred_val = np.argmax(self.model.predict(self.x_val), axis=1)


        f1_train = f1_score(y_true_train, y_pred_train, average='micro')
        precision_train = precision_score(y_true_train, y_pred_train, average='micro')
        recall_train = recall_score(y_true_train, y_pred_train, average='micro')

        f1_val = f1_score(y_true_val, y_pred_val, average='micro')
        precision_val = precision_score(y_true_val, y_pred_val, average='micro')
        recall_val = recall_score(y_true_val, y_pred_val, average='micro')

        self.history['loss'].append(logs['loss'])
        self.history['val_loss'].append(logs['val_loss'])
        self.history['train_accuracy'].append(logs['accuracy'])
        self.history['val_accuracy'].append(logs['val_accuracy'])
        self.history['train_precision'].append(precision_train)
        self.history['val_precision'].append(precision_val)
        self.history['train_recall'].append(recall_train)
        self.history['val_recall'].append(recall_val)
        self.history['train_f1'].append(f1_train)
        self.history['val_f1'].append(f1_val)

        # Save to JSON after each epoch
        # with open('Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/DIRT_NET_Results/training_history_DIRTNet_X_6_days_52925_42Days.json', 'w') as f:
        with open('Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/Test_Results/training_history_DIRTNet_X_6_days_52925_42Days.json','w') as f:
            json.dump(self.history, f)


# Define ResNet block
def resnet_block(input_tensor, filters, strides=(1, 1)):
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    if strides != (1, 1):
        input_tensor = Conv2D(filters, (1, 1), strides=strides, padding='same')(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.ReLU()(x)
    return x


# Define ResNet18
def build_resnet18(input_shape):
    input_tensor = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    x = resnet_block(x, 128, strides=(2, 2))
    x = resnet_block(x, 128)

    x = resnet_block(x, 256, strides=(2, 2))
    x = resnet_block(x, 256)

    x = resnet_block(x, 512, strides=(2, 2))
    x = resnet_block(x, 512)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    model = Model(inputs=input_tensor, outputs=x)
    return model


# Define VGG-like model
def build_vgg_like_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    return model


# Define hybrid model
def build_hybrid_model(input_shape, num_classes):
    input_signal = Input(shape=input_shape)

    vgg_like = build_vgg_like_model(input_shape)
    vgg_output = vgg_like(input_signal)

    resnet18 = build_resnet18(input_shape)
    resnet_output = resnet18(input_signal)

    concatenated_features = concatenate([vgg_output, resnet_output])

    # gru_output = GRU(64, activation='relu', return_sequences=False)(tf.expand_dims(concatenated_features, axis=1))

    from keras import ops

    gru_output = GRU(64, activation='relu', return_sequences=False)(
        ops.expand_dims(concatenated_features, axis=1)
    )

    # from tensorflow.keras.layers import Lambda
    #
    # expanded = Lambda(lambda x: tf.expand_dims(x, axis=1))(concatenated_features)
    # gru_output = GRU(64, activation='relu', return_sequences=False)(expanded)


    # working for 92%
    x = Dense(1024, activation='relu')(gru_output)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_signal, outputs=output)
    return model




# Read CSV file
file_path = "Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/cornData_final_data.xlsx"
days=25
sample_length=6
noise_label=0.1

## Augmented trained data: X_train and y_train. Before augmented trained data: X_train1 and y_train1, Test data (un-augmented): X_test and y_test
X_train1,  y_train1, X_train,  y_train, X_test, y_test = cornData_preprocessV1.all_data(days,sample_length,noise_label,file_path)
num_classes = len(np.unique(y_test)) # for depth
print("Classes:",num_classes)

# Ensure data is a NumPy array with the correct data type
X_augmented = np.array(X_train, dtype=np.float32)
y_augmented = np.array(y_train, dtype=np.int32)
X_test1 = np.array(X_test, dtype=np.float32)
y_test1 = np.array(y_test, dtype=np.int32)

# Check the shapes
print("trainX shape:", X_augmented.shape)
print("trainY shape:", y_augmented.shape)
print("testX shape:", X_test1.shape)
print("testY shape:", y_test1.shape)

# Load and preprocess data
num_tran = X_augmented.shape[1]
n = X_augmented.shape[2]


# Define input shape and number of classes
# input_shape = (len(X_train), num_tran, n)
input_shape = ( num_tran, n,n)
# num_classes = 10

# Build and compile the model
model = build_hybrid_model(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model_variant = 2

# Print the model summary
model.summary()

# Define and compile the ResNet model
# resnet_model = resnet(num_classes, num_tran, n)

X_train = np.expand_dims(X_train, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

X_train1 = np.expand_dims(X_train1, axis=-1)
y_train1 = tf.keras.utils.to_categorical(y_train1, num_classes)

X_test = np.expand_dims(X_test, axis=-1)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# %% get the working path
current_file = Path(__file__).stem
print(current_file)
model_name = current_file
logdir = Path("tensorboard_logs", "Trial")

# Define the custom callback
# metrics_callback = MetricsCallback()

X_train = X_train.astype('float32') # augmented
X_train1 = X_train1.astype('float32') # before augmentation
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')  # or int if you're using sparse labels
y_train1 = y_train1.astype('float32')  # or int if you're using sparse labels
y_test = y_test.astype('float32')


metrics_callback = MetricsCallback(X_train, y_train,X_test, y_test)

# model_path = Path(logdir, model_name, 'model/RootMonitoringNet_depth' + str(model_variant) + '_.h5')

# checkpoint_path = "best_model.h5"  # You can change the name and path as needed
# model_path="Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/DIRT_NET_Results/my_DIRNet_model.keras"
model_path="Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/Test_Results/my_DIRNet_model.keras"

checkpoint = ModelCheckpoint(
    filepath=model_path,
    monitor='val_accuracy',  # You can also monitor 'val_loss' or 'val_f1' if custom
    save_best_only=True,
    save_weights_only=False,  # Set to True if you only want weights
    mode='max',
    verbose=1
)


# Train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint,metrics_callback]
)
# model_path = Path(logdir, model_name, 'model/RootMonitoringNet_depth' + str(model_variant) + '_.h5')
# model.save(model_path)
# # Load the best saved model
# model = tf.keras.models.load_model(checkpoint_path)
model = tf.keras.models.load_model(model_path)

# Save results
y_pred_classes = np.argmax(model.predict(X_test), axis=1)
y_true_classes = np.argmax(y_test, axis=1)
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='micro')
recall = recall_score(y_true_classes, y_pred_classes, average='micro')
f1 = f1_score(y_true_classes, y_pred_classes, average='micro')
accuracy = accuracy_score(y_true_classes, y_pred_classes)



print("Test accuracy", accuracy)
print("Confusion matrix", confusion_mtx)
print("Prediction", y_pred_classes)
print("Actual", y_true_classes)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)

print("testX shape:", X_test.shape)
print("trainX shape:", X_train.shape)

my_list2 = X_test.flatten().tolist()

# # path='Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/DIRT_NET_Results_'
# with open(
#         f'Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/DIRT_NET_Results/pred_resultPlots_{model_variant}.txt',
#         'w') as filehandle:
#     json.dump(y_pred_classes.tolist(), filehandle)
# with open(
#         f'Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/DIRT_NET_Results/actual_resultPlots_{model_variant}.txt',
#         'w') as filehandle:
#     json.dump(y_true_classes.tolist(), filehandle)
# with open(
#         f'Y:/Kabir Hossain/Works Kabir/Fiber_sensor_2nd_paper/DIRT_NET_Results/x_test_{model_variant}.txt',
#         'w') as filehandle:
#     json.dump(my_list2, filehandle)
