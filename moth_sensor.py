import pandas as pd

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

from datetime import datetime
import skvideo.io

def load_n_process_datasets(current_file, torque_file, video_file):
    # Load current values and remove last record (180,4.48E-07)
    ti = pd.read_csv(current_file).values
    time, current = ti[:-1, 0], ti[:-1, 1]
    
    # Load and downsample torque values every 100 entries
    torque = pd.read_csv(torque_file).values[:, 1]
    torque = torque[::100]

    ct = np.stack((current, torque), axis=1)
    new_ct = np.expand_dims(ct, axis=(-1, -2))
    new_ct = np.tile(new_ct, (1, 80, 320, 1))

    # Load, downsample and crop video frames
    # TODO: frames invalid after 16870 
    video = skvideo.io.vread(video_file)[::10, 20:180, 8:328, 0]
    video = np.expand_dims(video, axis=-1)
    ctv_last = video + new_ct

    # Create shifted frames
    return ctv_last[:-1], video[1:]

current_file = 'datasets/moth/I-t_run_1.csv'
torque_file = 'datasets/moth/moth_152_run_1_trial_01_SOS_torque.csv'
video_file = 'datasets/moth/moth_152_run_1_trial_01_SOS.avi'
x, y = load_n_process_datasets(current_file, torque_file, video_file)

# Inspect the dataset.
print("Dataset Shapes: " + str(x.shape) + ", " + str(y.shape))

WINDOW_LENGTH = 5

# Create windows of data that include features and target
def window_data(X, Y, window=10):
    '''
    New length will be len(dataset)-window such that all samples have the window.
    '''
    x = []
    y = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
        y.append(Y[i])
    return np.array(x), np.array(y)
    
x_w, y_w = window_data(x, y, window=WINDOW_LENGTH)

n_train = int(len(x)*.85)
x_train_w, y_train_w = x_w[:n_train], y_w[:n_train]
x_test_w, y_test_w = x_w[n_train:], y_w[n_train:]

# Construct the input layer with no definite frame size.
inp = tf.keras.layers.Input(shape=(None, *x_train_w.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = tf.keras.layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=False,
    activation="relu",
)(x)
x = tf.keras.layers.Conv2D(
    filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"
)(x)

# Next, we will build the complete model and compile it.
model = tf.keras.models.Model(inp, x)
model.compile(
    loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(),
)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") #Support for tensorboard tracking!
logging = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Define some callbacks to improve training.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 5
batch_size = 2

# Fit the model to the training data.
model.fit(
    x_train_w,
    y_train_w,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test_w, y_test_w),
    callbacks=[logging, early_stopping, reduce_lr],
)