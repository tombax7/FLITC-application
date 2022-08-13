# Finds the Best HyperParameters for the Feeder Identification NN

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import os
import joblib
import tensorflow
import numpy as np
import pandas as pd
from tensorflow.keras.utils import plot_model
from ast import literal_eval
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
import matplotlib.pyplot as plt
import logging
from tqdm.keras import TqdmCallback
import random
import time

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)
cce = tensorflow.keras.losses.CategoricalCrossentropy()

# GPU usage
tensorflow.get_logger().setLevel(logging.INFO)
physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
directory = os.path.abspath(os.path.dirname(__file__))
results_dic = directory + r'\results\Feeder_ID'
I_CWT_path = directory + r'\I_CWT.joblib'
I_CWT_not_full_path = directory + r'\I_CWT_not_full.joblib'
Class_path = directory + r'\Feeder_Output_4_Outputs.joblib'
Rs_path = directory + r'\Rs_FFNN.joblib'
Duration_path = directory + r'\Duration_FFNN.joblib'
best_models_loc = results_dic + r'\best_cwt_feeder_id_not_full_v1.csv'

# Change load file for different dataset
# I_CWT = joblib.load(I_CWT_path)
I_CWT = joblib.load(I_CWT_not_full_path)
Feeder_Output = joblib.load(Class_path)
Rs = joblib.load(Rs_path)
Duration = joblib.load(Duration_path)


def shuffle_dataset(dataset, output_class, rs, dur):
    x_trn, x_tst, y_trn, y_tst = [], [], [], []
    rs_trn, rs_tst, dur_trn, dur_tst = [], [], [], []

    scenario_length = [44580, 114420, 114420, 114420, 12264, 12264, 12264, 28608, 28608, 28608, 9804, 14689]

    for idx in range(len(scenario_length)):
        x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = train_test_split(
            dataset[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            output_class[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            rs[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            dur[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            test_size=0.2, random_state=rand_num)

        for j in x_train:
            x_trn.append(j)
        for j in x_test:
            x_tst.append(j)
        for j in y_train:
            y_trn.append(j)
        for j in y_test:
            y_tst.append(j)
        for j in rs_train:
            rs_trn.append(j)
        for j in rs_test:
            rs_tst.append(j)
        for j in dur_train:
            dur_trn.append(j)
        for j in dur_test:
            dur_tst.append(j)

    temp1 = list(zip(x_trn, y_trn, rs_trn, dur_trn))
    temp2 = list(zip(x_tst, y_tst, rs_tst, dur_tst))

    random.shuffle(temp1)
    random.shuffle(temp2)

    x_trn, y_trn, rs_trn, dur_trn = zip(*temp1)
    x_tst, y_tst, rs_tst, dur_tst = zip(*temp2)

    x_trn = np.stack(x_trn, axis=0)
    x_tst = np.stack(x_tst, axis=0)
    y_trn = np.stack(y_trn, axis=0)
    y_tst = np.stack(y_tst, axis=0)
    rs_trn = np.stack(rs_trn, axis=0)
    rs_tst = np.stack(rs_tst, axis=0)
    dur_trn = np.stack(dur_trn, axis=0)
    dur_tst = np.stack(dur_tst, axis=0)

    return x_trn, x_tst, y_trn, y_tst, rs_trn, rs_tst, dur_trn, dur_tst


def create_model(input_shapes, rates, kernel_initializer, no_outputs, no_layer, unit):
    # Keras model
    model = Sequential()
    # First layer specifies input_shape
    model.add(Conv2D(64, 5, activation='relu', padding='same', input_shape=input_shapes,
                     kernel_initializer=kernel_initializer, data_format='channels_last'))
    model.add(Dropout(rate=rates))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 5, activation='relu', padding='same', kernel_initializer=kernel_initializer,
                     data_format='channels_last'))
    model.add(Dropout(rate=2*rates))
    model.add(MaxPooling2D())
    model.add(Flatten())

    # Max 5 Full connected hidden layers
    for idx in range(no_layer - 1):
        model.add(Dense(units=unit[idx], activation='relu', kernel_initializer=kernel_initializer))
        rt = rates*(3 + float(idx))
        model.add(Dropout(rate=rt))

    model.add(Dense(no_outputs, activation='softmax'))

    return model


def evaluate_model(dataset, output, rs, dur, model_df):

    epochs = 200
    history = np.empty((len(model_df), 4, epochs))
    x_axis = range(1, epochs+1)
    for i in range(len(model_df)):
        model = model_df.iloc[i]
        print(model)
        batch_size, layers = int(model['batch_size']), int(model['layers'])
        rate, units = int(model['rate']), literal_eval(model['units'])
        x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = shuffle_dataset(dataset,
                                                                                                   output, rs, dur)

        # define model
        num_features, num_outputs = x_train.shape[1], y_train.shape[1]
        # reshape into subsequences (samples, time steps, rows, cols, channels)
        shape_input = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

        # define model
        verbose = 1

        # define model
        best_model = create_model(shape_input, rate, "he_normal", num_outputs, layers, units)

        best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        best_model.summary()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

        history_i = best_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
                                   batch_size=batch_size, verbose=verbose, callbacks=[es, TqdmCallback(verbose=0)])

        best_model.save(results_dic + r'\model_' + str(i+1))
        len_of_history = len(history_i.history['accuracy'])
        history[i, 0, :len_of_history] = history_i.history['accuracy']
        if len_of_history < epochs:
            history[i, 0, len_of_history:] = [history[i, 0, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[i, 1, :len_of_history] = history_i.history['loss']
        if len_of_history < epochs:
            history[i, 1, len_of_history:] = [history[i, 1, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[i, 2, :len_of_history] = history_i.history['val_accuracy']
        if len_of_history < epochs:
            history[i, 2, len_of_history:] = [history[i, 2, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[i, 3, :len_of_history] = history_i.history['val_loss']
        if len_of_history < epochs:
            history[i, 3, len_of_history:] = [history[i, 3, len_of_history-1] for _ in range(epochs - len_of_history)]

    max_loss = np.amax(history[:, 1, :])
    max_val_loss = np.amax(history[:, 3, :])

    plt.rc('font', size=7)
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=250, figsize=(10, 10), sharex=True)
    fig.suptitle('History Data of Model with ReLU in Hidden Layers')

    axs[0, 0].plot(x_axis, 100 * history[0, 0, :], label='1st model', color='red', linewidth=3)
    axs[0, 0].plot(x_axis, 100 * history[1, 0, :], label='2nd model', color='green', linewidth=3)
    axs[0, 0].plot(x_axis, 100 * history[2, 0, :], label='3rd model', color='blue', linewidth=3)
    axs[0, 0].set_title('Accuracy of models')
    axs[0, 0].set_ylim([-5, 105])
    axs[0, 0].set_xlabel('Trials')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].legend(loc='lower right', fontsize='small')

    axs[0, 1].plot(x_axis, 100 * history[0, 1, :], label='1st model', color='red', linewidth=3)
    axs[0, 1].plot(x_axis, 100 * history[1, 1, :], label='2nd model', color='green', linewidth=3)
    axs[0, 1].plot(x_axis, 100 * history[2, 1, :], label='3rd model', color='blue', linewidth=3)
    axs[0, 1].set_title('Loss of models')
    # axs[0, 1].set_ylim([-5, 100*max_loss + 5])
    axs[0, 1].set_ylim([-5, 105])
    axs[0, 1].set_xlabel('Trials')
    axs[0, 1].set_ylabel('Loss (%)')
    axs[0, 1].legend(loc='upper right', fontsize='small')

    axs[1, 0].plot(x_axis, 100 * history[0, 2, :], label='1st model', color='red', linewidth=3)
    axs[1, 0].plot(x_axis, 100 * history[1, 2, :], label='2nd model', color='green', linewidth=3)
    axs[1, 0].plot(x_axis, 100 * history[2, 2, :], label='3rd model', color='blue', linewidth=3)
    axs[1, 0].set_title('Validation Accuracy of models')
    axs[1, 0].set_ylim([-5, 105])
    axs[1, 0].set_xlabel('Trials')
    axs[1, 0].set_ylabel('Validation Accuracy (%)')
    axs[1, 0].legend(loc='lower right', fontsize='small')

    axs[1, 1].plot(x_axis, 100 * history[0, 3, :], label='1st model', color='red', linewidth=3)
    axs[1, 1].plot(x_axis, 100 * history[1, 3, :], label='2nd model', color='green', linewidth=3)
    axs[1, 1].plot(x_axis, 100 * history[2, 3, :], label='3rd model', color='blue', linewidth=3)
    axs[1, 1].set_title('Validation Loss of models')
    # axs[1, 1].set_ylim([-5, 100*max_val_loss + 5])
    axs[1, 1].set_ylim([-5, 105])
    axs[1, 1].set_xlabel('Trials')
    axs[1, 1].set_ylabel('Validation Loss (%)')
    axs[1, 1].legend(loc='upper right', fontsize='small')

    plt.savefig(results_dic + r'\model' + r'.jpg')
    plt.close()
    return history


# Load Data
best_models = pd.read_csv(best_models_loc, index_col=0)

history = evaluate_model(I_CWT, Feeder_Output, Rs, Duration, best_models)
joblib.dump(history, results_dic + r'\history.joblib')

