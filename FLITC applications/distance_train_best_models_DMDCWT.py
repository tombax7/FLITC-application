import os
import pandas as pd
import numpy as np
import tensorflow
import logging
import joblib
from ast import literal_eval
import create_topology as top
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
from tqdm.keras import TqdmCallback

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)
cce = tensorflow.keras.losses.CategoricalCrossentropy()

# GPU usage
tensorflow.get_logger().setLevel(logging.INFO)
physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
directory = os.getcwd()
results_dic = directory + r'\results\Distance_ID'
Distance_Class_sorted_path = directory + r'\Distance_Class_sorted.joblib'
Rs_path = directory + r'\Rs_FCNN.joblib'
Duration_path = directory + r'\Duration_FCNN.joblib'

tree, nodes, grid_length = top.create_grid()
leaf_nodes = top.give_leaves(nodes[0])

# Find all leaf nodes
grid_path = []
branch_distance = []
for leaf in leaf_nodes:
    arr, dist = top.give_path(nodes[0], leaf)
    grid_path.append(arr)
    branch_distance.append(dist)
feeders_num = len(nodes[0].children)  # Number of root's children: 3
branches_num = len(grid_path)  # Number of grid's branches: 9
metrics_num = len(nodes)  # Number of voltage meters: 33
distance_sections = 5  # Number of sections a branch is divided: 5

# Change load file for different dataset

Distance_Class_sorted = np.array(joblib.load(Distance_Class_sorted_path))
Rs = joblib.load(Rs_path)
Duration = joblib.load(Duration_path)


def shuffle_dataset(dataset, output_class, rs, dur):

    x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = train_test_split(dataset, output_class,
                                                                                                rs, dur, test_size=0.2,
                                                                                                random_state=rand_num)
    return x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test


def reshape_class(y_hat):
    re_y = [0 for _ in range(len(y_hat))]
    segments = len(y_hat[0])
    for i in range(len(y_hat)):
        re_y[i] = float(np.argmax(y_hat[i]))/(segments-1)

    return np.array(re_y)


def create_model(input_shapes, rates, kernel_initializer, no_layer, unit):
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

    model.add(Dense(1, activation='sigmoid'))

    return model


def evaluate_model(round_i, dataset, output, rs, dur, model_df):
    epochs = 200
    history = np.empty((len(model_df), epochs))
    x_axis = range(1, epochs + 1)
    for i in range(len(model_df)):
        model = model_df.iloc[i]
        print(model)
        batch_size, layers = int(model['batch_size']), int(model['layers'])
        rate, units = int(model['rate']), literal_eval(model['units'])
        x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = shuffle_dataset(dataset,
                                                                                                   output, rs, dur)
        x_train, x_test = tensorflow.convert_to_tensor(x_train), tensorflow.convert_to_tensor(x_test)
        y_train, y_test = tensorflow.convert_to_tensor(y_train), tensorflow.convert_to_tensor(y_test)

        # define model
        # reshape into subsequences (samples, time steps, rows, cols, channels)
        shape_input = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

        # define model
        verbose = 1

        # define model
        best_model = create_model(shape_input, rate, "he_normal", layers, units)
        best_model.compile(optimizer='adam', loss="mean_squared_error")
        best_model.summary()

        es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=25)

        history_i = best_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
                                   batch_size=batch_size, verbose=verbose, callbacks=[es, TqdmCallback(verbose=0)])

        best_model.save(results_dic + r'\model_' + str(100 - 10 * round_i) + '_' + str(i+1))
        len_of_history = len(history_i.history['loss'])
        history[i, :len_of_history] = history_i.history['loss']
        if len_of_history < epochs:
            history[i, len_of_history:] = [history[i, len_of_history - 1] for _ in range(epochs - len_of_history)]

    plt.rc('font', size=7)
    plt.plot(100 * history[0, :], label='1st model', color='red')
    plt.plot(100 * history[1, :], label='2nd model', color='green')
    plt.plot(100 * history[2, :], label='3rd model', color='blue')
    plt.title('Loss of models')
    plt.ylabel('Loss (%)')
    plt.xlabel('Trials')
    plt.legend(loc='upper right', fontsize='small')
    plt.show()

    plt.savefig(results_dic + r'\model_' + str(100 - 10 * round_i) + '.jpg')
    plt.close()
    return history


best_models_loc = results_dic + r'\best_dmdcwt_distance_id_not_full_v1.csv'
best_models = pd.read_csv(best_models_loc, index_col=0)

for part in range(8):
    V_branch_i_DMDCWT_not_full_path = directory + r'\V_branch_DMDCWT' + str(100-10*part) + '_not_full.joblib'
    V_branch_DMDCWT = joblib.load(V_branch_i_DMDCWT_not_full_path)
    history_path = results_dic + r'\history_' + str(100-10*part) + '.joblib'

    # Run Data
    history_distance = evaluate_model(part, V_branch_DMDCWT, Distance_Class_sorted, Rs, Duration, best_models)
    joblib.dump(history_distance, history_path)
