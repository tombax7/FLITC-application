# Finds the Best HyperParameters for the Fault Type Identification NN

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import os
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import create_topology as top
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.backend import clear_session
import logging
from tqdm.keras import TqdmCallback
import random
import time

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)
cce = tf.keras.losses.CategoricalCrossentropy()

# GPU usage
tf.get_logger().setLevel(logging.INFO)
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
directory = os.getcwd()
results_dic = directory + r'\results\Fault_ID'

V_branch_DMDCWT_not_full_path = directory + r'\V_branch_DMDCWT_not_full.joblib'
Fault_Class_sorted_path = directory + r'\Fault_Class_sorted_reduced.joblib'
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
V_branch_DMDCWT = joblib.load(V_branch_DMDCWT_not_full_path)
Fault_Class_sorted = np.array(joblib.load(Fault_Class_sorted_path))
Rs = np.array(joblib.load(Rs_path))
Duration = np.array(joblib.load(Duration_path))


def shuffle_dataset(dataset, output_class, rs, dur):
    x_trn, x_tst, y_trn, y_tst = [], [], [], []
    rs_trn, rs_tst, dur_trn, dur_tst = [], [], [], []

    scenario_length = [114420, 114420, 114420, 12264, 12264, 12264, 28608, 28608, 28608, 9804, 14689]

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


def set_space(num_of_max_hidden_layers):

    end_space = {'layers': scope.int(hp.quniform('layers', 1, num_of_max_hidden_layers, 1))}

    for idx in range(num_of_max_hidden_layers):
        unit_name = 'units' + str(idx+1)
        end_space[unit_name] = scope.int(hp.quniform(unit_name, 16, 256, 16))

    return end_space


# Create the variables space of exploration
max_hidden_layers = 5
space = set_space(max_hidden_layers)
results_of_fnn = []


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
    model.add(Dropout(rate=rates/2))
    model.add(MaxPooling2D())
    model.add(Flatten())

    # Max 5 Full connected hidden layers
    for idx in range(no_layer - 1):
        model.add(Dense(units=unit[idx], activation='relu', kernel_initializer=kernel_initializer))
        rt = rates/(3 + float(idx))
        model.add(Dropout(rate=rt))

    model.add(Dense(no_outputs, activation='softmax'))

    return model


def create_model_v2(input_shapes, rates, kernel_initializer, kernel_regularizer, no_outputs, no_layer, unit):
    # Keras model
    model = Sequential()
    # First layer specifies input_shape
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shapes,
                     data_format='channels_last', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D())
    model.add(Flatten())

    # Max 5 Full connected hidden layers
    for idx in range(no_layer - 1):
        model.add(Dense(units=unit[idx], activation='relu', kernel_initializer=kernel_initializer))
        rt = rates/(1 + float(idx))
        model.add(Dropout(rate=rt))

    model.add(Dense(no_outputs, activation='softmax'))

    return model


def best_csv(dataframe, best_csv_loc):
    sorted_df = dataframe.sort_values(by=['Val_Loss (%)'])
    best_df = sorted_df.head(3)
    write_csv(best_df, best_csv_loc)
    return best_df


def write_csv(dataframe, csv_loc):
    try:
        outfile = open(results_dic + csv_loc, 'wb')
        dataframe.to_csv(outfile)
        outfile.close()
    except IOError as error:
        print(error)


# Define the function that defines model
def f_nn(params):
    dataset, output_class = V_branch_DMDCWT, Fault_Class_sorted
    x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = shuffle_dataset(dataset, output_class,
                                                                                               Rs, Duration)
    # x_train, x_test = tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test)
    # y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)

    # define model
    num_outputs = y_train.shape[1]
    print('n_outputs=', num_outputs)
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    shape_input = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    layer = params['layers']
    b_size, rt = 25, 0.05

    unt = [params['units' + str(idx + 1)] for idx in range(1, layer)]

    model = create_model_v2(shape_input, rt, "he_normal", L2(l=0.01), num_outputs, layer, unt)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()
    plot_model(model, "Fault_ID_training_model.png", show_shapes=True)
    es = EarlyStopping(monitor='val_loss', mode='min',
                       verbose=0, patience=25)
    start_time = time.time()
    result = model.fit(x_train, y_train,
                       verbose=0,
                       validation_split=0.2,
                       batch_size=b_size,
                       epochs=200,
                       callbacks=[es, TqdmCallback(verbose=0)])
    end_time = time.time()
    dt = end_time - start_time
    number_of_epochs_it_ran = len(result.history['loss'])
    avg_time = dt / number_of_epochs_it_ran
    # Get the highest accuracy of the training epochs
    acc = round(np.amax(result.history['accuracy']), 5)
    # Get the lowest loss of the training epochs
    loss = round(np.amin(result.history['loss']), 5)
    # Get the highest validation accuracy of the training epochs
    val_acc = round(np.amax(result.history['val_accuracy']), 5)
    # Get the lowest validation loss of the training epochs
    val_loss = round(np.amin(result.history['val_loss']), 5)

    print('Best validation loss of epoch:', val_loss)

    val_length = int(0.2 * len(y_test))
    x_val, y_val = x_train[-val_length:], y_train[-val_length:]
    yp = model.predict(x_val)
    cross = cce(y_val, yp).numpy()
    clear_session()
    print('Cross-entropy loss of eval:', cross)
    _, score = model.evaluate(x_train, y_train, verbose=0)

    print("Train accuracy: %.2f%%" % (100 * score))
    results_of_fnn.append([100 * acc, 100 * loss, 100 * val_acc, 100 * val_loss, b_size, rt, unt, layer,
                           number_of_epochs_it_ran, avg_time, dt])

    names = ['Accuracy (%)', 'Loss (%)', 'Val_ Accuracy (%)', 'Val_Loss (%)', 'batch_size', 'rate', 'units', 'layers',
             'Number of Epochs', 'Average Time per Epoch (s)', 'Total Time']
    df = pd.DataFrame(results_of_fnn, columns=names)

    # Change command according to the training set
    write_csv(df, r'\dmdcwt_fault_id_not_full_v2.csv')
    best_csv(df, r'\best_dmdcwt_fault_id_not_full_v2.csv')

    return {'loss': cross, 'status': STATUS_OK, 'model': model, 'params': params}


trials = Trials()
best = fmin(f_nn,
            space,
            algo=tpe.suggest,
            max_evals=200,  # 200
            trials=trials)  # max_evals=200
# print(results_of_fnn)

best_model = trials.results[np.argmin([r['loss'] for r in
                                       trials.results])]['model']
best_params = trials.results[np.argmin([r['loss'] for r in
                                        trials.results])]['params']
worst_model = trials.results[np.argmax([r['loss'] for r in
                                        trials.results])]['model']
worst_params = trials.results[np.argmax([r['loss'] for r in
                                         trials.results])]['params']

print(trials.trials)
print(best_model)
print(best_params)
print(worst_model)
print(worst_params)

print("Best estimate parameters", best)
batch_size, rate, epochs, layers = 25, 0.05, 200, int(best['layers'])
trainX, testX, trainY, testY, trainRs, testRs, trainDuration, testDuration = shuffle_dataset(V_branch_DMDCWT,
                                                                                             Fault_Class_sorted, Rs,
                                                                                             Duration)
# trainX, testX = tf.convert_to_tensor(trainX), tf.convert_to_tensor(testX)
# trainY, testY = tf.convert_to_tensor(trainY), tf.convert_to_tensor(testY)
n_outputs = trainY.shape[1]
units = [int(best['units'+str(idx+1)]) for idx in range(1, layers)]
input_shape = (trainX.shape[1], trainX.shape[2], trainX.shape[3])

# fit and evaluate the best model

# define model
verbose = 1
print('batch_size=%s' % batch_size)
print('epochs=%s' % epochs)
print('layers=%s' % layers)
print('units=%s' % units)
print('n_outputs=%s' % n_outputs)

# define model
last_model = create_model_v2(input_shape, rate, "he_normal", L2(l=0.01), n_outputs, layers, units)

last_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
last_model.summary()
plot_model(last_model, "Fault_ID_best_model.png", show_shapes=True)
last_model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
_, accuracy = last_model.evaluate(testX, testY, batch_size=batch_size, verbose=0)

print("Accuracy of the best model: %.3f%%" % (100 * accuracy))
