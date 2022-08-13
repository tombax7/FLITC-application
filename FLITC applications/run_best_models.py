import os
import joblib
import tensorflow
import numpy as np
import pandas as pd
import create_topology as top
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

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)
cce = tensorflow.keras.losses.CategoricalCrossentropy()

# GPU usage
tensorflow.get_logger().setLevel(logging.INFO)
physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

# Location initialization data
directory = os.path.abspath(os.path.dirname(__file__))
results_dic = directory + r'\results'
feeder_dic = results_dic + r'\Feeder_ID'
branch_dic = results_dic + r'\Branch_ID'
fault_dic = results_dic + r'\Fault_ID'
distance_dic = results_dic + r'\Distance_ID'

# Scenario length initialization
scenario_length = [44580, 114420, 114420, 114420, 12264, 12264, 12264, 28608, 28608, 28608, 9804, 14689]

# LVDG topology recreation
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


def shuffle_dataset(dataset, output_class, rs, dur, scen_len):
    x_trn, x_tst, y_trn, y_tst = [], [], [], []
    rs_trn, rs_tst, dur_trn, dur_tst = [], [], [], []

    for idx in range(len(scen_len)):
        x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = train_test_split(
            dataset[sum(scen_len[:idx]):sum(scen_len[:idx + 1])],
            output_class[sum(scen_len[:idx]):sum(scen_len[:idx + 1])],
            rs[sum(scen_len[:idx]):sum(scen_len[:idx + 1])],
            dur[sum(scen_len[:idx]):sum(scen_len[:idx + 1])],
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


def quick_shuffle(dataset, output_class, rs, dur):

    x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = train_test_split(dataset, output_class,
                                                                                                rs, dur, test_size=0.2,
                                                                                                random_state=rand_num)
    return x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test


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


def create_model_v3(input_shapes, rates, kernel_initializer, no_layer, unit):
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


def evaluate_model(dataset, output, rs, dur, model, direc, scen_len):
    if direc == branch_dic:
        if len(output[0]) == 4:
            idx = 1
        elif len(output[0]) == 3:
            idx = 2
        else:
            idx = 3

    epochs = 200
    history = np.empty([4, epochs], dtype=float)
    x_axis = range(1, epochs+1)
    print(model)
    batch_size, layers = int(model['batch_size']), int(model['layers'])
    rate, units = int(model['rate']), literal_eval(model['units'])
    if direc == branch_dic:
        x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = quick_shuffle(dataset, output,
                                                                                                 rs, dur)
    else:
        x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = shuffle_dataset(dataset, output,
                                                                                                   rs, dur, scen_len)

    # define model
    if direc != distance_dic:
        num_outputs = y_train.shape[1]
    shape_input = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

    # define model
    verbose = 1

    # define model
    if direc == branch_dic and (idx == 1 or idx == 2):
        best_model = create_model_v2(shape_input, rate, "he_normal", L2(l=0.01), num_outputs, layers, units)
    elif direc == distance_dic:
        best_model = create_model_v3(shape_input, rate, "he_normal", layers, units)
    else:
        best_model = create_model(shape_input, rate, "he_normal", num_outputs, layers, units)

    if direc == distance_dic:
        best_model.compile(optimizer='adam', loss="mean_squared_error")
        best_model.summary()
        es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=25)
    else:
        best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        best_model.summary()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

    history_i = best_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
                               batch_size=batch_size, verbose=verbose, callbacks=[es, TqdmCallback(verbose=0)])

    best_model.save(direc + r'\model')

    if direc == distance_dic:
        len_of_history = len(history_i.history['loss'])
        history[:len_of_history] = history_i.history['loss']
        if len_of_history < epochs:
            history[len_of_history:] = [history[len_of_history - 1] for _ in range(epochs - len_of_history)]

        plt.rc('font', size=7)
        plt.plot(x_axis, 100 * history[0, :], linewidth=3)
        plt.title('Loss of model')
        plt.ylabel('Loss (%)')
        plt.xlabel('Trials')
        plt.show()

    else:
        len_of_history = len(history_i.history['accuracy'])
        history[0, :len_of_history] = history_i.history['accuracy']
        if len_of_history < epochs:
            history[0, len_of_history:] = [history[0, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[1, :len_of_history] = history_i.history['loss']
        if len_of_history < epochs:
            history[1, len_of_history:] = [history[1, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[2, :len_of_history] = history_i.history['val_accuracy']
        if len_of_history < epochs:
            history[2, len_of_history:] = [history[2, len_of_history-1] for _ in range(epochs - len_of_history)]
        history[3, :len_of_history] = history_i.history['val_loss']
        if len_of_history < epochs:
            history[3, len_of_history:] = [history[3, len_of_history-1] for _ in range(epochs - len_of_history)]

        plt.rc('font', size=7)
        fig, axs = plt.subplots(nrows=2, ncols=2, dpi=250, figsize=(10, 10), sharex=True)
        if direc == branch_dic:
            fig.suptitle('History Data of Model ' + str(idx))
        else:
            fig.suptitle('History Data of Model')

        axs[0, 0].plot(x_axis, 100 * history[0, :], linewidth=3)
        axs[0, 0].set_title('Accuracy of models')
        axs[0, 0].set_ylim([-5, 105])
        axs[0, 0].set_xlabel('Trials')
        axs[0, 0].set_ylabel('Accuracy (%)')

        axs[0, 1].plot(x_axis, 100 * history[1, :], linewidth=3)
        axs[0, 1].set_title('Loss of models')
        axs[0, 1].set_ylim([-5, 105])
        axs[0, 1].set_xlabel('Trials')
        axs[0, 1].set_ylabel('Loss (%)')

        axs[1, 0].plot(x_axis, 100 * history[2, :], linewidth=3)
        axs[1, 0].set_title('Validation Accuracy of models')
        axs[1, 0].set_ylim([-5, 105])
        axs[1, 0].set_xlabel('Trials')
        axs[1, 0].set_ylabel('Validation Accuracy (%)')

        axs[1, 1].plot(x_axis, 100 * history[3, :], linewidth=3)
        axs[1, 1].set_title('Validation Loss of models')
        axs[1, 1].set_ylim([-5, 105])
        axs[1, 1].set_xlabel('Trials')
        axs[1, 1].set_ylabel('Validation Loss (%)')

        if direc == branch_dic:
            plt.savefig(direc + r'\model_' + str(idx) + '.jpg')
        else:
            plt.savefig(direc + r'\model.jpg')

        plt.close()
    return history


# Load Feeder_ID data
I_CWT_not_full_path = directory + r'\I_CWT_not_full.joblib'
feeder_class_path = directory + r'\Feeder_Output_4_Outputs.joblib'
Rs_path = directory + r'\Rs_FFNN.joblib'
Duration_path = directory + r'\Duration_FFNN.joblib'
feeder_best_models_loc = feeder_dic + r'\best_cwt_feeder_id_not_full.csv'
I_CWT = joblib.load(I_CWT_not_full_path)
Feeder_Output = joblib.load(feeder_class_path)
Feeder_Rs = joblib.load(Rs_path)
Feeder_Duration = joblib.load(Duration_path)
feeder_best_models = pd.read_csv(feeder_best_models_loc, index_col=0)

# Run Feeder Evaluation
feeder_history = evaluate_model(I_CWT, Feeder_Output, Feeder_Rs, Feeder_Duration,
                                feeder_best_models.iloc[0], feeder_dic, scenario_length)
joblib.dump(feeder_history, feeder_dic + r'\feeder_history.joblib')

# Load Branch_ID data
branch_best_models_loc = [[] for _ in range(3)]
branch_best_models_loc[0] = branch_dic + r'\best_dmdcwt_branch_id_1_not_full_v2.csv'
branch_best_models_loc[1] = branch_dic + r'\best_dmdcwt_branch_id_2_not_full_v2.csv'
branch_best_models_loc[2] = branch_dic + r'\best_dmdcwt_branch_id_3_not_full_v1.csv'

V_feeder_CWT, Branch_Output_sorted, branch_best_models = [[] for _ in range(feeders_num)], \
                                                         [[] for _ in range(feeders_num)], \
                                                         [[] for _ in range(feeders_num)]
Branch_Rs, Branch_Duration = [[] for _ in range(feeders_num)], [[] for _ in range(feeders_num)]
for index in range(3):  # feeder_num
    data_path = directory + r'\V_feeder_DMDCWT_reduced_' + str(index + 1) + '_not_full.joblib'
    branch_class_path = directory + r'\Branch_Output_sorted_' + str(index + 1) + '.joblib'
    Rs_path = directory + r'\Rs_FBNN_' + str(index + 1) + '.joblib'
    Duration_path = directory + r'\Duration_FBNN_' + str(index + 1) + '.joblib'
    V_feeder_CWT[index] = joblib.load(data_path)
    Branch_Output_sorted[index] = np.array(joblib.load(branch_class_path))
    Branch_Rs[index] = joblib.load(Rs_path)
    Branch_Duration[index] = joblib.load(Duration_path)
    branch_best_models[index] = pd.read_csv(branch_best_models_loc[index], index_col=0)

# Run Branch Evaluation
branch_f1_history = evaluate_model(V_feeder_CWT[0], Branch_Output_sorted[0], Branch_Rs[0], Branch_Duration[0],
                                   branch_best_models[0].iloc[0], branch_dic, scenario_length)
joblib.dump(branch_f1_history, branch_dic + r'\branch_f1_history.joblib')
branch_f2_history = evaluate_model(V_feeder_CWT[1], Branch_Output_sorted[1], Branch_Rs[1], Branch_Duration[1],
                                   branch_best_models[1].iloc[0], branch_dic, scenario_length)
joblib.dump(branch_f2_history, branch_dic + r'\branch_f2_history.joblib')
branch_f3_history = evaluate_model(V_feeder_CWT[2], Branch_Output_sorted[2], Branch_Rs[2], Branch_Duration[2],
                                   branch_best_models[2].iloc[0], branch_dic, scenario_length)
joblib.dump(branch_f3_history, branch_dic + r'\branch_f3_history.joblib')

# Load Fault_ID data
fault_best_models_loc = fault_dic + r'\best_dmdcwt_fault_id_not_full_v2.csv'
V_branch_DMDCWT_path = directory + r'\V_branch_DMDCWT.joblib'
Fault_Class_sorted_path = directory + r'\Fault_Class_sorted_reduced.joblib'
Rs_path = directory + r'\Rs_FCNN.joblib'
Duration_path = directory + r'\Duration_FCNN.joblib'
V_branch_DMDCWT = joblib.load(V_branch_DMDCWT_path)
Fault_Class_sorted = np.array(joblib.load(Fault_Class_sorted_path))
fault_best_models = pd.read_csv(fault_best_models_loc, index_col=0)
Fault_Rs = np.array(joblib.load(Rs_path))
Fault_Duration = np.array(joblib.load(Duration_path))

# Run Branch Evaluation
fault_history = evaluate_model(V_branch_DMDCWT, Fault_Class_sorted, Fault_Rs, Fault_Duration, fault_best_models.iloc[0],
                               fault_dic, scenario_length[1:])
joblib.dump(fault_history, fault_dic + r'\fault_history.joblib')

# Load Distance_ID data
distance_best_models_loc = distance_dic + r'\best_dmdcwt_distance_id_not_full_v1.csv'
V_branch_DMDCWT_path = directory + r'\V_branch_DMDCWT.joblib'
Distance_Class_sorted_path = directory + r'\Distance_Class_sorted.joblib'
Rs_path = directory + r'\Rs_FCNN.joblib'
Duration_path = directory + r'\Duration_FCNN.joblib'
V_branch_DMDCWT = joblib.load(V_branch_DMDCWT_path)
Distance_Class_sorted = np.array(joblib.load(Distance_Class_sorted_path))
distance_best_models = pd.read_csv(distance_best_models_loc, index_col=0)
Distance_Rs = np.array(joblib.load(Rs_path))
Distance_Duration = np.array(joblib.load(Duration_path))

# Run Distance Evaluation
distance_history = evaluate_model(V_branch_DMDCWT, Distance_Class_sorted, Distance_Rs, Distance_Duration,
                                  distance_best_models.iloc[0], distance_dic, scenario_length[1:])
joblib.dump(distance_history, distance_dic + r'\distance_history.joblib')
