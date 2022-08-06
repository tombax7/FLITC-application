import os
import joblib
import numpy as np
import random
import create_topology as top

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)

# Load data
directory = os.path.abspath(os.path.dirname(__file__))
V_rescaled_path = directory + r'\V_rescaled.joblib'
I_CWT_path = directory + r'\I_CWT.joblib'
Feeder_Class_path = directory + r'\Feeder_Output_4_Outputs.joblib'
Branch_Class_path = directory + r'\Branch_Output.joblib'
Rs_FFNN_path = directory + r'\Rs_FFNN.joblib'
Duration_FFNN_path = directory + r'\Duration_FFNN.joblib'

V_rescaled = joblib.load(V_rescaled_path)
I = joblib.load(I_CWT_path)
V_rescaled = np.transpose(V_rescaled, (0, 1, 3, 2))
Feeder_Output = joblib.load(Feeder_Class_path)
Branch_Output = joblib.load(Branch_Class_path)
Rs_FFNN = joblib.load(Rs_FFNN_path)
Duration_FFNN = joblib.load(Duration_FFNN_path)

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
branches_num = len(grid_path)         # Number of grid's branches: 9
metrics_num = len(nodes)              # Number of voltage meters: 33
distance_sections = 5                 # Number of sections a branch is divided: 5


def right_feeder(feeder_class):
    outp = []
    for scenario in range(len(feeder_class)):
        index = np.argmax(feeder_class[scenario])
        outp.append(index)
    return outp        


def distributer(dataset, rs, duration, feeder_class, branch_class, num_of_feeders, leafs):
    new_dataset = [[] for _ in range(num_of_feeders)]
    new_class = [[] for _ in range(num_of_feeders)]
    new_rs = [[] for _ in range(num_of_feeders)]
    new_duration = [[] for _ in range(num_of_feeders)]
   
    branch_per_feeder = []
    for idx in range(num_of_feeders):
        path = []
        for lf in leafs:
            ar, _ = top.give_path(nodes[1+idx], lf)
            if len(ar) != 0:
                path.append(ar)
        branch_per_feeder.append(len(path))
        
    offset = 0    
    for idx in range(len(feeder_class)):
        if feeder_class[idx] == num_of_feeders:
            offset += 1
        else:
            feeder = feeder_class[idx]
            new_dataset[feeder].append(dataset[idx,
                                       sum(branch_per_feeder[:feeder]):sum(branch_per_feeder[:feeder+1]), :, :])
            new_class[feeder].append(branch_class[idx - offset])
            new_rs[feeder].append(rs[idx])
            new_duration[feeder].append(duration[idx])
            
    return new_dataset, new_class, new_rs, new_duration


def distributer_i(dataset, feeder_class, num_of_feeders):
    new_dataset = [[] for _ in range(num_of_feeders)]

    offset = 0
    for idx in range(len(feeder_class)):
        if feeder_class[idx] == num_of_feeders:
            offset += 1
        else:
            feeder = feeder_class[idx]
            new_dataset[feeder].append(dataset[idx])

    return new_dataset


squeezed_class = right_feeder(Feeder_Output)
V_feeder, Branch_Output_sorted, Rs_FBNN, Duration_FBNN = distributer(
    V_rescaled, Rs_FFNN, Duration_FFNN, squeezed_class, Branch_Output, feeders_num, leaf_nodes)
V_feeder = np.array(V_feeder, dtype=object)
# I_CWT = distributer_i(I, squeezed_class, feeders_num)

# for i in range(feeders_num):
#     i_data_path = directory + r'\I_CWT_' + str(i + 1) + '.joblib'
#     joblib.dump(I_CWT[i], i_data_path)

for i in range(feeders_num):
    data_path = directory + r'\V_feeder_' + str(i+1) + '.joblib'
    class_path = directory + r'\Branch_Output_sorted_' + str(i+1) + '.joblib'
    Rs_FBNN_path = directory + r'\Rs_FBNN_' + str(i+1) + '.joblib'
    Duration_FBNN_path = directory + r'\Duration_FBNN_' + str(i+1) + '.joblib'
    joblib.dump(V_feeder[i], data_path)
    joblib.dump(Branch_Output_sorted[i], class_path)
    joblib.dump(Rs_FBNN[i], Rs_FBNN_path)
    joblib.dump(Duration_FBNN[i], Duration_FBNN_path)
    