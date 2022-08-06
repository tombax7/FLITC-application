import os
import joblib
import numpy as np

# Load data
directory = os.getcwd()

Fault_Class_path = directory + r'\Fault_Class_sorted.joblib'
Fault_Class_reduced_path = directory + r'\Fault_Class_sorted_reduced.joblib'


Fault_Class = np.array(joblib.load(Fault_Class_path))


scenario_length = [1907, 1907, 1907, 205, 205, 205, 477, 477, 477, 163, 245]
array = np.array(scenario_length) * 3
scenario_length = list(array)

Fault_Class_reduced = np.zeros([Fault_Class.shape[0], 7], dtype=int)

for scenario in range(Fault_Class.shape[0]):
    max_val = np.argmax(Fault_Class[scenario])
    if max_val<6:
        Fault_Class_reduced[scenario,max_val] = 1
    elif max_val<10:
        Fault_Class_reduced[scenario,max_val-3] = 1
    else:
        Fault_Class_reduced[scenario,max_val-4] = 1
        
joblib.dump(Fault_Class_reduced, Fault_Class_reduced_path)