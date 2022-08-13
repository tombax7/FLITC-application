import os
import joblib
import numpy as np

# Load data
directory = os.getcwd()

Fault_Class_path = directory + r'\Fault_Class_sorted.joblib'
Fault_Class_reduced_path = directory + r'\Fault_Class_sorted_reduced.joblib'


Fault_Class = np.array(joblib.load(Fault_Class_path))


scenario_length = [114420, 114420, 114420, 12264, 12264, 12264, 28608, 28608, 28608, 9804, 14689]

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