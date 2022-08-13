import os
import joblib
import numpy as np

def sort_meta(dataset):
    
    Rs = dataset['Fault_Resistance']
    Duration = dataset['Duration']
    
    Rs_rescaled = np.empty(sum(array), dtype=float)
    Duration_rescaled = np.empty(sum(array), dtype=float)
    
    
    for i in range(len(scenario_length)):
        for j in range(scenario_length[i]):
            sum_i = sum(scenario_length[:i])
            Rs_rescaled[sum_i+j] = Rs[i][j]
            Duration_rescaled[sum_i+j] = Duration[i][j]
    
    return Rs_rescaled, Duration_rescaled

directory = os.getcwd()

# scenario_length = datapoints for each scenario
scenario_length = [44580, 114420, 114420, 114420, 12264, 12264, 12264, 28608, 28608, 28608, 9804, 14689]

# Open  dataset file
dataset_path = directory + r'\Dataset.joblib'
Rs_rescaled_path = directory + r'\Rs_FFNN.joblib'
Duration_rescaled_path = directory + r'\Duration_FFNN.joblib'
Dataset = joblib.load(dataset_path)

Rs_rescaled, Duration_rescaled = sort_meta(Dataset)

joblib.dump(Rs_rescaled, Rs_rescaled_path)
joblib.dump(Duration_rescaled, Duration_rescaled_path)