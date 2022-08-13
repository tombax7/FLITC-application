import os
import joblib
import mat73
import numpy as np

def filter_dataset(dataset):
    time = dataset['Time']
    Dur = [[] for _ in range(12)]
    for idx in range(12):
        duration = time[idx][1][0]
        Dur[idx] = duration
    
    rs = dataset['Rs']
    Rs = [[] for _ in range(12)]
    for idx in range(12):
        res = rs[idx][0]
        Rs[idx] = res
    
    filtered_dataset = {
        "Name": dataset['Name'],
        "Duration": Dur,
        "Fault_Resistance": Rs,
        "Class": dataset['Class'],
        "Output": dataset['Output']
    }
    
    return filtered_dataset

def create_dataset():
    # Load data
    PATH = os.getcwd()
    directory = str(PATH)
    
    print("Directory is ",directory)
    
    # scenario_length = datapoints for each scenario
    scenario_length = [44580, 114420, 114420, 114420, 12264, 12264, 12264, 28608, 28608, 28608, 9804, 14689]
    
    Name = []
    Loads = [[] for _ in range(12)]
    Output = [[] for _ in range(12)]
    Class   = [[] for _ in range(12)]
    Time   = [[[] for _ in range(2)] for _ in range(12)]
    Rs   = [[] for _ in range(12)]
    
    # Open  dataset file
    parent_file = os.path.abspath(os.path.join(path, os.pardir))
    datasets_file = parent_file + r'\Dataset Creation'
    filename = datasets_file + r'\Int_Scenario.mat'
    print("Data location is ",filename)
    if os.path.exists(filename):
        Data_i = mat73.loadmat(filename)
        # Data_i = scipy.io.loadmat(filename) #, variable_names=['Output','Class'])
        # For the number of instances of each scenario add the mat files data in a new Dataset file optimized for ML
        
        for idx in range(len(scenario_length)):
            Name.append(Data_i['Name'][idx])
            Time[idx][0].append(Data_i['Time'][0][0][:scenario_length[idx]])
            Time[idx][1].append(Data_i['Time'][1][0][:scenario_length[idx]])
            Rs[idx].append(Data_i['Rs'][idx][:scenario_length[idx]])
            for counter in range(scenario_length[idx]):
                Loads[idx].append(Data_i['Loads'][idx][counter])
                V_val = Data_i['Output'][idx][counter][0]
                I_val = Data_i['Output'][idx][counter][1]
                for meter in range(99):
                    for time in range(250):
                        check_nan = V_val[time][meter]
                        if np.isnan(check_nan): 
                            if time == 0: V_val[time][meter] = 0
                            else: V_val[time][meter] = V_val[time-1][meter]
                for meter in range(9):
                    for time in range(250):
                        check_nan = I_val[time][meter]
                        if np.isnan(check_nan): 
                            if time == 0: I_val[time][meter] = 0
                            else: I_val[time][meter] = I_val[time-1][meter]
                
                Output[idx].append([V_val,I_val])
                Class[idx].append(Data_i['Class'][idx][counter])
    
    Dataset = {
      "Name": Name,
      "Loads": Loads,
      "Time": Time,
      "Rs": Rs,
      "Class": Class,
      "Output": Output
    }
    
    Filtered_Dataset = filter_dataset(Dataset)
    joblib.dump(Filtered_Dataset, directory + r'\Dataset.joblib')
    joblib.dump(Class, directory + r'\Class.joblib')


create_dataset()