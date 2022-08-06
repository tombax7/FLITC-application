import os
import joblib
import numpy as np


def avg_sample(sample):
    sample_shape = sample.shape
    sample_reduced_shape = (sample_shape[0], sample_shape[1], int(sample_shape[2]/5))
    sample_reduced = np.ndarray(sample_reduced_shape, dtype=float)
    idx_shape = (int(sample_shape[2]/5), 5)
    idx = np.ndarray(idx_shape, dtype=int)
    
    val=[]
    for i in range(int(sample_shape[2]/15)):
        k = 15*i
        for j in range(3):
            l = k+j
            val.append(l)
    idx[:,0] = val
    
    for i in range(int(sample_shape[2]/5)):
        for j in range(1,5):
            idx[i,j] = idx[i,j-1] + 3
        
    for i in range(int(sample_shape[2]/5)):
        temp = sample[:,:,idx[i,:]]
        temp_avg = np.average(temp, axis=2)
        sample_reduced[:,:,i] = temp_avg
    return sample_reduced

def avg_dataset(dataset):
    dataset_shape = dataset.shape
    dataset_reduced_shape = (dataset_shape[0], dataset_shape[1], dataset_shape[2], int(dataset_shape[3]/5))
    dataset_reduced = np.ndarray(dataset_reduced_shape, dtype=float)
    for i in range(dataset_shape[0]):
        sample = dataset[i]
        sample_reduced = avg_sample(sample)
        dataset_reduced[i] = sample_reduced
    return dataset_reduced

def main():
    
    # Access path where dataset is stored
    directory = os.path.abspath(os.path.dirname(__file__))
    
    V_feeder_CWT_1_path = directory + '\V_feeder_1_CWT_not_full_rescaled.joblib'
    V_feeder_CWT_2_path = directory + '\V_feeder_2_CWT_not_full_rescaled.joblib'
    V_feeder_CWT_3_path = directory + '\V_feeder_3_CWT_not_full_rescaled.joblib'

    V_feeder_CWT_1_reduced_path = directory + '\V_feeder_CWT_reduced_1_not_full_rescaled.joblib'
    V_feeder_CWT_2_reduced_path = directory + '\V_feeder_CWT_reduced_2_not_full_rescaled.joblib'
    V_feeder_CWT_3_reduced_path = directory + '\V_feeder_CWT_reduced_3_not_full_rescaled.joblib'
    
    V_feeder_CWT_1 = joblib.load(V_feeder_CWT_1_path)
    print("V_feeder_CWT_1_not_full loaded...")
    V_feeder_CWT_1_reduced = avg_dataset(V_feeder_CWT_1)
    print("V_feeder_CWT_1_not_full reduced...")
    joblib.dump(V_feeder_CWT_1_reduced, V_feeder_CWT_1_reduced_path)
    print("V_feeder_CWT_1_reduced_not_full saved...")
    
    
    V_feeder_CWT_2 = joblib.load(V_feeder_CWT_2_path)
    print("V_feeder_CWT_2_not_full loaded...")
    V_feeder_CWT_2_reduced = avg_dataset(V_feeder_CWT_2)
    print("V_feeder_CWT_2_not_full reduced...")
    joblib.dump(V_feeder_CWT_2_reduced, V_feeder_CWT_2_reduced_path)
    print("V_feeder_CWT_2_reduced_not_full saved...")
    
    V_feeder_CWT_3 = joblib.load(V_feeder_CWT_3_path)
    print("V_feeder_CWT_3_not_full loaded...")
    V_feeder_CWT_3_reduced = avg_dataset(V_feeder_CWT_3)
    print("V_feeder_CWT_3_not_full reduced...")
    joblib.dump(V_feeder_CWT_3_reduced, V_feeder_CWT_3_reduced_path)
    print("V_feeder_CWT_3_reduced_not_full saved...")
    
if __name__ == "__main__":
    main()