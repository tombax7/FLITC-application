import os
import joblib
import numpy as np


def rescaler(choose_data, rescale):
    directory = os.getcwd()

    # Access path where dataset is stored
    Iabc_path = directory + r'\Iabc.joblib'
    voltage_int_path = directory + r'\V_int.joblib'
    V_rescaled_path = directory + r'\V_rescaled.joblib' 
    I_rescaled_path = directory + r'\I_rescaled.joblib'
    
    # For Voltage rescaling    
    if not os.path.exists(V_rescaled_path):
        
        V_int = joblib.load(voltage_int_path)
        V_int=np.array(V_int)
    
        if rescale == True:
            x_max = V_int.max()  
            x_min = V_int.min()   
            d_range = x_max - x_min
            
            # transform data
            V_int[:, :, 0, :] = V_int[:, :, 1, :] #Remove settling of measurement values of grid simulation
            V_rescaled = np.empty(V_int.shape, dtype=float)
            for scenario in range(len(V_int)):
                for node in range(len(V_int[0])):
                    for time in range(len(V_int[0,0])):
                        for meter in range(len(V_int[0,0,0])):
                            V_rescaled[scenario, node, time, meter] = (V_int[scenario, node, time, meter] - x_min)/d_range
                                
        else: V_rescaled = V_int
        joblib.dump(V_rescaled, V_rescaled_path)
            
    else:
        V_rescaled = joblib.load(V_rescaled_path)
    
    
    # For Current rescaling 
    if not os.path.exists(I_rescaled_path):
        
        Iabc = joblib.load(Iabc_path)
        Iabc=np.array(Iabc)
        Iabc[:, 0, :] = Iabc[:, 1, :] #Remove settling of measurement values of grid simulation
        if rescale == True:
            x_max = Iabc.max()  
            x_min = Iabc.min()   
            d_range = x_max - x_min
        
            # transform data
            I_rescaled = np.empty(Iabc.shape, dtype=float)
            for scenario in range(len(Iabc)):
                for time in range(len(Iabc[0])):
                    for meter in range(len(Iabc[0,0])):
                        I_rescaled[scenario, time, meter] = (Iabc[scenario, time, meter] - x_min)/d_range
                        
            I_rescaled = np.asarray(I_rescaled)            
        else:
            I_rescaled = Iabc
            I_rescaled = np.asarray(Iabc)
        
        I_rescaled = I_rescaled.transpose(0,2,1)
        
        joblib.dump(I_rescaled, I_rescaled_path)
            
    else:
        I_rescaled = joblib.load(I_rescaled_path)
    
    if choose_data == 'V': return V_rescaled
    elif choose_data == 'I': return I_rescaled