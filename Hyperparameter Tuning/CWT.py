import os
import joblib
import pywt
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


# Access path where dataset is stored
directory = os.path.abspath(os.path.dirname(__file__))
I_rescaled_path = directory + r'\I_rescaled.joblib'
I_rescaled_CWT_path = directory + r'\I_CWT.joblib'
I_rescaled_CWT_not_full_path = directory + r'\I_CWT_not_full.joblib'

I_rescaled_DMD_path = directory + r'\Psi_I.joblib'
I_rescaled_DMDCWT_path = directory + r'\I_DMDCWT.joblib'
I_rescaled_DMDCWT_not_full_path = directory + r'\I_DMDCWT_not_full.joblib'

V_feeder_DMD_1_path = directory + '\Psi_feeder_1.joblib'
V_feeder_DMDCWT_1_path = directory + '\V_feeder_DMDCWT_1.joblib'
V_feeder_DMDCWT_1_not_full_path = directory + r'\V_feeder_DMDCWT_1_not_full.joblib'

V_feeder_DMD_2_path = directory + '\Psi_feeder_2.joblib'
V_feeder_DMDCWT_2_path = directory + '\V_feeder_DMDCWT_2.joblib'
V_feeder_DMDCWT_2_not_full_path = directory + r'\V_feeder_DMDCWT_2_not_full.joblib'

V_feeder_DMD_3_path = directory + '\Psi_feeder_3.joblib'
V_feeder_DMDCWT_3_path = directory + '\V_feeder_DMDCWT_3.joblib'
V_feeder_DMDCWT_3_not_full_path = directory + r'\V_feeder_DMDCWT_3_not_full.joblib'

V_branch_path = directory + r'\V_branch.joblib'
V_branch_CWT_path = directory + r'\V_branch_CWT.joblib'
V_branch_CWT_not_full_path = directory + r'\V_branch_CWT_not_full.joblib'

V_branch_DMD_path = directory + r'\Psi_V_b.joblib'
V_branch_DMDCWT_path = directory + r'\V_branch_DMDCWT.joblib'
V_branch_DMDCWT_not_full_path = directory + r'\V_branch_DMDCWT_not_full.joblib'


V_feeder_1_path = directory + r'\V_feeder_1.joblib'
V_feeder_1_CWT_path = directory + r'\V_feeder_1_CWT.joblib'
V_feeder_1_CWT_not_full_path = directory + r'\V_feeder_1_CWT_not_full.joblib'

V_feeder_2_path = directory + r'\V_feeder_2.joblib'
V_feeder_2_CWT_path = directory + r'\V_feeder_2_CWT.joblib'
V_feeder_2_CWT_not_full_path = directory + r'\V_feeder_2_CWT_not_full.joblib'

V_feeder_3_path = directory + r'\V_feeder_3.joblib'
V_feeder_3_CWT_path = directory + r'\V_feeder_3_CWT.joblib'
V_feeder_3_CWT_not_full_path = directory + r'\V_feeder_3_CWT_not_full.joblib'

def create_cwt_images(X, n_scales, rescale_size, wavelet_name = "morl"):
    n_samples = X.shape[0]
    n_nodes = X.shape[1]
    n_timesteps = X.shape[2]
    # range of scales from 1 to n_scales
    scales = np.arange(1, n_timesteps + 1) 
    
    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_nodes))
    
    for sample in range(n_samples):
        for node in range(n_nodes):
            series = X[sample, node, :]
            coeff, freq = pywt.cwt(series, scales, wavelet_name, 1)
            coeff_ = resize(coeff, (rescale_size, rescale_size), mode = 'constant')
            X_cwt[sample, :, :, node] = coeff_

    return X_cwt

def create_cwt_images_not_full(X, n_scales, rescale_size, wavelet_name = "morl"):
    n_samples = X.shape[0]
    n_nodes = X.shape[1]
    # range of scales from 1 to n_scales
    scales = np.arange(1, n_scales + 1) 
    
    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_nodes))
    
    for sample in range(n_samples):
        for node in range(n_nodes):
            series = X[sample, node, :]
            coeff, freq = pywt.cwt(series, scales, wavelet_name, 1)
            coeff_ = resize(coeff, (rescale_size, rescale_size), mode = 'constant')
            X_cwt[sample, :, :, node] = coeff_
                        
    return X_cwt

def normalize_img(X):
    
    X_norm = np.ndarray(X.shape, dtype=float)
    
    for i in range(X.shape[0]):
        Xmin = X[i].min()
        Xmax = X[i].max()
        X_norm[i] = (X[i] - Xmin) / (Xmax - Xmin)
    
    return X_norm

# amount of pixels in X and Y 
rescale_size = 32
# determine the max scale size
n_scales = 32


# Feeder Current CWT processing
I_rescaled = joblib.load(I_rescaled_path)
I_rescaled_CWT = create_cwt_images(I_rescaled, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of I_rescaled_CWT: {I_rescaled_CWT.shape}")        
joblib.dump(I_rescaled_CWT, I_rescaled_CWT_path)

# Feeder Current DMDCWT processing
I_rescaled_DMD = joblib.load(I_rescaled_DMD_path)
I_rescaled_DMDCWT = create_cwt_images(I_rescaled_DMD, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of I_rescaled_DMDCWT: {I_rescaled_DMDCWT.shape}")
joblib.dump(I_rescaled_DMDCWT, I_rescaled_DMDCWT_path)


# Feeder_1 Voltage DMDCWT processing
V_feeder_DMD_1 = joblib.load(V_feeder_DMD_1_path)
V_feeder_DMDCWT_1 = create_cwt_images(V_feeder_DMD_1, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_DMDCWT_1: {V_feeder_DMDCWT_1.shape}")        
joblib.dump(V_feeder_DMDCWT_1, V_feeder_DMDCWT_1_path)

# Feeder_2 Voltage DMDCWT processing
V_feeder_DMD_2 = joblib.load(V_feeder_DMD_2_path)
V_feeder_DMDCWT_2 = create_cwt_images(V_feeder_DMD_2, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_DMDCWT_2: {V_feeder_DMDCWT_2.shape}")        
joblib.dump(V_feeder_DMDCWT_2, V_feeder_DMDCWT_2_path)

# Feeder_3 Voltage DMDCWT processing
V_feeder_DMD_3 = joblib.load(V_feeder_DMD_3_path)
V_feeder_DMDCWT_3 = create_cwt_images(V_feeder_DMD_3, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_DMDCWT_3: {V_feeder_DMDCWT_3.shape}")        
joblib.dump(V_feeder_DMDCWT_3, V_feeder_DMDCWT_3_path)

# Branch Voltage CWT processing
V_branch = joblib.load(V_branch_path)
V_branch_CWT = create_cwt_images(V_branch, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_branch_CWT: {V_branch_CWT.shape}")        
joblib.dump(V_branch_CWT, V_branch_CWT_path)

# Branch Voltage DMDCWT processing
V_branch_DMD = joblib.load(V_branch_DMD_path)
V_branch_DMDCWT = create_cwt_images(V_branch_DMD, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_branch_DMDCWT: {V_branch_DMDCWT.shape}")        
joblib.dump(V_branch_DMDCWT, V_branch_DMDCWT_path)



# Feeder Current CWT not full processing
I_rescaled = joblib.load(I_rescaled_path)
I_rescaled_CWT_not_full = create_cwt_images_not_full(I_rescaled, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of I_rescaled_CWT_not_full: {I_rescaled_CWT_not_full.shape}")        
joblib.dump(I_rescaled_CWT_not_full, I_rescaled_CWT_not_full_path)

# Feeder Current DMDCWT not full processing
I_rescaled_DMD = joblib.load(I_rescaled_DMD_path)
I_rescaled_DMDCWT_not_full = create_cwt_images_not_full(I_rescaled_DMD, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of I_rescaled_DMDCWT_not_full: {I_rescaled_DMDCWT_not_full.shape}")        
joblib.dump(I_rescaled_DMDCWT_not_full, I_rescaled_DMDCWT_not_full_path)

# Feeder_1 Voltage DMDCWT not full processing
V_feeder_DMD_1 = joblib.load(V_feeder_DMD_1_path)
V_feeder_DMDCWT_1_not_full = create_cwt_images_not_full(V_feeder_DMD_1, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_DMDCWT_1_not_full: {V_feeder_DMDCWT_1_not_full.shape}")        
joblib.dump(V_feeder_DMDCWT_1_not_full, V_feeder_DMDCWT_1_not_full_path)

# Feeder_2 Voltage DMDCWT not full processing
V_feeder_DMD_2 = joblib.load(V_feeder_DMD_2_path)
V_feeder_DMDCWT_2_not_full = create_cwt_images_not_full(V_feeder_DMD_2, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_DMDCWT_2_not_full: {V_feeder_DMDCWT_2_not_full.shape}")        
joblib.dump(V_feeder_DMDCWT_2_not_full, V_feeder_DMDCWT_2_not_full_path)

# Feeder_3 Voltage DMDCWT not full processing
V_feeder_DMD_3 = joblib.load(V_feeder_DMD_3_path)
V_feeder_DMDCWT_3_not_full = create_cwt_images_not_full(V_feeder_DMD_3, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_DMDCWT_3_not_full: {V_feeder_DMDCWT_3_not_full.shape}")        
joblib.dump(V_feeder_DMDCWT_3_not_full, V_feeder_DMDCWT_3_not_full_path)

# Branch Voltage CWT not full processing
V_branch = joblib.load(V_branch_path)
V_branch_CWT_not_full = create_cwt_images_not_full(V_branch, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_branch_CWT_not_full: {V_branch_CWT_not_full.shape}")        
joblib.dump(V_branch_CWT_not_full, V_branch_CWT_not_full_path)

# Branch Voltage DMDCWT not full processing
V_branch_DMD = joblib.load(V_branch_DMD_path)
V_branch_DMDCWT_not_full = create_cwt_images_not_full(V_branch_DMD, n_scales, rescale_size)
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_branch_DMDCWT_not_full: {V_branch_DMDCWT_not_full.shape}")        
joblib.dump(V_branch_DMDCWT_not_full, V_branch_DMDCWT_not_full_path)



# Feeder_1 Voltage CWT not full processing
V_feeder_1 = np.array(joblib.load(V_feeder_1_path))
V_feeder_1 = V_feeder_1.reshape(V_feeder_1.shape[:-3] + (-1, V_feeder_1.shape[3]))
V_feeder_1_CWT = normalize_img(create_cwt_images(V_feeder_1, n_scales, rescale_size))
V_feeder_1_CWT_not_full = normalize_img(create_cwt_images_not_full(V_feeder_1, n_scales, rescale_size))
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_1_CWT_not_full: {V_feeder_1_CWT_not_full.shape}")  
joblib.dump(V_feeder_1_CWT, V_feeder_1_CWT_path)      
joblib.dump(V_feeder_1_CWT_not_full, V_feeder_1_CWT_not_full_path)

# Feeder_2 Voltage CWT not full processing
V_feeder_2 = np.array(joblib.load(V_feeder_2_path))
V_feeder_2 = V_feeder_2.reshape(V_feeder_2.shape[:-3] + (-1, V_feeder_2.shape[3]))
V_feeder_2_CWT = normalize_img(create_cwt_images(V_feeder_2, n_scales, rescale_size))
V_feeder_2_CWT_not_full = normalize_img(create_cwt_images_not_full(V_feeder_2, n_scales, rescale_size))
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_2_CWT_not_full: {V_feeder_2_CWT_not_full.shape}")  
joblib.dump(V_feeder_2_CWT, V_feeder_2_CWT_path)        
joblib.dump(V_feeder_2_CWT_not_full, V_feeder_2_CWT_not_full_path)

# Feeder_3 Voltage CWT not full processing
V_feeder_3 = np.array(joblib.load(V_feeder_3_path))
V_feeder_3 = V_feeder_3.reshape(V_feeder_3.shape[:-3] + (-1, V_feeder_3.shape[3]))
V_feeder_3_CWT = normalize_img(create_cwt_images(V_feeder_3, n_scales, rescale_size))
V_feeder_3_CWT_not_full = normalize_img(create_cwt_images_not_full(V_feeder_3, n_scales, rescale_size))
print(f"shapes (n_samples, n_nodes, x_img, y_img) of V_feeder_3_CWT_not_full: {V_feeder_3_CWT_not_full.shape}")  
joblib.dump(V_feeder_3_CWT, V_feeder_3_CWT_path)        
joblib.dump(V_feeder_3_CWT_not_full, V_feeder_3_CWT_not_full_path)

I_rescaled_CWT_not_full = joblib.load(I_rescaled_CWT_not_full_path)
I_rescaled_CWT_not_full_rescaled = normalize_img(I_rescaled_CWT_not_full)
I_rescaled_CWT_not_full_rescaled_path = directory + r'\I_rescaled_CWT_not_full_rescaled.joblib'
joblib.dump(I_rescaled_CWT_not_full_rescaled, I_rescaled_CWT_not_full_rescaled_path)

V_feeder_1_CWT_not_full = joblib.load(V_feeder_1_CWT_not_full_path)
V_feeder_1_CWT_not_full_rescaled = normalize_img(V_feeder_1_CWT_not_full)
V_feeder_1_CWT_not_full_rescaled_path = directory + r'\V_feeder_1_CWT_not_full_rescaled.joblib'
joblib.dump(V_feeder_1_CWT_not_full_rescaled, V_feeder_1_CWT_not_full_rescaled_path)

V_feeder_2_CWT_not_full = joblib.load(V_feeder_2_CWT_not_full_path)
V_feeder_2_CWT_not_full_rescaled = normalize_img(V_feeder_2_CWT_not_full)
V_feeder_2_CWT_not_full_rescaled_path = directory + r'\V_feeder_2_CWT_not_full_rescaled.joblib'
joblib.dump(V_feeder_2_CWT_not_full_rescaled, V_feeder_2_CWT_not_full_rescaled_path)

V_feeder_3_CWT_not_full = joblib.load(V_feeder_3_CWT_not_full_path)
V_feeder_3_CWT_not_full_rescaled = normalize_img(V_feeder_3_CWT_not_full)
V_feeder_3_CWT_not_full_rescaled_path = directory + r'\V_feeder_3_CWT_not_full_rescaled.joblib'
joblib.dump(V_feeder_3_CWT_not_full_rescaled, V_feeder_3_CWT_not_full_rescaled_path)

V_branch_CWT_not_full = joblib.load(V_branch_CWT_not_full_path)
V_branch_CWT_not_full_rescaled = normalize_img(V_branch_CWT_not_full)
V_branch_CWT_not_full_rescaled_path = directory + r'\V_branch_CWT_not_full_rescaled.joblib'
joblib.dump(V_branch_CWT_not_full_rescaled, V_branch_CWT_not_full_rescaled_path)