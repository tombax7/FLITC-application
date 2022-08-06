import os
import joblib as jb
import numpy as np
import matplotlib.pyplot as plt


def DMD(data, r):
    """Dynamic Mode Decomposition (DMD) algorithm."""
    
    ## Build data matrices
    X1 = data[:,:-1]
    X2 = data[:,1:]
    ## Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(X1, full_matrices = False)
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    ## Perform eigenvalue decomposition on A_tilde
    Phi, Q = np.linalg.eig(A_tilde)
    ## Compute the coefficient matrix
    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ Q
    A = Psi @ np.diag(Phi) @ np.linalg.pinv(Psi)
    
    return A, A_tilde, Phi, Psi

def DMD_I(I_path):
    I=np.array(jb.load(I_path))
    dataset_shape = I.shape

    r=dataset_shape[1] #rank truncation
    
    X = np.ndarray([dataset_shape[0], dataset_shape[1], dataset_shape[2]-1], dtype=float)
    Psi_dataset = np.ndarray([dataset_shape[0], r, r], dtype=float)
    # remove first value that repeats
    for i in range(len(I)):
        temp = I[i,:,1:]
        X[i,:,:] = temp
        [A, A_tilde, Phi, Psi] = DMD(temp, r)
        Psi_dataset[i,:,:] = Psi
    
    return Psi_dataset

def DMD_feeder(V_f_path):
    V_f=np.array(jb.load(V_f_path))
    dataset_shape = V_f.shape
    dataset_shape_new = (dataset_shape[0], dataset_shape[1]*dataset_shape[2], 
                         dataset_shape[3])
    V_f_new = np.reshape(V_f, dataset_shape_new)

    r=dataset_shape[1]*dataset_shape[2] #rank truncation
    
    X = np.ndarray([dataset_shape[0], dataset_shape[1]*dataset_shape[2], dataset_shape[3]-1], dtype=float)
    Psi_dataset = np.ndarray([dataset_shape[0], r, r], dtype=float)
    # remove first value that repeats
    for i in range(len(V_f_new)):
        temp = V_f_new[i,:,1:]
        X[i,:,:] = temp
        [A, A_tilde, Phi, Psi] = DMD(temp, r)
        Psi_dataset[i,:,:] = Psi
    
    return Psi_dataset

def DMD_branch(V_b_path):
    V_b=np.array(jb.load(V_b_path))
    dataset_shape = V_b.shape

    r=dataset_shape[1] #rank truncation
    
    X = np.ndarray([dataset_shape[0], dataset_shape[1], dataset_shape[2]-1], dtype=float)
    Psi_dataset = np.ndarray([dataset_shape[0], r, r], dtype=float)
    # remove first value that repeats
    for i in range(len(V_b)):
        temp = V_b[i,:,1:]
        X[i,:,:] = temp
        [A, A_tilde, Phi, Psi] = DMD(temp, r)
        Psi_dataset[i,:,:] = Psi
    
    return Psi_dataset

def main():
    directory = os.path.abspath(os.path.dirname(__file__))
    
    I_path = directory + r'\I_rescaled.joblib'
    Psi_I_path = directory + r'\Psi_I.joblib'
    V_f1_path = directory + r'\V_feeder_1.joblib'
    Psi_f1_path = directory + r'\Psi_feeder_1.joblib'
    V_f2_path = directory + r'\V_feeder_2.joblib'
    Psi_f2_path = directory + r'\Psi_feeder_2.joblib'
    V_f3_path = directory + r'\V_feeder_3.joblib'
    Psi_f3_path = directory + r'\Psi_feeder_3.joblib'
    V_b_path = directory + r'\V_branch.joblib'
    Psi_V_b_path = directory + r'\Psi_V_b.joblib'
    

    if not os.path.exists(Psi_I_path):
        Psi_I = DMD_I(I_path)
        jb.dump(Psi_I, Psi_I_path)
    else:
        Psi_I = jb.load(Psi_I_path)
        
    if not os.path.exists(Psi_f1_path):
        Psi_f1 = DMD_feeder(V_f1_path)
        jb.dump(Psi_f1, Psi_f1_path)
    else:
        Psi_f1 = jb.load(Psi_f1_path)
        
    if not os.path.exists(Psi_f2_path):
        Psi_f2 = DMD_feeder(V_f2_path)
        jb.dump(Psi_f2, Psi_f2_path)
    else:
        Psi_f2 = jb.load(Psi_f2_path)
    
    if not os.path.exists(Psi_f3_path):
        Psi_f3 = DMD_feeder(V_f3_path)
        jb.dump(Psi_f3, Psi_f3_path)
    else:
        Psi_f3 = jb.load(Psi_f3_path)
    
    if not os.path.exists(Psi_V_b_path):
        Psi_V_b = DMD_branch(V_b_path)
        jb.dump(Psi_V_b, Psi_V_b_path)
    else:
        Psi_V_b = jb.load(Psi_V_b_path)
        
    return Psi_I, Psi_f1, Psi_f2, Psi_f3, Psi_V_b

if __name__ == "__main__":
    ans = main()