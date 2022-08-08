# Preprocessing/Hyperparameter Tuning Process

This is the second step in the FLITC application: Preprocessing/Hyperparameter Tuning Process

The folder contains multiple scripts in order to firstly preprocess the raw data coming from the first step and secondly feed them into the FLITC DNN models for exploring the best possible performance by tuning their hyperparameters.
For the code to run smoothly, and since this is the first version of the application on GitHub, please pay extra attention to your directory environment (folder/subfolders) and make slight changes to the lines which read/write to specific locations in your environment.
Below, a process line is shown explaining the order of the scripts' execution.

## The order of execution is: 
1. *Dataset_Parser.py* handles the direct manipulation of the *.mat* file coming from *MATLAB* and splits it into a Voltage measurement file, a Current measurement file and a file containing each scenario's metadata, such as fault characteristics, in a *.joblib* format.
2. *create_topology.py* recreates the LVDG model into a graph structure, so that it is possible to get information regarding branch length, node parent/children, leaf nodes, etc.
3. *preprocessing_stage.py* is the first actual Preprocessing Stage which handles data dimensionality structuring and Branch Voltage Measurements Interpolation to fixed-sized Branch Virtual-Node Measurements.
4. *rescale_data.py* gives the user the choice to normalize the Voltage/Current data (The data is already in *per unit* format from the Simulink Environment).
5. *sort_metadata.py* simply handles the metadata mentioned in step 1.
6. *class_encoding.py* creates the classification label files used in the Supervised Learning process by the four FLITC DNNs.
7. *reduce_fault_class.py* reduces the number of classes in the Fault Class Neural Network (FCNN) from eleven (A-G, B-G, C-G, A-B, A-C, B-C, A-B-G, A-C-G, B-C-G, A-B-C, A-B-C-G) to (A-G, B-G, C-G, A-B, A-C, B-C, A-B-C) \*.
8. *sort_branches.py* creates a number of dataset files equal to the LVDG feeders containing only the feeder's branches where the fault occurred.
9. *sort_classes.py* creates a dataset file containing only the faulty branch voltage measurements.
10. *sort_distances.py* creates a dataset file similarly to *sort_classes.py*.
11. *DMD.py* applies the [Dynamic Mode Decomposition](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition) Dimensionality Reduction algorithm to the datasets.
12. *CWT.py* applies the [Continuous Wavelet Transform](https://en.wikipedia.org/wiki/Continuous_wavelet_transform) algorithm to the datasets.
13. *reduce_branch_dataset.py* applies Dimensionality Reduction to the faulted feeder branch measurements by averaging each branch node per phase.
14. *Feeder_HyperOpt_CWT_not_full_v1.py* searches the best-suited hyperparameters for the Faulty Feeder Neural Network (FFNN).
15. *Branch_HyperOpt_DMDCWT_1_not_full_v2.py* searches the best-suited hyperparameters for the Feeder A - Faulty Branch Neural Network (FBNN_1).
16. *Branch_HyperOpt_DMDCWT_2_not_full_v2.py* searches the best-suited hyperparameters for the Feeder B - Faulty Branch Neural Network (FBNN_2).
17. *Branch_HyperOpt_DMDCWT_3_not_full_v1.py* searches the best-suited hyperparameters for the Feeder C - Faulty Branch Neural Network (FBNN_3).
18. *Fault_HyperOpt_DMDCWT_not_full_v2.py* searches the best-suited hyperparameters for the Fault Class Neural Network (FCNN).
19. *Distance_HyperOpt_DMDCWT_not_full_v1.py* searches the best-suited hyperparameters for the Fault Distance Neural Network (FDNN).


\* This happens due to the system's inability to differentiate between a line-to-line and line-to-line-ground fault (lack of ground measuring equipment)
