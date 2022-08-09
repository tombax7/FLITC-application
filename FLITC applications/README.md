# FLITC-application Process

This is the final part of the FLITC application.

The folder contains code for each type of implemented DNN which is designed to run each component of the FLITC Fault Diagnosis Toolkit.

In the second step of the application, the [Hyperparameter Tuning](https://github.com/tombax7/FLITC-application/tree/main/Hyperparameter%20Tuning) step, an algorithm is used to search the optimal hyperparameters of each DNN's model to maximize the DNN's accuracy. In this step, the best 3 hyperparameter sets are retrained to showcase each DNN's performance.

 
## The following code handles:
1. *branch_train_best_model_DMDCWT.py* trains and evaluates the best 3 FLITC FBNN models for each feeder of the LVDG.
2. *distance_train_best_models_DMDCWT.py* trains and evaluates the best 3 FLITC FDNN models of the LVDG.
3. *fault_train_best_models_DMDCWT.py* trains and evaluates the best 3 FLITC FCNN models of the LVDG.
4. *feeder_train_best_models_CWT.py* trains and evaluates the best 3 FLITC FFNN models of the LVDG.

Finally, the last script, named *run_best_models.py*, trains and evaluates only the best model for each of the application's DNN, as these are the Neural Networks used by the FLITC system.
