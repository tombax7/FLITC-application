# Dataset Creation Process

This is the first step in the FLITC application: The Dataset Creation Process

The folder contains the Simulink created *.slx* file of the created LVDG model as used by the application. The other files contain the code neccessary for running the simulation as well as a *.mat* file with real Active Power measurements from consumers and PV microgeneration producers spanning one month.

## First Step: run *Create_network.m*

*Create_network.m* handles the creation of simulated environment's variables such as: 
- Consumer/Producer Loads 
- Fault Type 
- Fault Resistance 
- Fault Duration 
- Fault Time

## Second Step: run *simulation_main.m*

*simulation_main.m* is the main loop function simulating all fault scenarios created at the previous step. After running each scenario, it saves the Voltage/Current measurements in a .mat file that will be used later in the next FLITC's process.

Lastly, the folder contains the *Generate_Fault_Data.m* function that is called during the second step. This function is responsible for changing the values of the Consumer/Producer Loads during the simulation by interpolating neighboring scenario's load values. Furthermore, it places the Three Phase Fault Block to a randomly selected location in the LVDG with characteristics specified in the first step. Lastly, it runs the simulation and collects the data for each scenario, which will be returned to the main loop's environment in the second step.

