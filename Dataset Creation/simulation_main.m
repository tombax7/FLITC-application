%% Load Scenario structure
clear variables
clc
Scenario = load([pwd,'\Int_Scenario.mat']);

%% Call Generate_Fault_Data function 
Per_Fault=[3715,9535,9535,9535,1022,1022,1022,2384,2384,2384,817,1225];
batch_length=[743,1907,1907,1907,205,205,205,477,477,477,163,245];
Int_Per_Fault=12*Per_Fault;
Int_Per_Fault(12)=Int_Per_Fault(12)-11;
for fault_type = 1 : length(Scenario.Name)  
    fault_name = Scenario.Name{1,fault_type};
    for idx = 1:length(Scenario.Loads{1,fault_type})
        curr_loads = Scenario.Loads{1,fault_type}{idx,1};
        next_loads = Scenario.Loads{1,fault_type}{idx,2};
        fault_pv = Scenario.Loads{1,fault_type}{idx,3};
        fault_resistance = Scenario.Rs{1,fault_type}(idx);
        fault_start = Scenario.Time{1,fault_type}(idx);
        fault_duration = Scenario.Time{2,fault_type}(idx);
        tic
        Data_out = Generate_Fault_Data(fault_name, curr_loads, next_loads, fault_pv, fault_resistance, fault_start, fault_duration);
        toc
        if fault_type==1
            idx_start=0;
        else
            idx_start=sum(Int_Per_Fault(1:fault_type-1));
        end
        scenario_idx=idx_start+idx; 

        Scenario.Output{1,scenario_idx} = {Data_out{1,1} Data_out{1,2}};
        Scenario.Class{1,scenario_idx} = Data_out{1,3};
        save('Int_Scenario.mat', '-struct', 'Scenario');
    end
end

%% Save Dataset

save('Int_Scenario.mat', '-struct', 'Scenario');