%% Initialize
clear variables
clc
% Sampling time of node metrics
Tm = 20e-3;
% Stop time of simulation
Ts = 5.005;
% Load file containing loads and PVs of a month with a time step of 1 min
% which is used a static instance of our model for each simulation
Active_Power = load('Active_Power.mat');

% Power Factors of Loads and PVs
PF_load = 0.95;
PF_pv = 1;

% Shuffle the values retaining the same instance from the file above
Active_Loads = Active_Power.arr(:, 1:48);
PV = Active_Power.arr(:, 56:60);
% Each scenario corresponds to a minute of time and we want to create 
% scenarios for every 5 seconds of the recorded loads by interpolating them
timestep = 60/5; 

curr_gran = 1:1:length(Active_Loads);
end_gran = 1:(1/timestep):length(Active_Loads);
Int_Active_Loads = zeros(length(end_gran),48);
Int_PV = zeros(length(end_gran),5);
for i=1:48
    Int_Active_Loads(:,i) = interpn(curr_gran,Active_Loads(:,i),end_gran,'makima');
end
for i=1:5
    Int_PV(:,i) = interpn(curr_gran,PV(:,i),end_gran,'makima');
end

% Check for non-positive or very low values in Int_Active_Loads and convert them to mean
for i=1:48
    for j=1:length(Int_Active_Loads)
        if j==1
            if Int_Active_Loads(j,i)<=10
                Int_Active_Loads(j,i) = mean(Int_Active_Loads(:,i));
            end
        elseif Int_Active_Loads(j,i)<=10
            Int_Active_Loads(j,i) = Int_Active_Loads(j-1,i);
        end
    end
end

% Check for non-positive or very low values in Int_PV_Loads and convert them to mean
for i=1:5
    for j=1:length(Int_PV)
        if Int_PV(j,i)<0
            Int_PV(j,i) = 0;
        end
    end
end

Int_Reactive_Loads = Int_Active_Loads * tan(acos(PF_load));

Per_Fault=[3715,9535,9535,9535,1022,1022,1022,2384,2384,2384,817,1225];
Int_Per_Fault=12*Per_Fault;
Int_Per_Fault(12)=Int_Per_Fault(12)-11;
% Specify every possible fault we are going to simulate
Scenario.Name = {'Healthy','A-G','B-G','C-G','A-B','B-C','A-C','A-B-G','B-C-G','A-C-G','A-B-C','A-B-C-G'};

Load = cell(length(Int_Active_Loads),3);
for i=1:length(Int_Active_Loads)
    if i~=length(Int_Active_Loads)
        Load_i={Int_Active_Loads(i,:),Int_Active_Loads(i+1,:),Int_PV(i,:)};
    else
        Load_i={Int_Active_Loads(i,:),Int_Active_Loads(i,:),Int_PV(i,:)};
    end
    Load(i,:)=Load_i;
end

random_idx = randperm(length(Int_Active_Loads));
Load = Load(random_idx,:);

% Fault Resistance - LogNormal Distribution Generator
% Makes 
m = 50;     % mean
v = 4000;   % variance
mu = log((m^2)/sqrt(v+m^2));
sigma = sqrt(log(v/(m^2)+1));

Rs = lognrnd(mu, sigma, length(Int_Active_Loads), 1);

% For each scenario make a cell containing Rs

% Fault Duration - Weibull Distribution Generator

a = 0.3;    % scale parameter
b = 1.2;    % shape parameter

Fault_Duration = wblrnd(a, b, length(Int_Active_Loads), 1);
for i=1:length(Int_Active_Loads)
    while Fault_Duration(i)<=20e-3
        Fault_Duration(i) = wblrnd(a, b);
    end
end
Fault_Start = Tm + rand(length(Int_Active_Loads),1).*(Ts - Tm - Fault_Duration);

% For each scenario make a cell containing Fault_Duration and Fault_Start

Scenario.Loads = cell(1,length(Scenario.Name));
Scenario.Time = cell(2,length(Scenario.Name));
Scenario.Rs = cell(1,length(Scenario.Name));
for i=1:12
    if i==1
        idx_start=1;
    else
        idx_start=sum(Int_Per_Fault(1:i-1))+1;
    end
    idx_stop=sum(Int_Per_Fault(1:i));
    Scenario.Loads(1,i)={Load(idx_start:idx_stop,:)};
    Scenario.Time(:,i)={Fault_Start(idx_start:idx_stop), Fault_Duration(idx_start:idx_stop)};
    Scenario.Rs(1,i)={Rs(idx_start:idx_stop)};
end   

Scenario.Output = cell(1,length(Int_Active_Loads));
Scenario.Class  = cell(1,length(Int_Active_Loads));
%% Save Scenario structure for exporting

save('Int_Scenario.mat', '-struct', 'Scenario');
