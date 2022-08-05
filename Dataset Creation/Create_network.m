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

% Check for non-positive or very low values in Active_Loads and convert them to mean
for i=1:48
    for j=1:44580
        if Active_Loads(j,i)<=10
            Active_Loads(j,i) = mean(Active_Loads(:,i));
        end
    end
end

Reactive_Loads = Active_Loads * tan(acos(PF_load));
PV = Active_Power.arr(:, 56:60);
random_idx = randperm(size(Active_Loads,1));
Shuffled_Active_Loads = Active_Loads(random_idx,:);
Shuffled_PV = PV(random_idx,:);

% Specify every possible fault we are going to simulate
Scenario.Name = {'Healthy','A-G','B-G','C-G','A-B','B-C','A-C','A-B-G','B-C-G','A-C-G','A-B-C','A-B-C-G'};

% For each scenario make a cell containing loads and PVs
Load1 = {Shuffled_Active_Loads(1:3715, :), Shuffled_PV(1:3715, :)};
Load2 = {Shuffled_Active_Loads(3716:13250, :), Shuffled_PV(3716:13250, :)};
Load3 = {Shuffled_Active_Loads(13251:22785, :), Shuffled_PV(13251:22785, :)};
Load4 = {Shuffled_Active_Loads(22786:32320, :), Shuffled_PV(22786:32320, :)};
Load5 = {Shuffled_Active_Loads(32321:33342, :), Shuffled_PV(32321:33342, :)};
Load6 = {Shuffled_Active_Loads(33343:34364, :), Shuffled_PV(33343:34364, :)};
Load7 = {Shuffled_Active_Loads(34365:35386, :), Shuffled_PV(34365:35386, :)};
Load8 = {Shuffled_Active_Loads(35387:37770, :), Shuffled_PV(35387:37770, :)};
Load9 = {Shuffled_Active_Loads(37771:40154, :), Shuffled_PV(37771:40154, :)};
Load10= {Shuffled_Active_Loads(40155:42538, :), Shuffled_PV(40155:42538, :)};
Load11= {Shuffled_Active_Loads(42539:43355, :), Shuffled_PV(42539:43355, :)};
Load12= {Shuffled_Active_Loads(43356:44580, :), Shuffled_PV(43356:44580, :)};

Scenario.Loads = {Load1, Load2, Load3, Load4, Load5, Load6, Load7, Load8, Load9, Load10, Load11, Load12};

% Fault Resistance - LogNormal Distribution Generator
% Makes 
m = 50;     % mean
v = 4000;   % variance
mu = log((m^2)/sqrt(v+m^2));
sigma = sqrt(log(v/(m^2)+1));

Rs = lognrnd(mu, sigma, 44580, 1);

% For each scenario make a cell containing Rs
R1 = Rs(1:3715);
R2 = Rs(3716:13250);
R3 = Rs(13251:22785);
R4 = Rs(22786:32320);
R5 = Rs(32321:33342);
R6 = Rs(33343:34364);
R7 = Rs(34365:35386);
R8 = Rs(35387:37770);
R9 = Rs(37771:40154);
R10= Rs(40155:42538);
R11= Rs(42539:43355);
R12= Rs(43356:44580);

Scenario.Rs = {R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12};

% Fault Duration - Weibull Distribution Generator

a = 0.3;    % scale parameter
b = 1.2;    % shape parameter

Fault_Duration = wblrnd(a, b, 44580, 1);
for i=1:44580
    while Fault_Duration(i)<=20e-3
        Fault_Duration(i) = wblrnd(a, b);
    end
end
Fault_Start = Tm + rand(44580,1).*(Ts - Tm - Fault_Duration);

% For each scenario make a cell containing Fault_Duration and Fault_Start
T1 = {Fault_Start(1:3715), Fault_Duration(1:3715)};
T2 = {Fault_Start(3716:13250), Fault_Duration(3716:13250)};
T3 = {Fault_Start(13251:22785), Fault_Duration(13251:22785)};
T4 = {Fault_Start(22786:32320), Fault_Duration(22786:32320)};
T5 = {Fault_Start(32321:33342), Fault_Duration(32321:33342)};
T6 = {Fault_Start(33343:34364), Fault_Duration(33343:34364)};
T7 = {Fault_Start(34365:35386), Fault_Duration(34365:35386)};
T8 = {Fault_Start(35387:37770), Fault_Duration(35387:37770)};
T9 = {Fault_Start(37771:40154), Fault_Duration(37771:40154)};
T10= {Fault_Start(40155:42538), Fault_Duration(40155:42538)};
T11= {Fault_Start(42539:43355), Fault_Duration(42539:43355)};
T12= {Fault_Start(43356:44580), Fault_Duration(43356:44580)};

Scenario.Time = {T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12};

% We need 15 (743x108)x1000 matrices to include all 44580 scenarios
% Each matrix constists of ((4x250)=20sec) rows x ((99 V + 9 A) = 108
% features x 743 scenarios = 80244) columns
Scenario.Output = cell(4,743,15);
Scenario.Class  = cell(4,743,15);
%% Save Scenario structure for exporting

save('Int_Scenario.mat', '-struct', 'Scenario');