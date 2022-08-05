function out = Generate_Fault_Data(fault_name, curr_loads, next_loads, fault_pv, fault_resistance, fault_start, fault_duration)
clc

%% Model to be tested

sys = 'LV_grid';
% Sampling time of node metrics
T_sample = 20e-3;
% Step time of simulation
T_step = 5e-4;
% Stop time of simulation
T_stop = 5.0005;
%Power factor of loads
PF_load = 0.95;

curr_Active_Loads = curr_loads;
next_Active_Loads = next_loads;
PV_Loads = fault_pv;

%% Fault characteristics

FaultType     = fault_name;
Rground       = fault_resistance;
TonFault      = fault_start;
ToffFault     = fault_start + fault_duration;

%% Create points where Load changes

Load_changes=randi([2, 5],1,length(curr_Active_Loads));
int_Active_Loads = zeros(length(curr_Active_Loads),5);
RL_int=zeros(length(curr_Active_Loads),6,3);
int_Reactive_Loads=zeros(length(curr_Active_Loads),5);
change_log = zeros(1,length(curr_Active_Loads));
t=T_step:T_step:T_stop;
counter=1;

V_nom = 230;
f_nom = 50;

for i=1:48
    
    Load_Times= T_step + rand(1,Load_changes(i)-1)*(T_stop-5*T_step);
    Load_Times_Rounded_i = interp1(t,t,Load_Times,'nearest','extrap');
    Load_Times_Rounded_i = sort(Load_Times_Rounded_i);
    len = length(Load_Times_Rounded_i);
    change_log(i) = len;
    if len==5
        RL_int(i,:,1)=[0 Load_Times_Rounded_i];
    else
        lst=Load_Times_Rounded_i(end)*ones(1,5-len);
        RL_int(i,:,1)=[0 Load_Times_Rounded_i lst];
    end
    
    counter = counter + Load_changes(i)-1;
    int_Active_Loads(i,1)=curr_Active_Loads(i);
    if Load_changes(i)==2
        int_Active_Loads(i,2:end)=next_Active_Loads(i);
        RL_int(i,1:2,2)=[V_nom^2./curr_Active_Loads(i) V_nom^2./next_Active_Loads(i)];
        RL_int(i,3:end,2)=V_nom^2./next_Active_Loads(i);
        Q_val1 = curr_Active_Loads(i)*tan(acos(PF_load));
        Q_val2 = next_Active_Loads(i)*tan(acos(PF_load));
        RL_int(i,1:2,3)=[V_nom^2./(2*pi*f_nom*Q_val1) V_nom^2./(2*pi*f_nom*Q_val2)];
        RL_int(i,3:end,3)=V_nom^2./(2*pi*f_nom*Q_val2);
    else
        x=1:2;
        step=1/(Load_changes(i)-1);
        v=[curr_Active_Loads(i),next_Active_Loads(i)];
        xq=1:step:2;
        P_val=interp1(x,v,xq);
        Q_val=P_val*tan(acos(PF_load));
        R_val = V_nom^2./P_val;
        L_val = V_nom^2./(2*pi*f_nom*Q_val);

        R_lst=R_val(end)*ones(1,5-len);
        L_lst=L_val(end)*ones(1,5-len);
        RL_int(i,:,2)=[R_val R_lst];
        RL_int(i,:,3)=[L_val L_lst];
        
        int_Active_Loads(i,1:length(P_val))=P_val;
        int_Active_Loads(i,(length(P_val)+1):end)=P_val(end);
    end
    int_Reactive_Loads(i,:)=int_Active_Loads(i,:) * tan(acos(PF_load));
end
RL_2 = RL_int(:,:,2);
RL_2( RL_2 < 20 ) = 20;
RL_int(:,:,2) = RL_2;
RL_3 = RL_int(:,:,3);
RL_3( RL_3 < 0.06 ) = 0.06;
RL_int(:,:,3) = RL_3;

%% Set parameters for Loads and Microgenerations

open_system(sys)
t1 = RL_int(1,1:change_log(1)+1,1);
    if length(t1) ~= length(unique(t))
        t1 = unique_val(t1, T_step);
    end
t1 = sort(t1);
r1 = RL_int(1,1:change_log(1)+1,2);LV_grid
l1 = RL_int(1,1:change_log(1)+1,3);
T1='[';
R1='[';
L1='[';
for i=1:length(t1)
    t_char=num2str(t1(i));
    r_char=num2str(r1(i));
    l_char=num2str(l1(i));
    if i~=length(t1)
        T1 = [T1 t_char ','];
        R1 = [R1 r_char ','];
        L1 = [L1 l_char ','];
    else
        T1 = [T1 t_char ']'];
        R1 = [R1 r_char ']'];
        L1 = [L1 l_char ']'];
    end
end

set_param('LV_grid/Stair Generator', 't', T1, 'e', R1, 'Ts', num2str(T_step));
set_param('LV_grid/Stair Generator48', 't', T1, 'e', L1, 'Ts', num2str(T_step));

for i=1:47
    t = RL_int(i+1,1:change_log(i+1)+1,1);
    if length(t) ~= length(unique(t))
        t = unique_val(t, T_step);
    end
    t = sort(t);
    r = RL_int(i+1,1:change_log(i+1)+1,2);
    l = RL_int(i+1,1:change_log(i+1)+1,3);

    T='[';
    R='[';
    L='[';
    for j=1:length(t)
        t_char=num2str(t(j));
        r_char=num2str(r(j));
        l_char=num2str(l(j));
        if j~=length(t)
            T = [T t_char ','];
            R = [R r_char ','];
            L = [L l_char ','];
        else
            T = [T t_char ']'];
            R = [R r_char ']'];
            L = [L l_char ']'];
        end
    end

    path_r = strcat('LV_grid/Stair Generator',int2str(i));
    path_l = strcat('LV_grid/Stair Generator',int2str(i+48));
    set_param(path_r, 't', T, 'e', R, 'Ts', num2str(T_step));
    set_param(path_l, 't', T, 'e', L, 'Ts', num2str(T_step));
end

for i=1:5
    p=PV_Loads(i);
    P_Ph = num2str(p);
    path = strcat('LV_grid/AC Voltage Source',int2str(i));
    set_param(path, 'Pref', P_Ph);
end

%% Characteristics of lines where faults can be applied (total of 38 fault locations)

%			   1	 2	   3	 4	   5	 6	   7	 8	   9
Lines.Name = {'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', ...
...            10	  11     12     13     14     15     16     17
              'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', ...
...            18	  19     20     21     22     23     24     25                
              'L18', 'L19', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25', ...
...            26	  27     28     29     30     31     32                  
              'L26', 'L27', 'L28', 'L29', 'L30', 'L31', 'L32'} ; % Line name
Lines.SName = Lines.Name;
Lines.Nphase = 3*ones(1,32); %number of phases
Lines.Nsection = [ 1 1 1 2 2 1 5 3 5 2 2 5 2 1 2 1 3 2 1 3 5 2 5 4 4 2 2 5 5 2 5 4] ; %number of sections


%% Selection of faulted lines

nLineSelect = 1:32; % run through all lines ex: [1,5]


%% Start

LineFaulted.Name     = {Lines.Name{nLineSelect}};
LineFaulted.SName    = {Lines.SName{nLineSelect}};
LineFaulted.Nphase   =  Lines.Nphase(nLineSelect);
LineFaulted.Nsection =  Lines.Nsection(nLineSelect);

iLineFaulted = randi(length(LineFaulted.Name));

%% Add Fault block in faulted line

LineName = [sys,'/',LineFaulted.Name{iLineFaulted}];
open_system(LineName) % open Line subsystem

Nphase		   = LineFaulted.Nphase(iLineFaulted);
Nsection	   = LineFaulted.Nsection(iLineFaulted);
FaultBlockName = [LineName, '/Fault'];

try
    hfb = add_block('powerlib/Elements/Three-Phase Fault',FaultBlockName);
catch ME
    % a Fault block is already present
    % delete it and replace with a new one at a different location (to avoid reconnection)
    warning(ME.identifier,'%s\n',ME.message)
    Position = get_param(FaultBlockName,'Position');
    LineHandles = get_param(FaultBlockName,'LineHandles');
    delete_line(LineHandles.LConn)
    delete_block(FaultBlockName);
    add_block('powerlib/Elements/Three-Phase Fault',FaultBlockName,'Position',Position+10);
end

%% Program fault resistance, type and timing
set_param(FaultBlockName,'GroundResistance',num2str(Rground));

if contains(FaultType,'A')
    set_param(FaultBlockName,'FaultA','on');
else
    set_param(FaultBlockName,'FaultA','off');
end

if contains(FaultType,'B')
    set_param(FaultBlockName,'FaultB','on');
else
    set_param(FaultBlockName,'FaultB','off');
end

if contains(FaultType,'C')
    set_param(FaultBlockName,'FaultC','on');
else
    set_param(FaultBlockName,'FaultC','off');
end

if contains(FaultType,'G')
    set_param(FaultBlockName,'GroundFault','on');
else
    set_param(FaultBlockName,'GroundFault','off');
end

set_param(FaultBlockName,'SwitchTimes',['[',num2str(TonFault),' ',num2str(ToffFault),']']);

%% Apply fault at line-section and terminal
iSection = randi(Nsection);
BlockName=[LineName,'/L',num2str(iSection)];
fprintf('Fault location: %s\n',BlockName);
hFault = get_param(FaultBlockName,'PortHandles');
hBlock = get_param(BlockName,'PortHandles');
hLine  = zeros(1,Nphase);
for iphase=1:Nphase
    hLine(iphase)=add_line(LineName,hFault.LConn(iphase),hBlock.RConn(iphase),'autorouting','smart');
end           

%% Simulate model 
mdl = 'LV_grid';
simOut = sim(mdl);

simOut.Vabc  = {simOut.Vabc(2:end, :)};
simOut.Iabc  = {simOut.Iabc(2:end, :)};
simOut.Gen   = {simOut.Gen(2:end, :)};
simOut.Class = [FaultType,'-',LineFaulted.Name{iLineFaulted}];

out = {simOut.Vabc simOut.Iabc simOut.Class};

%% Delete Block after simulation

LineHandles = get_param(FaultBlockName,'LineHandles');
delete_line(LineHandles.LConn)
delete_block(FaultBlockName);
bdclose(mdl);