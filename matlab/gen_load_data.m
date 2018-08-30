close all;
clear all;
clc;
format compact;
path2tools = 'PR_Toolbox\';
addpath(path2tools);

% Load case
%define_constants;
mpc = loadcase('case30');

% Retrieve number of load buses and their indicesbus
n_loadbus = sum(mpc.bus(:,2) == 1);

loads = load_prep_data('data\nyiso_2017\*.csv', n_loadbus);

%%
loads1 = loads(1:10,:)
total1 = get_state_vars_with_load( mpc, loads1 );

%%
loads2 = loads(10001:16424,:)
total2 = get_state_vars_with_load(mpc, loads2)

%%
clear all;
close all;
clc;
load nyiso_load_statevars

%%
H = total.H(:,:,1);
R_inv = total.R_inv(:,:,1);
z = total.z(:,1);
x_rad = total.x_rad(:,1);

c = zeros(size(H,2),1);
c(1) = 2;
c(5) = 2;
a = H*c;
z_a = z+a;

x = (H'*R_inv*H)^-1 * H'*R_inv * z;
x_bad = (H'*R_inv*H)^-1 * H'*R_inv * z_a;

norm(z-H*x)
norm(z_a-H*x_bad)

%%
clear all;
close all;
clc;
load nyiso_load_statevars

timesteps = 16;
iters = floor(size(X,1) / timestep);

for iter=1:iter
   for timestep=1:timesteps
       
   end
end
for i=1:size(total.z, 1)
    if mod(i, 16) == 0
        z_a = total.z(i,:) + ((total.H(:,:,i) * (2 * ones(size(total.H(:,:,i),2),1))))';
        X(i,:) = z_a;
        Y(i,:) = 1;
    else
        X(i,:) = total.z(i,:);
        %Y(i,:) = 0;
    end
end

usable = floor(size(X,1) / step);
X = X(1:(usable*step),:);



%%
loads = loads';
%loads = loads(1:1000,:);
z_loads = zscore(loads);
z_loads(:,any(isnan(z_loads),1))=[];    % Delete NaN columns

input.x = z_loads;
input.targets = ones(size(input.x, 1), 1);
input.labels = cell(1, size(input.x, 2));
[data, coeff] = calcPrincipalComponents_matlab(input, 2);

plotData(data);
title('load zscore');

input2.x = loads;
input2.targets = ones(size(input2.x, 1), 1);
input2.labels = cell(1, size(input.x, 2));
[data2, coeff] = calcPrincipalComponents_matlab(input2, 2);

plotData(data2);
title('load');

%%
input2.x = total.z';
input2.targets = ones(size(input2.x, 1), 1);
input2.labels = cell(1, size(input.x, 2));
[data2, coeff] = calcPrincipalComponents_matlab(input2, 2);

plotData(data2);
title('measurements z')

%%
se = total.x_rad';
se(any(isnan(se),2),:)=[];

input.x = se;
input.targets = ones(size(input.x,1),1);
input2.labels = cell(1, size(input.x, 2))
[data3, coeff] = calcPrincipalComponents_matlab(input, 2);

plotData(data3);
title('state estimation x');



    





