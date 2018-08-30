function [ total ] = get_state_vars_with_load( mpc, loads )
%GET_STATE_VARS Summary of this function goes here
%   Detailed explanation goes here

define_constants;

% Retrieve number of load buses and their indices
n_loadbus = sum(mpc.bus(:,2) == 1);
idx_loadbus = find(mpc.bus(:,2) == 1);

% Retrieve number of gen buses and their indices
n_genbus = sum(mpc.bus(:,2) == 2);
idx_genbus = find(mpc.bus(:,2) == 2);

t0 = tic;
% Iterate over all load sets
for set=1:size(loads,1)
    % Iterate over each measurement in the load set
    for msmt=1:size(loads,2)
       mpc.bus(idx_loadbus(msmt),PD) = loads(set,msmt);
       mpc.bus(idx_loadbus(msmt),QD) = 0;
       mpc.bus(idx_loadbus(msmt),GS) = 0;
       mpc.bus(idx_loadbus(msmt),BS) = 0;
    end

    current_load = sum(mpc.bus(:,3));
    current_gen = sum(mpc.gen(:,2));

    % Calculate, what amount the gen has to be increased to match the new
    % load
    inc_per_gen = (current_load - current_gen) / n_genbus;

    % Increase the gen for each of the generators
    for msmt=2:n_genbus+1
        mpc.gen(msmt,2) = mpc.gen(msmt,2) + inc_per_gen;
    end

    mpc.baseMVA = max(mpc.bus(:,3));
    
    %fprintf('Total load: %.5f\n', current_load);
    %fprintf('Total gen: %.5f\n', current_gen);
    
    results = rundcpf(mpc, mpoption('out.all',0));
    [ x, H, R_Inv, z ] = dc_state_est(mpc, results);
    
    disp(set);
    
    total.H(:,:,set) = full(H);
    total.R_Inv(:,:,set) = R_Inv;
    total.z(set,:) = z;
    total.x_se(set,:) = x;
    total.x_pf(set,:) = (pi/180) .* results.bus(2:end,9);
    total.x_diff(set,:) = total.x_se(set,:) - total.x_pf(set,:);
end

toc(t0)

end

