%% Project Part 2
clc
close all
clear

load("iddata-13.mat");

u_id = id.InputData{1};
y_id = id.OutputData{1};

u_val = val.InputData{1};
y_val = val.OutputData{1};

N_id = length(y_id);
N_val = length(y_val);

Ts = id.Ts{1};

figure
subplot(221); plot(u_id); title("Identification input:");
xlabel("k"); ylabel("u\_id(k)"); grid on;
subplot(222); plot(y_id); title("Identification output:");
xlabel("k"); ylabel("y\_id(k)"); grid on;
subplot(223); plot(u_val); title("Validation input:");
xlabel("k"); ylabel("u\_val(k)"); grid on;
subplot(224); plot(y_val); title("Validation output:");
xlabel("k"); ylabel("y\_val(k)"); grid on;

nk = 1; % time delay
n = 3; % maximum order for na=nb
m = 10; % maximum polynomial degree

mse_min = Inf;
na_best = 1;
deg_best = 1;
y_pred_id_best = [];
y_pred_val_best = [];
y_sim_id_best = [];
y_sim_val_best = [];
y_id_best = [];
y_val_best = [];

mse_pred_id = zeros(n,m);
mse_pred_val = zeros(n,m);
mse_sim_id = zeros(n,m);
mse_sim_val = zeros(n,m);

for na = 1:n
    nb = na;

    for deg = 1:m

        % % % IDENTIFICATION DATASET % % %
        PHI_id = [];

        % PREDICTION + compute theta

        for k = 1:N_id
            d_id_pred = []; %vector of delayed outputs and inputs
            for i = 1:na
                if (k-i > 0)
                    d_id_pred = [d_id_pred y_id(k-i)];
                else
                    d_id_pred = [d_id_pred 0];
                end
            end
            for j = 1:nb
                if (k-nk-j+1 > 0)
                    d_id_pred = [d_id_pred u_id(k-nk-j+1)];
                else
                    d_id_pred = [d_id_pred 0];
                end
            end

            phi_id_pred = computePhi(d_id_pred, deg);
            PHI_id = [PHI_id; phi_id_pred'];
        end

        theta = PHI_id \ y_id;
        y_pred_id = PHI_id * theta; % output prediction
        
        % MSE for prediction - id 
        err_pred_id = y_pred_id - y_id;
        mse_pred_id(na,deg) = sum(err_pred_id.^2) / N_id;

        % SIMULATION 

        y_sim_id = zeros(N_id,1);

        for k = 1:N_id
            d_id_sim = [];
            for i = 1:na
                if (k-i > 0)
                    d_id_sim = [d_id_sim y_sim_id(k-i)];
                else
                    d_id_sim = [d_id_sim 0];
                end
            end
            for j = 1:nb
                if (k-nk-j+1 > 0)
                    d_id_sim = [d_id_sim u_id(k-nk-j+1)];
                else
                    d_id_sim = [d_id_sim 0];
                end
            end

            phi_id_sim = computePhi(d_id_sim, deg);
            y_sim_id(k) = phi_id_sim' * theta; % output simulation
        end

        % MSE for simulation - id
        err_sim_id = y_sim_id - y_id;
        mse_sim_id(na, deg) = sum(err_sim_id.^2) / N_id;

        % % % VALIDATION DATASET % % %

        PHI_val = [];

        % PREDICTION

        for k = 1:N_val
            d_val_pred = [];
            for i = 1:na
                if (k-i > 0)
                    d_val_pred = [d_val_pred y_val(k-i)];
                else
                    d_val_pred = [d_val_pred 0];
                end
            end
            for j = 1:nb
                if (k-nk-j+1 > 0)
                    d_val_pred = [d_val_pred u_val(k-nk-j+1)];
                else
                    d_val_pred = [d_val_pred 0];
                end
            end
            
            phi_val_pred = computePhi(d_val_pred, deg);
            
            PHI_val = [PHI_val; phi_val_pred'];
        end
        y_pred_val = PHI_val * theta; % output prediction

        % MSE for prediction - val
        err_pred_val = y_pred_val - y_val;
        mse_pred_val(na,deg) = sum(err_pred_val.^2) / N_val;

        % SIMULATION

        y_sim_val = zeros(N_val,1);

        for k = 1:N_val
            d_val_sim = [];
            for i = 1:na
                if (k-i > 0)
                    d_val_sim = [d_val_sim y_sim_val(k-i)];
                else
                    d_val_sim = [d_val_sim 0];
                end
            end
            for j = 1:nb
                if (k-nk-j+1 > 0)
                    d_val_sim = [d_val_sim u_val(k-nk-j+1)];
                else
                    d_val_sim = [d_val_sim 0];
                end
            end
        
            phi_val_sim = computePhi(d_val_sim, deg);
            y_sim_val(k) = phi_val_sim' * theta; % output simulation
        end

        % MSE for simulation - val
        err_sim_val = y_sim_val-y_val;
        mse_sim_val(na, deg) = sum(err_sim_val.^2) / N_val;

        % best model

        if(mse_sim_val(na,deg) < mse_min)
            mse_min = mse_sim_val(na,deg);
            na_best = na;
            deg_best = deg;
            y_pred_id_best = y_pred_id;
            y_pred_val_best = y_pred_val;
            y_sim_id_best = y_sim_id;
            y_sim_val_best = y_sim_val;
            y_id_best = y_id;
            y_val_best = y_val;

        end
     end
end

fprintf("na_best = %d\n", na_best);
fprintf("deg_best = %d\n", deg_best);

% plot MSE for optimal degree (na,deg)
degNames = arrayfun(@(d)sprintf('deg=%d',d), 1:m, "UniformOutput",false);
naNames = arrayfun(@(a)sprintf('na=%d',a), 1:n, "UniformOutput",false);
valTableStr = arrayfun(@(x)sprintf('%.6g',x), mse_sim_val, "UniformOutput",false);

[valMin,idx] = min(mse_sim_val(:));
[rowMin,colMin] = ind2sub(size(mse_sim_val),idx);
S = uistyle("BackgroundColor","green");
sLeft = uistyle("HorizontalAlignment","Left");

f = figure;

% ID
uicontrol(f,"Style","text","String","mse_sim_id", ...
    "Units","normalized","Position",[0.00 0.95 1.00 0.05], ...
    "FontSize",12,"FontWeight","bold");
tid = uitable(f,"Data",mse_sim_id,"ColumnName",degNames,"RowName",naNames, ...
    "Units","normalized","Position",[0.00 0.50 1.00 0.45]);
addStyle(tid,sLeft,"column",1:m);

% VAL
uicontrol(f,"Style","text","String","mse_sim_val", ...
    "Units","normalized","Position",[0.00 0.45 1.00 0.05], ...
    "FontSize",12,"FontWeight","bold");
tval = uitable(f,"Data",mse_sim_val,"ColumnName",degNames,"RowName",naNames, ...
    "Units","normalized","Position",[0.00 0.00 1.00 0.45]);
addStyle(tval,S,"cell",[rowMin,colMin]);
addStyle(tval,sLeft,"column",1:m);

% Id dataset - approximated model vs real outputs - pred
figure
subplot(211)
plot(1:N_id,y_id_best, "Color","b","LineWidth",1.5)
hold on
plot(1:N_id,y_pred_id_best,"Color","r","LineWidth",1.5)
title("Id - prediction vs real output")
xlabel("k"); ylabel("Output"); legend("Real output","Prediction")
grid on

% Id dataset - approximated model vs real outputs - sim
subplot(212)
plot(1:N_id,y_id_best, "Color","b","LineWidth",1.5)
hold on
plot(1:N_id,y_sim_id_best,"Color","r","LineWidth",1.5)
title("Id - simulation vs real output")
xlabel("k"); ylabel("Output"); legend("Real output","Simulation")
grid on

% Val dataset - approximated model vs real outputs - pred
figure
subplot(211)
plot(1:N_val,y_val_best, "Color","b","LineWidth",1.5)
hold on
plot(1:N_val,y_pred_val_best,"Color","r","LineWidth",1.5)
title("Val - prediction vs real output")
xlabel("k"); ylabel("Output"); legend("Real output","Prediction")
grid on

% Val dataset - approximated model vs real outputs - sim
subplot(212)
plot(1:N_val,y_val_best, "Color","b","LineWidth",1.5)
hold on
plot(1:N_val,y_sim_val_best,"Color","r","LineWidth",1.5)
title("Val - simulation vs real output")
xlabel("k"); ylabel("Output"); legend("Real output","Simulation")
grid on

% using compare for linear arx
id = iddata(y_id,u_id,Ts);
val = iddata(y_val,u_val,Ts);
model = arx(id,[na_best na_best nk]);

figure
subplot(211)
compare(id,model,1); % prediction
xlabel("k"); ylabel("Output"); grid on;
subplot(212)
compare(val,model,1); % prediction
xlabel("k"); ylabel("Output"); grid on;

modelOE = oe(id,[na_best na_best nk]); % for simulation

figure
subplot(211)
compare(id,modelOE,Inf);
xlabel("k"); ylabel("Output"); grid on;
subplot(212)
compare(val,modelOE,Inf);
xlabel("k"); ylabel("Output"); grid on;

% plot errors
err_pred_id_best = y_pred_id_best - y_id_best;
err_sim_id_best = y_sim_id_best - y_id_best;

err_pred_val_best = y_pred_val_best - y_val_best;
err_sim_val_best = y_sim_val_best - y_val_best;

figure
subplot(211)
plot(1:N_id, err_pred_id_best,"LineWidth",1.5)
xlabel("k"); ylabel("Error:"); grid on;
legend("err\_pred\_id\_best");
title("Prediction errors - ID dataset (best model)")

subplot(212)
plot(1:N_id, err_sim_id_best,"LineWidth",1.5)
xlabel("k"); ylabel("Error:"); grid on;
legend("err\_sim\_id\_best")
title("Simulation errors - ID dataset (best model)")

figure
subplot(211)
plot(1:N_val, err_pred_val_best,"LineWidth",1.5)
xlabel("k"); ylabel("Error:"); grid on;
legend("err\_pred\_val\_best");
title("Prediction errors - VAL dataset (best model)")

subplot(212)
plot(1:N_val, err_sim_val_best,"LineWidth",1.5)
xlabel("k"); ylabel("Error:"); grid on;
legend("err\_sim\_val\_best")
title("Simulation errors - VAL dataset (best model)")

function phi = computePhi(d,m)
    Nd = length(d);
    phi = 1; % degree 0

    for p = 1:m
        idx = ones(1,p);
        while true
            % add the monomial d(idx(1))*d(idx(2)) *...* d(idx(p))
            phi = [phi; prod(d(idx))];
    
            % move to the next nondecreasing index tuple
            i = p;
            while i >= 1 && idx(i) == Nd
                i = i - 1;
            end
    
            if (i == 0)
                break; % finished all combinations for this degree p
            end
    
            idx(i) = idx(i) + 1;
            idx(i+1:end) = idx(i); % keep it nondecreasing (avoids duplicates)
        end
    end
end