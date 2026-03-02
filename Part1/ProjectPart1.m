%% Project Part 1
clc
close all
clear

load("proj_fit_11.mat");

figure
surf(id.X{1},id.X{2},id.Y)
xlabel("x1")
ylabel("x2")
zlabel("y")
title('The given dataset:');

n1 = length(id.X{1});
n2 = length(val.X{2});

min_id = 10;
min_val = 10;

m = 35; % configurable degree
mse_id = zeros(1,m);
mse_val = zeros(1,m);

for d = 1:m 

   % exponent pairs for degree d
    exps = [];
    for total = 0:d
        for e1 = 0:total
            e2 = total - e1;
            exps = [exps; e1 e2];
        end
    end

    P = size(exps,1);
    PHI_id = zeros(n1*n1, P);
    Y_column_vector = zeros(n1*n1, 1);

    % id data set - determine theta parameters
    row = 0;
    for i = 1:n1
        for j = 1:n1
            row = row + 1;
            x1 = id.X{1}(i);
            x2 = id.X{2}(j);

            for k = 1:P
                PHI_id(row,k) = x1^exps(k,1) * x2^exps(k,2);
            end

            Y_column_vector(row) = id.Y(i,j);
        end
    end

    theta = PHI_id \ Y_column_vector;
    % N=nxn; PHI=Nxp  Y=Nx1
    
    % compute y_aprox_id 
    Y_aprox_id = PHI_id * theta;

    % compute MSE for id data set
    mse_id(d) = mean((Y_aprox_id - Y_column_vector).^2); % mean of errors corresponding to grade d

    if (mse_id(d) < min_id)
        min_id = mse_id(d);
        Y_aprox_id_optim = Y_aprox_id;
    end

    % val data set - determine phi_val and y_aprox_val
    PHI_val = zeros(n2*n2, P);
    Y_val_column_vector = zeros(n2*n2, 1);

    row = 0;
    for i = 1:n2
        for j = 1:n2
            row = row + 1;
            x1 = val.X{1}(i);
            x2 = val.X{2}(j);

            for k = 1:P
                PHI_val(row,k) = x1^exps(k,1) * x2^exps(k,2);
            end

            Y_val_column_vector(row) = val.Y(i,j);
        end
    end

    % calculate y_aprox_val (corresponding to d)
    Y_aprox_val = PHI_val * theta;

    % compute MSE for val data set
    mse_val(d) = mean((Y_aprox_val - Y_val_column_vector).^2); % mean of errors corresponding to grade: d

    if (mse_val(d) < min_val)
        min_val = mse_val(d);
        Y_aprox_val_optim = Y_aprox_val;
    end
end

% plot the errors and determine the optim grade
grad = 1:m;
figure
plot(grad,mse_id); hold on;
plot(grad,mse_val);
hold on;
title('MSE-ID & MSE-VAL & Best degree');

% extract the coordinates for minimum mse_val; plot the minimum
[ymin, xmin] = min(mse_val);
fprintf("Optimal polynomial degree: %d\n", xmin);
plot(xmin, ymin, 'gx','MarkerSize',20);
legend('Identification','Validation','Optimal degree');

% report mse for both data sets
fprintf('Mse_id= %.4f\n', min_id);
fprintf('Mse_val= %.4f\n', min_val);

% following the comparison on id & val
Y_aprox_id_optim_matrix = reshape(Y_aprox_id_optim,n1,n1);
Y_aprox_val_optim_matrix = reshape(Y_aprox_val_optim,n2,n2);

[x2id, x1id] = meshgrid(id.X{2}, id.X{1});
[x2val, x1val] = meshgrid(val.X{2}, val.X{1}); 

% identification data set - true values VS approximation
figure
subplot(221); 
surf(x1id,x2id,id.Y');
title('Identification: True values')

subplot(222); 
surf(x1id,x2id,Y_aprox_id_optim_matrix);
title('Identification: Approximation'); 

% validation data set - true values VS approximation
subplot(223);
surf(x1val,x2val,val.Y'); 
title('Validation: True values');

subplot(224);
surf(x1val, x2val, Y_aprox_val_optim_matrix); 
title('Validation: Approximation');

%plots on the same graph
% identification data set - true values VS approximation
figure
subplot(221); 
surf(x1id,x2id,id.Y', 'FaceColor', [0 0 1]);
hold on
surf(x1id,x2id,Y_aprox_id_optim_matrix, 'FaceColor', [1 0 0]);
title('ID: True(blue) VS Approximation(red)')

% validation data set - true values VS approximation
subplot(222);
surf(x1val,x2val,val.Y', 'FaceColor', [0 0 1]);
hold on
surf(x1val, x2val, Y_aprox_val_optim_matrix,'FaceColor', [1 0 0] ); 
title('VAL: True(blue) VS Approximation(red)');