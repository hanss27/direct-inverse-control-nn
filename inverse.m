
clear;clc; close all;
% y(t) = f[y(t-1), y(t-2),..., u(t), t(t-1), u(t-2),...]
% y(k) = 1/(1+y(k-1)^2) + 250u(k) -300u(k-1);


% y(k) = 0.04251*u(k-1) + 0.04044*u(k-2) + 1.778*y(k-1) - 0.8607*y(k-2)
data = readtable('data.xlsx');
input = table2array(data(:,1));
output = table2array(data(:,2));

% Z Score data
in_zscore = zeros(1,3010);
out_zscore = zeros(1, 3010);
for i = 1 : 3010
    in_zscore(i) = (input(i) - mean(input))/std(input);
    out_zscore(i) = (output(i) - mean(output))/std(output);
end

% Min Max Data
b = out_zscore;
a = in_zscore;
b_new = zeros(1,length(b));
a_new = zeros(1, length(a));
for i = 1 : length(b)
    b_new(i) = -1 + (((b(i)-min(b))*(2))/(max(b)-min(b)));
    a_new(i) = -1 + (((a(i)-min(a))*(2))/(max(a)-min(a)));
end
out_zscore = b_new;
in_zscore = a_new;
%u(t) u(t-1) y(t-3) t(t-2)
% 5 Features, [u(t), u(t-1), y(t), y(t-2), y(t-3)]
features = zeros(3010,5);
for i = 1 : 3007
   features(i,:) = [in_zscore(i+2) in_zscore(i+1) out_zscore(i+3) out_zscore(i+2) out_zscore(i+1)];
end

out_zscore = [0 in_zscore(2:end)];

% Splitting X_train, Y_train from dataset with ratio 75%
[row,col] = size(features);
X_train = zeros(round(row *0.75), col);
Y_train = zeros(round(row *0.75), 1);
k = 1;
j = 1;
for i = 1: row
    if (i <= round(row * 0.75) )
        X_train(k,:) = features(i,:);
        Y_train(k,:) = out_zscore(i);
        k = k + 1;
    end
end

% BackProp Model
n = 5; % Input Layer
p = 4; % Hidden Layer
m = 1;  % Output Layer
a = -0.5;
V = a + rand(n+1, p);
W = a + rand(p+1, m);

% Weight Initialization
beta = 0.7 * (p)^(1/n);
Vij = rand(n,p);
norm_Vij = sqrt(sum(Vij.^2));
for j = 1 : p
    Vij(:,j) = beta*Vij(:,j)/norm_Vij(j);
end
Voj = rand(1,p) * (2.*beta) - beta;

beta = 0.7*m^(1/p) ;
Wjk = rand(p,m);
norm_Wjk = sqrt(sum(Wjk.^2));
for j = 1 : m
    Wjk(:,j) = beta*Wjk(:,j)/norm_Wjk(j);
end
Wok = rand(1,m) * (2.* beta) - beta;

Error = zeros(length(X_train),1);
sum_Error = 0;
error_stop = 0.0001;
epoch = 1;
epoch_max = 1000;
alpha = 0.001;
miu = 0.5;
delta_wjk_old = 0;
delta_wok_old = 0;
delta_vij_old = 0;
delta_voj_old = 0;
error_epoch = zeros(1, epoch_max);
stop_toggle = 0;
Y_train_pred = zeros(length(X_train),m);

while stop_toggle == 0 && epoch <= epoch_max
    for i=1:length(X_train)
        % Input to Hidden Layer
        z_inj = Voj + (X_train(i,:) * Vij) ;
        z_j = zeros(1,p);
        
        % ReLu activation function
        for j = 1:p
            if (z_inj(j) >= 0)
                z_j(j) = z_inj(j);
            else
                z_j(j) = 0;
            end
        end
        
        % Hidden Layer into Output Layer
        y_ink = Wok + (z_j*Wjk ) ;
        y_k = zeros(1,m);
        % Bipolar Sigmoid activation function
        for j = 1:m
            y_k(j) = (1- exp(-y_ink(j)))/(1+ exp(-y_ink(j)));
        end

        Y_train_pred(i,:) = y_k;

        % Quadratic Error
        Error(i) = 0.5*sum((Y_train(i,:)-y_k).^2);

        % Back Propagation
        % Wjk dan Wok delta update
        do_k = (Y_train(i,:) - y_k);
        delta_wjk = alpha * (z_j' * do_k) + miu * delta_wjk_old;
        delta_wok = alpha * do_k + miu * delta_wok_old;
        delta_wjk_old = delta_wjk;
        delta_wok_old = delta_wok;
        
        % Vij & Voj delta Update
        do_j = zeros(1,p);   
        % ReLu derivative activation function
        for j = 1:p
             if (z_inj(j) >= 0)
                do_j(1,j) = 1;
            else
                do_j(1,j) = 0;
            end
        end
        
        do_j = sum(do_k * Wjk' .* do_j);
        delta_vij = alpha *  (X_train(i,:)' * do_j) + miu * delta_vij_old;
        delta_voj = alpha * do_j + miu * delta_voj_old;
        delta_vij_old = delta_vij;
        delta_voj_old = delta_voj;
        
        % Update weight
        Vij = Vij + delta_vij;
        Voj = Voj + delta_voj;
        Wjk = Wjk + delta_wjk;
        Wok = Wok + delta_wok;
    end
    % Update the error per epoch
    error_epoch(epoch) = sum(Error)/length(X_train);
    if ( isnan(error_epoch(epoch))|| error_epoch(epoch) < error_stop)
        stop_toggle = 1;
    end
    epoch = epoch +1;
end
MSE_train = (sum((Y_train-Y_train_pred).^2))/length(Y_train);
fprintf("MSE Training : %f\n", MSE_train);
error_all= zeros(length(features),m);
Y_all_pred = zeros(length(out_zscore),m);
for i=1:length(features) 
        % Input to Hidden Layer
        z_inj = Voj + (features(i,:) * Vij) ;
        z_j = zeros(1,p);
        
        % ReLu activation function
        for j = 1:p
            if (z_inj(j) >= 0)
                z_j(1,j) = z_inj(j);
            else
                z_j(1,j) = 0;
            end
        end
        
        % Hidden Layer into Output Layer
        y_ink = Wok + (z_j*Wjk ) ;
        y_k = zeros(1,m);
        % Softmax activation function
        %for j = 1:m
        %    y_k(1,j) = exp(y_ink(j))/sum(exp(y_ink));
        %end
        
        % Bipolar Sigmoid activation function
        for j = 1:m
            y_k(j) = (1- exp(-y_ink(j)))/(1+ exp(-y_ink(j)));
        end
        Y_all_pred(i,:) = y_k;
    
    % Check the prediction by index    

    % Quadratic Error
    error_all(i) = 0.5*sum((out_zscore(:,i)-y_k).^2);
 
end

MSE_all = (sum((out_zscore'-Y_all_pred).^2))/length(out_zscore);
fprintf("MSE Testing: %f\n", MSE_all);
fprintf("Error: %f\n",error_epoch(epoch-1));

figure;
plot(error_epoch);
title('Error per Epoch');
figure;
plot(error_all);
title('Error All');
figure;
plot([Y_train,Y_train_pred]);
legend('Train Data','Train Prediction')
title('Train Model');

figure;
plot([out_zscore',Y_all_pred]);
legend('Target','Predicted')
title('Model');

figure;
hold on;
scatter(1:length(out_zscore),out_zscore');
scatter(1:length(out_zscore),Y_all_pred);
legend('Target','Predicted')
title('Scatter Model');

delta_vij_in = delta_vij;
delta_vij_old_in = delta_vij_old;
delta_voj_in = delta_voj;
delta_voj_old_in = delta_voj_old;
delta_wjk_in = delta_wjk;
delta_wjk_old_in = delta_wjk_old;
delta_wok_in = delta_wok;
delta_wok_old_in = delta_wok_old;
Vij_in = Vij;
Voj_in = Voj;
Wjk_in = Wjk;
Wok_in = Wok;
save inverse.mat delta_vij_in delta_vij_old_in delta_voj_in  delta_voj_old_in  delta_wjk_in  delta_wjk_old_in  delta_wok_in delta_wok_old_in Vij_in  Voj_in  Wjk_in  Wok_in 
