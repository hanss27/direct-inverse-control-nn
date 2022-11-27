
clear;clc; close all;
% y(t) = f[y(t-1), y(t-2),..., u(t), t(t-1), u(t-2),...]
% y(k) = 1/(1+y(k-1)^2) + 250u(k) -300u(k-1);

load inverse.mat
load identification.mat

% y(k) = 0.04251*u(k-1) + 0.04044*u(k-2) + 1.778*y(k-1) - 0.8607*y(k-2)
u_sinusoid = sind(0:3009);
y_sinusoid = zeros(3010,1);

for i = 2:length(u_sinusoid)
 if i == 1 
    y_sinusoid(i)=0;
 end
y_sinusoid(i)=(1/(1+y_sinusoid(i-1)^2))+ 0.25*u_sinusoid(i)-0.3* u_sinusoid(i-1);
end

input = u_sinusoid;
output = y_sinusoid;



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

% 5 Features INVERSE, [ u(t-1), u(t-2), y(t), y(t-1), y(t-2)]
features_in = zeros(3010,5);
for i = 1 : 3007
   features_in(i,:) = [0 0 out_zscore(i+3) out_zscore(i+2) out_zscore(i+1)];
end

out_zscore_in = [0 in_zscore(2:end)];

% Splitting X_train, Y_train from dataset with ratio 75%

[row,col] = size(features_in);

X_train_in = zeros(round(row *1), col);
Y_train_in = zeros(round(row *1), 1);
k = 1;
j = 1;
for i = 1: row
    if (i <= round(row * 1) )
        X_train_in(k,:) = features_in(i,:);
        Y_train_in(k,:) = out_zscore_in(i);
        k = k + 1;
    end
end

% 5 Features IDENTIFICATION, [u(t), u(t-1), y(t-1), y(t-2), y(t-3)]
features_id = zeros(3010,5);
for i = 1 : 3007
   features_id (i,:) = [0 0 0 0 0]; 
end

% Splitting X_train, Y_train from dataset with ratio 75%
[row,col] = size(features_id );
X_train_id  = zeros(round(row *1), col);
Y_train_id  = zeros(round(row *1), 1);
k = 1;
j = 1;
for i = 1: row
    if (i <= round(row * 1) )
        X_train_id (k,:) = features_id (i,:);
        Y_train_id (k,:) = out_zscore(i);
        k = k + 1;
    end
end
% BackProp Model
n = 5; % Input Layer
p = 4; % Hidden Layer
m = 1;  % Output Layer
a = -0.5;

Error_in = zeros(length(X_train_in),1);
sum_Error_in = 0;
error_stop = 0.0001;
epoch = 1;
epoch_max = 1000;
alpha = 0.001;
miu = 0.5;

error_epoch_in = zeros(1, epoch_max);
stop_toggle = 0;
Y_train_pred_in  = zeros(length(X_train_in ),m);

error_epoch_id = zeros(1, epoch_max);
Y_train_pred_id  = zeros(length(X_train_id ),m);

while stop_toggle == 0 && epoch <= epoch_max
    for i=3:length(X_train_id )
        %INVERSE
        % Input to Hidden Layer
        z_inj = Voj_in + (X_train_in(i,:) * Vij_in) ;
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
        y_ink = Wok_in + (z_j*Wjk_in ) ;
        y_k = zeros(1,m);
        % Bipolar Sigmoid activation function
        for j = 1:m
            y_k(j) = (1- exp(-y_ink(j)))/(1+ exp(-y_ink(j)));
        end
        X_train_in(i,2) = X_train_in(i,1);
        X_train_in(i,1) = y_k;

        Y_train_pred_in(i,:) = y_k;

        % Quadratic Error
        Error_in(i) = 0.5*sum((Y_train_in(i,:)-y_k).^2);

        % Back Propagation
        % Wjk dan Wok delta update
        do_k = (Y_train_in(i,:) - y_k);
        delta_wjk_in = alpha * (z_j' * do_k) + miu * delta_wjk_old_in ;
        delta_wok_in = alpha * do_k + miu * delta_wok_old_in ;
        delta_wjk_old_in = delta_wjk_in ;
        delta_wok_old_in  = delta_wok_in ;
        
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
        
        do_j = sum(do_k * Wjk_in' .* do_j);
        delta_vij_in  = alpha *  (X_train_in(i,:)' * do_j) + miu * delta_vij_old_in ;
        delta_voj_in  = alpha * do_j + miu * delta_voj_old_in ;
        delta_vij_old_in  = delta_vij_in ;
        delta_voj_old_in  = delta_voj_in ;
        
        % Update weight
        Vij_in  = Vij_in  + delta_vij_in ;
        Voj_in  = Voj_in  + delta_voj_in ;
        Wjk_in  = Wjk_in  + delta_wjk_in ;
        Wok_in  = Wok_in  + delta_wok_in ;

        % IDENTIFICATION
        X_train_id(i,1) = X_train_in(i,1);
        X_train_id(i,2) = X_train_in(i-1,1);


        % Input to Hidden Layer
        z_inj = Voj_id + (X_train_id(i,:) * Vij_id) ;
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
        y_ink = Wok_id + (z_j*Wjk_id ) ;
        y_k = zeros(1,m);
        % Bipolar Sigmoid activation function
        for j = 1:m
            y_k(j) = (1- exp(-y_ink(j)))/(1+ exp(-y_ink(j)));
        end

        X_train_id(i,5) = X_train_id(i,4);
        X_train_id(i,4) = X_train_id(i,3);
        X_train_id(i,3) = y_k;

        Y_train_pred_id(i,:) = y_k;

        % Quadratic Error
        Error_id(i) = 0.5*sum((Y_train_id(i,:)-y_k).^2);

        % Back Propagation
        % Wjk dan Wok delta update
        do_k = (Y_train_id(i,:) - y_k);
        delta_wjk_id = alpha * (z_j' * do_k) + miu * delta_wjk_old_id ;
        delta_wok_id = alpha * do_k + miu * delta_wok_old_id ;
        delta_wjk_old_id = delta_wjk_id ;
        delta_wok_old_id  = delta_wok_id ;
        
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
        
        do_j = sum(do_k * Wjk_id' .* do_j);
        delta_vij_id  = alpha *  (X_train_id(i,:)' * do_j) + miu * delta_vij_old_id ;
        delta_voj_id  = alpha * do_j + miu * delta_voj_old_id ;
        delta_vij_old_id  = delta_vij_id ;
        delta_voj_old_id  = delta_voj_id ;
        
        % Update weight
        Vij_id  = Vij_id  + delta_vij_id ;
        Voj_id  = Voj_id  + delta_voj_id ;
        Wjk_id  = Wjk_id  + delta_wjk_id ;
        Wok_id  = Wok_id  + delta_wok_id ;

    end
    % Update the error per epoch
    error_epoch_id(epoch) = sum(Error_id)/length(X_train_id);
    if ( isnan(error_epoch_id(epoch))|| error_epoch_id(epoch) < error_stop)
        stop_toggle = 1;
    end
    epoch = epoch +1;
end
MSE_train_id = (sum((Y_train_id-Y_train_pred_id).^2))/length(Y_train_id);
fprintf("MSE Training : %f\n", MSE_train_id);
error_all_id= zeros(length(features_id),m);
Y_all_pred_id = zeros(length(out_zscore),m);
for i=1:length(features_id) 
        % Input to Hidden Layer
        z_inj = Voj_id  + (features_id(i,:) * Vij_id ) ;
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
        y_ink = Wok_id  + (z_j*Wjk_id  ) ;
        y_k = zeros(1,m);
        % Softmax activation function
        %for j = 1:m
        %    y_k(1,j) = exp(y_ink(j))/sum(exp(y_ink));
        %end
        
        % Bipolar Sigmoid activation function
        for j = 1:m
            y_k(j) = (1- exp(-y_ink(j)))/(1+ exp(-y_ink(j)));
        end
        Y_all_pred_id(i,:) = y_k;
    
    % Check the prediction by index    

    % Quadratic Error
    error_all_id(i) = 0.5*sum((out_zscore(:,i)-y_k).^2);
 
end

figure;
plot([out_zscore',Y_train_pred_id]);
legend('Target ID','Predicted ID')
title('Model ID');

figure;
hold on;
scatter(1:length(out_zscore),out_zscore');
scatter(1:length(out_zscore),Y_train_pred_id);


