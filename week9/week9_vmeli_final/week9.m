clear all
close all

%% Good setting

%N = 2000;
%data = getDataNN(2,N,0.2,1);

%[w_hidden, w_output, bias ] = nn_train(X_train, y_train, 0.01,1000 );
% [error,weigths]= nn_train_flex(X_train, y_train,0.01,1000 ,4);
%weigths= nn_train_flex_mb(X_train, y_train, 0.1,1000 ,4,100);

%
%%

% collect the data
N = 2000;
data = getDataNN(2,N,0.4,1);

%assign random weights
w_hidden = rand(2,4);
w_output = rand(4,2);




% Setup the samples from a regular grid
line_x1 = -6:0.1:6;
line_x2 = -6:0.1:6;
[grid_X1,grid_X2] = meshgrid(line_x1,line_x2);
x1 = reshape(grid_X1, [length(line_x1)^2,1]);
x2 = reshape(grid_X2, [length(line_x2)^2,1]);
input = [x1 x2];
%%

bias = rand(1,2);
% Predict on the sample from the grid
y = nn_predict(input,w_hidden, w_output, bias);
probs = y(:,1) >= 0.5;
%%
% figure
% imagesc(reshape(probs, [length(line_x1),length(line_x2)]))

r=150;

idx=randperm(N);
data=data(idx,:);
X_train=data(1:r,1:2);
y_train= data(1:r,3:4);
X_test=data(r+1:N,1:2);
y_test= data(r+1:N,3:4);

[w_hidden, w_output, bias ] = nn_train(X_train, y_train, 0.03,10000 );
y = nn_predict(X_test,w_hidden, w_output, bias);
probs = y(:,1) >= 0.5;
y_est=[probs,~probs];
% 
% 
figure(2);
subplot(1,2,1)
plot(X_test(probs,1), X_test(probs,2),'b.',X_test(~probs,1), X_test(~probs,2),'r.');
title('Estimation - output of NN')

subplot(1,2,2)
probs2 = y_test(:,1) >= 0.5;
plot(X_test(probs2,1), X_test(probs2,2),'b.',X_test(~probs2,1), X_test(~probs2,2),'r.');
title('Groundtruth ')
    
%error
err=norm(y_est-y_test)/sqrt(N)*100;
%%
r=1500;

idx=randperm(N);
data=data(idx,:);
X_train=data(1:r,1:2);
y_train= data(1:r,3:4);
X_test=data(r+1:N,1:2);
y_test= data(r+1:N,3:4);
% [error,weigths]= nn_train_flex(X_train, y_train,0.03,10000 ,4);
[error,error_ep,weigths]= nn_train_flex_mb(X_train, y_train, 0.03,10000 ,[8,6,6],20);
y_e = nn_predict_flex(X_test,weigths);
probs = y_e(:,1) >= 0.5;

y_est=[probs,~probs];



figure(3);
subplot(1,2,1)
plot(X_test(probs,1), X_test(probs,2),'b.',X_test(~probs,1), X_test(~probs,2),'r.');
title('Estimation - output of NN')
subplot(1,2,2)
probs2 = y_test(:,1) >= 0.5;
plot(X_test(probs2,1), X_test(probs2,2),'b.',X_test(~probs2,1), X_test(~probs2,2),'r.');
title('Groundtruth ')

figure(4)
plot(error);
title('Error in % in function of iterations')
%error
err=norm(y_est-y_test)/sqrt(N)*100;

figure(5)
plot(error_ep);
title('Error in % in function of iterations')