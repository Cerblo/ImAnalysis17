%% Store activation function

act_func= struct('h',{},'h_der',{});

% 
syms a;
% h1 = max(0,a);
% h1_der=max(0,a);
% act_func(1).h=h1;
% act_func(1).h_der=h1_der;
h1=softmax_func(a);
h1_der=diff(h1);
act_func(1).h=h1;
act_func(1).h_der=h1_der;

h2=softmax_func(a);
h2_der=diff(h2);
act_func(2).h=h2;
act_func(2).h_der=h2_der;

h3=softmax_func(a);
h3_der=diff(h3);
act_func(3).h=h3;
act_func(3).h_der=h3_der;

% vpa(subs(y,x,2))


%% Load data

clear all 
close all

load MNIST.mat

data=double(data)/255;
X_train=data(1:50000,:);
X_test=data(50001:60000,:);
y_train=label(1:50000,:);
y_test=label(50001:60000,:);
%Train
[error,err_ep,weigths]= nn_train_flex_mb( X_train, y_train, 0.08,4000,200,100);

%Predict
y_e = nn_predict_flex(X_test,weigths);
y_est=y_e>=0.5;

err=norm(y_est-y_test)/sqrt(10000)*100;

figure(1)
plot(error);
title('Error in % function of iterations')

figure(2)

plot(err_ep);
title('Error  in % function of epochs');
