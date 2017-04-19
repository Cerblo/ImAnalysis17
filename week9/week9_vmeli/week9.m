clear all
close all

% collect the data
N = 1000;
data = getDataNN(2,N,0.3,1);

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

feat=data(:,1:2);
targ= data(:,3:4);

[w_hidden, w_output, bias ] = nn_train(feat, targ, 0.01,200 )
y = nn_predict(input,w_hidden, w_output, bias);
probs = y(:,1) >= 0.5;
% 
% 
figure(2);
plot(x1(probs), x2(probs),'b.',x1(~probs), x2(~probs),'r.');
%%
idx=randperm(N);
data=data(idx,:,:,:,:);
feat=data(:,1:2);
targ= data(:,3:4);
% weigths= nn_train_flex(feat, targ, 0.0001,5000 ,4);
weigths= nn_train_flex_mb(feat, targ, 0.0001,1500 ,4,100);
zfin = nn_predict_flex(input,weigths);
figure(3);
probs = zfin(:,1) >= 0.5;
plot(x1(probs), x2(probs),'b.',x1(~probs), x2(~probs),'r.');

