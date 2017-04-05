clearvars
%assign random weights
w_hidden = rand(4,2);
w_output = rand(2,5);
bias = rand();

% collect the data
N = 100;
data = getDataNN(2,N,0.3,1);

% Setup the samples from a regular grid
line_x1 = -6:0.1:6;
line_x2 = -6:0.1:6;
[grid_X1,grid_X2] = meshgrid(line_x1,line_x2);
x1 = reshape(grid_X1, [length(line_x1)^2,1]);
x2 = reshape(grid_X2, [length(line_x2)^2,1]);
input = [x1 x2];

% Predict on the sample from the grid
y = nn_predict(input,w_hidden, w_output, bias);
probs = y(:,1) >= 0.5;

% figure
% imagesc(reshape(probs, [length(line_x1),length(line_x2)]))

% [w_hidden, w_output, bias ] = nn_train(data(:,1:2), data(:,3:4), 0.01,1000 )
% y = nn_predict(input,w_hidden, w_output, bias);
% probs = y(:,1) >= 0.5;


[w_hidden, w_output, bias ] = nnV2_train(data(:,1:2), data(:,3:4), 0.01,1000, 30 )
y = nn_predict(input,w_hidden, w_output, bias);
probs = y(:,1) >= 0.5;

w = nnV3_train(data(:,1:2), data(:,3:4), 0.01, 1000, [5,4,3])
plot(x1(probs), x2(probs),'b.')
plot(x1(~probs), x2(~probs),'r.')

