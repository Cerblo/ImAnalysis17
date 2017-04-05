function z2 = nn_predict(x, w_hidden, w_output, bias)

[n,~] = size(x);
a1 = x * w_hidden';
%RELU hidden layer
z1 = max(0,a1);
a2 = [z1, bias*ones(n,1)] * w_output';
%Softmax output layer
col_sum = sum(exp(a2),2);
sums = repmat(col_sum, 1, 2);
z2 = exp(a2)./sums;
