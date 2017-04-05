function [w_hidden, w_output, bias ] = nn_train(x, t, lrate,nb_iter )
w_hidden = rand(4,2);
w_output = rand(2,5);
bias = rand();
[n,~] = size(x);

for iter=1:nb_iter
    a1 = x * w_hidden';
    %RELU hidden layer
    z1 = max(0,a1);
    a2 = [z1, bias*ones(n,1)] * w_output';
    %Softmax output layer
    col_sum = sum(exp(a2),2);
    sums = repmat(col_sum, 1, 2);
    z2 = exp(a2)./sums;

    delta2 = z2 - t;
    %delta1 = [z1, bias*ones(n,1)].* (delta2 * w_output);
    delta1 = [max(0, a1 ./abs(a1)), bias*ones(n,1)] .* (delta2 * w_output);

    bias = bias - lrate * sum(delta1(:,5));
    w_hidden = w_hidden - lrate * delta1(:,1:4)' * x;
    w_output = w_output - lrate * delta2' * [z1, bias * ones(n,1)];
end

end

