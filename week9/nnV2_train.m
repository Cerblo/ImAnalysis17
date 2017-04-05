function [w_hidden, w_output, bias ] = nnV2_train(x, t, lrate,nb_iter, nb_neurons)
w_hidden = rand(nb_neurons-1,2);
w_output = rand(2,nb_neurons);
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
    delta1 = [a1 > 0, bias*ones(n,1)] .* (delta2 * w_output);

    bias = bias - lrate * sum(delta1(:,end));
    w_hidden = w_hidden - lrate * delta1(:,1:end-1)' * x;
    w_output = w_output - lrate * delta2' * [z1, bias * ones(n,1)];
end

end

