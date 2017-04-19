function [w_hidden, w_output, bias ] = nn_train(x, t, lrate,nb_iter )
w_hidden = rand(2,4);
w_output = rand(4,2);
[n,~] = size(x);
bias = rand(1,2);




for iter=1:nb_iter
    a1 = x * w_hidden;
    %RELU hidden layer
    z1 = max(0,a1);
    a2 = z1 * w_output+repmat(bias,n,1);
    %Softmax output layer
    z2 = softmax_func( a2 );

    delta2 = z2 - t;
    delta1 = max(0, a1 ./abs(a1)) .* (delta2 * w_output');
    
    bias = bias - lrate * mean(delta2,1);
    
    w_hidden = w_hidden - lrate * x'*delta1;
    w_output = w_output - lrate * z1'*delta2;
end

end

