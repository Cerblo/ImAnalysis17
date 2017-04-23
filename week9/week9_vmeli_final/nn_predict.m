function z2 = nn_predict(x, w_hidden, w_output, bias)

[n,~]=size(x);
a1 = x * w_hidden;
%RELU hidden layer
z1 = max(0,a1);
a2 = z1 * w_output+repmat(bias,n,1);
%Softmax output layer
z2 = softmax_func( a2 );
