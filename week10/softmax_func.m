function z = softmax_func( a )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    col_sum = sum(exp(a),2);
    sums = repmat(col_sum, 1, size(a,2));
    z = exp(a)./sums;

end

