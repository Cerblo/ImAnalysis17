function [vec,val] = normalized_cut_from_W(W,k)
%NORMALIZED_CUT_FROM_W   Normalized eignevectors from affinity matrix.
%
% Iput:
%   W, n-times-n affinity matrix
%   k, number of eigenvectors
% Output:
%   vec, n-times-k matrix containing k leading eigenvectors
%   val, corresponding eigenvalues
%
% Author: vand@dtu.dk

% normalization with the degree matrix
n = size(W,1); % size of the problem
Dinvsqrt = spdiags(1./sqrt(sum(W,2)+eps),0,n,n); % degree matrix ^ -1/2
W_SYM = Dinvsqrt*W*Dinvsqrt;

% solving eigenproblem of the system equivalent to normalized cuts
[z,s] = eigs(W_SYM,k,'LM'); % largest magnitude eigenvalues of W_SYM
vec = Dinvsqrt*z; % transforminmg back to solution of W_RW

% sorting eigenvalues and eigenvectors
[val,ind] = sort(1-diag(s));
vec = vec(:,ind);

% normalizing each of eigenvectors
norm_vec = (sum(vec.^2)).^0.5;
vec = n^0.5 * vec./(ones(n,1)*norm_vec);

end