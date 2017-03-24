function seg = kmeans_discretize(vec)
%KMEANS_DISCRETIZE   k-segmentation from k-eigenvectors by k-means.
%
% Iput:
%   vec, n-times-k matrix containing k eigenvectors
% Output:
%   seg, length n vector containing segmentation index form 1 to k
% Author: vand@dtu.dk

k = size(vec,2); % as many clusters as eigenvectors

% normalizing for each vertex
vm = (sum(vec.^2,2)).^0.5;
vec = vec./(vm*ones(1,k));

% k means
seg = kmeans(vec,k);

end