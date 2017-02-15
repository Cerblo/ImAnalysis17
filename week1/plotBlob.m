function plotBlob(im, blobMat, blobThres)
% Function for plotting scale-space blobs
% 
%   function plotBlob(im, blobMat, blobThres)
% 
% Input
%   im - image where blobs are found
%   blobMat - matrix with cloumns: [row, col, scale, blobResponse]
%   blobThres - optional threshold parameter that will not plot blobs with
%       response below blobThres
% 
% Anders Lindbjerg Dahl - February 2012
% 

if ( nargin == 2 )
    blobThres = -inf;
end

cVal = (2*pi*(1:100)/99)';
uCirc = [cos(cVal), sin(cVal)];

figure
imagesc(im)
colormap gray
hold on
axis equal

for i = 1:size(blobMat,1)
    if ( blobMat(i,4) > blobThres )
        plot(uCirc(:,1)*sqrt(blobMat(i,3))*2 + blobMat(i,2), ...
            uCirc(:,2)*sqrt(blobMat(i,3))*2 + blobMat(i,1), 'r', 'LineWidth', 1.5);
    end
end








