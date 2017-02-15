 function [final_blobPts] = getBlob2(im, nScales)
 blobPts = [];
 range = 10:floor((200 - 10)/nScales):200;
 for t=range
     
 % choose a scale and a support interval
 x = -round(5 * sqrt(t)):round(5 * sqrt(t));
 % Gaussian filter
 g = 1/(sqrt(2*pi*t)).*exp(-(x.*x)/(2*t));
 % Gaussian derivative filter
 dg = -x./(t*sqrt(2*pi*t)).*exp(-(x.*x)/(2*t));
 % Gaussian second derivative filter
 ddg = (x - sqrt(t)) .* (x + sqrt(t)) / sqrt(2 * pi * t^5) .* exp(-x.*x/(2*t));
 % convolve the image in the x-direction
 Lxx = filter2(ddg, im);
 Lxx = filter2(g', Lxx);
 % convolve the image in the y-direction
 Lyy = filter2(ddg',im);
 Lyy = filter2(g, Lyy);
normL = - t* (Lxx + Lyy);
BW = imregionalmax(normL, 26);
[row,col] = find(BW);

for i = 1:length(row)
    blobPts = [blobPts;row(i), col(i), t, normL(row(i), col(i))];
end
 end
 max_grad = max(blobPts(:,4));
 index = find(blobPts(:,4)==max_grad);
 auto_scale = blobPts(index,3);
 final_blobPts = blobPts(blobPts(:,3)==auto_scale,:);
 end