 function [blobPts, BW] = getBlob(im, t)
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
 % compute the negative norm
normL = -t* (Lxx + Lyy);
% finds the maximum of this norm
BW = imregionalmax(normL);
[row,col] = find(BW);
blobPts = [];
for i = 1:length(row)
    blobPts = [blobPts;row(i), col(i), t, normL(row(i), col(i))];
end
 end