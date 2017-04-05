clearvars
close all
figure
imshow('frame10.png')
figure
imshow('frame11.png')

I1 = double(rgb2gray(imread('frame10.png')));
I2 = double(rgb2gray(imread('frame11.png')));

% Central difference filters
% used to compute pixel derivatives in x- and y-directions
centrDiff = [-1,0,1];   % for x-direction
centrDiff_t = [-1;0;1]; % for y-direction
% Pixel difference
pixDiff = [-1,1;-1,1];
pixDiff_t = [-1,-1;1,1];

% Central difference
img_dx = imfilter(I1, centrDiff, 'same', 'conv') + imfilter(I2, centrDiff, 'same', 'conv');
img_dy = imfilter(I1, centrDiff_t, 'same', 'conv') + imfilter(I2, centrDiff_t, 'same', 'conv');

% Pixel difference
% img_dx = imfilter(I1, pixDiff, 'same', 'conv') + imfilter(I2, pixDiff, 'same', 'conv');
% img_dy = imfilter(I1, pixDiff_t, 'same', 'conv') + imfilter(I2, pixDiff_t, 'same', 'conv');

% Derivative of Gauss
% Gauss = fspecial('gaussian'); % 3x3 Gaussian filter with sigma=0.5
% [GaussDiff, GaussDiff_t] = gradient(Gauss);
% img_dx = imfilter(I1, -GaussDiff, 'same', 'conv') + imfilter(I2, -GaussDiff, 'same', 'conv');
% img_dy = imfilter(I1, -GaussDiff_t, 'same', 'conv') + imfilter(I2, -GaussDiff_t, 'same', 'conv');


It = double(I2 - I1);

u = zeros(size(I1));
v = zeros(size(I1));
alpha = 100;

alpha_2 = alpha^2 * ones(size(I1));

sigma = 0.5;
for k = 1:100
    u_avg = imgaussfilt(u, sigma);
    v_avg = imgaussfilt(v, sigma);
    u = u_avg - img_dx .* (img_dx .* u_avg + img_dy .* v_avg + It) ./ (alpha_2 + img_dx.^2 + img_dy.^2);
    v = v_avg - img_dy .* (img_dx .* u_avg + img_dy .* v_avg + It) ./ (alpha_2 + img_dx.^2 + img_dy.^2);
end

line_x1 = 1:640;
line_x2 = 1:480;
[grid_X1,grid_X2] = meshgrid(line_x1,line_x2);

figure
imshow('frame10.png')
hold on
quiver(grid_X1, grid_X2, u, v, 20, 'b')
