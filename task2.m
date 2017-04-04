%% Assignment #8 - Optical Flow
%% Task 2
clear
close all

I1 = double(rgb2gray(imread('frame10.png')));
I2 = double(rgb2gray(imread('frame11.png')));

% Central difference filters
% used to compute pixel derivatives in x- and y-directions
centrDiff = [-1,0,1];   % for x-direction
centrDiff_t = [-1;0;1]; % for y-direction
% Pixel difference
pixDiff = [-1,1;-1,1];
pixDiff_t = [-1,-1;1,1];

A = zeros(9,2);
b = zeros(9,1);

% Optical flow vectors for the whole image (cropping the edges)
[r,c] = size(I1);
X = zeros(r);
Y = zeros(r);
U = zeros(r);
V = zeros(r);

% Central difference
% img_dx = imfilter(I1, centrDiff, 'same', 'conv') + imfilter(I2, centrDiff, 'same', 'conv');
% img_dy = imfilter(I1, centrDiff_t, 'same', 'conv') + imfilter(I2, centrDiff_t, 'same', 'conv');

% Pixel difference
% img_dx = imfilter(I1, pixDiff, 'same', 'conv') + imfilter(I2, pixDiff, 'same', 'conv');
% img_dy = imfilter(I1, pixDiff_t, 'same', 'conv') + imfilter(I2, pixDiff_t, 'same', 'conv');

% Derivative of Gauss
Gauss = fspecial('gaussian'); % 3x3 Gaussian filter with sigma=0.5
[GaussDiff, GaussDiff_t] = gradient(Gauss);

img_dx = imfilter(I1, -GaussDiff, 'same', 'conv') + imfilter(I2, -GaussDiff, 'same', 'conv');
img_dy = imfilter(I1, -GaussDiff_t, 'same', 'conv') + imfilter(I2, -GaussDiff_t, 'same', 'conv');

for y_i = 2:r-1
    for x_i = 2:c-1
        win1 = [I1(y_i-1,x_i-1:x_i+1); I1(y_i,x_i-1:x_i+1); I1(y_i+1,x_i-1:x_i+1)];
        win2 = [I2(y_i-1,x_i-1:x_i+1); I2(y_i,x_i-1:x_i+1); I2(y_i+1,x_i-1:x_i+1)];

        win1_dx = [img_dx(y_i-1,x_i-1:x_i+1); img_dx(y_i,x_i-1:x_i+1); img_dx(y_i+1,x_i-1:x_i+1)];
        win1_dy = [img_dy(y_i-1,x_i-1:x_i+1); img_dy(y_i,x_i-1:x_i+1); img_dy(y_i+1,x_i-1:x_i+1)];
                        
        r = 1;
        
        for r_i = 1:3
            for c_i = 1:3
                A(r,1) = win1_dx(r_i,c_i);
                A(r,2) = win1_dy(r_i,c_i);
                
                b(r) = win2(r_i,c_i) - win1(r_i,c_i);
                
                r = r + 1;
            end
        end
        
        opt_flow = (A' * A)\(A' * b);
        
        X(y_i,x_i) = x_i;
        Y(y_i,x_i) = y_i;
        U(y_i,x_i) = opt_flow(1);
        V(y_i,x_i) = opt_flow(2);
    end
end

figure(3)
% imshow(rgb2gray(imread('frame10.png')))
imshow('frame10.png')
hold on
quiver(X(2:3:end,2:3:end),Y(2:3:end,2:3:end),U(2:3:end,2:3:end),V(2:3:end,2:3:end), 20, 'b');

figure(4)
% imshow(rgb2gray(imread('frame11.png')))
imshow('frame11.png')
hold on
quiver(X(2:3:end,2:3:end),Y(2:3:end,2:3:end),U(2:3:end,2:3:end),V(2:3:end,2:3:end), 20, 'b');
