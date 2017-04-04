%% Assignment #8 - Optical Flow
%% Task 1
clear
close all

I1 = double(imread('composedIm_1.png'));
I2 = double(imread('composedIm_2.png'));

% First patch centered at (2,6)
win1 = [I1(1,5:7); I1(2,5:7); I1(3,5:7)];
% Second patch centered at (5,4)
win2 = [I2(4,3:5); I2(5,3:5); I2(6,3:5)];

% Central difference filters
% used to compute pixel derivatives in x- and y-directions
centrDiff = [-1,0,1];   % for x-direction
centrDiff_t = [-1;0;1]; % for y-direction
% Pixel difference
pixDiff = [-1,1;-1,1];
pixDiff_t = [-1,-1;1,1];

% Window derivatives
win1_dx = imfilter(I1, centrDiff, 'same', 'conv') + imfilter(I2, centrDiff, 'same', 'conv');
win1_dy = imfilter(I1, centrDiff_t, 'same', 'conv') + imfilter(I2, centrDiff_t, 'same', 'conv');

A = zeros(9,2);
b = zeros(9,1);

r = 1;

for y_i = 1:3
    for x_i = 1:3
        A(r,1) = win1_dx(y_i,x_i);
        A(r,2) = win1_dy(y_i,x_i);
        
        b(r) = win2(y_i,x_i) - win1(y_i,x_i);
        
        r = r + 1;
    end
end

opt_flow = (A' * A)\(A' * b)

% Optical flow vectors for the whole image (cropping the edges)
[r,c] = size(I1);
U = zeros(r);
V = zeros(r);

% Central difference
% img_dx = imfilter(I1, centrDiff, 'same', 'conv') + imfilter(I2, centrDiff, 'same', 'conv');
% img_dy = imfilter(I1, centrDiff_t, 'same', 'conv') + imfilter(I2, centrDiff_t, 'same', 'conv');

% Pixel difference
img_dx = imfilter(I1, pixDiff, 'same', 'conv') + imfilter(I2, pixDiff, 'same', 'conv');
img_dy = imfilter(I1, pixDiff_t, 'same', 'conv') + imfilter(I2, pixDiff_t, 'same', 'conv');

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
        
        U(y_i,x_i) = opt_flow(1);
        V(y_i,x_i) = opt_flow(2);
    end
end

figure(1)
imshow('composedIm_1.png')
hold on
quiver(U,V);

figure(2)
imshow('composedIm_2.png')
hold on
quiver(U,V);