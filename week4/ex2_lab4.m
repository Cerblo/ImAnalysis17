%%  Build images and segmentation Q2.1
clear all; 
close all;


C = imread('circles.png'); % a built-in binary image
S_gt = double(C(1:100,1:100))+1; % ground truth two-label segmentation
S_gt(C(101:200,101:200)) = 3; % adding a third label
mu = [70,130,190]; % mean intensities of three classes



D = zeros(size(S_gt)); % data (image)
for i=1:3
D(S_gt==i) = mu(i); % clean data (three label image)
end
D = D + 20*randn(size(D)); % adding noise to data
D = min(max(round(D),0),255); % truncating to [0,255]

 figure;
hold on
for k=1:3
    plot(0:255,hist(D(S_gt==k),0:255))
end



S_t = zeros(100,100);
S_t(D < 100) = 1;
S_t(D >= 100 & D <= 160) = 2;
S_t(D > 160) = 3;

 S_m = medfilt2(S_gt,[5 5],'symmetric');
 %% Displays 
 
 figure;
 subplot(1,4,1);
 imagesc(D);
 colormap parula;
 title('Initial noisy image');
 
 subplot(1,4,2);
imagesc(S_gt);
title('S_gt');
 
 subplot(1,4,3);
 imagesc(S_t);
 colormap parula;
 title('Segmentation 1 image');
colorbar;
 
 subplot(1,4,4);
 imagesc(S_m);
 colormap parula;
 title('Segmentation 2 image');
 colorbar;
 

 
 %% Likelihood energy Q:2.2
 alpha=0.0005;

mu = [70, 130, 190];
  %S_gt
Ulh_Sgt=sum(sum(alpha*(mu(S_gt)-D).^2));

 
 %Seg1

Ulh_St=alpha*sum(sum((mu(S_t)-D).^2)); %Most probable
 
 %Seg2

Ulh_Sm=alpha*sum(sum((mu(S_m)-D).^2));
%% Finding energy by only considering prior energy Q2.3%%

%S_gt
Up_Sgt_r = nnz(diff(S_gt));
Up_Sgt_c = nnz(diff(S_gt,1,2));
Up_Sgt = Up_Sgt_r+Up_Sgt_c;

%S_t
Up_St_r = nnz(diff(S_t));
Up_St_c = nnz(diff(S_t,1,2));
Up_St = Up_St_r+Up_St_c;

%S_m
Up_Sm_r = nnz(diff(S_m));
Up_Sm_c = nnz(diff(S_m,1,2));
Up_Sm = Up_Sm_r+Up_Sm_c;


%% Total energy/posterior energy  Q2.4%%
% 
pe_S_gt = Up_Sgt+Ulh_Sgt; % Now this is most probable
pe_S_t = Up_St+Ulh_St;
pe_S_m = Up_Sm+Ulh_Sm; % This is the second most probable

%% Q2.5: Yes! Minimizing energy seems to lead to good segmentation


