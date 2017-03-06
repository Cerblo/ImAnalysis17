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
 subplot(2,2,1);
 imagesc(D);
 colormap parula;
 title('Initial noisy image');
 
 subplot(2,2,2);
imagesc(S_gt);
title('S_gt');
 
 subplot(2,2,3);
 imagesc(S_t);
 colormap parula;
 title('Segmentation 1 image');
 
 subplot(2,2,4);
 imagesc(S_m);
 colormap parula;
 title('Segmentation 2 image');

 

 
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


 %% ICM Q2.6
 S=S_t;
 
 for i=1:10
     [LE] = label_energies(S,D,mu,alpha);
     [M_S,S]=min(LE,[],3);
   
 end
 

%% Display 
 figure;
 subplot(1,2,1);
 imagesc(S);
 title('Segmentation after ICM Parralel 10 iterations');

  [LE] = label_energies(S,D,mu,alpha);
 [M_S,S]=min(LE,[],3);
 subplot(1,2,2);
 imagesc(S);
 title('Segmentation after ICM Parralel 11 iterations');
 
  %% ICM Q2.7

 
 
 C_b = checkerboard(1,size(S_t,1)/2);
 C_b=(C_b>0.5);
 S_even=S_t;
 S_odd=S_t;
 for i=1:10
     
     [LE_even] = label_energies(S_odd,D,mu,alpha);
     [M_even,S_even1]=min(LE_even,[],3);
     S_even2=C_b.*S_even1;
     S_even=(1-C_b).*S_odd+S_even2;
     
     [LE_odd] = label_energies(S_even,D,mu,alpha);
     [M_odd,S_odd1]=min(LE_odd,[],3);
     S_odd2=(1-C_b).*S_odd1;
     S_odd=C_b.*S_even+S_odd2;
 end
 
S=S_even2+S_odd2;
  figure;
  subplot(1,2,1)
 imagesc(S);
 title('Segmentation after ICM Parallel 10 iterations');
 
[LE_even] = label_energies(S_odd,D,mu,alpha);
[M_even,S_even1]=min(LE_even,[],3);
S_even2=C_b.*S_even1;
S_even=(1-C_b).*S_odd+S_even2;

[LE_odd] = label_energies(S_even,D,mu,alpha);
[M_odd,S_odd1]=min(LE_odd,[],3);
S_odd2=(1-C_b).*S_odd1;
S_odd=C_b.*S_even+S_odd2;
 
  subplot(1,2,2)
 imagesc(S);
 title('Segmentation after ICM Parallel 11 iterations');
 %% Questions 2.8 and 2.9
 
PE_ICM=sum(sum(M_S));
PE_ICM2=sum(sum((1-C_b).*M_odd+C_b.*M_even));

%% Other initializations

S_rand = randi(3,size(S_t,1),size(S_t,2));
S=S_rand;

C_b = checkerboard(1,size(S,1)/2);
 C_b=(C_b>0.5);
 S_even=S;
 S_odd=S;
 for i=1:10
     
     [LE_even] = label_energies(S_odd,D,mu,alpha);
     [M_even,S_even1]=min(LE_even,[],3);
     S_even2=C_b.*S_even1;
     S_even=(1-C_b).*S_odd+S_even2;
     
     [LE_odd] = label_energies(S_even,D,mu,alpha);
     [M_odd,S_odd1]=min(LE_odd,[],3);
     S_odd2=(1-C_b).*S_odd1;
     S_odd=C_b.*S_even+S_odd2;
 end
 
S=S_even2+S_odd2;
  figure;
  subplot(1,2,1)
 imagesc(S);
 title('Random init Seg ICM Parralel 10 iterations');
 

 [LE_even] = label_energies(S_odd,D,mu,alpha);
 [M_even,S_even1]=min(LE_even,[],3);
 S_even2=C_b.*S_even1;
 S_even=(1-C_b).*S_odd+S_even2;

 [LE_odd] = label_energies(S_even,D,mu,alpha);
 [M_odd,S_odd1]=min(LE_odd,[],3);
 S_odd2=(1-C_b).*S_odd1;
 S_odd=C_b.*S_even+S_odd2;
     

  subplot(1,2,2)
 imagesc(S);
 title('Random init Seg ICM Parralel 11 iterations');
     
 PE_ICM3=sum(sum((1-C_b).*M_odd+C_b.*M_even));
 
 %% Changing the weigth (alpha=0.005)
 
 
alpha=0.005;
S_rand = randi(3,size(S_t,1),size(S_t,2));
S=S_rand;

C_b = checkerboard(1,size(S,1)/2);
 C_b=(C_b>0.5);
 S_even=S;
 S_odd=S;
 for i=1:10
     
     [LE_even] = label_energies(S_odd,D,mu,alpha);
     [M_even,S_even1]=min(LE_even,[],3);
     S_even2=C_b.*S_even1;
     S_even=(1-C_b).*S_odd+S_even2;
     
     [LE_odd] = label_energies(S_even,D,mu,alpha);
     [M_odd,S_odd1]=min(LE_odd,[],3);
     S_odd2=(1-C_b).*S_odd1;
     S_odd=C_b.*S_even+S_odd2;
 end
 
S=S_even2+S_odd2;
  figure;
  subplot(1,2,1)
 imagesc(S);
 title('Random init + alpha=0.005 Seg after ICM Parralel 10 iterations');
 
      
     [LE_even] = label_energies(S_odd,D,mu,alpha);
     [M_even,S_even1]=min(LE_even,[],3);
     S_even2=C_b.*S_even1;
     S_even=(1-C_b).*S_odd+S_even2;
     
     [LE_odd] = label_energies(S_even,D,mu,alpha);
     [M_odd,S_odd1]=min(LE_odd,[],3);
     S_odd2=(1-C_b).*S_odd1;
     S_odd=C_b.*S_even+S_odd2;
     
    subplot(1,2,2)
 imagesc(S);
 title('Random init + alpha=0.005 Seg after ICM Parralel 11 iterations');   
 
 %% Changing the weigth (alpha=0.00005)
 
alpha=0.00005;
S_rand = randi(3,size(S_t,1),size(S_t,2));
S=S_rand;

C_b = checkerboard(1,size(S,1)/2);
 C_b=(C_b>0.5);
 S_even=S;
 S_odd=S;
 for i=1:10
     
     [LE_even] = label_energies(S_odd,D,mu,alpha);
     [M_even,S_even1]=min(LE_even,[],3);
     S_even2=C_b.*S_even1;
     S_even=(1-C_b).*S_odd+S_even2;
     
     [LE_odd] = label_energies(S_even,D,mu,alpha);
     [M_odd,S_odd1]=min(LE_odd,[],3);
     S_odd2=(1-C_b).*S_odd1;
     S_odd=C_b.*S_even+S_odd2;
 end
 
S=S_even2+S_odd2;
  figure;
  subplot(1,2,1)
 imagesc(S);
 title('Random init + alpha=0.00005 Seg after ICM Parralel 10 iterations');
 
      [LE_even] = label_energies(S_odd,D,mu,alpha);
     [M_even,S_even1]=min(LE_even,[],3);
     S_even2=C_b.*S_even1;
     S_even=(1-C_b).*S_odd+S_even2;
     
     [LE_odd] = label_energies(S_even,D,mu,alpha);
     [M_odd,S_odd1]=min(LE_odd,[],3);
     S_odd2=(1-C_b).*S_odd1;
     S_odd=C_b.*S_even+S_odd2;
     
       subplot(1,2,2)
 imagesc(S);
 title('Random init + alpha=0.00005 Seg after ICM Parralel 11 iterations');