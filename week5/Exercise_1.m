%% Exercises for week 5 of Advanced Image Analysis %%
clear all
close all

%% Exercise 1 %%
%defining alpha and beta and number of nodes, N
alpha=1;
beta = 100;
N=6;

%Defining nodes and 1-clique potentials (likelihoods)
edge_t = [1 14^2 4; 2 81 49; 3 17^2 1; 4 9 19^2; 5 100 36; 6 0 16^2];
%Defining neighboring pairs and 2-clique potentials (priors)
edge_n = [1 2 beta beta; 2 3 beta beta; 3 4 beta beta; 4 5 beta beta; 5 6 beta beta];

%Doing graph cut
[Scut,flow] = GraphCutMex(N,edge_t,edge_n);
%scut = nodes belonging to sink
%flow is the cost of the cut

%with beta = 10, node 5 also belongs to the sink, because cutting between
%node 4 and four + between source and node 5 is =10+36+10 = 56, while
%cutting between node 5 and sink is 10^2 = 100.

%% Exercise 2 %%

pic1 = imread('V12_10X_x502.png');

% figure(1)
% imagesc(pic1);
% colorbar

%Making the picture a double and dividing by 2^16-1
dpic1 = double(pic1);
dpic1 = dpic1/(2^16-1);

%Plotting the picture
figure(1)
imagesc(dpic1);
colorbar
title('Original image of rat bone')

%Making and plotting histogram
hdpic1 = histcounts(dpic1, 0:0.01:1);
figure(2)
bar(0.01:0.01:1, hdpic1)
title('Histogram of rat bone picture')

%bins 0.41 and 0.72 should be class centers - .4 is sink 
mu = [0.41 0.72];

%Likelihood energies (sink and source)
Lh1 = (dpic1-mu(1)).^2;
Lh2 = (dpic1-mu(2)).^2;

dim = size(dpic1);
idxs = reshape(1:prod(dim), dim);
v_idxs = reshape(idxs, [], 1);

%defining beta and the total number of pixels, N
beta = 0.025;
N = dim(1)*dim(2);

%define edge_t with the likelihoods
edge_t = [v_idxs, reshape(Lh1, [], 1), reshape(Lh2, [], 1)];

%defining edge_n
l = [dim(1):dim(1):length(v_idxs)];
tmp = v_idxs;
tmp(l)=[];
en(:, 1) = tmp;
en(:, 2) = tmp+1;
tmp = idxs(:, 1:(end-1));
tmp = reshape(tmp, [], 1);
en2(:, 1) = tmp;
en2(:, 2) = tmp+dim(1);
en_f = [en;en2];

edge_n = [en_f(:,1), en_f(:,2), beta*ones(length(en_f),1), beta*ones(length(en_f), 1)];

%Do the graph cut
[Scut,flow] = GraphCutMex(N,edge_t,edge_n);

%Making a binary result/classification
pic_n = zeros(size(pic1));
pic_n(Scut) = 1;

%plotting it
figure(3)
imagesc(pic_n)
title('Image of rat bone 2-part segmentation')

%Histograms of air and bone
hist_bone = histcounts(dpic1(Scut), 0:0.01:1);
figure(4)
bar(0.01:0.01:1, hist_bone);
title('Histogram of bone segments')

v_idxs_2 = v_idxs;
v_idxs_2(Scut) = [];
hist_air = histcounts(dpic1(v_idxs_2), 0:0.01:1);
figure(5)
bar(0.01:0.01:1, hist_air)
title('Histogram of air segments')

%% Exercise 3 %%

% FROM LAST WEEK %%%%%%%%%%%%%%%%%%%%%%%%%%%
C = imread('circles.png'); % a built-in binary image

% Plot the image
figure(6)
imagesc(C)
title('Circle.png image')

S_gt = double(C(1:100,1:100))+1; % ground truth two-label segmentation
S_gt(C(101:200,101:200)) = 3; % adding a third label
mu = [70,130,190]; % mean intensities of three classes
D = zeros(size(S_gt)); % data (image)

for i=1:3
    D(S_gt==i) = mu(i); % clean data (three label image)
end

D = D + 20*randn(size(D)); % adding noise to data
D = min(max(round(D),0),255); % truncating to [0,255]

figure(7)
imagesc(D)
title('Circles with noise')

% Plotting! %
figure(8)
hold on

for k=1:3
    plot(0:255,hist(D(S_gt==k),0:255))
end

title('Histogram of circles-image')
hold off

% NO MORE LAST WEEK %%%%%%%%%%%%%%%%%%%%%%%
beta = 2000;
mu = [70 128 192]; %values defined from looking at the histogram plot

%finding 1-clique potentials
Lh1 = (D-mu(1)).^2;
Lh2 = (D-mu(2)).^2;
Lh3 = (D-mu(3)).^2;

Lh1 = reshape(Lh1, [], 1);
Lh2 = reshape(Lh2, [], 1);
Lh3 = reshape(Lh3, [], 1);

%U is a matrix containing 1-clique potentials for each pixel in the rows
U = [Lh1 Lh2 Lh3];

dim = size(D); %Dimensions of the image

%Doing multilabel MRF
[S, iter] = multilable_MRF(U,dim,beta);

%Plotting result and original images w.o. noise
figure(9)
imagesc(S)
title(['Multilabel MRF using beta = ', num2str(beta)])

figure(10)
imagesc(S_gt)
title('Original clean image')

%Likelihood energies
V1_S=sum(sum((mu(S)-D).^2)); 

%Prior energies
V2r = beta*nnz(diff(S_gt));
V2c = beta*nnz(diff(S_gt,1,2));
Sum1 = V2r+V2c;

%Posterior/total energies
Total_energy = Sum1+V1_S;

%% Exercise 4 %%

%Setting beta and mean values
beta=0.002; %Increasing this makes neighbors the same value
mu2 = [0.41 0.44 0.72];

%finding 1-clique potentials/likelihoods
Lh1 = (dpic1-mu2(1)).^2;
Lh2 = (dpic1-mu2(2)).^2;
Lh3 = (dpic1-mu2(3)).^2;

%Reshaping likelihoods to be vectors instead of matrices
Lh1 = reshape(Lh1, [], 1);
Lh2 = reshape(Lh2, [], 1);
Lh3 = reshape(Lh3, [], 1);

%U is a matrix containing 1-clique potentials for each pixel in the rows
U = [Lh1 Lh2 Lh3];

dim = size(dpic1); %Dimensions of the image

%Doing multilabel MRF
[S2, iter] = multilable_MRF(U,dim,beta);

%Plotting result and original images w.o. noise
figure(11)
imagesc(S2)
title({'Multilabel MRF of 1st rat picture using:'; ['beta = ', num2str(beta)]; ['And \mu = ', num2str(mu2)]})

figure(12)
imagesc(dpic1)
title('Original clean image')

%% Exercise 4 - part 2 %%
%Trying exactly same approach for a new image

pic2 = imread('V8_10X_x502.png');

% figure(1)
% imagesc(pic1);
% colorbar

%Making the picture a double and dividing by 2^16-1
dpic2 = double(pic2);
dpic2 = dpic2/(2^16-1);

%Plotting the picture
figure(13)
imagesc(dpic2);
colorbar
title('Original image of rat bone')

%Making and plotting histogram:
%WE SEE MEAN VALUES ARE CLEARLY DIFFERENT IN THIS PICTURE THAN THE 1ST
hdpic2 = histcounts(dpic2, 0:0.01:1);
figure(14)
bar(0.01:0.01:1, hdpic2)
title('Histogram of second rat bone picture')

%Setting beta and mean values
% beta=0.004; %Increasing this makes neighbors the same value
mu3 = mu2 %[0.38 0.415 0.68];
%WE CAN'T RE-USE SAME VALUES, BUT WITH NEW MEANS, ONLY SMALL TWEAKS HAS TO
%BE DONE FOR beta

%finding 1-clique potentials/likelihoods
Lh1 = (dpic2-mu3(1)).^2;
Lh2 = (dpic2-mu3(2)).^2;
Lh3 = (dpic2-mu3(3)).^2;

%Reshaping likelihoods to be vectors instead of matrices
Lh1 = reshape(Lh1, [], 1);
Lh2 = reshape(Lh2, [], 1);
Lh3 = reshape(Lh3, [], 1);

%U is a matrix containing 1-clique potentials for each pixel in the rows
U = [Lh1 Lh2 Lh3];

dim = size(dpic2); %Dimensions of the image

%Doing multilabel MRF
[S3, iter] = multilable_MRF(U,dim,beta);

%Plotting result
figure(15)
imagesc(S3)
title({'Multilabel MRF of 2nd rat picture using:'; ['beta = ', num2str(beta)]; ['And \mu = ', num2str(mu3)]})

