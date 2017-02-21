clear all;
close all;

addpath sift/mex/mexw64
% Load images and plot them %
A = imread('./data/ukbench00000.jpg');
B = imread('./data/ukbench00001.jpg');
C = imread('./data/ukbench00004.jpg');

figure(1)
imagesc(A)
title('Picture A');
figure(2)
imagesc(B)
title('Picture B');
figure(3)
imagesc(C)
title('Picture C');

% Convert to singles and grayscale %
Ag = single(rgb2gray(A));
Bg = single(rgb2gray(B));
Cg = single(rgb2gray(C));

% SIFT on the images to find localizations and descriptors %
[FA,DA] = vl_sift(Ag); 
[FB,DB] = vl_sift(Bg);
[FC,DC] = vl_sift(Cg);
    % F contains  4 values: [x-coord, y-coord, scale, orientation]
    % D contains descriptors, 128 values which corresponds to:
        % 4x4 cells each with 8 gradient values => 4x48=128.

% Plot of 10 randomly selected descriptors in picture A & B%
sel = randperm(size(DA,2)); %Shuffle an index to select random descriptors
sel = sel(1:10); %Select first 10 random indexes
figure(5)
imagesc(A); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DA(:,sel), FA(:,sel))
title('10 random descriptors in picture A')

figure(6)
imagesc(B); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DB(:,sel), FB(:,sel))
title('10 random descriptors in picture B')

% L2 normalization %
DAn = double(DA)./sqrt((ones(size(DA,1),1)*sum(DA.*DA)));
DBn = double(DB)./sqrt((ones(size(DB,1),1)*sum(DB.*DB)));
DCn = double(DC)./sqrt((ones(size(DC,1),1)*sum(DC.*DC)));

% Calculating euclidian distances %
% This calculates distances between all columns (1&2), (1&3), (1&4) ...
    % And therefore if DAn is 128xN and DBn is 128xM distAB will be NxM %
distAB = pdist2(DAn', DBn');
distAC = pdist2(DAn', DCn');

% Sorted to find similarity (eta) %
% Rows are sorted from smallest to largest values %
sdAB=sort(distAB, 2);
sdAC=sort(distAC, 2);

% Calculate eta and find indexes for eta's below 0.6 %
% This is done to make sure a match is unique.
eta_AB = sdAB(:,1)./sdAB(:,2); %Dividing largest and 2nd largest value of each row
eta_AC = sdAC(:,1)./sdAC(:,2);

thresh=0.6;

% Finding the index number (in rows) where eta is below the threshold %
idx_AB=find(eta_AB<thresh); 
idx_AC=find(eta_AC<thresh);

% !Nothing in common for pictures A and C, so this comparison is dropped! %

% Finding the matching descriptors between pictures A and B %
c_AB = distAB(idx_AB, :); %Isolating the rows, where a match was found 
[c_ABmin, idxc_AB] = min(c_AB, [], 2); %Finding the corresponding column

% Matrix of indexes of all the matching descriptors %
matches_AB = [idx_AB, idxc_AB]; 
    % 'idx_AB' contains the descriptor-numbers of picture A
    % 'idxc_AB' contains the matching descriptor-numbers of picture B
    % Meaning 'idx_AB(1)' matches 'idxc_AB(1)'

% Plot the k first matches on the pictures %
k=20

figure(7)
imagesc(A); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DA(:,idx_AB(1:k)), FA(:,idx_AB(1:k)))
title(['The ', num2str(k), ' first mathing descriptors (picture A)']);

figure(8)
imagesc(B); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DB(:,idxc_AB(1:k)), FB(:,idxc_AB(1:k)))
title(['The ', num2str(k), ' first mathing descriptors (picture B)']);