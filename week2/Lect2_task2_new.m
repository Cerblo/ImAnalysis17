clear all
close all

%% Selecting 100 random descriptors from training images
% And storing it in matrix 'TrainD'

nti = 150; %number of training images
sel_k = 100; %the number of descriptors from each image
TrainD = zeros(128,nti*sel_k); % ALLOCATING SPACE FOR ALL DESCRIPTORS

g = 1; %Counter for skipping every 4th image
d=0; %Counter for saving in correct indexes of TrainD

for i=0:199
    if g<4
        DaF = load(['./data/output/result_', sprintf('%05d', i), '.mat']);
        sel = randperm(size(DaF.D,2)); %Shuffle the descriptors
        sel = sel(1:sel_k); %Select the sel_k first descriptors (here 100)
        
        % Set indexes for storage
        d=d+1; % Count!
                
        c1 = (d*100)-99;
        c2 = d*100;
        
        %Storing training descriptors
        TrainD(:, c1:c2) = DaF.D(:, sel); 
                
        g=g+1; % Count!
    else
        g=1; % Restart count!
    end
end

% Save dat sjit!
save('./data/output/TrainD.mat', 'TrainD')

%% K-means clustering (euclidian distance - NOT Mahalanobis) %%
n_bins = 250; % Number of 'visual words'
[idx1, C] = kmeans(TrainD', n_bins); % Use 'kmeans' to find the 250 centers

%% Comparing all descriptors from all images to the visual vocabulary, 'C' %%
h=zeros(200, n_bins);
g = 1; %Counter for skipping every 4th image
d=0; %Counter for saving in correct indexes of Xtrain
e=0;  %Counter for saving in correct indexes of Xtest

for i=0:199    
    DaF = load(['./data/output/result_', sprintf('%05d', i), '.mat']);
    dists = pdist2(DaF.D', C);      % Computes the distance from all descriptors to all C's 
    [val, idx] = min(dists, [], 2); % Classify descriptors by finding the min. distances
    h1 = histcounts(idx, n_bins);   % Make a histogram for each picture
    h((i+1), :) = h1/length(dists); % normalize it by the number of descriptors
end

%% Defining Xtrain and Xtest %%
idxTrain = 1:200;
idxTest = 4:4:200;
idxTrain(idxTest) = []; % Make idxTrain a vector of 1:200 that skips every 4th img

Xtrain = h(idxTrain, :);
Xtest = h(idxTest, :);

%% Defining ytrain and ytest

ytrain=zeros(150,1);
ytest=[1:50]';

for i=1:50
    limLow=3*i-2;
    limHigh=3*i;
    ytrain(limLow:limHigh,1)=i*ones(3,1);
end

%% Computing distances between picture histograms to find which pictures match %%

%Computing cityblock distances and plotting them
dist_sum = pdist2(h, h, 'CityBlock');

figure()
imagesc(dist_sum)

%Find the 3 smallest distances between test and train histograms 
[min_val, idx_min] = pdist2(Xtrain, Xtest, 'euclidean', 'Smallest', 3);
classes_dist = ceil(idx_min/3); % Simple classification

% Classification using K-nearest neighbor:
% Just select the nearest class, and don't consider 2nd or 3rd nearest
class_nn = classes_dist(1,:);

%% Classification with LDA %%

class = classify(Xtest, Xtrain, ytrain,'diagLinear');

%% PCA on the data %%

% get mean
ncomp=50;
X = h';
m = mean(X,2); m = m(:);

% compute covariance matrix C
C = zeros(size(X,1), size(X,1));
for i=1:size(X,2)
     C = (X(:,i)-m(:))*(X(:,i)-m(:))'+C;
end

% compute eigenvectors and eigenvalues
[eVec_tr, eVal_tr] = eig(C);
eVec_tr = fliplr(eVec_tr); % flip to ensure largest first
eVec_tr = eVec_tr(:,1:ncomp); % ncomp: The number of components retained

% Apply transformation on training
pcScore_tr = eVec_tr'*((X - repmat(m, [1 size(X,2)])));

% Transpose to match 'classify' command input
Xtest_50 = pcScore_tr(:,idxTest)';
Xtrain_50 = pcScore_tr(:,idxTrain)';

% Finally we do LDA classification
class_pca = classify(Xtest_50, Xtrain_50, ytrain);


%% Confusion Matrices and plots %%

% Nearest neighbor
Cmat_nn = confusionmat(ytest, class_nn);
%The number of correctly classified images
corr_class_nn = length(find(diag(Cmat_nn))); 

figure()
imagesc(Cmat_nn)
title({'Confusion Matrix, nearest neighbor';['Correct classes = ', num2str(corr_class_nn)]})

% LDA only
Cmat = confusionmat(ytest,class);
%The number of correctly classified images
corr_class = length(find(diag(Cmat)));

figure()
imagesc(Cmat);
title({'Confusion Matrix, LDA only';['Correct classes = ', num2str(corr_class)]})

% With PCA
Cmat_pca = confusionmat(ytest,class_pca);
%The number of correctly classified images
corr_class_pca = length(find(diag(Cmat_pca)));

figure()
imagesc(Cmat_pca);
title({'Confusion Matrix with PCA';['ncomp = ', num2str(ncomp), '  Correct classes = ', num2str(corr_class_pca)]})
