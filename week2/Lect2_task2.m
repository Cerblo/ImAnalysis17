clear all
close all

%% Doing SIFT on all 200 images, and saving the results %%
for i=0:199
    input_name = strcat('ukbench', sprintf('%05d', i),'.jpg');
    raw_image =imread(['./data/', input_name]);
    s = size(raw_image);
    im = reshape(single(rgb2gray(raw_image)), s(1), []);
    [F,D] = vl_sift(im); %SIFT
    
    %Saving descriptors of each image in a new .mat file
    save(['./data/output/result_', sprintf('%05d', i), '.mat'], 'F', 'D')
end

%% Selecting 100 random descriptors from each image, and storing it into
% matrix 'TrainD'
nti = 150; %number of training images
sel_k = 100; %the number of descriptors from each image
TrainD = zeros(128,nti*sel_k); % ALLOCATING SPACE FOR ALL DESCRIPTORS

% TrainD = []; NO ALLOCATION OF SPACE

g = 1; %Counter for skipping every 4th image
d=0; %Counter for saving in correct indexes of TrainD

for i=0:199
    if g<4
        DaF = load(['./data/output/result_', sprintf('%05d', i), '.mat']);
        sel = randperm(size(DaF.D,2));
        sel = sel(1:sel_k);
        
        d=d+1;
        
        c1 = (d*100)-99;
        c2 = d*100;
        TrainD(:, c1:c2) = DaF.D(:, sel); 
        % TrainD = [TrainD DaF.D(:, sel)]; CODE FOR NO ALLOCATION
        
        g=g+1;
    else
        g=1;
    end
end

% Save dat sjit!
save('./data/output/TrainD.mat', 'TrainD')

%% Try K-means clustering (this is euclidian distance as standard - NOT
% Mahalanobis)
n_bins = 250;
[idx1, C] = kmeans(TrainD', n_bins);

%% Comparing all descriptors from all images to the visual vocabulary, 'C' %%
h=zeros(200, n_bins);
g = 1; %Counter for skipping every 4th image
d=0; %Counter for saving in correct indexes of Xtrain
e=0;  %Counter for saving in correct indexes of Xtest

Xtrain=zeros(150,250);
ytrain=zeros(150,1);
Xtest=zeros(50,250);

for i=0:199    
    DaF = load(['./data/output/result_', sprintf('%05d', i), '.mat']);
    dists = pdist2(DaF.D', C);      % Computes the distance from all descriptors to C 
    [val, idx] = min(dists, [], 2); % Classify descriptors by finding the min. distances
    h1 = histcounts(idx, n_bins);   % Make a histogram for each picture
    h((i+1), :) = h1/length(dists); % normalize it by the number of descriptors
    
    if g<4
        d=d+1;
        Xtrain(d,:)=h((i+1), :);
        [M,I]=max(Xtrain(d,:));
        ytrain(d,1)=I;
        g=g+1;
    else
        g=1;
        e=e+1;
        Xtest(e,:)=h((i+1), :);
    end
    % figure()
    % bar(h)
end


%% Computing distances between picture histograms to find which pictures match %%

dist_pic = pdist2(h, h);
figure()
imagesc(dist_pic)
colormap('jet')
colorbar

%% Classification

class = classify(Xtest, Xtrain, ytrain,'diagLinear');

%% Classification with PCA


