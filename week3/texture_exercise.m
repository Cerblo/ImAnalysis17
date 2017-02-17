clear all
close all
addpath('bif')

% Show BIF responses for different scales and epsilon
img = double(imread(fullfile('data', 'lena.png')));
scale = 4;
epsilon = 0;
bif_img = bif_colors(bif_idx(bif_response(img, scale, epsilon)));
figure(1)
imshow(bif_img);
title({'BIF response for';['Scale = ', num2str(scale), ', eps = ', num2str(epsilon)]})

img = double(imread(fullfile('data', 'lena.png')));
scale = 4;
epsilon = 0.05;
bif_img = bif_colors(bif_idx(bif_response(img, scale, epsilon)));
figure(2)
imshow(bif_img);
title({'BIF response for';['Scale = ', num2str(scale), ', eps = ', num2str(epsilon)]})

img = double(imread(fullfile('data', 'lena.png')));
scale = 8;
epsilon = 0;
bif_img = bif_colors(bif_idx(bif_response(img, scale, epsilon)));
figure(3)
imshow(bif_img);
title({'BIF response for';['Scale = ', num2str(scale), ', eps = ', num2str(epsilon)]})

% Show BIF histogram
img = double(imread(fullfile('data', 'lena.png')));
scales = [1 2 4 8];
epsilon = 0;
h = bif_hist(img, scales, epsilon);
figure(4)
bar(h)


% Compute histograms for the first five images in each class and show their
% distances
data_root = 'data';
class_dir = {fullfile('T01_bark1', 'T01_0'), ...
             fullfile('T04_wood1', 'T04_0'), ...
             fullfile('T08_granite', 'T08_0'), ...
             fullfile('T09_marble', 'T09_0'), ...
             fullfile('T20_upholstery', 'T20_0'), ...
             fullfile('T24_corduroy', 'T24_0')};

scales = [1 2 3 4];
n_classes = 6;
n_imgs_per_class = 5;
n_imgs = n_classes * n_imgs_per_class;
hists = zeros(n_imgs, 1296);
img_idx = 1;
for i = 1:n_classes
    for j = 1:n_imgs_per_class
        img = double(imread(fullfile(data_root, ...
                                     [class_dir{i} num2str(j) '.jpg'])));
        hists(img_idx, :) = bif_hist(img, scales, epsilon);
        img_idx = img_idx + 1;
    end
end

% Show the distance between histograms as a confusion matrix
% dists = zeros(n_imgs, n_imgs);

% Compute L1 distances between all histograms
dists2 = pdist2(hists, hists, 'cityblock');
% Plot dat sjit!
figure(5)
imagesc(dists2);
colormap('jet')
colorbar
title('Distances between textures');

%% Classifications %%
% Calculate the 'n_imgs' smallest distances from each image histogram 
[min_val, idx_min] = pdist2(hists, hists, 'cityblock', 'Smallest', n_imgs_per_class);
idx_min = ceil(idx_min(2:n_imgs_per_class, :)/n_imgs_per_class);
class = zeros(n_imgs, 1);
edges = [0.5:1:(n_imgs)+0.5];

% Counting most usual class of the 4 nearest classes
for k = 1:n_imgs
    h = histcounts(idx_min(:,k), edges);
    [mv, class_k] = max(h);
    class(k) = class_k;
end

Known = ones(n_imgs, 1);
Known(6:10) = 2;
Known(11:15) = 3;
Known(16:20) = 4;
Known(21:25) = 5;
Known(26:30) = 6;

% Calculate a confusionmatrix for the most usual of 4 nearest classes
CF1 = confusionmat(Known, class);
figure(6)
imagesc(CF1)
colorbar
title('Nearest 4 classification')

% Nearest neighbor
CF2 = confusionmat(Known, idx_min(1,:));
figure(7)
imagesc(CF2)
colorbar
title('Nearest Neighbor classification')

% THIS IS NOT SCALE INVARIANT!!! DIFFERENT SCALES WOULD OUTPUT DIFFERENT
% HISTOGRAMS.