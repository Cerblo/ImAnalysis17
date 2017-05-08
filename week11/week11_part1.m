clearvars
% Load the model
%run 'matconvnet-1.0-beta24 (1)'\matconvnet-1.0-beta24\matlab/vl_setupnn
net = load('imagenet-matconvnet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

% Preprocess the image
im = imread('ukbench00000.jpg');
im = single(im); % note: 255 range
imSize = net.meta.normalization.imageSize(1:2);
im = imresize(im, imSize);
avgIm = net.meta.normalization.averageImage;
if isvector(avgIm) %the avgerage value can be an entire image, or just three values (RGB)
avgIm = repmat(reshape(avgIm,1,1,3),imSize);
end
im = im - avgIm;

% Apply the model to the image
res = vl_simplenn(net, im) ;

% Display each layer with its dimensions
print_3d = @(vect) [num2str(vect(1)), ' x ', num2str(vect(2)), ' x ', num2str(vect(3))];
for i = 1:19
    layer = net.layers{i}.type;
    dim = size(res(i+1).x);
    if strcmp(layer,'conv')
        kernel_size = size(net.layers{i}.weights{1});
        depth = size(net.layers{i}.weights{1}, 4);
        disp([num2str(i), ' & ', layer, ' &      ', print_3d(dim), ' &       ', print_3d(kernel_size(1:3)), ' \\'])
    else
        disp([num2str(i), ' & ', layer, ' &      ', print_3d(dim), ' & \\'])
    end

%     disp([layer, ': ', num2str(dim)])
end

features = zeros(200, 4096);
for i=0:199
    input_name = strcat('ukbench', sprintf('%05d', i),'.jpg');
    % Preprocess the image
    im = imread(input_name);
    im = single(im); % note: 255 range
    imSize = net.meta.normalization.imageSize(1:2);
    im = imresize(im, imSize);
    avgIm = net.meta.normalization.averageImage;
    if isvector(avgIm) %the avgerage value can be an entire image, or just three values (RGB)
    avgIm = repmat(reshape(avgIm,1,1,3),imSize);
    end
    im = im - avgIm;

    % Apply the model to the image
    res = vl_simplenn(net, im) ;
    % Put the result in the features matrix
    features(i+1, :) = res(15 + 1).x;
end

% Apply the nearest neighbour to the dataset
nearest = zeros(1, 200);
for i = 1:200
    cloud = features([1:i-1, i+1:end], :);
    point = features(i, :);
    [indices,dists] = knnsearch(cloud, point);
    nearest(i) = indices;
end

preds = floor(nearest/4);
reals = floor((0:199)/4);
score = sum(preds == reals)/200;
    

for layer = 0:19
    dim = size(res(layer+1).x);
    features = zeros(200, numel(res(layer+1).x));
    for i=0:199
        input_name = strcat('ukbench', sprintf('%05d', i),'.jpg');
        % Preprocess the image
        im = imread(input_name);
        im = single(im); % note: 255 range
        imSize = net.meta.normalization.imageSize(1:2);
        im = imresize(im, imSize);
        avgIm = net.meta.normalization.averageImage;
        if isvector(avgIm) %the avgerage value can be an entire image, or just three values (RGB)
        avgIm = repmat(reshape(avgIm,1,1,3),imSize);
        end
        im = im - avgIm;

        % Apply the model to the image
        res = vl_simplenn(net, im) ;
        % Put the result in the features matrix  
        features_temp = res(layer + 1).x;
        features(i+1, :) = reshape(features_temp, [1, numel(features_temp)]);
    end

    % Apply the nearest neighbour to the dataset
    nearest = zeros(1, 200);
    for i = 1:200
        cloud = features([1:i-1, i+1:end], :);
        point = features(i, :);
        [indices,dists] = knnsearch(cloud, point);
        nearest(i) = indices;
    end

    preds = floor(nearest/4);
    reals = floor((0:199)/4);
    scores(layer+1) = sum(preds == reals)/200;
end
