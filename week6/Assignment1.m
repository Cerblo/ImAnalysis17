clear all
close all

%% Assignment 7 - 1%%
data = load('points_data.mat');
p1 = data.points{1,1};
p2 = data.points{1,2};
p3 = data.points{1,3};
p4 = data.points{1,4};
p5 = data.points{1,5};

%Plot all pointgroups
figure(1)
plot(p1(:,1), p1(:,2), 'ok') %3 groups

figure(2)
plot(p2(:,1), p2(:,2), 'ok') %3 groups

figure(3)
plot(p3(:,1), p3(:,2), 'ok') %2 groups

figure(4)
plot(p4(:,1), p4(:,2), 'ok') %3 groups

figure(5)
plot(p5(:,1), p5(:,2), 'ok') %2 groups

%% Question 2 %% 

sig = [0.5, 0.25, 0.02, 0.75, 0.05];

for i=1:5
    W.W{1,i} = exp(-pdist2(data.points{1,i}, data.points{1,i}).^2/(2*sig(i)^2));
end

%% Question 3 %%
close all

k = [3,3,2,3,2];
% col = ['r'; 'g'; 'b'];

for i = 1:5
    [W.eigvec{1,i}, W.eigval{1,i}] = normalized_cut_from_W(W.W{1,i}, k(i));
    W.seg{1,i} = kmeans_discretize(W.eigvec{1,i});

    figure(i)
    
    for j = 1:k(i)
        n = data.points{1,i}((find(W.seg{1,i}==j)), :);
        W.order{j,i} = n;
        plot(n(:,1), n(:,2), 'o')
        hold on
    end
    title(['Segmentation of pointcloud ', num2str(i)])
    hold off
end

%% Question 4 %%
close all 

for i=1:5
    W.o{1,i}=[W.order{1,i};W.order{2,i};W.order{3,i}];
    W.W2{1,i} = exp(-pdist2(W.o{1,i}, W.o{1,i}).^2/(2*sig(i)^2));
    [W.eigvec2{1,i}, W.eigval2{1,i}] = normalized_cut_from_W(W.W2{1,i}, k(i));
    
    figure(i*2-1)
    plot(W.eigvec{1,i}(:,2),'.')
    title(['Plot of eigvector ', num2str(i)])
    saveas(gcf, ['eig_' num2str(i) '.png'])
    figure(i*2)
    plot(W.eigvec2{1,i}(:,2),'.')
    title(['Plot of the ordered eigvector ', num2str(i)])
    saveas(gcf, ['eig_ord_' num2str(i) '.png'])
end

%% Question 5 %%
close all

for i=1:5
    figure(i*2-1)
    imagesc(W.W{1,i})
    title(['Affinity matrix ', num2str(i)])
    saveas(gcf, ['Aff_' num2str(i) '.png'])
    
    figure(i*2)
    imagesc(W.W2{1,i})
    title(['Ordered affinity matrix ', num2str(i)])
    saveas(gcf, ['Aff_ord_' num2str(i) '.png'])
end

%% Assignemnt 7 - 3 %%
%Setting up
close all
clear all

% I = imread('plane.jpg');
% I = im2double(imresize(I,0.2));
% 
% I = imread('onion.png');
% I = im2double(imresize(I,0.4));

I = imread('peppers.png');
I = im2double(imresize(I,0.2));

figure()
imshow(I)

dim1 = size(I,1);
dim2 = size(I,2);
dim3 = size(I,3);

%Reshaping intensities (RGB) into vector for input into pdist2 
I = reshape(I, dim1*dim2, dim3);

%Defining matrix indexes
vec1=floor([1:1/dim2:dim1+1]);
vec1(end)=[];
vec2=repmat(1:dim2, 1, dim1);
dists = [vec1;vec2]';

%% Trying different values of sigma and segmenting %%

sig1=0.2;
sig2=5;
classes=5;

%Similarity
S1 = exp(-pdist2(I, I, 'squaredeuclidean')/(2*sig1^2));
%Proximity
P1 = exp(-pdist2(dists, dists, 'squaredeuclidean')/(2*sig2^2));
%Affinity
W=S1*P1;

%Segmentation
[eigvec, eigval] = normalized_cut_from_W(W, classes);
seg = kmeans_discretize(eigvec);

seg = reshape(seg, dim1, dim2);
figure()
imagesc(seg)

%% Question 2 %%
% [idx,C] = kmeans(I,2);
% 
% seg_k = reshape(idx, dim1, dim2);
% figure()
% imagesc(seg_k)
