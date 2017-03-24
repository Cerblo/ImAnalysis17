%% %%%%%%%%%%%%%%%%%%EXERCICE 1 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Read data
% code non generic for the plotting of figures + legend
clear all;
close all;
points_data=load('points_data.mat');

size_set=length(points_data.points);

data_set=points_data.points;

figure(1);
for i=1:size_set
    data_set_i=data_set{1,i};
    subplot(2,3,i);
    plot(data_set_i(:,1),data_set_i(:,2),'.');
    title(['Set ', num2str(i)])
end


%% Affinity Matrix computation
affinity=struct('mat',[],'k',[2,3,2,3,2],'sigma',[]);

for i=1:size_set
    data_set_i=data_set{1,i};
    dist= pdist(data_set_i);
    affinity.sigma{1,i}=std(dist(:));
    affinity.mat{1,i}=squareform(exp(-dist.^2/(2*affinity.sigma{1,i}^2)));

end

%% Graph cut
clust=struct('vec',[],'val',[],'seg',[]);
cell_col={'r.','b.','y.'}; 
figure(2);
for i=1:size_set
    
    
   [clust.vec{1,i},clust.val{1,i}] = normalized_cut_from_W(affinity.mat{1,i},affinity.k(1,i));
   clust.seg{1,i} = kmeans_discretize(clust.vec{1,i});
  data_set_i=data_set{1,i};
  X=data_set_i(:,1);
  Y=data_set_i(:,2);
   subplot(2,3,i);
   for j=1:affinity.k(1,i)
       
        plot(X(clust.seg{1,i}==j,:),Y(clust.seg{1,i}==j,:),cell_col{1,j},'MarkerSize',12);
       hold on;
   end
  if affinity.k(1,i)==2
legend('Cluster 1','Cluster 2')

  else
    legend('Cluster 1','Cluster 2','Cluster 3')  
  end
 title(['Spectral Clust. , Set: ',num2str(i)]);
hold off
   
end


%% Ordering
 clust_ord=struct('vec',[],'val',[],'seg',[],'ind_ord',[]);
for i=1:size_set
    
    
    
[clust_ord.seg{1,i},I] = sort(clust.seg{1,i});
clust_ord.val{1,i}=clust.val{1,i};
clust_ord.ind_ord{1,i}=I;
Vec=clust.vec{1,i};

% Vec_full=zeros(size(Vec,1),affinity.k(1,i));
% 
% 
% for j=1:affinity.k(1,i)
%     res=Vec(:,j);
%   Vec_full(:,j) =res(I);
%   
% end

clust_ord.vec{1,i}=Vec(I,:);

end


%% Visualization of eigen vecotrs

figure(3);
for i=1:size_set
    subplot(2,5,i)
    
     Vec=clust.vec{1,i};
        plot(Vec(:,2),'.');
      title(['Unsorted 2nd Eigen Vector, Set: ',num2str(i)]);
    
    subplot(2,5,i+5)
    
   Vec2=clust_ord.vec{1,i};
       plot(Vec2(:,2),'.');
       title(['Sorted 2nd Eigen Vector, Set: ',num2str(i)]);
        
end

%% Visualisation of affinity matrix

figure(4);
for i=1:size_set
    subplot(2,3,i);
  
    imagesc(affinity.mat{1,i});
    colormap parula;
 title({' Affinity Matrix';[' Set:',num2str(i),' Sigma: ',num2str(affinity.sigma{1,i})]})    
end

%% Affinity Matrix after sorting computation
affinity_ord=struct('mat',[],'k',[2,3,2,3,2],'sigma',[]);

for i=1:size_set
    data_set_i=data_set{1,i};
    data_set_i_ord=data_set_i(clust_ord.ind_ord{1,i},:);
    dist= pdist(data_set_i_ord);
    affinity_ord.sigma{1,i}=std(dist(:));
    affinity_ord.mat{1,i}=squareform(exp(-dist.^2/(2*affinity_ord.sigma{1,i}^2)));
    
end

%% Visualisation of affinity matrix ordered

figure(5);
for i=1:size_set
    subplot(2,3,i);
  
    imagesc(affinity_ord.mat{1,i});
    colormap parula;
    title({'Ordered Affinity Matrix';[' Set:',num2str(i),' Sigma: ',num2str(affinity_ord.sigma{1,i})]})
    
end


%% K means results
clust_kmeans=struct('seg',[],'k',[2,3,2,3,2]);
cell_col={'b.','r.','y.'}; 

figure(6);
for i=1:size_set
    data_set_i=data_set{1,i};
[clust_kmeans.seg{1,i}] = kmeans(data_set_i,clust_kmeans.k(1,i));

 X=data_set_i(:,1);
 Y=data_set_i(:,2);
 subplot(2,3,i);
   for j=1:clust_kmeans.k(1,i)
       
        plot(X(clust_kmeans.seg{1,i}==j,:),Y(clust_kmeans.seg{1,i}==j,:),cell_col{1,j},'MarkerSize',12);
       hold on;
   end
  if clust_kmeans.k(1,i)==2
legend('Cluster 1','Cluster 2')

  else
    legend('Cluster 1','Cluster 2','Cluster 3')  
  end
 title (['Cluster kmeans, Set: ',num2str(i)]);
hold off
   

end


%% %%%%%%%%%%%%%%%%%%EXERCICE 2 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
Io = imread('bag.png');

I = double(Io(221:250,1:30))/255;
n=size(I,1);
m= size(I,2);

figure(7)
imshow(I);

%% Affinity Matrix computation

Ir=reshape(I,n*m,1);

dist_Ir= pdist(Ir);
sigmaIr=std(dist_Ir);

[X,Y] = meshgrid(1:n,1:m);
Xr=reshape(X,n*m,1);
Yr=reshape(Y,n*m,1);
coor=cat(2,Xr,Yr);
dist_c=pdist(coor);
sigma_c=std(coor(:));


W=squareform(exp(-dist_Ir.^2/(2*sigmaIr^2)-dist_c.^2/(2*sigma_c^2)));
Ws=squareform(exp(-dist_c.^2/(2*sigma_c^2)));
Wb=squareform(exp(-dist_Ir.^2/(2*sigmaIr^2)));

figure(8)
subplot(1,3,1)
imagesc(Ws);
title('Spatial Affinity matrix');
colormap parula;

subplot(1,3,2)
imagesc(Wb);
title('Brightness Affinity matrix');
colormap parula;

subplot(1,3,3)
imagesc(W);
title('Final Affinity matrix');
colormap parula;

% 
 [vec,val] = normalized_cut_from_W(W,2);
   seg = kmeans_discretize(vec);

seg_mat=reshape(seg,n,m);
vec2_mat=reshape(vec(:,2),n,m);

figure(9)
imagesc(seg_mat);
title('Spectral Clust.');
% 
figure(10)
imagesc(vec2_mat');
title('2nd eigen vector');

%% %%%%%%%%%%%%%%%%%%EXERCICE 3 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plane

clear all;
close all;
%Visualisatio 

k=2; %nb of clusters
I0 = imread('plane.jpg');
I = double(imresize(I0,0.2))/255;

n=size(I,1);
m= size(I,2);

figure(11)
imshow(I);
%Computation of the affinity matrix

Ir=reshape(I,n*m,3);

dist_Ir= pdist(Ir);
sigmaIr=std(dist_Ir);

[X,Y] = meshgrid(1:m,1:n);
Xr=reshape(X,n*m,1);
Yr=reshape(Y,n*m,1);
coor=cat(2,Xr,Yr);
dist_c=pdist(coor);
sigma_c=std(coor(:));


W=squareform(exp(-dist_Ir.^2/(2*sigmaIr^2)-dist_c.^2/(2*sigma_c^2)));
Ws=squareform(exp(-dist_c.^2/(2*sigma_c^2)));
Wb=squareform(exp(-dist_Ir.^2/(2*sigmaIr^2)));

figure(8)
subplot(1,3,1)
imagesc(Ws);
title('Spatial Affinity matrix');
colormap parula;

subplot(1,3,2)
imagesc(Wb);
title('Brightness Affinity matrix');
colormap parula;

subplot(1,3,3)
imagesc(W);
title('Final Affinity matrix');
colormap parula;
    
%Computation of eigen vectors, values + re ordering of
%affinity matrix 


 [vec,val] = normalized_cut_from_W(W,k);
   seg = kmeans_discretize(vec);
[seg_or,ind]=sort(seg);

Ir_ord=Ir(ind,:);
dist_Ir_ord= pdist(Ir_ord);
sigmaIr_ord=std(dist_Ir_ord);
W=squareform(exp(-dist_Ir_ord.^2/(2*sigmaIr_ord^2)-dist_c.^2/(2*sigma_c^2)));
Ws=squareform(exp(-dist_c.^2/(2*sigma_c^2)));
Wb=squareform(exp(-dist_Ir_ord.^2/(2*sigmaIr_ord^2)));
    
%Visualisation of affinity matrix ordered

figure(9)
subplot(1,3,1)
imagesc(Ws);
title('Spatial Affinity matrix');
colormap parula;

subplot(1,3,2)
imagesc(Wb);
title('Brightness Affinity matrix');
colormap parula;

subplot(1,3,3)
imagesc(W);
title('Final Affinity matrix');
colormap parula;
    
    
%Visualisation of eigen vector and segmentation    

seg_mat=reshape(seg,n,m);
vec2_mat=reshape(vec(:,2),n,m);

figure(10)
imagesc(seg_mat);
title('Spectral Clust.');
% 
figure(11)
imagesc(vec2_mat);
title('2nd eigen vector');


%kmeans results




   
[clust_kmeans] = kmeans(Ir,k);
clust_kmeans_mat=reshape(clust_kmeans,n,m);

figure(12);
imagesc(clust_kmeans_mat);
title('K means Clusterisation');


%% Onion

clear all;
close all;
%Visualisatio 

k=5; %nb of clusters
I0 = imread('onion.png');
I = double(imresize(I0,0.4))/255;

n=size(I,1);
m= size(I,2);

figure(13)
imshow(I);
%Computation of the affinity matrix

Ir=reshape(I,n*m,3);

dist_Ir= pdist(Ir);
sigmaIr=std(dist_Ir);

[X,Y] = meshgrid(1:m,1:n);
Xr=reshape(X,n*m,1);
Yr=reshape(Y,n*m,1);
coor=cat(2,Xr,Yr);
dist_c=pdist(coor);
sigma_c=std(coor(:));


W=squareform(exp(-dist_Ir.^2/(2*sigmaIr^2)-dist_c.^2/(2*sigma_c^2)));
Ws=squareform(exp(-dist_c.^2/(2*sigma_c^2)));
Wb=squareform(exp(-dist_Ir.^2/(2*sigmaIr^2)));

figure(14)
subplot(1,3,1)
imagesc(Ws);
title('Spatial Affinity matrix');
colormap parula;

subplot(1,3,2)
imagesc(Wb);
title('Brightness Affinity matrix');
colormap parula;

subplot(1,3,3)
imagesc(W);
title('Final Affinity matrix');
colormap parula;
    
%Computation of eigen vectors, values + re ordering of
%affinity matrix 


 [vec,val] = normalized_cut_from_W(W,k);
   seg = kmeans_discretize(vec);
[seg_or,ind]=sort(seg);

Ir_ord=Ir(ind,:);
dist_Ir_ord= pdist(Ir_ord);
sigmaIr_ord=std(dist_Ir_ord);
W=squareform(exp(-dist_Ir_ord.^2/(2*sigmaIr_ord^2)-dist_c.^2/(2*sigma_c^2)));
Ws=squareform(exp(-dist_c.^2/(2*sigma_c^2)));
Wb=squareform(exp(-dist_Ir_ord.^2/(2*sigmaIr_ord^2)));
    
%Visualisation of affinity matrix ordered

figure(15)
subplot(1,3,1)
imagesc(Ws);
title('Spatial Affinity matrix');
colormap parula;

subplot(1,3,2)
imagesc(Wb);
title('Brightness Affinity matrix');
colormap parula;

subplot(1,3,3)
imagesc(W);
title('Final Affinity matrix');
colormap parula;
    
    
%Visualisation of eigen vector and segmentation    

seg_mat=reshape(seg,n,m);
vec2_mat=reshape(vec(:,2),n,m);

figure(16)
imagesc(seg_mat);
title('Spectral Clust.');
% 
figure(17)
imagesc(vec2_mat);
title('2nd eigen vector');


%kmeans results

   
[clust_kmeans] = kmeans(Ir,k);
clust_kmeans_mat=reshape(clust_kmeans,n,m);

figure(18);
imagesc(clust_kmeans_mat);
title('K means Clusterisation');

 
%% Peppers

   
clear all;
close all;
%Visualisation 

k=5; %nb of clusters
I0 = imread('peppers.png');
I = double(imresize(I0,0.2))/255;

n=size(I,1);
m= size(I,2);

figure(19)
imshow(I);
%Computation of the affinity matrix

Ir=reshape(I,n*m,3);

dist_Ir= pdist(Ir);
sigmaIr=std(dist_Ir);

[X,Y] = meshgrid(1:m,1:n);
Xr=reshape(X,n*m,1);
Yr=reshape(Y,n*m,1);
coor=cat(2,Xr,Yr);
dist_c=pdist(coor);
sigma_c=std(coor(:));


W=squareform(exp(-dist_Ir.^2/(2*sigmaIr^2)-dist_c.^2/(2*sigma_c^2)));
Ws=squareform(exp(-dist_c.^2/(2*sigma_c^2)));
Wb=squareform(exp(-dist_Ir.^2/(2*sigmaIr^2)));

figure(20)
subplot(1,3,1)
imagesc(Ws);
title('Spatial Affinity matrix');
colormap parula;

subplot(1,3,2)
imagesc(Wb);
title('Brightness Affinity matrix');
colormap parula;

subplot(1,3,3)
imagesc(W);
title('Final Affinity matrix');
colormap parula;
    
%Computation of eigen vectors, values + re ordering of
%affinity matrix 


 [vec,val] = normalized_cut_from_W(W,k);
   seg = kmeans_discretize(vec);
[seg_or,ind]=sort(seg);

Ir_ord=Ir(ind,:);
dist_Ir_ord= pdist(Ir_ord);
sigmaIr_ord=std(dist_Ir_ord);
W=squareform(exp(-dist_Ir_ord.^2/(2*sigmaIr_ord^2)-dist_c.^2/(2*sigma_c^2)));
Ws=squareform(exp(-dist_c.^2/(2*sigma_c^2)));
Wb=squareform(exp(-dist_Ir_ord.^2/(2*sigmaIr_ord^2)));
    
%Visualisation of affinity matrix ordered

figure(21)
subplot(1,3,1)
imagesc(Ws);
title('Spatial Affinity matrix');
colormap parula;

subplot(1,3,2)
imagesc(Wb);
title('Brightness Affinity matrix');
colormap parula;

subplot(1,3,3)
imagesc(W);
title('Final Affinity matrix');
colormap parula;
    
    
%Visualisation of eigen vector and segmentation    

seg_mat=reshape(seg,n,m);
vec2_mat=reshape(vec(:,2),n,m);

figure(22)
imagesc(seg_mat);
title('Spectral Clust.');
% 
figure(23)
imagesc(vec2_mat);
title('2nd eigen vector');


%kmeans results

   
[clust_kmeans] = kmeans(Ir,k);
clust_kmeans_mat=reshape(clust_kmeans,n,m);

figure(24);
imagesc(clust_kmeans_mat);
title('K means Clusterisation');


