%% Image location matching

addpath mex/mexmaci64
%addpath sift

A=imread('data/ukbench00000.jpg');
B=imread('data/ukbench00001.jpg');
C=imread('data/ukbench00004.jpg');


figure;
imagesc(A);

A_db=single(rgb2gray(A));

figure;
imagesc(B);

B_db=single(rgb2gray(B));

figure;
imagesc(C);

C_db=single(rgb2gray(C));

[FA,DA] = vl_sift(A_db);

figure;
ha=vl_plotsiftdescriptor(DA,FA);


[FB,DB] = vl_sift(B_db);
figure;
hb=vl_plotsiftdescriptor(DB,FB);

[FC,DC] = vl_sift(C_db);

figure;
hc=vl_plotsiftdescriptor(DC,FC);

sela = randperm(size(DA,2));
sela = sela(1:10);
figure;
imagesc(A_db); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DA(:,sela), FA(:,sela))



selb = randperm(size(DB,2));
selb = selb(1:10);
figure;
imagesc(B_db); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DB(:,selb), FB(:,selb))




selc = randperm(size(DC,2));
selc = selc(1:10);
figure;
imagesc(C_db); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DC(:,selc), FC(:,selc))

%% Normalization

DAn = double(DA)./sqrt((ones(size(DA,1),1)*sum(DA.*DA)));
DBn = double(DB)./sqrt((ones(size(DB,1),1)*sum(DB.*DB)));
DCn = double(DC)./sqrt((ones(size(DC,1),1)*sum(DC.*DC)));

%% Distance

dist_AB = pdist2(DAn',DBn','euclidean');
dist_AC = pdist2(DAn',DCn','euclidean');

%% Find min distances


sort_AB=sort(dist_AB,2);
mins_dist_AB=sort_AB(:,1:2);
eta_AB=mins_dist_AB(:,1)./mins_dist_AB(:,2);
eta_AB=eta_AB.*(eta_AB<0.6);
idr_AB=find(eta_AB);

sort_AC=sort(dist_AC,2);
mins_dist_AC=sort_AC(:,1:2);
eta_AC=mins_dist_AC(:,1)./mins_dist_AC(:,2);
eta_AC=eta_AC.*(eta_AC<0.6);
idr_AC=find(eta_AC);

[minc_AB,idc_AB] = min(dist_AB(idr_AB,:),[],2);

[minc_AC,idc_AC] = min(dist_AC(idr_AC,:),[],2);




% plot the k first matches on the pictures %
k=20;

figure
imagesc(A); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DA(:,idr_AB(1:k)), FA(:,idr_AB(1:k)))

figure
imagesc(B); colormap gray, axis off, axis image
vl_plotsiftdescriptor(DB(:,idc_AB(1:k)), FB(:,idc_AB(1:k)))

