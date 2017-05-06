%% IMAGE ANALYSIS WEEK 12


clear all
close all

v = VideoReader('material/data/echiniscus.mp4');
i=1;

while hasFrame(v)
im.fr{1,i} = double(readFrame(v))/255;
im.y{1,i}=(2*im.fr{1,i}(:,:,3) - (im.fr{1,i}(:,:,1) + im.fr{1,i}(:,:,2)) + 2)/4;
imagesc(im.fr{1,i}), axis image,
pause(1/v.FrameRate);
i=i+1;
end
%% FOR 1 FRAME 
frame=194;
c=[210, 140];
r=50;
N=100;
S = make_circular_snake(c,r,N);
figure()
imagesc(im.fr{1,frame})
hold on
plot(S(:,1), S(:,2), 'r')

mask = double(poly2mask(S(:,1), S(:,2), size(im.fr{1,frame},1), size(im.fr{1,frame},2)));
% figure()
% imagesc(mask)
mask_in = (mask.*im.y{1,frame});
mask_out = (mask==0).*im.y{1,frame};

[~,~,v_in]=find(mask_in>0);
[~,~,v_out]=find(mask_out>0);

int_in=sum(sum(mask_in));
int_out=sum(sum(mask_out));

mean_int_in = int_in/length(v_in);
mean_int_out = int_out/length(v_out);

% Intensity at the snake points
s_point = interp2(im.y{1,frame}, S(:,1), S(:,2));
s_norm = snake_normals(S);

F_ext = repmat(0.5*(mean_int_in-mean_int_out)*(s_point-mean_int_in+s_point-mean_int_out),1,2).*s_norm;

quiver(S(:,1), S(:,2),F_ext(:,1), F_ext(:,2), 'r');

% % fr=double(rgb2gray(fr))/255; % For amoeba
% fr=double(fr)/255; % For echiniscus
% 
% %Converting to yellow-blue scale
% y=(2*fr(:,:,3) - (fr(:,:,1) + fr(:,:,2)) + 2)/4;
% % imagesc(y)

%% Snake on the video

%Random init of the snake
c=[210, 140];
r=50;
N=100;
S = make_circular_snake(c,r,N);
[B,A ]= regularization_matrix(N,alpha,beta);
% Read the video
v = VideoReader('material/data/echiniscus.mp4');

% Step init
tau=50;
alpha=0.001;
beta=0.4;

i=1;

figure()
while hasFrame(v)
im.fr{1,i} = double(readFrame(v))/255;
im.y{1,i}=(2*im.fr{1,i}(:,:,3) - (im.fr{1,i}(:,:,1) + im.fr{1,i}(:,:,2)) + 2)/4;

F_ext = ext_forc( S,im.y{1,i},im.fr{1,i} );
S=B*(S+tau*F_ext);
S = remove_crossings(S);
S=distribute_points(S,'number',N) ;
S = keep_snake_inside(S,[size(im.fr{1,i},1),size(im.fr{1,i},2)]);
imagesc(im.fr{1,i}),
hold on
plot(S(:,1), S(:,2), 'r'),
%quiver(S(:,1), S(:,2),F_ext(:,1), F_ext(:,2), 'r'),
axis image,
pause(1/v.FrameRate);
i=i+1;
end
