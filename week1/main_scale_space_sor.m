
%% Load and display image 

blobSp=imread('Data/blobSp.png');


im_ss=blobSp;

im_ss_d=double(im_ss);

size_im_ss=size(im_ss_d);

figure(1);
imagesc(im_ss_d);
colormap gray;
% 
% im_ss_vec=reshape(blobSp_db,size_im_ss(1,1)*size_im_ss(1,2),1);
% 
% x_m=1:size_im_ss(1,2);
% y_m=1:size_im_ss(1,1);
% 
% [X,Y,Z]=meshgrid(x_m,y_m,im_ss_vec);

figure(2);
mesh(im_ss);
colormap gray;


%% Gaussian Derivatives
t1=2;
t2=4;
t3=8;

xMax1= round(5*sqrt(t1));
x1=-xMax1:xMax1;
xMax2= round(5*sqrt(t2));
x2=-xMax2:xMax2;
xMax3= round(5*sqrt(t3));
x3=-xMax3:xMax3;

[g1, dG1, ddG1, dddG1] = getGaussFilt(t1);
[g2, dG2, ddG2, dddG2] = getGaussFilt(t2);
[g3, dG3, ddG3, dddG3] = getGaussFilt(t3);


figure(3);
subplot(4,1,1);
plot(x1,g1(x1),x2,g2(x2),x3,g3(x3));
title('Gaussian');
legend('g1','g2','g3');

subplot(4,1,2);
plot(x1,dG1(x1),x2,dG2(x2),x3,dG3(x3));
title('1st Derivative Gaussian');
legend('dg1','dg2','dg3');

subplot(4,1,3);
plot(x1,ddG1(x1),x2,ddG2(x2),x3,ddG3(x3));
title('2nd Derivative Gaussian');
legend('ddg1','ddg2','ddg3');

subplot(4,1,4);
plot(x1,dddG1(x1),x2,dddG2(x2),x3,dddG3(x3));
title('3rd Derivative Gaussian');
legend('dddg1','dddg2','dddg3');


%% Filtering

phi=45;

Lx = filter2(dG1(x1), im_ss_d);
Lx = filter2(g1(x1)', Lx);



Ly = filter2(dG1(x1)', im_ss_d);
Ly = filter2(g1(x1), Ly);

Lphi=Lx*cosd(phi)+Ly*sind(phi);

figure(4)
subplot(1,3,1);
imagesc(Lx);
colormap gray;
title('Lx t=2');

subplot(1,3,2);
imagesc(Ly);
colormap gray;
title('Ly t=2');

subplot(1,3,3);
imagesc(Lphi);
colormap gray;
title('Lphi t=2');

%% Scale Space

%%For blobs


 
Bout=im_ss_d;

  figure(5);
for n=1:20
    
   
   Bout = filter2(g1(x1), Bout);
   Bout = filter2(g1(x1)', Bout);

   
    switch (n)
        case 1
            subplot(1,4,1);
            imagesc(Bout);
            colormap gray;
            title('Scalespace step 1');
        case 5
            subplot(1,4,2);
            imagesc(Bout);
            colormap gray;
            title('Scalespace step 5');
            
        case 10
            subplot(1,4,3);
            imagesc(Bout);
            colormap gray;
            title('Scalespace step 10');
        case 20
            subplot(1,4,4);
            imagesc(Bout);
            colormap gray;
            title('Scalespace step 20');
    end
    
    
    
end


%% Edge Detector 

[edgeIm1] = getEdge(im_ss_d, 1);
[edgeIm5] = getEdge(im_ss_d, 5);
[edgeIm10] = getEdge(im_ss_d, 10);

figure(6);
subplot(1,3,1)
imagesc(edgeIm1);
title('Edge detector t=1')
colormap gray;

subplot(1,3,2)
imagesc(edgeIm5);
title('Edge detector t=5')
colormap gray;

subplot(1,3,3)
imagesc(edgeIm10);
title('Edge detector t=10')
colormap gray;

