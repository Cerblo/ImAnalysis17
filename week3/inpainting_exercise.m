img_train = double(imread(fullfile('data', 'lena.png')))/255;
img_test = double(imread(fullfile('data', 'lena40pctNoise.png')))/255;

patch_size = 5;
n_patches = 1000;
patches = extract_patches(img_train, n_patches, patch_size);
img_inpainted = inpaint(img_test, patches);
%number of different pixels
%sum(sum(img_test ~= img_train))/(512*512)
%sum(sum(img_inpainted ~= img_train))/(512*512)

%squared error
%sum(sum((img_test - img_train).^2))/(512*512)
%sum(sum((img_inpainted - img_train).^2))/(512*512)
figure(1)
imshow(img_inpainted)
