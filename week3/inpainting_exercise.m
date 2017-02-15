img_train = double(imread(fullfile('data', 'lena.png')))/255;
img_test = double(imread(fullfile('data', 'lena10pctNoise.png')))/255;

patch_size = 11;
n_patches = 1000;
patches = extract_patches(img_train, n_patches, patch_size);
img_inpainted = inpaint(img_test, patches);

figure(1)
imshow(img_inpainted)
