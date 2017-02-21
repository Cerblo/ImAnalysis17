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