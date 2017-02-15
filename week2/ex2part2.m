clearvars

%Question 1
for i=0:199
input_name = strcat('ukbench', sprintf('%05d', i),'.jpg');
raw_image =imread(input_name);
s = size(raw_image);
im = reshape(single(rgb2gray(raw_image)), s(1), []);
[F,D] = vl_sift(im);
output_name = strcat('data/output/result_', sprintf('%05d', i), '.txt'); 
dlmwrite(output_name,F)
dlmwrite(output_name, '-', '-append')
dlmwrite(output_name,D, '-append')
end

%Subset
test_index = strcat('data/output/result_', sprintf('%05d', 3:4:199), '.txt');
for i=0:199
    file_name = strcat('data/output/result_', sprintf('%05d', i), '.txt'); 
    if endsWith(filename, sprintf('%05d', i
copyfile(output_name, 


