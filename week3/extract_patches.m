function patches = extract_patches(img, n, patch_size)
    img_blocks = im2col(img, [patch_size patch_size]);
    n_blocks = size(img_blocks, 2);
    if n_blocks < n
        error('Image does not contain n patches.');
    end
    patches = img_blocks(:, randperm(n_blocks, n));
    patches = permute(patches, [2 1]);
    patches = reshape(patches, [n patch_size patch_size]);
end
