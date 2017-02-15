function img = inpaint(img, patches)
    patch_size = size(patches, 2);
    if mod(patch_size, 2) == 0
        error('Patch size should be odd.');
    end
    n_patches = size(patches, 1);
    step = floor(patch_size/2);
    patches = permute(patches, [2 3 1]);
    for iy = 1+step:size(img, 1)-step
        for ix = 1+step:size(img, 2)-step
            if img(iy, ix) == 0
                window = img(iy-step:iy+step, ix-step:ix+step);
                mask = window > 0;
                % Calculate L1 distance between local window and all patches
                dists = repmat(window, [1 1 n_patches]) - patches;
                dists = dists .* repmat(mask, [1 1 n_patches]);
                dists = sum(sum(abs(dists)));

                % Find the minimum distance between the window and all the
                % patches
                % Note that this may give several possibilities
                closest_candidates = find(dists==min(dists));
                % We only take the first candidate, the others are going to
                % be included anyway thanks to the next steps
                closest = closest_candidates(1);
                min_dist = dists(closest);
                
                % Here all the close enough patches are included 
                % epsilon is chosen as 0.1
                good_centers = patches(step+1,step+1,dists<=min_dist*1.1);
                good_distances = dists(1, 1, dists<=min_dist*1.1);
                
                % For practical reasons we make them be vectors
                set_centers = permute(good_centers, [3 1 2]);
                % The weights have been chosen to be a decreasing function
                % of the distance
                weights = 1 / (1 + permute(good_distances, [3 1 2]));
                % The missing pixel is finally replaced
                %img(iy, ix) = randsample(set_centers,1);
                img(iy, ix) = randsample(set_centers,1, true, weights);
            end
        end
    end
end
