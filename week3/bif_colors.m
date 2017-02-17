function img = bif_colors(idx)
    colors = zeros(7, 3);
    colors(1, :) = [84, 84, 84];
    colors(2, :) = [0, 0, 0];
    colors(3, :) = [255, 255, 255];
    colors(4, :) = [219, 219, 112];
    colors(5, :) = [0, 0, 128];
    colors(6, :) = [154, 205, 50];
    colors(7, :) = [255, 105, 180];
    colors = colors/255;

    img = zeros(size(idx, 1), size(idx, 2), 3);
    for iy = 1:size(idx, 1)
        for ix = 1:size(idx, 2)
            img(iy, ix, :) = colors(idx(iy, ix), :);
        end
    end
end
