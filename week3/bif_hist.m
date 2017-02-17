function h = bif_hist(img, scales, epsilon)
    n_scales = numel(scales);
    if epsilon > 0
        n_responses = 7;
    else
        n_responses = 6;
    end

    % Assemble multi-scale BIF structure indices
    joint_idx = zeros(size(img));
    for scale_idx = 1:n_scales
        scale = scales(scale_idx);
        idx = bif_idx(bif_response(img, scale, epsilon));
        
        joint_idx = joint_idx + (idx-1)*n_responses^(scale_idx-1);
     end

    % Count bin contributions
    % TODO: Compute number of bins and the histogram using the tricks from
    %        the exercise text.
    n_bins = n_responses^n_scales;
    h = histc(joint_idx(:), 0:n_bins-1);
    
    % Normalize histogram
    h = h/sum(h);
end
