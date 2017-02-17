function C = conv2_(im, h)
    % Does convolution with edge replication (not zero pading)
    padding = floor(size(h) / 2);
    im = padarray(im, padding, 'replicate');
    C = conv2(im, h, 'valid');
end