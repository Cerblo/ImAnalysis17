function resp = bif_response(img, scale, epsilon)
assert(isa(img, 'double') || isa(img, 'single'), ...
    'Only images of class DOUBLE or SINGLE are supported.');
assert(size(img, 3)==1, 'Only single channel images are supported.');

% Gaussian kernels
[g, dG, ddG] = getGaussFilt(scale);

% Scale-normalized image derivatives
s = conv2_(conv2_(img, g), g');
sx = scale * conv2_(conv2_(img, g'), dG);
sy = scale * conv2_(conv2_(img, dG'), g);
sxx = scale^2 * conv2_(conv2_(img, g'), ddG);
sxy = scale^2 * conv2_(conv2_(img, dG'), dG);
syy = scale^2 * conv2_(conv2_(img, ddG'), g);

% Calculate feature responses
if epsilon > 0
    n_responses = 7;
else
    n_responses = 6;
end
resp = zeros(size(img, 1), size(img, 2), n_responses);

lambda = sxx+syy;
gamma = sqrt((sxx-syy).^2+4*sxy.^2);

resp(:, :, 1) = 2*sqrt(sx.^2+sy.^2);        % slope
resp(:, :, 2) = lambda;                     % dark blob
resp(:, :, 3) = -lambda;                    % bright blob
resp(:, :, 4) = 2^(-0.5)*(gamma-lambda);    % ridge (bright line)
resp(:, :, 5) = 2^(-0.5)*(gamma+lambda);    % rut (dark line)
resp(:, :, 6) = gamma;                      % saddle
if n_responses == 7
    resp(:, :, 7) = epsilon*s;              % flat
end
end