function data = getDataNN(s, n, noise, showPoints)
% returns a data set as a n x 4 matrix. The first two columns are the
% coordinates and the last two is the label
% Anders B. Dahl, March 29, 2017
% 

if ( nargin == 3 )
    showPoints = false;
end
if ( s == 1 )
    % data set 1
    nT = round(n/2);
    data = zeros(n,4);
    data(:,1:2) = noise*randn(n,2);
    data(1:nT,1) = data(1:nT,1) + 1;
    data(1:nT,2) = data(1:nT,2) + 1;
    data(1:nT,3) = 1;
    data(nT+1:end,1) = data(nT+1:end,1) - 1;
    data(nT+1:end,2) = data(nT+1:end,2) - 1;
    data(nT+1:end,4) = 1;
elseif ( s == 2 )
    % data set 2
    nT = 2*round(n/3);
    x = rand(n,1)*2*pi;
    data = zeros(n,4);
    data(1:nT,:) = [noise*randn(nT,2) + [cos(x(1:nT)), sin(x(1:nT))], ones(nT,1), zeros(nT,1)];
    data(nT+1:end,:) = [noise*randn(n-nT,2), zeros(n-nT,1), ones(n-nT,1)];
else
    % data set 3
    nT = round(n/4);
    data = zeros(n,4);
    data(:,1:2) = noise*randn(n,2);
    data(1:nT,1) = data(1:nT,1) + 1;
    data(1:nT,2) = data(1:nT,2) + 1;
    data(1:nT,3) = 1;
    data(nT+1:2*nT,1) = data(nT+1:2*nT,1) + 1;
    data(nT+1:2*nT,2) = data(nT+1:2*nT,2) - 1;
    data(nT+1:2*nT,4) = 1;
    data(2*nT+1:3*nT,1) = data(2*nT+1:3*nT,1) - 1;
    data(2*nT+1:3*nT,2) = data(2*nT+1:3*nT,2) - 1;
    data(2*nT+1:3*nT,3) = 1;
    data(3*nT+1:end,1) = data(3*nT+1:end,1) - 1;
    data(3*nT+1:end,2) = data(3*nT+1:end,2) + 1;
    data(3*nT+1:end,4) = 1;
end
if ( showPoints )
    figure(1)
    id = find(data(:,3)==1);
    plot(data(id,1), data(id,2), 'b.', 'MarkerSize', 20);
    hold on
    id = find(data(:,4)==1);
    plot(data(id,1), data(id,2), 'r.', 'MarkerSize', 20);
end