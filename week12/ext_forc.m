function [ F_ext ] = ext_forc( S,y,fr )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

mask = double(poly2mask(S(:,1), S(:,2), size(fr,1), size(fr,2)));
mask_in = (mask.*y);
mask_out = (mask==0).*y;
[~,~,v_in]=find(mask_in>0);
[~,~,v_out]=find(mask_out>0);

int_in=sum(sum(mask_in));
int_out=sum(sum(mask_out));

mean_int_in = int_in/length(v_in);
mean_int_out = int_out/length(v_out);

% Intensity at the snake points
s_point = interp2(y, S(:,1), S(:,2));
s_norm = snake_normals(S);

F_ext = repmat(0.5*(mean_int_in-mean_int_out)*(s_point-mean_int_in+s_point-mean_int_out),1,2).*s_norm;


end

