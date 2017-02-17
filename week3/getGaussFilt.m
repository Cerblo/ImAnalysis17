function [g, dG, ddG, dddG] = getGaussFilt(t)
xMax = round(5*sqrt(t));
x = -xMax:xMax;

g    = exp(-x.^2/(2.*t))/sqrt(2.*pi.*t);
g    = g/sum(g);
dG   = exp(-x.^2/(2.*t))/sqrt(2.*pi.*t.^3).*(-x);
ddG  = exp(-x.^2/(2.*t))/sqrt(2.*pi.*t.^5).*(x.^2-t);
dddG = exp(-x.^2/(2.*t))/sqrt(2.*pi.*t.^7).*x.*(3.*t-x.^2);