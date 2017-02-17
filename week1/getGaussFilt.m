function [g, dG, ddG, dddG] = getGaussFilt(t)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

g=@(x) (1/(sqrt(2*pi*t))*exp(-x.^2/(2*t))); 
dG=@(x) (-x/(sqrt(2*pi*t^3)).*exp(-x.^2/(2*t))); 
ddG=@(x) (((x-sqrt(t)).*(x+sqrt(t)))/(sqrt(2*pi*t^5)).*exp(-x.^2/(2*t))); 
dddG=@(x) ((-x.*(x.^2-3*t))/(sqrt(2*pi*t^7)).*exp(-x.^2/(2*t))); 

end

