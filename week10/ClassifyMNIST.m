function [ y_e ] = ClassifyMNIST( data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
load ann.mat
data=double(data)/255;
y_e = nn_predict_flex(data,weigths);



end

