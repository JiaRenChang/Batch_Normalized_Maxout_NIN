clc
clear all

A = gpuArray(single(randn(4,4,2,3)));
B = gpuArray(single(randn(4,4,2,3)));

C= cat(3,A,B);

K = vl_maxout(C,2,2);

% B = gpuArray(single(randn(4,4,2,3)));
% 
% BP= vl_maxout(A,2,3,B);

%find(A(:)==K(4,4,5,2))