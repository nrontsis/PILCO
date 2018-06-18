%% pred.m
% *Summary:* Compute predictive (marginal) distributions of a trajecory
%
%   [M S] = pred(policy, plant, dynmodel, m, s, H)
%
% *Input arguments:*
%
%   policy             policy structure
%   plant              plant structure
%   dynmodel           dynamics model structure
%   m                  D-by-1 mean of the initial state distribution
%   s                  D-by-D covariance of the initial state distribution
%   H                  length of prediction horizon
%
% *Output arguments:*
%
%   M                  D-by-(H+1) sequence of predicted mean vectors
%   S                  D-by-D-(H+1) sequence of predicted covariance
%                      matrices
% 
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-01-23
%
%% High-Level Steps
% # Predict successor state distribution

function [M S] = pred(policy, plant, dynmodel, m, s, H)
%% Code

D = length(m); S = zeros(D,D,H+1); M = zeros(D,H+1);
M(:,1) = m; S(:,:,1) = s;
for i = 1:H
  [m s] = plant.prop(m, s, plant, dynmodel, policy);
  M(:,i+1) = m(end-D+1:end); 
  S(:,:,i+1) = s(end-D+1:end,end-D+1:end);
end
