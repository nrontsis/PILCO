%% predcost.m
% *Summary:* Compute trajectory of expected costs for a given set of 
% state distributions
%
% inputs:
% m0          mean of states, D-by-1 or D-by-K for multiple means
% S           covariance matrix of state distributions
% dynmodel    (struct) for dynamics model (GP)
% plant	      (struct) of system parameters
% policy      (struct) for policy to be implemented
% cost        (struct) of cost function parameters
% H           length of optimization horizon
%
% outputs:
% L            expected cumulative (discounted) cost
% s            standard deviation of cost
%
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2012-01-12
%
%% High-Level Steps
% # Predict successor state distribution
% # Predict corresponding cost distribution

function [L, s] = predcost(m0, S, dynmodel, plant, policy, cost, H)
%% Code 

L = zeros(size(m0,2),H); s = zeros(size(m0,2),H);
for k = 1:size(m0,2);
  m = m0(:,k);
  for t = 1:H
    [m, S] = plant.prop(m, S, plant, dynmodel, policy);	     % get next state
    [L(k,t), d1, d2, v] = cost.fcn(cost, m, S);              % compute cost
    s(k,t) = sqrt(v);
  end
end
L = mean(L,1); s = mean(s,1); 
