%% gp0.m
% *Summary:* Compute joint predictions for multiple GPs with uncertain inputs.
% If gpmodel.nigp exists, individial noise contributions are added.
% Predictive variances contain uncertainty about the function, but no noise.
%
%   function [M, S, V] = gp0(gpmodel, m, s)
%
% *Input arguments:*
%
%   gpmodel    GP model struct
%     hyp      log-hyper-parameters                                  [D+2 x  E ]
%     inputs   training inputs                                       [ n  x  D ]
%     targets  training targets                                      [ n  x  E ]
%     nigp     (optional) individual noise variance terms            [ n  x  E ]
%   m          mean of the test distribution                         [ D  x  1 ]
%   s          covariance matrix of the test distribution            [ D  x  D ]
%
% *Output arguments:*
%
%   M          mean of pred. distribution                            [ E  x  1 ]
%   S          covariance of the pred. distribution                  [ E  x  E ]
%   V          inv(s) times covariance between input and output      [ D  x  E ]
%
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-05-24
%
%% High-Level Steps
% # If necessary, compute kernel matrix and cache it
% # Compute predicted mean and inv(s) times input-output covariance
% # Compute predictive covariance matrix, non-central moments
% # Centralize moments

function [M, S, V] = gp0(gpmodel, m, s)
%% Code

persistent K iK beta oldX oldn;
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
[n, E] = size(gpmodel.targets);     % number of examples and number of outputs
X = gpmodel.hyp;                              % short hand for hyperparameters

% 1) if necessary: re-compute cashed variables
if numel(X) ~= numel(oldX) || isempty(iK) || sum(any(X ~= oldX)) || n ~= oldn
  oldX = X; oldn = n;                                               
  iK = zeros(n,n,E); K = zeros(n,n,E); beta = zeros(n,E);
  
  for i=1:E                                              % compute K and inv(K)
    inp = bsxfun(@rdivide,gpmodel.inputs,exp(X(1:D,i)'));
    K(:,:,i) = exp(2*X(D+1,i)-maha(inp,inp)/2);
    if isfield(gpmodel,'nigp')
      L = chol(K(:,:,i) + exp(2*X(D+2,i))*eye(n) + diag(gpmodel.nigp(:,i)))';
    else        
      L = chol(K(:,:,i) + exp(2*X(D+2,i))*eye(n))';
    end
    iK(:,:,i) = L'\(L\eye(n));
    beta(:,i) = L'\(L\gpmodel.targets(:,i));
  end
end

k = zeros(n,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E);

inp = bsxfun(@minus,gpmodel.inputs,m');                     % centralize inputs

% 2) compute predicted mean and inv(s) times input-output covariance
for i=1:E    
  iL = diag(exp(-X(1:D,i))); % inverse length-scales
  in = inp*iL;
  B = iL*s*iL+eye(D); 
  
  t = in/B;
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tiL = t*iL;
  c = exp(2*X(D+1,i))/sqrt(det(B));
  
  M(i) = sum(lb)*c;                                            % predicted mean
  V(:,i) = tiL'*lb*c;                    % inv(s) times input-output covariance
  k(:,i) = 2*X(D+1,i)-sum(in.*in,2)/2;
end

% 3) ompute predictive covariance, non-central moments
for i=1:E                 
  ii = bsxfun(@rdivide,inp,exp(2*X(1:D,i)'));
  
  for j=1:i
    R = s*diag(exp(-2*X(1:D,i))+exp(-2*X(1:D,j)))+eye(D); 
    t = 1/sqrt(det(R));
    ij = bsxfun(@rdivide,inp,exp(2*X(1:D,j)'));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    if i==j
      S(i,i) = t*(beta(:,i)'*L*beta(:,i) - sum(sum(iK(:,:,i).*L)));
    else
      S(i,j) = beta(:,i)'*L*beta(:,j)*t; 
      S(j,i) = S(i,j);
    end  
  end
  
  S(i,i) = S(i,i) + exp(2*X(D+1,i));
end

% 4) centralize moments
S = S - M*M';                                              
