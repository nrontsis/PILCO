%% gp2.m
% *Summary:* Compute joint predictions and derivatives for multiple GPs
% with uncertain inputs. Does not consider the uncertainty about the underlying
% function (in prediction), hence, only the GP mean function is considered.
% Therefore, this representation is equivalent to a regularized RBF
% network.
% If gpmodel.nigp exists, individial noise contributions are added.
%
%
%   function [M, S, V] = gp2(gpmodel, m, s)
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
% Last modified: 2013-03-05
%
%% High-Level Steps
% # If necessary, re-compute cached variables
% # Compute predicted mean and inv(s) times input-output covariance
% # Compute predictive covariance matrix, non-central moments
% # Centralize moments

function [M, S, V] = gp2(gpmodel, m, s)
%% Code
persistent iK oldX oldIn oldOut beta oldn;
D = size(gpmodel.inputs,2);    % number of examples and dimension of inputs
[n, E] = size(gpmodel.targets);      % number of examples and number of outputs

input = gpmodel.inputs;  target = gpmodel.targets; X = gpmodel.hyp;

% 1) if necessary: re-compute cached variables
if numel(X) ~= numel(oldX) || isempty(iK) ||  n ~= oldn || ...
    sum(any(X ~= oldX)) || sum(any(oldIn ~= input)) || ...
    sum(any(oldOut ~= target))
  oldX = X; oldIn = input; oldOut = target; oldn = n;
  K = zeros(n,n,E); iK = K; beta = zeros(n,E);
  
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

inp = bsxfun(@minus,gpmodel.inputs,m');                    % centralize inputs

% 2) Compute predicted mean and inv(s) times input-output covariance
for i=1:E
  iL = diag(exp(-X(1:D,i))); % inverse length-scales
  in = inp*iL;
  B = iL*s*iL+eye(D);
  
  t = in/B;
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*iL;
  c = exp(2*X(D+1,i))/sqrt(det(B));
  
  M(i) = sum(lb)*c;                                            % predicted mean
  V(:,i) = tL'*lb*c;                   % inv(s) times input-output covariance
  k(:,i) = 2*X(D+1,i)-sum(in.*in,2)/2;
end

% 3) Compute predictive covariance, non-central moments
for i=1:E
  ii = bsxfun(@rdivide,inp,exp(2*X(1:D,i)'));
  
  for j=1:i
    R = s*diag(exp(-2*X(1:D,i))+exp(-2*X(1:D,j)))+eye(D);
    t = 1/sqrt(det(R));
    ij = bsxfun(@rdivide,inp,exp(2*X(1:D,j)'));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    S(i,j) = t*beta(:,i)'*L*beta(:,j); S(j,i) = S(i,j);
  end
  
  S(i,i) = S(i,i) + 1e-6;          % add small jitter for numerical reasons
  
end

% 4) Centralize moments
S = S - M*M';
