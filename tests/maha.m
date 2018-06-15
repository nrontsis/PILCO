%% maha.m
% *Summary:* Point-wise squared Mahalanobis distance (a-b)*Q*(a-b)'.
% Vectors are row-vectors
%
%    function K = maha(a, b, Q)                         
%
% *Input arguments:*
%  
%   a   matrix containing n row vectors                                 [n x D]
%   b   matrix containing n row vectors                                 [n x D]
%   Q   weight matrix. Default: eye(D)                                  [D x D]
%
%
% *Output arguments:*
%  K    point-wise squared distances                                    [n x n]
%
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-21

function K = maha(a, b, Q)                         
%% Code

if nargin == 2                                                  % assume unit Q
  K = bsxfun(@plus,sum(a.*a,2),sum(b.*b,2)')-2*a*b';
else
  aQ = a*Q; K = bsxfun(@plus,sum(aQ.*a,2),sum(b*Q.*b,2)')-2*aQ*b';
end