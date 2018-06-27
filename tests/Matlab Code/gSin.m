%% gSin.m
% *Summary:* Compute moments of the saturating function $e*sin(x(i))$,
% where $x \sim\mathcal N(m,v)$ and $i$ is a (possibly empty) set of $I$
% indices. The optional  scaling factor $e$ is a vector of length $I$.
% Optionally, compute derivatives of the moments.
%
%    function [M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv] = gSin(m, v, i, e)
%
% *Input arguments:*
%
%   m     mean vector of Gaussian                                    [ d       ]
%   v     covariance matrix                                          [ d  x  d ]
%   i     vector of indices of elements to augment                   [ I  x  1 ]
%   e     (optional) scale vector; default: 1                        [ I  x  1 ]
%
% *Output arguments:*
%
%   M     output means                                               [ I       ]
%   V     output covariance matrix                                   [ I  x  I ]
%   C     inv(v) times input-output covariance                       [ d  x  I ]
%   dMdm  derivatives of M w.r.t m                                   [ I  x  d ]
%   dVdm  derivatives of V w.r.t m                                   [I^2 x  d ]
%   dCdm  derivatives of C w.r.t m                                   [d*I x  d ]
%   dMdv  derivatives of M w.r.t v                                   [ I  x d^2]
%   dVdv  derivatives of V w.r.t v                                   [I^2 x d^2]
%   dCdv  derivatives of C w.r.t v                                   [d*I x d^2]
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-25

function [M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv] = gSin(m, v)
%% Code

i = [linspace(1,length(m), length(m))];
d = length(m); I = length(i);
if nargin == 2, e = ones(I,1); else e = e(:); end          % unit column default
mi(1:I,1) = m(i); vi = v(i,i); vii(1:I,1) = diag(vi);      % short-hand notation

M = e.*exp(-vii/2).*sin(mi);                                              % mean

lq = -bsxfun(@plus,vii,vii')/2; q = exp(lq);
V = (exp(lq+vi)-q).*cos(bsxfun(@minus,mi,mi')) - ...
                                      (exp(lq-vi)-q).*cos(bsxfun(@plus,mi,mi'));
V = e*e'.*V/2;                                                        % variance

C = zeros(d,I); C(i,:) = diag(e.*exp(-vii/2).*cos(mi));       % inv(v) times cov

if nargout > 3                                            % compute derivatives?
  dVdm = zeros(I,I,d); dCdm = zeros(d,I,d); dVdv = zeros(I,I,d,d);
  dCdv = zeros(d,I,d,d); dMdm = C';
  U1 = -(exp(lq+vi)-q).*sin(bsxfun(@minus,mi,mi'));
  U2 = (exp(lq-vi)-q).*sin(bsxfun(@plus,mi,mi'));
  for j = 1:I
    u = zeros(I,1); u(j) = 1/2;
    dVdm(:,:,i(j)) = e*e'.*(U1.*bsxfun(@minus,u,u') + U2.*bsxfun(@plus,u,u'));
    dVdv(j,j,i(j),i(j)) = exp(-vii(j)) * ...
                                (1+(2*exp(-vii(j))-1)*cos(2*mi(j)))*e(j)*e(j)/2;
    for k = [1:j-1 j+1:I]
      dVdv(j,k,i(j),i(k)) = (exp(lq(j,k)+vi(j,k)).*cos(mi(j)-mi(k)) + ...
                            exp(lq(j,k)-vi(j,k)).*cos(mi(j)+mi(k)))*e(j)*e(k)/2;
      dVdv(j,k,i(j),i(j)) = -V(j,k)/2;
      dVdv(j,k,i(k),i(k)) = -V(j,k)/2;
    end
    dCdm(i(j),j,i(j)) = -M(j);
    dCdv(i(j),j,i(j),i(j)) = -C(i(j),j)/2;
  end
  dMdv = permute(dCdm,[2 1 3])/2;

  dMdv = reshape(dMdv,[I d*d]);
  dVdv = reshape(dVdv,[I*I d*d]); dVdm = reshape(dVdm,[I*I d]);
  dCdv = reshape(dCdv,[d*I d*d]); dCdm = reshape(dCdm,[d*I d]);
end
