function [lse, sm] = logsumexp(x, dim)
%LOGSUMEXP Log-sum-exp function.
%    lse = LOGSUMEXP(x) returns the log-sum-exp function evaluated at
%    the vector x, defined by lse = log(sum(exp(x)).
%    [lse,sm] = LOGSUMEXP(x) also returns the softmax function evaluated
%    at x, defined by sm = exp(x)/sum(exp(x)).
%    The functions are computed in a way that avoids overflow and
%    optimizes numerical stability.
%    Reference:
%    P. Blanchard, D. J. Higham, and N. J. Higham.
%    Accurately computing the log-sum-exp and softmax functions.
%    IMA J. Numer. Anal., Advance access, 2020.

% If no dimension is supplied, choose the first singleton dimension
if (nargin < 2)
    dim = find(size(x) > 1, 1, 'first');
    if isempty(dim)
        dim = 1; % Default to first dimension for scalars
    end
end

xmax = max(x, [], dim);
xmax(xmax == -Inf) = 0;

e = exp(x-xmax);
s = sum(e, dim);

if (nargout > 1)
    sm = e ./ s;
end

s = s-1; % Subtract exp(0) == 1 here to use the more stable log1p
lse = xmax + log1p(s);

end