function D = sinkhornDistance(r, C, varargin)
%SINKHORNDISTANCE Fast computation of the Sinkhorn distance, i.e. entropy
%regularized Earth Mover's Distance (EMD), following 'Sinkhorn Distances:
%Lightspeed Computation of Optimal Transport' by Marco Cuturi (2013). This
%method assumes that the probability signatures are defined on the same
%input point sets. Technically, this method computes the dual Sinkhorn
%distance.
%
%   INPUT PARAMETERS:
%
%       - r:        numPts x 1 probability vector.
%
%       - C:        numPts x numDists matrix, where each column C(:,j) is a
%                   probability vector.
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)- Pairs:
%
%       - ('Lambda', lambda = -1): A positive semi-definite Lagrange
%       multiplier enforcing the entropy constraint of Sinkhorn distances.
%       As lambda -> 0 we recover the fully independent transport policy (r
%       * c.'). As lambda -> infinity we approach the true EMD. Smaller
%       lambda typically results in faster compute times. Typically, for
%       lambda > 50 we recover values close to the EMD. If a non-valid
%       lambda is supplied we automatically pick one using a heuristic from
%       Cuturi (2013).
%
%       - ('CostMatrix', M = []): A numPts x numPts cost matrix where
%       M(i,j) is the cost of mapping r to C(:,j). For approximation to the
%       classic EMD this should be Euclidean distances between
%       points.
%
%       - ('PointLocations', X = []): A numPts x dim matrix containing the
%       locations of the points on which the probability signatures are
%       defined. This is only needed if the cost matrix M is undefined.
%
%       - ('CostType', costType = 'euclidean'): The type of distance
%       used to assemble the cost matrix if M is not supplied by the user.
%       See 'pdist2' documentation for more options.
%
%       - ('MaxIterations', maxIter = 1000): The maximum number of
%       optimization iterations. Optimization terminates when the iteration
%       count exceeds this number.
%
%       - ('MaxDelta', maxDelta = 1e-4): The maximum fractional change
%       of any distance between optimization iterations below which
%       optimization stops. Explicitly, optimization terminates when
%       ||d_t/d_{t-1}-1||_{\infty} < maxDelta.
%
%       - ('Stability', stability = 'stable'): Whether to perform a stable
%       computation using logsumexp, a semi-stable computation in the log
%       domain, or an unstable computation. The stable computation is much
%       slower, but the unstable computation produces almost unusable
%       result for a broad range of parameters. Right now the stable
%       computation only supports vector C input.
%
%       - ('useGPU', useGPU = true): Whether to perform the computation
%       on the GPU.
%
%       - ('Verbose', verbose = false): Whether to produce verbose output.
%
%   OUTPUT PARAMETERS:
%
%       - D:        1 x numDists row vector, where D(j) is the Sinkhorm
%                   distace between r and C(:,j)
%
%   by Dillon Cislo 2024/08/19

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(r, {'numeric'}, {'vector', 'finite', 'real', ...
    'nonnegative'}, 'sinkhornDistance', 'r');
if (size(r,2) ~= 1), r = r.'; end
numPoints = numel(r);
if (abs(1-sum(r)) > 1e-12)
warning('r is not properly normalized');
end

validateattributes(C, {'numeric'}, {'2d', 'finite', 'real', ...
    'nonnegative'}, 'sinkhornDistance', 'C');
if (size(C,1) ~= numPoints)
    C = C.';
    assert(size(C,1) == numPoints, 'C matrix is improperly sized');
end
numDists = size(C,2);
if any(abs(1-sum(C,1)) > 1e-12)
    warning('C is not properly normalized');
end

% OPTIONAL INPUT PROCESSING -----------------------------------------------
lambda = -1;
M = [];
X = [];
costType = 'euclidean';
maxIter = 1000;
maxDelta = 1e-4;
stability = 'stable';
useGPU = true;
verbose = false;

supportedOptions = {'Lambda', 'CostMatrix', 'PointLocations', ...
    'CostType', 'MaxIterations', 'MaxDelta', 'UseGPU', 'Verbose', ...
    'Stability'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:numel(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'Lambda')
        lambda = varargin{i+1};
        if ~isempty(lambda) 
            validateattributes(lambda, {'numeric'}, {'scalar', ...
                'nonnegative', 'finite', 'real'}, ...
                'sinkhornDistance', 'lambda');
        end
    end

    if strcmpi(varargin{i}, 'CostMatrix')
        M = varargin{i+1};
        if ~isempty(M)
            validateattributes(M, {'numeric'}, {'2d', 'finite', ...
                'nonnegative', 'real', 'nrows', numPoints, ...
                'ncols', numPoints}, 'sinkhornDistance', 'M');
        end
    end

    if strcmpi(varargin{i}, 'PointLocations')
        X = varargin{i+1};
        if ~isempty(X)
            validateattributes(X, {'numeric'}, {'2d', 'finite', ...
                'real', 'nrows', numPoints}, 'sinkhornDistance', 'X');
        end
    end

    if strcmpi(varargin{i}, 'CostType')
        costType = varargin{i+1};
        validateattribtues(costType, {'char'}, {'vector'}, ...
            'sinkhornDistance', 'costType');
    end

    if strcmpi(varargin{i}, 'MaxIterations')
        maxIter = varargin{i+1};
        validateattributes(maxIter, {'numeric'}, {'scalar', ...
            'positive', 'nonnan', 'real', 'integer'}, ...
            'sinkhornDistance', 'maxIter');
    end

    if strcmpi(varargin{i}, 'MaxDelta')
        maxDelta = varargin{i+1};
        validateattributes(maxDelta, {'numeric'}, {'scalar', ...
            'positive', 'nonnan', 'real'},'sinkhornDistance', 'maxDelta');
    end

    if strcmpi(varargin{i}, 'Stability')
        stability = varargin{i+1};
        validateattributes(stability, {'char'}, {'vector'}, ...
            'sinkhornDistance', 'stability');
    end

    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'}, ...
            'sinkhornDistance', 'UseGPU');
    end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'}, ...
            'sinkhornDistance', 'verbose');
    end

end

if isempty(M)
    assert(~isempty(X), ['You must supply either a cost matrix ' ...
        'or a set of point locations']);
    M = pdist2(X, X, costType);
end

if (isempty(lambda) || (lambda < 0))
    lambda = 10 / median(M(:));
end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

if (strcmpi(stability, 'Stable') && (size(C,2) > 1))
    stability = 'unstable';
    if verbose
        warning(['Stable computation only supports vector C input. ' ...
            'Switching to unstable.']);
    end
end

if strcmpi(stability, 'Stable')

    %----------------------------------------------------------------------
    % COMPUTE SINKHORN DISTANCES (STABLE COMPUTATION)
    %----------------------------------------------------------------------

    incIDx = (r > 0);
    r = r(incIDx);
    M = M(incIDx, :);
    u = ones(numel(r), numDists) ./ numel(r);

    if useGPU

        r = gpuArray(r);
        C = gpuArray(C);
        M = gpuArray(M);
        u = gpuArray(u);
        lambda = gpuArray(lambda);

    end

    mlM = -lambda .* M;
    mlMtilde = mlM - log(r);
    logC = permute(log(C), [3 1 2]);
    logu = permute(log(u), [1 3 2]);
    logv = logC - logsumexp(mlM + logu, 1);

    curIter = 0;
    while true

        prevLogU = logu;
        prevLogV = logv;

        logu = -logsumexp(mlMtilde + logv, 2);
        logv = logC - logsumexp(mlM + logu, 1);

        curIter = curIter + 1;
        if (curIter >= maxIter)
            if verbose
                disp(['Distance computation terminated: max iteration ' ...
                    'count exceeded']);
            end
            break;
        end

        curDelta = max(abs(logu ./ prevLogU)-1);
        if (curDelta < maxDelta)
            if verbose
                disp(['Distance computation terminated: insufficient ' ...
                    'change']);
            end
            break;
        end

        if any(isnan(logu(:))) || any(isnan(logv(:)))
            logu = prevLogU;
            logv = prevLogV;
            if verbose
                disp('Distance computation terminated: NaN values detected');
            end
            break;
        end

    end

    u = squeeze(exp(logu));
    v = squeeze(exp(logv));
    if (size(v,1) == 1), v = v.'; end

    D = gather(sum(u .* ((exp(mlM) .* M) * v)));

elseif strcmpi(stability, 'semi-stable')

    D = NaN;
    error('Semi-stable computation is not yet implemented');

elseif strcmpi(stability, 'unstable')

    %----------------------------------------------------------------------
    % COMPUTE SINKHORN DISTANCES (UNSTABLE)
    %----------------------------------------------------------------------

    incIDx = (r > 0);
    r = r(incIDx).';
    M = M(incIDx, :);
    K = exp(-lambda .* M);
    u = ones(numel(r), numDists) ./ numel(r);
    Ktilde = bsxfun(@rdivide, K, r); % == diag(1 ./ r) * K

    D = nan(1, numDists);
    if useGPU

        C = gpuArray(C);
        M = gpuArray(M);
        K = gpuArray(K);
        u = gpuArray(u);
        Ktilde = gpuArray(Ktilde);
        D = gpuArray(D);

    end

    KM = K .* M;
    KT = K.';
    v = C ./ (KT * u);

    curIter = 0;
    while true

        prevD = D;

        u = 1 ./ (Ktilde * v);
        v = C ./ (KT * u);
        D = sum(u .* (KM * v));

        curIter = curIter + 1;
        if (curIter >= maxIter)
            if verbose
                disp(['Distance computation terminated: max iteration ' ...
                    'count exceeded']);
            end
            break;
        end

        curDelta = max(abs(D ./ prevD)-1);
        if (curDelta < maxDelta)
            if verbose
                disp(['Distance computation terminated: insufficient ' ...
                    'change']);
            end
            break;
        end

    end

    D = gather(D);

else

    error('Invalid stability option');

end

end

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
e = exp(x - xmax);
s = sum(e, dim);

if (nargout > 1)
    sm = e ./ s;
end

s = s-1; % Subtract exp(0) == 1 here to use the more stable log1p
lse = xmax + log1p(s);

end

