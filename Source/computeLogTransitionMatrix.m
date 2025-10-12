function [logT, volumeElement] = computeLogTransitionMatrix(X, U, dt, varargin)
%COMPUTELOGTRANSITIONMATRIX Compute the log of the transition matrix that
%discretizes the short time Fokker-Planck drift-diffusion dynamics on a set
%of points subject to a user defined potential. Explicitly, T(i,j) defines
%the probability (NOT probability density) of transitioning from X(j) ->
%X(i). It is assumed that the points themselves are sampled from a
%drift-diffusion process (NOT necessarily the one defined by U!) defined by
%a potential U0. The log transition matrix is nominally just log(T), but
%computed in a numerically stable way that enables exploration of very
%small temperatures/scalar metrics.
%
%   INPUT PARAMETERS:
%
%       - X:    #N x dim set of input points
%
%       - U:    #N x 1 scalar potential defined on the input points that
%               defines the drift-diffusion dynamics
%
%       - dt:   The short time step over which the transition matrix
%               approximates the drift-diffusion dynamics
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('PointPotential', U0 = U): The auxilliary potential defining the
%       equilibrium distribution from which the point set is assumed to be
%       sampled. By default, we assume this is equal to the dynamical
%       potential
%
%       - ('ScalarMetric', scalarMetric = []): The conformal factor of a
%       scalar metric, defined on each input point, that re-scales the
%       dynamical velocity (i.e. v = -(1/scalarMetric) * \nabla U)
%
%       - ('DiffusionCoefficient', D = 1): The diffusion coefficient for
%       the dynamical drift-diffusion process
%
%       - ('PointDiffusionCoefficient', D0 = 1): The diffusion coefficient
%       for the drift-diffusion process assumed to generate the input point
%       set
%
%       - ('ClipThreshold', clipThreshold = 0): Exponential distributions
%       produce insanely small values. Entries |T(i,j)| < this threshold
%       are just set to zero. NOTE: This is a threshold on the elements of
%       T, NOT logT! It is transformed into log-space for the computation.
%       BE CAREFUL HERE - I HAVE NOT TESTED THIS THOROUGHLY YET
%
%       - ('DistanceMatrix', distMatrix = []): #N x #N pairwise distance
%       matrix, i.e. distMatrix(i,j) is the distance between cell i and
%       cell j.
%
%       - ('UseGPU', useGPU = true): Whether or not to perform computations
%       on a GPU
%
%       - ('VolumeElementType', volumeType = 'graphLaplacian'): The type of
%       volume element used to ensure the transition matrix operates on
%       discrete probabilities. Possible types are 'graphLaplacian' or
%       'laplaceBeltrami'.
%
%       - ('VolumeElement', volumeElement = []): A precomputed volume
%       element for each point
%
%       - ('PrecomputeBaseT', precompT = []): #N x #N matrix with elements
%       equal to -distMatrix.^2 ./ (4 * D * dt). This is not really
%       intended for most use cases, but instead to speed up repeated
%       computations when this function is called as a subroutine of a
%       larger method.
%
%       - ('VectorField', vecField = []): #N x dim vector field defined on
%       each point. This is an experimental feature that lets the user
%       build a transition matrix associated to the following theoretical
%       full vector field: -g^{-1) \nabla U + vecField. This allows for the
%       exploration of more general, non-gradient vector fields using this
%       method (e.g. periodic orbits).
%
%   OUTPUT PARAMETERS:
%
%       - logT:             #N x #N (left Markov) log transition matrix
%
%       - volumeElement:    #N x 1 volume element for each point
%
%   by Dillon Cislo 2024/010/29

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if ~isempty(X)
    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'}, ...
        'computeLogTransitionMatrix', 'X');
    numPoints = size(X,1); dim = size(X,2);
else
    numPoints = -1;
end

validateattributes(U, {'numeric'}, {'vector', 'finite', 'real'}, ...
    'computeLogTransitionMatrix', 'U');
if (size(U,2) ~= 1), U = U.'; end
if (numPoints > 0)
    assert(numel(U) == numPoints, 'Scalar potential is improperly sized');
else
    numPoints = numel(U);
end

validateattributes(dt, {'numeric'}, ...
    {'scalar', 'positive', 'finite', 'real'}, ...
    'computeLogTransitionMatrix', 'dt');

% OPTIONAL INPUT PROCESSING -----------------------------------------------

U0 = U;
scalarMetric = [];
D = 1;
D0 = 1;
clipThreshold = 0;
distMatrix = [];
precompT = [];
useGPU = true;
volumeType = 'graphlaplacian';
volumeElement = [];
vecField = [];

allVolumeTypes = {'graphlaplacian', 'laplacebeltrami'};

supportedOptions = {'PointPotential', 'ScalarMetric', ...
    'DiffusionCoefficient', 'PointDiffusionCoefficient', ...
    'ClipThreshold', 'DistanceMatrix', 'UseGPU', 'VolumeElementType', ...
    'VolumeElement', 'PrecomputeBaseT', 'VectorField'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'PointPotential')
        U0 = varargin{i+1};
        validateattributes(U0, {'numeric'}, {'vector', ...
            'finite', 'real', 'numel', numPoints}, ...
            'computeLogTransitionMatrix', 'U0');
        if (size(U0,2) ~= 1), U0 = U0.'; end
    end
    
    if strcmpi(varargin{i}, 'ScalarMetric')
        scalarMetric = varargin{i+1};
    end
    
    if strcmpi(varargin{i}, 'DiffusionCoefficient')
        D = varargin{i+1};
        validateattributes(D, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'}, ...
            'computeLogTransitionMatrix', 'D');
    end
    
    if strcmpi(varargin{i}, 'PointDiffusionCoefficient')
        D0 = varargin{i+1};
        validateattributes(D0, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'}, ...
            'computeLogTransitionMatrix', 'D0');
    end

    if strcmpi(varargin{i}, 'ClipThreshold')
        clipThreshold = varargin{i+1};
        validateattributes(clipThreshold, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'}, ...
            'computeLogTransitionMatrix', 'clipThreshold');
    end
    
    if strcmpi(varargin{i}, 'DistanceMatrix')
        distMatrix = varargin{i+1};
        if ~isempty(distMatrix)
            validateattributes(distMatrix, {'numeric'}, {'2d', ...
                'nonnegative', 'finite', 'real', ...
                'ncols', numPoints, 'nrows', numPoints}, ...
                'computeLogTransitionMatrix', 'distMatrix')
        end
    end
    
    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'}, ...
            'computeLogTransitionMatrix', 'useGPU');
    end

    if strcmpi(varargin{i}, 'VolumeElementType')
        volumeType = lower(varargin{i+1});
        validateattributes(volumeType, {'char'}, {'vector'}, ...
            'computeLogTransitionMatrix', 'volumeType');
        assert(ismember(volumeType, allVolumeTypes), ...
            'Invalid volume element type');
    end

    if strcmpi(varargin{i}, 'VolumeElement')
        volumeElement = varargin{i+1};
        if ~isempty(volumeElement)
            validateattributes(volumeElement, {'numeric'}, ...
                {'vector', 'finite', 'real', 'positive', ...
                'numel', numPoints}, 'computeLogTransitionMatrix', ...
                'volumeElement');
            if (size(volumeElement, 2) ~= 1)
                volumeElement = volumeElement.';
            end
        end
    end

    if strcmpi(varargin{i}, 'PrecomputeBaseT')
        precompT = varargin{i+1};
        if ~isempty(precompT)
            validateattributes(precompT, {'numeric'}, {'2d', ...
                '<=', 0, 'finite', 'real', ...
                'ncols', numPoints, 'nrows', numPoints}, ...
                'computeLogTransitionMatrix', 'precompT')
        end
    end

    if strcmpi(varargin{i}, 'VectorField')
        vecField = varargin{i+1};
        if ~isempty(vecField)
            validateattributes(vecField, {'numeric'}, {'2d', ...
                'nrows', numPoints, 'ncols', dim, 'finite', 'real'}, ...
                'computeLogTransitionMatrix', 'vecField');
        end
    end
    
end

if ~isempty(scalarMetric)

    % If the metric is constant over all space it is faster to simply
    % re-scale U than to perform the full metric calculation
    if isscalar(scalarMetric)

        validateattributes(scalarMetric, {'numeric'}, {'scalar', ...
            'finite', 'positive', 'real'}, ...
            'computeLogTransitionMatrix', 'scalarMetric');

        U = U ./ scalarMetric;
        scalarMetric = [];

    else

        validateattributes(scalarMetric, {'numeric'}, {'vector', ...
            'finite', 'positive', 'real', 'numel', numPoints}, ...
            'computeLogTransitionMatrix', 'scalarMetric');
        if (size(scalarMetric,2) ~= 1)
            scalarMetric = scalarMetric.';
        end

    end

end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

assert(~(isempty(X) && isempty(distMatrix) && isempty(precompT)), ...
    ['You have to supply either a complete input point set or ' ...
    'a distance matrix or a precomputed sum-square-distance matrix ' ...
    '(precompT)']);

%--------------------------------------------------------------------------
% COMPUTE LOG TRANSITION MATRIX
%--------------------------------------------------------------------------
if useGPU

    X = gpuArray(X);
    U = gpuArray(U);
    U0 = gpuArray(U0);
    scalarMetric = gpuArray(scalarMetric);
    
    if ~isempty(precompT), precompT = gpuArray(precompT); end
    if ~isempty(vecField), vecField = gpuArray(vecField); end
    if ~isempty(distMatrix), distMatrix = gpuArray(distMatrix); end
    if ~isempty(volumeElement), volumeElement = gpuArray(volumeElement);end

end

% Compute squared Euclidean distance matrix and add to operator
if ~isempty(precompT)
    
    logT = precompT;
    
elseif ~isempty(distMatrix)

    logT = -distMatrix.^2 ./ (4 * D * dt);

else

    % Fast computation
    % logT = X * X.';
    % logT = -(diag(logT) + diag(logT).' - 2 .* logT) ./ (4 * D * dt);

    % Stable/conservative memory computation
    logT = -pdist2(X, X, 'squaredeuclidean') ./ (4 * D * dt);

end
logT(1:(numPoints+1):numel(logT)) = 0; % Ensure zero diagonal elements

% Add gradient dynamics term to operator
if isempty(scalarMetric)
    
    U = U ./ (2 * D);
    logT = logT + (U.' - U);
    
else
    
    U = U ./ D;
    logT = logT + ((U.' - U) ./ (scalarMetric.' + scalarMetric));
    
end

% Add external forcing vector field terms to operator
if ~isempty(vecField)

    XDiffArr = repmat(permute(X, [1 3 2]), [1 numPoints 1]) - ...
        repmat(permute(X, [3 1 2]), [numPoints 1 1]);
    vecField = repmat(permute(vecField, [3 1 2]), [numPoints 1 1]);

    logT = logT + squeeze(dot(XDiffArr, vecField, 3)) ./ (2 * D);

end

% Handle log volume element
if ~isempty(volumeElement)

    logVol = log(volumeElement);
    if (nargout > 1)
        volumeElement = gather(volumeElement ./ numPoints);
    end

else

    if strcmpi(volumeType, 'graphlaplacian')

        logVol = U0 ./ D0;
        if (nargout > 1), volumeElement = gather(exp(logVol)); end

    elseif strcmpi(volumeType, 'laplacebeltrami')

        affinityOptions = struct();
        affinityOptions.Sigma = dt;
        affinityOptions.NumNeighbors = numPoints;
        affinityOptions.Verbose = false;
        if ~isempty(distMatrix)
            affinityOptions.DistanceMatrix = distMatrix;
        end
        K = affinityMatrix(gather(X), affinityOptions);

        mapOptions = struct();
        mapOptions.Normalization = 'LaplaceBeltrami';
        mapOptions.NumVectors = 0;
        mapOptions.Verbose = false;
        [~, ~, ~, ~, DAlpha] = diffusionMap(K, mapOptions);
        DAlpha = full(DAlpha);
        if useGPU, DAlpha = gpuArray(DAlpha); end

        logVol = log(DAlpha);
        if (nargout > 1), volumeElement = gather(DAlpha); end

    else

        error('Invalid volume element type');

    end

end
logT = logVol + logT;

% Normalize log transition matrix so that the true transition matrix is a
% right Markov matrix
nanIDx = isnan(logT(:));
if any(nanIDx)
    warning('\nLog transition matrix contains NaN prior to normalization');
    logT(nanIDx) = 0;
end

infIDx = isinf(logT(:));
if any(infIDx)
    warning('\nLog transition matrix contains Inf prior to normalization');
    logT(infIDx) = 0;
end

if (clipThreshold > 0), logT(logT(:) < log(clipThreshold)) = 0; end

normLogT = logsumexp(logT, 1);
assert(~any(isinf(normLogT) | isnan(normLogT)), ...
    'Column-wise normalization constant is Inf/NaN');
logT = logT - normLogT;

logT = gather(logT);


end

