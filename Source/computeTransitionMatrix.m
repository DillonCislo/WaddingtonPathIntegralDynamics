function T = computeTransitionMatrix(X, U, dt, varargin)
%COMPUTETRANSITIONMATRIX Compute the transition matrix that discretizes the
%short time Fokker-Planck drift-diffusion dynamics on a set of points
%subject to a user defined potential. Explicitly, T(i,j) defines the
%probability (NOT probability density) of transitioning from X(j) -> X(i).
%It is assumed that the points themselves are sampled from a
%drift-diffusion process (NOT necessarily the one defined by U!) defined by
%a potential U0. NOTE: You are allowed to set the various diffusion
%coefficients for the sake of generalizability, but we STRONGLY recommend
%you just leave them equal to one.
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
%       - ('ClipThreshold', clipThreshold = 0): Exponential
%       distributions produce insanely small values. Entries |T(i,j)| <
%       this threshold are just set to zero. BE CAREFUL HERE - I HAVE NOT
%       TESTED THIS THOROUGHLY YET
%
%       - ('StrictNormalization', strictNormalization = true): Whether or
%       not to distribute round-off error in the column-wise normalization.
%
%       - ('DistanceMatrix', distMatrix = []): #N x #N pairwise distance
%       matrix, i.e. distMatirx(i,j) is the distance between cell i and
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
%   OUTPUT PARAMETERS:
%
%       - T:    #N x #N (left Markov) transition matrix
%
%   by Dillon Cislo 2024/03/21

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if ~isempty(X)
    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'}, ...
        'computeTransitionMatrix', 'X');
    numPoints = size(X,1); % dim = size(X,2);
else
    numPoints = -1;
end

validateattributes(U, {'numeric'}, {'vector', 'finite', 'real'}, ...
    'computeTransitionMatrix', 'U');
if (size(U,2) ~= 1), U = U.'; end
if (numPoints > 0)
    assert(numel(U) == numPoints, 'Scalar potential is improperly sized');
else
    numPoints = numel(U);
end

validateattributes(dt, {'numeric'}, ...
    {'scalar', 'positive', 'finite', 'real'}, ...
    'computeTransitionMatrix', 'dt');

% OPTIONAL INPUT PROCESSING -----------------------------------------------

U0 = U;
scalarMetric = [];
D = 1;
D0 = 1;
clipThreshold = 0;
strictNormalization = true;
distMatrix = [];
useGPU = true;
volumeType = 'graphlaplacian';
volumeElement = [];

allVolumeTypes = {'graphlaplacian', 'laplacebeltrami'};

supportedOptions = {'PointPotential', 'ScalarMetric', ...
    'DiffusionCoefficient', 'PointDiffusionCoefficient', ...
    'ClipThreshold', 'StrictNormalization', 'DistanceMatrix', ...
    'UseGPU', 'VolumeElementType', 'VolumeElement'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'PointPotential')
        U0 = varargin{i+1};
        validateattributes(U0, {'numeric'}, {'vector', ...
            'finite', 'real', 'numel', numPoints}, ...
            'computeTransitionMatrix', 'U0');
        if (size(U0,2) ~= 1), U0 = U0.'; end
    end
    
    if strcmpi(varargin{i}, 'ScalarMetric')
        scalarMetric = varargin{i+1};
    end
    
    if strcmpi(varargin{i}, 'DiffusionCoefficient')
        D = varargin{i+1};
        validateattributes(D, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'}, ...
            'computeTransitionMatrix', 'D');
    end
    
    if strcmpi(varargin{i}, 'PointDiffusionCoefficient')
        D0 = varargin{i+1};
        validateattributes(D0, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'}, ...
            'computeTransitionMatrix', 'D0');
    end

    if strcmpi(varargin{i}, 'ClipThreshold')
        clipThreshold = varargin{i+1};
        validateattributes(clipThreshold, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'}, ...
            'computeTransitionMatrix', 'clipThreshold');
    end

    if strcmpi(varargin{i}, 'StrictNormalization')
        strictNormalization = varargin{i+1};
        validateattributes(strictNormalization, {'logical'}, {'scalar'}, ...
            'computeTransitionMatrix', 'strictNormalization');
    end
    
    if strcmpi(varargin{i}, 'DistanceMatrix')
        distMatrix = varargin{i+1};
        if ~isempty(distMatrix)
            validateattributes(distMatrix, {'numeric'}, {'2d', ...
                'nonnegative', 'finite', 'real', ...
                'ncols', numPoints, 'nrows', numPoints}, ...
                'computeTransitionMatrix', 'distMatrix')
        end
    end
    
    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'}, ...
            'computeTransitionMatrix', 'useGPU');
    end

    if strcmpi(varargin{i}, 'VolumeElementType')
        volumeType = lower(varargin{i+1});
        validateattributes(volumeType, {'char'}, {'vector'}, ...
            'computeTransitionMatrix', 'volumeType');
        assert(ismember(volumeType, allVolumeTypes), ...
            'Invalid volume element type');
    end

    if strcmpi(varargin{i}, 'VolumeElement')
        volumeElement = varargin{i+1};
        if ~isempty(volumeElement)
            validateattributes(volumeElement, {'numeric'}, ...
                {'vector', 'finite', 'real', 'positive', ...
                'numel', numPoints}, 'computeTransitionMatrix', ...
                'volumeElement');
            if (size(volumeElement, 2) ~= 1)
                volumeElement = volumeElement.';
            end
        end
    end
    
end

if ~isempty(scalarMetric)

    % If the metric is constant over all space it is faster to simply
    % re-scale U than to perform the full metric calculation
    if isscalar(scalarMetric)

        validateattributes(scalarMetric, {'numeric'}, {'scalar', ...
            'finite', 'positive', 'real'}, ...
            'computeTransitionMatrix', 'scalarMetric');

        U = U ./ scalarMetric;
        scalarMetric = [];

    else

        validateattributes(scalarMetric, {'numeric'}, {'vector', ...
            'finite', 'positive', 'real', 'numel', numPoints}, ...
            'computeTransitionMatrix', 'scalarMetric');
        if (size(scalarMetric,2) ~= 1)
            scalarMetric = scalarMetric.';
        end

    end

end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

assert(~(isempty(X) && isempty(distMatrix)), ['You have to supply ' ...
    'either a complete input point set or a distance matrix']);

%--------------------------------------------------------------------------
% COMPUTE TRANSITION MATRIX
%--------------------------------------------------------------------------
if useGPU

    X = gpuArray(X);
    U = gpuArray(U);
    U0 = gpuArray(U0);
    distMatrix = gpuArray(distMatrix);
    scalarMetric = gpuArray(scalarMetric);

    if ~isempty(volumeElement)
        volumeElement = gpuArray(volumeElement);
    end

end

% Compute squared Euclidean distance matrix and add to operator
if isempty(distMatrix)

    % Fast computation
    % T = X * X.';
    % T = -(diag(T) + diag(T).' - 2 .* T) ./ (4 * D * dt);

    % Stable/conservative memory computation
    T = -pdist2(X, X, 'squaredeuclidean') ./ (4 * D * dt);

else

    T = -distMatrix.^2 ./ (4 * D * dt);

end
T(1:(numPoints+1):numel(T)) = 0; % Ensure zero diagonal elements

% Add gradient dynamics term to operator
if isempty(scalarMetric)
    
    U = U ./ (2 * D);
    T = T + (U.' - U);
    
else
    
    U = U ./ D;
    T = T + ((U.' - U) ./ (scalarMetric.' + scalarMetric));
    
end

T = exp(T);

% Handle volume element
if ~isempty(volumeElement)

    T = volumeElement .* T;

else

    if strcmpi(volumeType, 'graphlaplacian')

        T = exp(U0 ./ D0) .* T;

    elseif strcmpi(volumeType, 'laplacebeltrami')

        affinityOptions = struct();
        affinityOptions.Sigma = dt;
        affinityOptions.NumNeighbors = numPoints;
        affinityOptions.Verbose = false;
        K = affinityMatrix(gather(X), affinityOptions);

        mapOptions = struct();
        mapOptions.Normalization = 'LaplaceBeltrami';
        mapOptions.NumVectors = 0;
        mapOptions.Verbose = false;
        [~, ~, ~, ~, DAlpha] = diffusionMap(K, mapOptions);
        DAlpha = full(DAlpha);
        if useGPU, DAlpha = gpuArray(DAlpha); end

        T = DAlpha .* T;

    else

        error('Invalid volume element type');

    end

end

% Normalize transition matrix to be a right Markov matrix
nanIDx = isnan(T(:));
if any(nanIDx)
    warning('\nTransition matrix contains NaN prior to normalization');
    T(nanIDx) = 0;
end

infIDx = isinf(T(:));
if any(infIDx)
    warning('\nTransition matrix contains Inf prior to normalization');
    T(infIDx) = 0;
end

if (clipThreshold > 0), T(T(:) < clipThreshold) = 0; end

normT = sum(T,1);
assert(~any(normT == 0), 'Column-wise normalization constant equals 0');
T = T ./ normT;

if strictNormalization
    normT = sum(T,1);
    for i = 1:size(T, 2)
        if (normT(i) > 1)
            [~, maxID] = max(T(:, i));
            T(maxID, i) = T(maxID, i) + (1 - normT(i));
        end
    end
end

assert(~any(T(:) < 0), 'Negative transition probabilities on output');

T = gather(T);
    
end
