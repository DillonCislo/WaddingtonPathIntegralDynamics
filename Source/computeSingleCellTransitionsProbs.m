function [T, volumeElement] = computeSingleCellTransitionsProbs( ...
    X, U, dt, initIDx, varargin)
%COMPUTESINGLECELLTRANSITIONSPROBS Computes the single step transition
%probabilities for a set of cells that live on the same discrete dynamical
%manifold. In short, suppose that each cell moves on the manifold subject
%to a dynamical landscape. For each cell, one can define a transition
%matrix that discretizes the short time Fokker-Planck drift-diffusion
%dynamics on a set of points. Explicitly, T(i,j|U_c) defines the
%probability (NOT probability density) for cell c to transition from
%X(j) -> X(i). It is wasteful to construct this whole transition matrix if
%all we want to do is run a bunch of Markov chain simulation for cells
%bouncing around the points in the manifold. This function basically just
%constructs the single column of T(i,j|U_c) that enables us to efficiently
%update each cell according to its own dynamical landscape. See also
%'computeTransitionMatrix' to get a sense of how the whole matrix is built.
%It is assumed that the points comprising the manifold are sampled from a
%drift-diffusion process (NOT necessarily the one defined by U!) defined by
%a potential U0. 
%
%   INPUT PARAMETERS:
%
%       - X:        #N x dim set of input points. This must be the same
%                   for all cells.
%
%       - U:        #N x #C scalar potentials for each cell defined on the
%                   input points that defines the drift-diffusion dynamics.
%
%       - dt:       The short time step over which the transition matrix
%                   approximates the drift-diffusion dynamics. This must
%                   be the same for all cells.
%
%       - initIDx:  #C x 1 vector of indices into X definition the current
%                   position of each cell on the discrete manifold (i.e.
%                   the fixed column from the per-cell transition matrix
%                   that will be constructed).
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('PointPotential', U0 = []): The #N x 1 auxilliary potential
%       defining the equilibrium distribution from which the point set is
%       assumed to be sampled. This must be the same for all cells. Either
%       a point potential, a volume element must be supplied, or the volume
%       element type must be set to 'LaplaceBeltrami'.
%
%       - ('ScalarMetric', scalarMetric = []): The conformal factor of a
%       scalar metric, defined on each input point, that re-scales the
%       dynamical velocity (i.e. v = -(1/scalarMetric) * \nabla U). This
%       can be empty (identity metric assumed), a #C x 1 vector of scalars
%       per cell, or an #N x #C vector per cell.
%
%       - ('DiffusionCoefficient', D = 1): The diffusion coefficient for
%       the dynamical drift-diffusion process for each cell. This can be a
%       single scalar (for all cells) or a 1 x #C row vector of scalars per
%       cell.
%
%       - ('PointDiffusionCoefficient', D0 = 1): The diffusion coefficient
%       for the drift-diffusion process assumed to generate the input point
%       set. This must be the same for all cells.
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
%       matrix, i.e. distMatrix(i,j) is the distance between manifold point
%       i and manifold point j.
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
%       element for each point. This must be the same for all cells. Either
%       a point potential, a volume element must be supplied, or the volume
%       element type must be set to 'LaplaceBeltrami'.
%
%       - ('PrecomputeBaseT', precompT = []): #N x #C matrix with elements
%       equal to -||x_i - x_c||.^2 ./ (4 * D(c) * dt). Notice that we
%       expect the per-cell diffusion constant to already be correctly
%       applied. This is not really intended for most use cases, but
%       instead to speed up repeated computations when this function is
%       called as a subroutine of a larger method.
%
%       - ('VectorField', vecField = []): #N x dim x #C array of vector
%       fields defined on each point in the manifold for each cell. This is
%       an experimental feature that lets the user build a transition
%       matrix associated to the following theoretical full vector field:
%       -g^{-1) \nabla U + vecField. This allows for the exploration of
%       more general, non-gradient vector fields using this method (e.g.
%       periodic orbits).
%
%   OUTPUT PARAMETERS:
%
%       - T:                #N x #C array of transition probabilities. Each
%                           column is normalized to 1. T(i,j) is the
%                           probability for the jth cell to transition to
%                           the ith point in the manifold.
%
%       - volumeElement:    #N x 1 volume element for each point.
%
%   by Dillon Cislo 2026/01/16

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if ~isempty(X)
    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'}, ...
        'computeSingleCellTransitionsProbs', 'X');
    numPoints = size(X,1); dim = size(X,2);
else
    numPoints = -1;
end

validateattributes(U, {'numeric'}, {'2d', 'finite', 'real'}, ...
    'computeSingleCellTransitionsProbs', 'U');
if (numPoints > 0)
    assert(size(U,1) == numPoints, 'Scalar potential is improperly sized');
else
    numPoints = numel(U);
end
numCells = size(U,2);

validateattributes(dt, {'numeric'}, ...
    {'scalar', 'positive', 'finite', 'real'}, ...
    'computeSingleCellTransitionsProbs', 'dt');

validateattributes(initIDx, {'numeric'}, ...
    {'vector', 'positive', 'finite', 'real', 'integer', ...
    'numel', numCells, '<=', numPoints}, ...
    'computeSingleCellTransitionsProbs', 'initIDx');
if (size(initIDx,2) ~= 1), initIDx = initIDx.'; end

% Convert into a row vector of linear indices into a #N x #C matrix. In
% other words if initIDx(j) = i is the index of the initial location of
% cell j in the manifold, then potIDx is a 1 x #C vector which grabs the
% (i,j)th value from a #N x #C matrix (i.e. the potential array)
potIDx = sub2ind([numPoints, numCells], initIDx, (1:numCells).').';

% OPTIONAL INPUT PROCESSING -----------------------------------------------

U0 = [];
scalarMetric = [];
D = ones(1, numCells);
D0 = 1;
clipThreshold = 0;
strictNormalization = true;
distMatrix = [];
precompT = [];
useGPU = true;
volumeType = 'graphlaplacian';
volumeElement = [];
vecField = [];

allVolumeTypes = {'graphlaplacian', 'laplacebeltrami'};

supportedOptions = {'PointPotential', 'ScalarMetric', ...
    'DiffusionCoefficient', 'PointDiffusionCoefficient', ...
    'ClipThreshold', 'StrictNormalization', 'DistanceMatrix', ...
    'UseGPU', 'VolumeElementType', 'VolumeElement', 'PrecomputeBaseT', ...
    'VectorField'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'PointPotential')
        U0 = varargin{i+1};
        validateattributes(U0, {'numeric'}, {'vector', ...
            'finite', 'real', 'numel', numPoints}, ...
            'computeSingleCellTransitionsProbs', 'U0');
        if (size(U0,2) ~= 1), U0 = U0.'; end
    end
    
    if strcmpi(varargin{i}, 'ScalarMetric')
        scalarMetric = varargin{i+1};
    end
    
    if strcmpi(varargin{i}, 'DiffusionCoefficient')
        D = varargin{i+1};
        validateattributes(D, {'numeric'}, ...
            {'positive', 'finite', 'real'}, ...
            'computeSingleCellTransitionsProbs', 'D');
        if isvector(D)
            assert(numel(D) == numCells, ['Diffusion coefficient ' ...
                'vector is improperly sized']);
            % Ensure D is a row vector
            if (size(D,1) ~= 1), D = D.'; end
        else
            assert(isscalar(D), ['Diffusion coefficient must be ' ...
                'supplied as a vector or a scalar']);
            % Just convert this to a vector for unified handling
            D = repmat(D, [1 numCells]);
        end
    end
    
    if strcmpi(varargin{i}, 'PointDiffusionCoefficient')
        D0 = varargin{i+1};
        validateattributes(D0, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'}, ...
            'computeSingleCellTransitionsProbs', 'D0');
    end

    if strcmpi(varargin{i}, 'ClipThreshold')
        clipThreshold = varargin{i+1};
        validateattributes(clipThreshold, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'}, ...
            'computeSingleCellTransitionsProbs', 'clipThreshold');
    end

    if strcmpi(varargin{i}, 'StrictNormalization')
        strictNormalization = varargin{i+1};
        validateattributes(strictNormalization, {'logical'}, {'scalar'}, ...
            'computeSingleCellTransitionsProbs', 'strictNormalization');
    end
    
    if strcmpi(varargin{i}, 'DistanceMatrix')
        distMatrix = varargin{i+1};
        if ~isempty(distMatrix)
            validateattributes(distMatrix, {'numeric'}, {'2d', ...
                'nonnegative', 'finite', 'real', ...
                'ncols', numPoints, 'nrows', numPoints}, ...
                'computeSingleCellTransitionsProbs', 'distMatrix')
        end
    end
    
    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'}, ...
            'computeSingleCellTransitionsProbs', 'useGPU');
    end

    if strcmpi(varargin{i}, 'VolumeElementType')
        volumeType = lower(varargin{i+1});
        validateattributes(volumeType, {'char'}, {'vector'}, ...
            'computeSingleCellTransitionsProbs', 'volumeType');
        assert(ismember(volumeType, allVolumeTypes), ...
            'Invalid volume element type');
    end

    if strcmpi(varargin{i}, 'VolumeElement')
        volumeElement = varargin{i+1};
        if ~isempty(volumeElement)
            validateattributes(volumeElement, {'numeric'}, ...
                {'vector', 'finite', 'real', 'positive', ...
                'numel', numPoints}, 'computeSingleCellTransitionsProbs', ...
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
                'ncols', numCells, 'nrows', numPoints}, ...
                'computeSingleCellTransitionsProbs', 'precompT')
        end
    end

    if strcmpi(varargin{i}, 'VectorField')
        vecField = varargin{i+1};
        if ~isempty(vecField)
            validateattributes(vecField, {'numeric'}, { ...
                'nrows', numPoints, 'ncols', dim, 'finite', 'real'}, ...
                'computeSingleCellTransitionsProbs', 'vecField');
            if ndims(vecField) == 3
                assert(isequal(size(vecField), [numPoints, dim, numCells]), ...
                    '3D vector field array is improperly sized');
            elseif ismatrix(vecField)
                assert(isequal(size(vecField), [numPoints, dim]), ...
                    '2D vector field array is improperly sized');
                vecField = repmat(vecField, [1 1 numCells]);
            else
                error('Vector field has incorrect dimensions');
            end
        end
    end
    
end

if ~isempty(scalarMetric)

    % If the metric is constant over all space it is faster to simply
    % re-scale U than to perform the full metric calculation
    if isscalar(scalarMetric)
        
        % Validate the case where there is one scalar metric for all cells
        validateattributes(scalarMetric, {'numeric'}, {'scalar', ...
            'finite', 'positive', 'real'}, ...
            'computeSingleCellTransitionsProbs', 'scalarMetric');

        U = U ./ scalarMetric;
        scalarMetric = [];

    elseif isvector(scalarMetric)

        validateattributes(scalarMetric, {'numeric'}, {'vector', ...
            'finite', 'positive', 'real'}, ...
            'computeSingleCellTransitionsProbs', 'scalarMetric');
        if (size(scalarMetric,2) ~= 1)
            scalarMetric = scalarMetric.';
        end

        if numel(scalarMetric) == numCells
            % Assume this means that a #C x 1 vector of scalars per cell
            % was supplied. Rescale per-cell potentials.
            U = U ./ repmat(scalarMetric.', [numPoints, 1]);
            scalarMetric = [];

        elseif numel(scalarMetric) == numPoints
            % This means that we have a single conformal metric for a
            % single cell
            assert(numCells == 1, ...
                'Conformal metric vector is improperly sized.')

        end

    else

        % This is a fully general unique conformal metric fo each
        % individual simulated cell
        validateattributes(scalarMetric, {'numeric'}, {'2d', ...
            'finite', 'positive', 'real', 'nrows', numPoints, ...
            'ncols', numCells}, 'computeSingleCellTransitionsProbs', ...
            'scalarMetric');
        
    end

end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

if (isempty(U0) && isempty(volumeElement))
    assert(strcmpi(volumeType, 'laplacebeltrami'), ['If neither a point ' ...
        'potential nor a volume element are supplied, the volume ' ...
        'element type must be ''LaplaceBeltrami''.']);
end

assert(~(isempty(X) && isempty(distMatrix) && isempty(precompT)), ...
    ['You have to supply either a complete input point set or ' ...
    'a distance matrix or a precomputed sum-square-distance matrix ' ...
    '(precompT)']);

% Explicitly resize the diffusion coefficient to #N x #C
D = repmat(D, [numPoints, 1]);

%--------------------------------------------------------------------------
% COMPUTE TRANSITION MATRIX
%--------------------------------------------------------------------------
if useGPU

    X = gpuArray(X);
    U = gpuArray(U);
    D = gpuArray(D);

    if ~isempty(U0), U0 = gpuArray(U0); end
    if ~isempty(scalarMetric), scalarMetric = gpuArray(scalarMetric); end
    if ~isempty(precompT), precompT = gpuArray(precompT); end
    if ~isempty(vecField), vecField = gpuArray(vecField); end
    if ~isempty(distMatrix), distMatrix = gpuArray(distMatrix); end
    if ~isempty(volumeElement), volumeElement = gpuArray(volumeElement);end

end

% Compute squared Euclidean distance matrix and add to operator
if ~isempty(precompT)

    T = precompT; % #N x #C
    T(potIDx) = 0; % Ensure distance to current point is zero

else

    if ~isempty(distMatrix)

        T = -distMatrix.^2 ./ (4 * dt); % #N x #N

    else

        % Fast computation
        % T = X * X.';
        % T = -(diag(T) + diag(T).' - 2 .* T) ./ (4 * dt); % #N x #N

        % Stable/conservative memory computation
        T = -pdist2(X, X, 'squaredeuclidean') ./ (4 * dt); % #N x #N

    end

    T(1:(numPoints+1):numel(T)) = 0; % Ensure zero diagonal elements
    T = T(:, initIDx); % Reslice to #N x #C

    % Divide by diffusion coefficient
    T = T ./ D;

end

% Add gradient dynamics term to operator
if isempty(scalarMetric)
    
    % This branch handles all cases with uniform scalar metric
    U = U ./ (2 * D); % #N x #C
    initU = repmat(U(potIDx), [numPoints, 1]); % #N x #C
    T = T + (initU - U);
    
else
    
    % Metric is #N x #C conformal metric
    initMetric = repmat(scalarMetric(potIDx), [numPoints, 1]); % #N x #C
    U = U ./ D; % #N x #C
    initU = repmat(U(potIDx), [numPoints, 1]); % #N x #C
    T = T + ((initU - U) ./ (initMetric + scalarMetric));
    
end

% Add external forcing vector field terms to operator
if ~isempty(vecField)

    error('Arbitrary vector field handling is not yet implemented.')

    % % This works for a single #N x dim vector field
    % XDiffArr = repmat(permute(X, [1 3 2]), [1 numPoints 1]) - ...
    %     repmat(permute(X, [3 1 2]), [numPoints 1 1]);
    % vecField = repmat(permute(vecField, [3 1 2]), [numPoints 1 1]);
    % 
    % T = T + squeeze(dot(XDiffArr, vecField, 3)) ./ (2 * D);

end

T = exp(T);

% Handle volume element
if ~isempty(volumeElement)

    T = volumeElement .* T;
    if (nargout > 1), volumeElement = gather(volumeElement); end

else

    if strcmpi(volumeType, 'graphlaplacian')
        
        assert(~isempty(U0), ['Point potential must be supplied ' ...
            'to use the GraphLaplacian volume element type.']);

        T = exp(U0 ./ D0) .* T;
        if (nargout > 1)
            volumeElement = gather(exp(U0 ./ D0) ./ numPoints);
        end

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

        T = DAlpha .* T;
        if (nargout > 1), volumeElement = gather(DAlpha); end

    else

        error('Invalid volume element type');

    end

end

% Normalize column sums of T to one
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

