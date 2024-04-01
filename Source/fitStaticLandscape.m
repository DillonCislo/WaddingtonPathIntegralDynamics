function [optKLDErr, optFixHeights, optScalarMetric, ...
    optTimeScale, optTimes] = fitStaticLandscape(X, dataProb, ...
    dataTimes, dt, allPaths, varargin)
%FITSTATICLANDSCAPE Fits a single static dynamical landscape to a set of
%input time-series data by minimizing an average (symmetric) K-L divergence
%over each data point and its corresponding simulation point. The degrees
%of freedom to be fit are (1) the heights of potential minima/saddles
%(stored in the same vector) (2) a uniform scalar metric (literally a
%single scalar constant), and (3) an (optional) constant re-scaling of time
%that matches simulation time to physical time. Options are included to fix
%the scalar metric or a (subset) of the minima/saddle heights to user
%specified values.
%
%   INPUT PARAMETERS:
%
%       - X:            #N x dim set of input points on which the dynamics
%                       are defined
%
%       - dataProb:     #N x #DT matrix of measured probability vectors.
%                       Each column in the matrix corresponds to the
%                       probability of being in one of the possible states
%                       at a particular time. A cell array containing
%                       multiple experimental runs can also be supplied.
%                       Each experiment can have a different number of data
%                       points.
%
%       - dataTimes:    1 x #DT vector of physical times at which the
%                       data in 'dataProb' were measured. If multiple data
%                       sets are supplied, the size of the cell array must
%                       match the size of 'dataProb'
%
%       - dt:           The uniform simulation time step. It is assumed
%                       that all simulation runs use the same time step, 
%                       i.e. the times for all simulations are given by
%                       [0 1 2 ... (numSimTimes-1)] * dt
%
%       - allPaths:     #P x 1 cell array. allPaths{i} is an an ordered
%                       et of point IDs defining that path. The end points
%                       of each allPaths{i} correspond to the
%                       minima/saddles in the point set. The ordering of
%                       the values in the output 'fixHeights' is determined
%                       by the sorted, unique IDs of the path end points
%
%   OPTIONAL INPUT ARGUMENTS (Name, Value)-Pairs:
%
%       - ('InitialGuess', initGuess = []); An initial guess for the
%       parameters supplied as a vector of the form:
%       [ (heights); (scalarMetric); (timeScale) ] -OR-
%       [ (heights); (scalarMetric) ] depending on the simulation time
%       handling options. The user is responsible for ensuring the initial
%       guess is feasible given the various imposed constraints
%
%       - ('InitialConditions', initConditions = {}): For each data set
%       where an initial condition is not supplied (i.e. dataTimes{i}(1) ~=
%       0) an initial condition must be supplied. This can either be a full
%       initial probability defined on all points OR a (pointID, probSigma)
%       pair that creates a normalized Gaussian cloud with variance
%       'probSigma' around the specified point in 'X'
%
%       - ('NumSimTimes', numSimTimes = 500): The number of time steps to
%       run each simulation forward
%
%       - ('IsSaddle', isSaddle = []): A logical vector indicating which
%       points in the fixed point height list are saddles. Ordering is
%       determined from the 'allPaths' variable. This field must be
%       specified in order to enforce the saddles.
%
%       - ('EnforceSaddles', enforceSaddles = true): Whether or not to
%       enforce the constraint that index-1 saddles have a greater
%       potential height than the corresponding minima (i.e. prevent
%       saddles from becoming minima)
%
%       - ('ConstHeightSum', constHeightSum = []): A user supplied constant
%       that constrains the total sum of the minima/saddle heights.
%
%       - ('SimTimeHandling', simTimeHandling = 'none'): The method used to
%       constrain the relationship between simulation time and physical
%       time:
%
%           - 'none': No constraints are enforced. Error is computed by
%           matching data points to the simulation time point that globally
%           minimizaes the error. Results may be non-causal.
%
%           - 'causal': Within each supplied data set, Error is computed by
%           matching data points to a set of simulation time points that
%           matches the data chronology (i.e. the temporal ordering of the
%           data points must match the temporal ordering of the simulation
%           points). WARNING: This is currently only implemented in an
%           approximate way. No guarantees are provided that this is the
%           optimal causal matching
%
%           - 'constant': Enforce a constand rescaling of physical time
%           matching it to simulation time, i.e. (simulation time) = 
%           timeScale * (physical time). The variable 'timeScale' only has
%           meaning if this option is selected.
%
%       - ('ConstTimeScale', constTimeScale = []): A user supplied constant
%       time scale that maps simulation time into physical time. If
%       supplied, this property is held fixed over optimization
%
%       - ('ConstFixedHeights', constFixHeights = []): A set of user
%       supplied values for a subset of the minima/saddle heights, supplied
%       a (numFixedPoints) x 1 vector. Non-NaN entries correspond to
%       specified values. If supplied, these fields are held fixed over
%       optimization
%
%       - ('ConstScalarMetric', constScalarMetric = []): The uniform scalar
%       metric, (i.e. the same for all input points) that re-scales the
%       dynamical velocity (i.e. v = -(1/scalarMetric) * \nabla U). If
%       supplied, this property is held fixed over optimization
%
%       - ('OptimizationOptions', optOptions = {}): A cell array containing
%       options that can be supplied to a MATLAB 'optimoptions' object to
%       define solver behavior
%
%   Physical Constants ----------------------------------------------------
%   You are allowed to set these for the sake of completeness, but you are
%   strongly advised to just leave them equal to one
%
%       - ('DiffusionCoefficient', D = 1): The diffusion coefficient for
%       the dynamical drift-diffusion process
%
%       - ('PointDiffusionCoefficient', D0 = 1): The diffusion coefficient
%       for the drift-diffusion process assumed to generate the input point
%       set
%
%   Manifold/Point Set/Potential Properties -------------------------------
%   These fields are computed directly from (X, dt) if they are not
%   supplied
%
%       - ('PointPotential', U0 = []): The auxilliary potential defining
%       the equilibrium distribution from which the point set is assumed to
%       be sampled.
%
%       - ('BasePotential', UB = U0): The base potential that is added to
%       the interpolation potential to build the dynamical potential
%
%       - ('Laplacian', L = []): #N x #N point cloud Laplace-Beltrami
%       operator
%
%       - ('MassMatrix', M = []): #N x #N diagonal mass matrix
%       corresponding to L
%
%   'interpolatePotentialKHarmonic' Options -------------------------------
%
%       - ('TikhonovRegularization', regSigma = 1e-12): A small positive
%       quantity used to make the quadratic problem positive definite.
%
%       - ('RemoveOutliers', removeOutliers = true): Whether or not to
%       remove outliers generated during the interpolation process.
%
%       - ('OutlierThreshold', outlierThreshold = []): The threshold
%       above and below which outliers are removed. If empty and
%       removeOutliers == true, this is automatically set from the values
%       of interpVals
%
%       - ('OutlierNeighbors', outlierNNSize = 10): The number of neighbors
%       over which to average to in order to find the new value for any
%       removed outliers.
%
%       - ('NormalizeMassMatrix', normalizeMassMatrix = true): Whether or
%       not to normalize the mass matrix by its largest value (see
%       'kharmonic.m')
%
%   'interpolateValuesAlongPath' Options ----------------------------------
%
%       - ('PathLengths', allPathLengths = {}): A #P x 1 cell array
%       containing the length of each each edge in the associated path
%       (This doesn't have to just be physical length - you can supply any
%       set of positive weights)
%
%       - ('PathInterpolationMethod', pathInterpMethod = 'weighted'):
%       Whether to interpolate using the path length weights or to just
%       interpolate according to the number of points along the path. If no
%       path lengths are supplied, 'weighted' and 'unweighted' are
%       equivalent
%
%       - ('PathCollisionMethod', pathCollisionMethod = 'mean'): How to
%       handle the case where multiple paths intersect at a subset of the
%       points.
%
%   'computeTransitionMatrix' Options -------------------------------------
%
%       - ('ClipThreshold', clipThreshold = 1e-14): Exponential
%       distributions produce insanely small values. Entries |T(i,j)| <
%       this threshold are just set to zero. BE CAREFUL HERE - I HAVE NOT
%       TESTED THIS THOROUGHLY YET
%
%       - ('StrictNormalization', strictNormalization = true): Whether or
%       not to distribute round-off error in the column-wise normalization.
%
%   General Options -------------------------------------------------------
%
%       - ('UseGPU', useGPU = true): Whether or not to use the GPU to
%       accelerate time evolution
%
%       - ('Verbose', verbose = false): Whether or not to produce verbose
%       progress ouput. The output from the actual optimization routine
%       will be overridden by the corresponding field in 'optOptions' if
%       that field is specified.
%
%   OUTPUT PARAMETERS:
%
%       - optKLDErr:        The average K-L divergence error between the
%                           optimized simulated time courses and their
%                           corresponding data sets
%
%       - optFixHeights:    1 x #H list of optimizes minima/saddle heights.
%                           Ordering in this list is determined from the
%                           'allPaths' input variable
%
%       - optScalarMetric:  The optimized uniform scalar metric
%
%       - optTimeScale:     The optimized time scale (NaN if
%                           'simTimeHandling' is not 'constant')
%
%       - optTimes:         The optimally matched simulation time points
%                           for each data point in each data set. Behavior
%                           is determined the 'simTimeHandling' field.
%
%   by Dillon Cislo 2024/04/01

%==========================================================================
% INPUT PROCESSING
%==========================================================================
validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1); dim = size(X,2);

% This is probably not the most efficient way to format this, but it does
% streamline multiple experiment handling
if ~iscell(dataProb), dataProb = {dataProb}; end
if ~iscell(dataTimes), dataTimes = {dataTimes}; end

% More explicit data handling is performed after optional input handling
validateattributes(dataProb, {'cell'}, {'vector'});
numDataSets = numel(dataProb);
validateattributes(dataTimes, {'cell'}, {'vector', 'numel', numDataSets});

validateattributes(dt, {'numeric'}, {'scalar', 'positive', ...
    'finite', 'real'});

validateattributes(allPaths, {'cell'}, {'vector'});
numPaths = numel(allPaths);
cellfun(@(x) validateattributes(x, {'numeric'}, {'vector', 'integer', ...
    'positive', 'finite', 'real'}), allPaths, 'Uni', false);
assert(all(cellfun(@(x) numel(x) > 1, allPaths, 'Uni', true)), ...
    'Paths must have at least two points');

% fixPointIDx: The unique set of indices in 'X' corresponding to
% minima/saddles
% fixInPathIDx: #Px2 array of indices into 'fixPointIDx' mapping
% minima/saddles back into their corresponding paths
fixInPathIDx = cellfun(@(x) [x(1); x(end)], allPaths, 'Uni', false);
fixInPathIDx = mat2cell(fixInPathIDx);
[fixPointIDx, ~, fixInPathIDx] = unique(reshape(fixInPathIDx.', [], 1));
fixInPathIDx = reshape(fixInPathIDx, [2 numPaths]).';
numFixPoints = numel(fixPointIDx);

%--------------------------------------------------------------------------
% OPTIONAL INPUT PROCESSING
%--------------------------------------------------------------------------

% Optimization options
initGuess = [];
initConditions = cell(numDataSets, 1);
numSimTimes = 500;
isSaddle = false(1, numFixPoints);
enforceSaddles = true;
constHeightSum = [];
simTimeHandling = 'none';
constFixHeights = nan(numFixPoints, 1);
constScalarMetric = [];
optOptions = {};

simTimeHandlingOptions = {'none', 'causal', 'constant'};

% Physical constants
D = 1;
D0 = 1;

% Manifold/point set/potential properties
U0 = [];
UB = [];
L = [];
M = [];

% 'interpolatePotentialKHarmonic' options
regSigma = 1e-12;
removeOutliers = true;
outlierThreshold = [];
outlierNNSize = 10;
normalizeMassMatrix = true;

% 'interpolateValuesAlongPath' options
allPathLengths = {};
pathInterpMethod = 'weighted';
pathCollisionMethod = 'mean';

% 'computeTransitionMatrix' options
clipThreshold = 1e-14;
strictNormalization = true;

% General options
useGPU = true;
verbose = false;

supportedOptions = {'InitialConditions', 'NumSimTimes', 'IsSaddle', ...
    'EnforceSaddles', 'ConstHeightSum', 'SimTimeHandling', ...
    'ConstFixedHeights', 'ConstScalarMetric', 'Verbose', ...
    'OptimizationOptions', 'DiffusionCoefficient', ...
    'PointDiffusionCoefficient', 'PointPotential', 'BasePotential', ...
    'Laplacian', 'MassMatrix', 'TikhonovRegularization', ...
    'RemoveOutliers', 'OutlierThreshold', 'OutlierNeighbors', ...
    'NormalizeMassMatrix', 'PathLengths', 'PathInterpolationMethod', ...
    'PathCollisionMethod', 'ClipThreshold', 'StrictNormalization', ...
    'UseGPU', 'InitialGuess'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end


    % Optimization Options ------------------------------------------------

    if strcmpi(varargin{i}, 'InitialGuess')
        initGuess = varargin{i+1};
        if ~isempty(initGuess)
            validateattributes(initGuess, {'numeric'}, ...
                {'vector', 'finite', 'real'});
            if (size(initGuess, 2) ~= 1), initGuess = initGuess.'; end
        end
    end

    if strcmpi(varargin{i}, 'InitialConditions')
        initConditions = varargin{i+1};
        if isempty(initConditions)
            initConditions = cell(numDataSets, 1);
        else
            validateattributes(initConditions, {'cell'}, ...
                {'vector', 'numel', numDataSets});
        end
    end

    if strcmpi(varargin{i}, 'NumSimTimes')
        numSimTimes = varargin{i+1};
        validateattributes(numSimTimes, {'numeric'}, ...
            {'scalar', 'positive', 'integer', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'IsSaddle')
        isSaddle = varargin{i+1};
        if islogical(isSaddle)
            assert(isvector(isSaddle) && ...
                (numel(isSaddle) == numFixPoints), ['Logical saddle ' ...
                'input is improperly sized']);
            if (size(isSaddle, 2) ~= 1), isSaddle = isSaddle.'; end
        else
            validateattributes(isSaddle, {'numeric'}, ...
                {'vector', 'integer', 'positive', 'finite', 'real', ...
                '<=', numFixedPoints});
            isSaddle = ismember((1:numFixedPoints).', isSaddle);
        end
    end

    if strcmpi(varargin{i}, 'EnforceSaddles')
        enforceSaddles = varargin{i+1};
        validateattributes(enforceSaddles, {'logical'}, {'scalar'});
    end

    if strcmpi(varargin{i}, 'ConstHeightSum')
        constHeightSum = varargin{i+1};
        if ~isempty(constHeightSum)
            validateattributes(constHeightSum, {'numeric'}, ...
                {'scalar', 'finite', 'real'});
        end
    end

    if strcmpi(varargin{i}, 'SimTimeHandling')
        simTimeHandling = lower(varargin{i+1});
        validateattributes(simTimeHandling, {'char'}, {'vector'});
        assert(ismember(simTimeHandling, simTimeHandlingOptions), ...
            'Invalid simulation time handling option supplied');
    end
    
    if strcmpi(varargin{i}, 'ConstFixedHeights')
        constFixHeights = varargin{i+1};
        if ~isempty(constFixHeights)
            assert(isvector(constFixHeights) && ...
                numel(constFixHeights) == numFixedHeights, ...
                'Constrained heights are improperly sized');
        end
    end

    if strcmpi(varargin{i}, 'ConstScalarMetric')
        constScalarMetric = varargin{i+1};
        if ~isempty(constScalarMetric)
            validateattributes(constScalarMetric, {'numeric'}, ...
                {'scalar', 'positive', 'finite', 'real'});
        end
    end

    if strcmpi(varargin{i}, 'OptimizationOptions')
        optOptions = varargin{i};
        validateattribtues(optOptions, {'cell'}, {'vector'});
    end

    % Physical Constants --------------------------------------------------
    
    if strcmpi(varargin{i}, 'DiffusionCoefficient')
        D = varargin{i+1};
        validateattributes(D, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'});
    end
    
    if strcmpi(varargin{i}, 'PointDiffusionCoefficient')
        D0 = varargin{i+1};
        validateattributes(D0, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'});
    end

    % Manifold/Point Set/Potential Properties -----------------------------

    if strcmpi(varargin{i}, 'PointPotential')
        U0 = varargin{i+1};
        if ~isempty(U0)
            validateattributes(U0, {'numeric'}, {'vector', ...
                'finite', 'real', 'numel', numPoints});
            if (size(U0,2) ~= 1), U0 = U0.'; end
        end
    end

    if strcmpi(varargin{i}, 'BasePotential')
        UB = varargin{i+1};
        if ~isempty(UB)
            validateattribtues(UB, {'numeric'}, {'vector', ...
                'finite', 'real', 'numel', numPoints});
            if (size(UB,2) ~= 1), UB = UB.'; end
        end
    end

    if strcmpi(varargin{i}, 'Laplacian'), L = varargin{i+1}; end

    if strcmpi(varargin{i}, 'MassMatrix'), M = varargin{i+1}; end

    % 'interpolatePotentialKHarmonic' Options -----------------------------

    if strcmpi(varargin{i}, 'TikhonovRegularization')
        regSigma = varargin{i+1};
        validateattributes(regSigma, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'RemoveOutliers')
        removeOutliers = varargin{i+1};
        validateattributes(removeOutliers, {'logical'}, {'scalar'});
    end

    if strcmpi(varargin{i}, 'OutlierThreshold')
        outlierThreshold = varargin{i+1};
        if ~isempty(outlierThrehsold)
            validateattribtues(outlierThreshold, {'numeric'}, ...
                {'vector', 'numel', 2, 'finite', 'real'});
            assert(outlierThreshold(2) > outlierThreshold(1), ...
                ['Outlier threshold must have a second element ' ...
                'that is greater than its first element']); 
        end
    end

    if strcmpi(varargin{i}, 'OutlierNeighbors')
        outlierNNSize = varargin{i+1};
        validateattributes(outlierNNSize, {'numeric'}, ...
            {'positive', 'integer', 'scalar', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'NormalizeMassMatrix')
        normalizeMassMatrix = varargin{i+1};
        validateattributes(normalizeMassMatrix, {'logical'}, {'scalar'});
    end

    % 'interpolateValuesAlongPath' Options --------------------------------

    if strcmpi(varargin{i}, 'PathLengths')
        allPathLengths = varargin{i+1};
        validateattributes(allPathLengths, {'cell'}, ...
            {'vector', 'numel', numPaths});
    end

    if strcmpi(varargin{i}, 'PathInterpolationMethod')
        pathInterpMethod = lower(varargin{i+1});
        validateattributes(pathInterpMethod, {'char'}, {'vector'});
    end

    if strcmpi(varargin{i}, 'PathCollisionMethod')
        pathCollisionMethod = lower(varargin{i+1});
        validateattributes(pathCollisionMethod, {'char'}, {'vector'});
    end

    % 'computeTransitionMatrix' Options -----------------------------------

    if strcmpi(varargin{i}, 'ClipThreshold')
        clipThreshold = varargin{i+1};
        validateattribtues(clipThreshold, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'StrictNormalization')
        strictNormalization = varargin{i+1};
        validateattributes(strictNormalization, {'logical'}, {'scalar'});
    end

    % General Options -----------------------------------------------------

    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'});
    end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end

end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

%--------------------------------------------------------------------------
% OPTIMIZATION INPUT PROCESSING
%--------------------------------------------------------------------------

if ~any(isSaddle), enforceSaddles = false; end
numConstHeights = sum(~isnan(constFixHeights));

% Process the supplied initial guess --------------------------------------
if ~isempty(initGuess)

    assert(numel(initGuess) == (numFixedPoints+1), ...
        'Initial guess is improperly sized');

    if ~isempty(constScalarMetric)
        initGuess(end) = constScalarMetric;
    end
    assert(initGuess(end) > 0, ['Scalar metric ' ...
        'must be positive in the initial guess']);

    if (numConstHeights > 0)
        initGuess(~isnan(constFixHeights)) = ...
            constFixHeights(~isnan(constFixHeights));
    end

    if ~isempty(constHeightSum)
        assert(abs(sum(initGuess(1:numFixPoints))-constHeightSum) < 1e-12, ...
            ['User supplied initial guess does not adhere to the ' ...
            'fixed point height sum constraint']);
    end

end

% Process the data points/data times/initial conditions -------------------

numDataPointsPerSet = zeros(1, numDataSets);
for i = 1:numDataSets

    validateattributes(dataProb{i}, {'numeric'}, {'2d', 'finite', ...
        'real', 'nonnegative', 'nrows', numPoints});
    numDataPointsPerSet(i) = size(dataProb{i}, 2);

    if isempty(dataTimes{i})

        dataTimes{i} = 0:(numDataPointsPerSet(i)-1);

    else

        validateattributes(dataTimes{i}, {'numeric'}, {'vector', 'real', ...
            'nonnegative', 'finite', 'numel', numDataPointsPerSet(i)});
        if (size(dataTimes{i}, 1) ~= 1), dataTimes{i} = dataTimes{i}.'; end

        % Sort the input data points so that time is monotonically
        % increasing
        [dataTimes{i}, DI] = sort(dataTimes{i});
        dataProb{i} = dataProb{i}(:, DI);

    end

    assert(isequal(dataTimes{i}, unique(dataTimes{i})), ...
        ['Multiple data points have been supplied with the same ' ...
        'physical time for data set %d'], i);

    if (numDataPointsPerSet(i) == 1)
        assert(dataTimes{i}(1) > 0, ['Only the initial condition ' ...
            'was supplied for data set %d'], i);
    end

    if isempty(initConditions{i})

        assert(dataTimes{i}(1) == 0, ['No initial condition supplied ' ...
            'for data set %d'], i);

        initConditions{i} = dataProb{i}(:,1);

    else

        validateattributes(initConditions{i}, {'numeric'}, {'vector', ...
            'nonnegative', 'finite', 'real'});

        if (numel(initConditions{i}) == 2)

            assert(ismember(initConditions{i}(1), (1:numPoints)), ...
                'Invalid index for initial condition in data set %d', i);
            assert(initConditions{i}(2) > 0, ...
                'Invalid variance for initial condition in data set %d', i);

            curIC = X-repmat(X(initConditions{i}(1), :), [numPoints, 1]);
            curIC = sum(curIC.^2, 2);
            curIC = exp(-curIC ./ (2 * initConditions{i}(2)^2));
            curIC  = curIC ./ (2 * pi * initConditions{i}(2)^2)^(dim/2);
            curIC = exp(U0/D0) .* curIC ./ numPoints;
            curIC = curIC ./ sum(curIC(:));
            initConditions{i} = curIC(:);

        elseif (numel(initConditions{i}) ~= numPoints)

            error(['Initial conditions for data set %d ' ...
                'are improperly sized'], i);
        
        end

    end

end

numTotalDataPoints = sum(numDataPointsPerSet);

%--------------------------------------------------------------------------
% MANIFOLD/POINT SET/POTENTIAL OPTION PROCESSING
%--------------------------------------------------------------------------

% Estimate point set pseudo potential from point cloud
if isempty(U0)
    if verbose, disp('Computing point set potential:'); end
    U0 = -D0 * log(gaussianKDE(X, X, [], sqrt(2 * dt), verbose));
end

if isempty(UB), UB = U0; end

% Validity checks for user-supplied values are performed within
% 'interpolatePotentialKHarmonic'
if (isempty(L) || isempty(M))

    if verbose, fprintf('Building Laplacian/mass matrix... '); end
    [L, M] = diffusionMapLaplacian(X, dt);
    if verbose, fprintf('Done\n'); end

end

%==========================================================================
% BUILD OPTIMIZATION PROBLEM
%==========================================================================

% Build inequality constraints --------------------------------------------

A = []; b = [];

if enforceSaddles

    saddleIDx = find(isSaddle);
    saddleInPath = ismember(fixInPathIDx, saddleIDx);
    assert(all(ismember(sum(saddleInPath, 2), [0 1])), ...
        'Some input paths pair saddles to saddles');

    A = fixInPathIDx(any(saddleInPath, 2), :);
    A(~saddleInPath(:,1), :) = A(~saddleInPath(:,1), [2 1]);

    A = full(sparse( repmat((1:size(A,1)).', [1 2]), A, ...
        [ones(size(A,1), 1), -ones(size(A,1), 1)], ...
        size(A,1), numFixPoints+1 ));

    b = -1e-12 * ones(size(A,1), 1);

end

% Build equality/bound constraints ----------------------------------------

Aeq = []; beq = [];

if (numConstHeights > 0)

    Aeq = full(sparse(1:numConstHeights, find(~isnan(constFixHeights)), ...
        1, numConstHeights, numFixPoints+1));
    beq = reshape(constFixHeights(~isnan(constFixHeights)), [], 1);

end

if ~isempty(constHeightSum)

    Aeq = [Aeq; ones(1, numFixPoints) 0];
    beq = [beq; constHeightSum];

end

if ~isempty(constScalarMetric)

    Aeq = [Aeq; full(sparse(1, numFixPoints+1, 1, 1, numFixPoints+1))];
    beq = [beq; constScalarMetric];

end

lb = [-inf(numFixPoints, 1); 1e-12];
ub = inf(numFixPoints+1, 1);

%==========================================================================
% RUN OPTIMIZATION
%==========================================================================
if strcmpi(simTimeHandling, 'constant')

    optFun = @simulateLandscapeDynamicsConstTimeScale;
else
    optFun = @simulateLandscapeDynamics;
end

options = optimoptions('fmincon', optOptions{:});
if options.useParallel, useGPU = false; end

[optOutput, optKLDErr] = fmincon(optFun, ...
    initGuess, A, b, Aeq, beq, lb, ub, [], options);

optFixHeights = optOutput(1:numFixPoints);
optScalarMetric = optOutput(numFixPoints+1);

%--------------------------------------------------------------------------
% FORMAT OUTPUT
%--------------------------------------------------------------------------

if verbose, fprintf('Consolidating output... '); end

% Convert fixed point height list into path end point values
optEndPointVals = optFixHeights(fixInPathIDx);
[optKnownU, optKnownIDx] = interpolateValuesAlongPath(optEndPointVals, ...
    allPaths, 'PathLengths', allPathLengths, ...
    'InterpolationMethod', pathInterpMethod, ...
    'CollisionMethod', pathCollisionMethod);

% Compute interpolated potential
optUI = interpolatePotentialKHarmonic(X, optKnownU, optKnownIDx, ...
    2, 'TikhonovRegularization', regSigma, 'RemoveOutliers', ...
    removeOutliers, 'OutlierThreshold', outlierThreshold, ...
    'OutlierNeighbors', outlierNNSize, 'Laplacian', L, ...
    'MassMatrix', M, 'TimeStep', dt, ...
    'NormalizeMassMatrix', normalizeMassMatrix);

optU = UB + optUI; % Combine to compute dynamical potential

optT = computeTransitionMatrix(X, optU, dt, ...
    'PointPotential', U0, 'ScalarMetric', optScalarMetric, ...
    'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0, ...
    'ClipThreshold', clipThreshold, ...
    'StrictNormalization', strictNormalization);

optSimProb = cell(numDataSets, 1);
for i = 1:numDataSets
    optSimProb{i} = evolveProbabilities(initConditions{i}, optT, ...
        (numSimTimes-1), 'NumViewTimes', -1, 'TimeStep', dt, ...
        'StrictNormalization', strictNormalization, ...
        'UseGPU', useGPU);
end

if strcmpi(simTimeHandling, 'constant')

    optTimeScale = fitProbSimTimeScale(dataProb, dataTimes, optSimProb, ...
        dt, 'UseGPU', useGPU);

    optTimes = cellfun(@(x) optTimeScale * x, dataTimes, 'Uni', false);
    
else

    optTimeScale = NaN;

    simTimes = (0:(numSimTimes-1)) * dt;
    optTimes = cell(1, numDataSets);
    for i = 1:numDataSets

        optMatchTimes = -ones(numDataPointsPerSet(i), 1);
        for ii = 1:numDataPointsPerSet(i)

            optKLD = dataProb{i}(:, ii) .* ...
                log(dataProb{i}(:, ii) ./ optSimProb{i}) + ...
                optSimProb{i} .* log(optSimProb{i} ./ dataProb{i}(:, ii));
            optKLD(isnan(optKLD)) = 0;
            optKLD = sum(optKLD, 1);

            if strcmpi(simTimeHandling, 'none')

                [~, optMatchTimes(ii)] = min(optKLD);

            elseif strcmpi(simTimeHandling, 'causal')

                if (ii > 1), optKLD(1:optMatchTimes(ii-1)) = inf; end
                [~, optMatchTimes(ii)] = min(optKLD);

            else

                error(['Invalid simulation time handling in ' ...
                    'output consolidation']);

            end

        end

        optTimes{i} = simTimes(optMatchTimes);

    end

end

if verbose, ('Done\n'); end

%**************************************************************************
%**************************************************************************
%                       SIMULATE LANDSCAPE DYNAMICS
%**************************************************************************
%**************************************************************************

    function E = simulateLandscapeDynamics(x)

        scalarMetric = x(end);
        fixHeights = x(1:numFixPoints);

        % Convert fixed point height list into path end point values
        endPointVals = fixHeights(fixInPathIDx);
        [knownU, knownIDx] = interpolateValuesAlongPath(endPointVals, ...
            allPaths, 'PathLengths', allPathLengths, ...
            'InterpolationMethod', pathInterpMethod, ...
            'CollisionMethod', pathCollisionMethod);

        % Compute interpolated potential
        UI = interpolatePotentialKHarmonic(X, knownU, knownIDx, ...
            2, 'TikhonovRegularization', regSigma, 'RemoveOutliers', ...
            removeOutliers, 'OutlierThreshold', outlierThreshold, ...
            'OutlierNeighbors', outlierNNSize, 'Laplacian', L, ...
            'MassMatrix', M, 'TimeStep', dt, ...
            'NormalizeMassMatrix', normalizeMassMatrix);

        U = UB + UI; % Combine to compute dynamical potential

        T = computeTransitionMatrix(X, U, dt, ...
            'PointPotential', U0, 'ScalarMetric', scalarMetric, ...
            'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', clipThreshold, ...
            'StrictNormalization', strictNormalization);

        % NOTE: We perform this operation serially so that gradients can be
        % estimated in parallel, if desired
        E = 0;
        % simTimes = (0:(numSimTimes-1)) * dt;
        for k = 1:numDataSets

            curDataProb = dataProb{k};
            % curDataTimes = dataTimes{k};
            initProb = initConditions{k};

            simProb = evolveProbabilities(initProb, T, (numSimTimes-1), ...
                'NumViewTimes', -1, 'TimeStep', dt, 'UseGPU', useGPU, ...
                'StrictNormalization', strictNormalization);

            matchTimes = -ones(numDataPointsPerSet(k), 1);
            for j = 1:numDataPointsPerSet(k)

                KLD = ...
                    curDataProb(:,j) .* log(curDataProb(:,j) ./ simProb) + ...
                    simProb .* log(simProb ./ curDataProb(:,j));
                KLD(isnan(KLD)) = 0;
                KLD = sum(KLD, 1);

                if strcmpi(simTimeHandling, 'none')
                    
                    E = E + min(KLD);

                elseif strcmpi(simTimeHandling, 'causal')

                    if (j > 1), KLD(1:matchTimes(j-1)) = inf; end
                    [minKLD, matchTimes(j)] = min(KLD);
                    E = E + minKLD;

                else

                    error(['Invalid simulation time handling in ' ...
                        'optimization function']);

                end

            end


        end

        E = E ./ numTotalDataPoints;

    end

%**************************************************************************
%**************************************************************************
%               SIMULATE LANDSCAPE DYNAMICS CONSTANT TIME SCALE
%**************************************************************************
%**************************************************************************

    function E = simulateLandscapeDynamicsConstTimeScale(x)

        scalarMetric = x(end);
        fixHeights = x(1:numFixPoints);

        % Convert fixed point height list into path end point values
        endPointVals = fixHeights(fixInPathIDx);
        [knownU, knownIDx] = interpolateValuesAlongPath(endPointVals, ...
            allPaths, 'PathLengths', allPathLengths, ...
            'InterpolationMethod', pathInterpMethod, ...
            'CollisionMethod', pathCollisionMethod);

        % Compute interpolated potential
        UI = interpolatePotentialKHarmonic(X, knownU, knownIDx, ...
            2, 'TikhonovRegularization', regSigma, 'RemoveOutliers', ...
            removeOutliers, 'OutlierThreshold', outlierThreshold, ...
            'OutlierNeighbors', outlierNNSize, 'Laplacian', L, ...
            'MassMatrix', M, 'TimeStep', dt, ...
            'NormalizeMassMatrix', normalizeMassMatrix);

        U = UB + UI; % Combine to compute dynamical potential

        T = computeTransitionMatrix(X, U, dt, ...
            'PointPotential', U0, 'ScalarMetric', scalarMetric, ...
            'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', clipThreshold, ...
            'StrictNormalization', strictNormalization);

        simProb = cell(numDataSets, 1);
        for k = 1:numDataSets
            simProb{k} = evolveProbabilities(initConditions{k}, T, ...
                (numSimTimes-1), 'NumViewTimes', -1, 'TimeStep', dt, ...
                'StrictNormalization', strictNormalization, ...
                'UseGPU', useGPU);
        end

        [~, E] = fitProbSimTimeScale(dataProb, dataTimes, simProb, dt, ...
            'UseGPU', useGPU);

    end

end