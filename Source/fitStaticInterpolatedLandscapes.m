function [optErr, optFixHeights0, optScalarMetric0, ...
    optFixHeights1, optScalarMetric1,optInterpParam, optTimeScales, ...
    optTimes, optOutput] = fitStaticInterpolatedLandscapes( ...
    X, dataProb, dataTimes, dt, allPaths, varargin)
%FITSTATICINTERPOLATEDLANDSCAPES Simultaneously fits three static dynamical
%landscapes to a set of input time-series data on a shared manifold. For
%the two "boundary" landscapes, the degrees of freedom to be fit are (1)
%the heights of potential minima/saddles (stored in the same vector) (2) a
%uniform scalar metric (literally a single scalar constant), and (3) an
%(optional) constant re-scaling of time that matches simulation time to
%physical time. The third landscape is produced by linearly interpolating
%between the "boundary" landscapes. Explicitly we have:
%
%   (U_b + U_int) = (1-c) * (U_b + U_i0) / g0 + c * (U_b + U_i1) / g1
%
%for some parameter 0 <= c <= 1. The inclusion of an explicit base
%potential in the dynamical potential for the interpolated dynamics is made
%for compatibility with underlying functions. Note that we absorb the
%scalar metric for the interpolated dynamics into the dynamical potential.
%Options are included to fix the scalar metrics, a subset of the
%minima/saddle heights, or the interpolation parameter to user specified
%values.
%
%   INPUT PARAMETERS:
%
%       - X:            #N x dim set of input points on which the dynamics
%                       are defined
%
%       - dataProb:     A three element cell array, corresponding to the
%                       dynamics for landscape 0, the interpolated
%                       landscape, and landscape 1, respectively. Each 
%                       element is an #N x #DT matrix of measured
%                       probability vectors. Each column in the matrix
%                       corresponds to the probability of being in one of
%                       the possible states at a particular time. A cell
%                       array containing multiple experimental runs can
%                       also be supplied. Each experiment can have a
%                       different number of data points.
%
%       - dataTimes:    A three element cell array, corresponding to the
%                       dynamics for landscape 0, the interpolated
%                       landscape, and landscape 1, respectively. Each
%                       element is a 1 x #DT vector of physical times at
%                       which the data in 'dataProb' were measured.
%                       If multiple data sets are supplied, the size of the
%                       cell array must match the size of 'dataProb'
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
%
%           [ (heights0); (scalarMetric0); (timeScale0); ...
%             (heights1); (scalarMetric1); (timeScale1);
%             (interpParam); (timeScaleInt) ] 
%
%                       -OR-
%
%           [ (heights0); (scalarMetric0); ...
%             (heights1); (scalarMetric1); (interpParam) ]
%
%       depending on the simulation time handling options. The user is
%       responsible for ensuring the initial guess is feasible given the
%       various imposed constraints
%
%       - ('InitialConditions', initConditions = {}): A three element cell
%       array, corresponding to the dynamics for landscape 0, the
%       interpolated landscape, and landscape 1, respectively. For each
%       data set where an initial condition is not supplied (i.e.
%       dataTimes{i}(1) ~= 0) an initial condition must be supplied. This
%       can either be a full initial probability defined on all points OR a
%       (pointID, probSigma) pair that creates a normalized Gaussian cloud
%       with variance 'probSigma' around the specified point in 'X'
%
%       - ('NumSimTimes', numSimTimes = 500): The number of time steps to
%       run each simulation forward
%
%       - ('IsSaddle', isSaddle = []): A logical vector indicating which
%       points in the fixed point height list are saddles. Ordering is
%       determined from the 'allPaths' variable. This field must be
%       specified in order to enforce the saddles.
%
%       - ('EnforcePositiveMetric', enforcePositiveMetric = true): Whether
%       or not to set a bound constraint on the scalar metrics. Be sure to
%       choose a good initial condition if you turn this off!
%
%       - ('EnforceUnitInterpParam', enforceUnitInterpParam = true):
%       Whether or not to bound the interpolation parameter between 0 and
%       1.
%
%       - ('EnforceSaddles', enforceSaddles = false): Whether or not to
%       enforce the constraint that index-1 saddles have a greater
%       potential height than the corresponding minima (i.e. prevent
%       saddles from becoming minima)
%
%       - ('ConstHeightSum', constHeightSum = []): A user supplied constant
%       that constrains the total sum of the minima/saddle heights. Can be
%       supplied as a two element vector (for the two "boundary"
%       landscapes), where NaN entries correspond to no constraint.
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
%       supplied, this property is held fixed over optimization Can be
%       supplied as a three element vector, where NaN entries correspond to
%       no constraint.
%
%       - ('ConstFixedHeights', constFixHeights = []): A set of user
%       supplied values for a subset of the minima/saddle heights, supplied
%       a (numFixedPoints) x 2 vector. Non-NaN entries correspond to
%       specified values. If supplied, these fields are held fixed over
%       optimization
%
%       - ('ConstScalarMetric', constScalarMetric = []): The uniform scalar
%       metric, (i.e. the same for all input points) that re-scales the
%       dynamical velocity (i.e. v = -(1/scalarMetric) * \nabla U). Can be
%       supplied as a two element vector, where NaN entries correspond to
%       no constraint. If supplied, this property is held fixed over
%       optimization
%
%       - ('ConstInterpParam', constInterpParam = []): The fixed
%       interpolation parameter. This should take a value between [0, 1]
%       (although this is not strictly enforced)
%
%       - ('PrecomputeQuadProg', precomputeQuadProg = true): Whether or not
%       to precompute the solver information needed to compute the
%       interpolated potential. This can help to significantly speed up the
%       code, but can also lead to OOM error for large problems run in
%       parallel.
%
%       - ('OptimizationOptions', optOptions = {}): A cell array containing
%       options that can be supplied to a MATLAB 'optimoptions' object to
%       define solver behavior
%
%       - ('UpperBounds', upperBounds = []): A set of upper bound
%       constraints on the optimization variables
%
%       - ('LowerBounds', lowerBounds = []): A set of lower bound
%       constraints on the optimization variables
%
%       - ('ErrorType', errorType = 'symKLD'): The type of error metric to
%       minimize. Total error is the average "per timepoint" error weighted
%       using dataset specific weights. See 'computeSimulationError' below
%       for more details. WARNING: Approximate earth mover's distance
%       computation ('EMD') is VERY slow
%
%       - ('DataSetWeights', dataSetWeights = []): The relative weight of
%       each data set to the total error computation. Weights should sum to
%       one. All time points within each data set will be weighted
%       identically
%
%   Physical Constants ----------------------------------------------------
%   You are allowed to set these for the sake of completeness, but you are
%   strongly advised to just leave them equal to one
%
%       - ('DiffusionCoefficient', D = 1): The diffusion coefficient for
%       the dynamical drift-diffusion process. Can also be supplied as a
%       three element vector, corresponding to each dynamical process.
%
%       - ('PointDiffusionCoefficient', D0 = 1): The diffusion coefficient
%       for the drift-diffusion process assumed to generate the input point
%       set
%
%   Manifold/Point Set/Potential Properties -------------------------------
%   These fields are computed directly from (X, dt) if they are not
%   supplied and are shared for all three dynamical processes
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
%       - ('VolumeElementType', volumeType = 'graphLaplacian'): The type of
%       volume element used to ensure the transition matrix operates on
%       discrete probabilities. Possible types are 'graphLaplacian' or
%       'laplaceBeltrami'.
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
%       - optErr:           The total error between the optimized simulated
%                           time courses and their corresponding data sets
%
%       - optFixHeights0:   1 x #H list of optimizes minima/saddle heights
%                           for landscape 0. Ordering in this list is 
%                           determined from the 'allPaths' input variable
%
%       - optScalarMetric0: The optimized uniform scalar metric for
%                           landscape 0
%
%       - optFixHeights1:   1 x #H list of optimizes minima/saddle heights
%                           for landscape 1. Ordering in this list is 
%                           determined from the 'allPaths' input variable
%
%       - optScalarMetric1: The optimized uniform scalar metric for
%                           landscape 1
%
%       - optInterpParam:   The optimized linear interpolation parameter
%                           defining the interpolated landscape in terms of
%                           landscape 0 and landscape 1
%
%       - optTimeScales:    The optimized time scale ([NaN NaN NaN] if
%                           'simTimeHandling' is not 'constant')
%
%       - optTimes:         The optimally matched simulation time points
%                           for each data point in each data set. Behavior
%                           is determined the 'simTimeHandling' field.
%
%       - optOutput:        A struct containing information about the
%                           optimization process. See 'fmincon' or
%                           'fminunc'
%
%   by Dillon Cislo 2024/06/25

%==========================================================================
% INPUT PROCESSING
%==========================================================================
validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1); dim = size(X,2);

validateattributes(dataProb, {'cell'}, {'vector', 'numel', 3});
validateattributes(dataTimes, {'cell'}, {'vector', 'numel', 3});
numDataSets = zeros(3, 1);
for i = 1:3
    
    % This is probably not the most efficient way to format this, but it
    % does streamline multiple experiment handling
    if ~iscell(dataProb{i}), dataProb{i} = dataProb(i); end
    if ~iscell(dataTimes{i}), dataTimes{i} = dataTimes(i); end
    
    % More explicit data handling is performed after optional input
    % handling
    validateattributes(dataProb{i}, {'cell'}, {'vector'});
    numDataSets(i) = numel(dataProb{i});
    validateattributes(dataTimes{i}, {'cell'}, ...
        {'vector', 'numel', numDataSets(i)});
    
end

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
fixInPathIDx = cell2mat(fixInPathIDx);
[fixPointIDx, ~, fixInPathIDx] = unique(reshape(fixInPathIDx.', [], 1));
fixInPathIDx = reshape(fixInPathIDx, [2 numPaths]).';
numFixPoints = numel(fixPointIDx);

%--------------------------------------------------------------------------
% OPTIONAL INPUT PROCESSING
%--------------------------------------------------------------------------

% Optimization options
initGuess = [];
initConditions = cell(3, 1);
for i = 1:3, initConditions{i} = cell(numDataSets(i), 1); end
numSimTimes = 500;
isSaddle = false(1, numFixPoints);
enforceSaddles = false;
enforcePositiveMetric = true;
enforceUnitInterpParam = true;
constHeightSum = [NaN NaN];
simTimeHandling = 'none';
constFixHeights = nan(numFixPoints, 2);
constScalarMetric = [NaN NaN];
constInterpParam = NaN;
precomputeQuadProg = true;
optOptions = {};
upperBounds = [];
lowerBounds = [];
errorType = 'symKLD';
dataSetWeights = cell(3, 1);

simTimeHandlingOptions = {'none', 'causal', 'constant'};
allErrorTypes = lower({'symKLD', 'dataKLD', 'simKLD', ...
    'MSE', 'geoSphere', 'EMD'});

% Physical constants
D = ones(1,3);
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
volumeType = 'graphlaplacian';
allVolumeTypes = {'graphlaplacian', 'laplacebeltrami'};

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
    'UseGPU', 'InitialGuess', 'EnforcePositiveMetric', ...
    'PrecomputeQuadProg', 'VolumeElementType', 'UpperBounds', ...
    'LowerBounds', 'ErrorType', 'DataSetWeights', ...
    'EnforceUnitInterpParam', 'ConstInterpParam'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end


    % Optimization Options ------------------------------------------------

    if strcmpi(varargin{i}, 'InitialGuess')
        initGuess = varargin{i+1};
        if ~isempty(initGuess)
            validateattributes(initGuess, {'numeric'}, ...
                {'vector', 'finite', 'real'}, ...
                'fitStaticInterpolatedLandScapes', 'initGuess');
            if (size(initGuess, 2) ~= 1), initGuess = initGuess.'; end
        end
    end

    if strcmpi(varargin{i}, 'InitialConditions')
        initConditions = varargin{i+1};
        if isempty(initConditions)
            initConditions = cell(3, 1);
            for ii = 1:3
                initConditions{ii} = cell(numDataSets(ii), 1);
            end
        else
            validateattributes(initConditions, {'cell'}, ...
                {'vector', 'numel', 3}, ...
                'fitStaticInterpolatedLandscapes', 'initConditions');
            for ii = 1:3
                validateattributes(initConditions{ii}, {'cell'}, ...
                    {'vector', 'numel', numDataSets(ii)}, ...
                    'fitStaticInterpolatedLandscapes', 'initConditions');
            end
        end
    end

    if strcmpi(varargin{i}, 'NumSimTimes')
        numSimTimes = varargin{i+1};
        validateattributes(numSimTimes, {'numeric'}, ...
            {'scalar', 'positive', 'integer', 'finite', 'real'}, ...
            'fitStaticInterpolatedLandscapes', 'numSimTimes');
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
                '<=', numFixedPoints}, ...
                'fitStaticInterpolatedLandscapes', 'isSaddle');
            isSaddle = ismember((1:numFixedPoints).', isSaddle);
        end
    end

    if strcmpi(varargin{i}, 'EnforceSaddles')
        enforceSaddles = varargin{i+1};
        validateattributes(enforceSaddles, {'logical'}, {'scalar'}, ...
            'fitStaticInterpolatedLandscapes', 'enforceSaddles');
    end

    if strcmpi(varargin{i}, 'EnforcePositiveMetric')
        enforcePositiveMetric = varargin{i+1};
        validateattributes(enforcePositiveMetric, {'logical'}, {'scalar'}, ...
            'fitStaticInterpolatedLandscapes', 'enforcePositiveMetric');
    end
    
    if strcmpi(varargin{i}, 'EnforceUnitInterpParam')
        enforceUnitInterpParam = varargin{i+1};
        validateattributes(enforceUnitInterpParam, {'logical'}, ...
            {'scalar'}, 'fitStaticInterpolatedLandscapes', ...
            'enforceUnitInterpParam');
    end

    if strcmpi(varargin{i}, 'ConstHeightSum')
        constHeightSum = varargin{i+1};
        if isempty(constHeightSum)
            constHeightSum = [NaN NaN];
        else
            validateattributes(constHeightSum, {'numeric'}, ...
                {'vector', 'numel', 2}, ...
                'fitStaticInterpolatedLandscapes', 'constHeightSum');
        end
    end
    
    if strcmpi(varargin{i}, 'ConstInterpParam')
        constInterpParam = varargin{i+1};
        if isempty(constInterpParam)
            constInterpParam = NaN;
        elseif ~isnan(constInterpParam)
            validateattributes(constInterpParam, {'numeric'}, ...
                {'scalar'}, 'fitStaticInterpolatedLandscapes', ...
                'constInterpParam');
        end
    end

    if strcmpi(varargin{i}, 'SimTimeHandling')
        simTimeHandling = lower(varargin{i+1});
        validateattributes(simTimeHandling, {'char'}, {'vector'}, ...
            'fitStaticInterpolatedLandscapes', 'simTimeHandling');
        assert(ismember(simTimeHandling, simTimeHandlingOptions), ...
            'Invalid simulation time handling option supplied');
    end
    
    if strcmpi(varargin{i}, 'ConstFixedHeights')
        constFixHeights = varargin{i+1};
        if isempty(constFixHeights)
            constFixHeights = nan(numFixPoints, 2);
        else
            assert(isequal(size(constFixHeights), [numFixPoints 2]), ...
                'Constrained heights are improperly sized');
        end
    end

    if strcmpi(varargin{i}, 'ConstScalarMetric')
        constScalarMetric = varargin{i+1};
        if isempty(constScalarMetric)
            constScalarMetric = [NaN NaN];
        else
            validateattributes(constScalarMetric, {'numeric'}, ...
                {'scalar', 'positive', 'finite', 'real'}, ...
                'fitStaticInterpolatedLandscapes', 'constScalarMetric');
        end
    end

    if strcmpi(varargin{i}, 'PrecomputeQuadProg')
        precomputeQuadProg = varargin{i+1};
        validateattributes(precomputeQuadProg, {'logical'}, {'scalar'}, ...
            'fitStaticInterpolatedLandscapes', 'precomputeQuadProg');
    end

    if strcmpi(varargin{i}, 'OptimizationOptions')
        optOptions = varargin{i+1};
        validateattributes(optOptions, {'cell'}, {'vector'}, ...
            'fitStaticInterpolatedLandscapes', 'optOptions');
    end

    if strcmpi(varargin{i}, 'UpperBounds')
        upperBounds = varargin{i+1};
        if ~isempty(upperBounds)
            validateattributes(upperBounds, {'numeric'}, {'vector', ...
                'numel', 2*numFixPoints+3, 'nonnan'}, ...
                'fitStaticInterpolatedLandscapes', 'upperBounds');
            if (size(upperBounds, 2) ~= 1)
                upperBounds = upperBounds .';
            end
        end
    end

    if strcmpi(varargin{i}, 'LowerBounds')
        lowerBounds = varargin{i+1};
        if ~isempty(lowerBounds)
            validateattributes(lowerBounds, {'numeric'}, {'vector', ...
                'numel', 2*numFixPoints+3, 'nonnan'}, ...
                'fitStaticInterpolatedLandscapes', 'lowerBounds');
            if (size(lowerBounds, 2) ~= 1)
                lowerBounds = lowerBounds .';
            end
        end
    end
    
    if strcmpi(varargin{i}, 'ErrorType')
        errorType = varargin{i+1};
        validateattributes(errorType, {'char'}, {'vector'}, ...
            'fitStaticInterpolatedLandscapes', 'errorType');
        assert(ismember(lower(errorType), allErrorTypes), ...
            'Invalid error type supplied');
    end
    
    if strcmpi(varargin{i}, 'DataSetWeights')
        dataSetWeights = varargin{i+1};
        if isempty(dataSetWeights)
            dataSetWeights = cell(3,1);
        else
            assert(iscell(dataSetWeights) && numel(dataSetWeights) == 3, ...
                'Invalid data set weight input');
            for ii = 1:3
                if isempty(dataSetWeights{ii}), continue; end
                validateattributes(dataSetWeights{ii}, {'numeric'}, ...
                    {'vector', 'finite', 'positive', 'real'}, ...
                    'fitStaticInterpolatedLandscapes', 'dataSetWeights');
                if (size(dataSetWeights{ii}, 1) ~= 1)
                    dataSetWeights{ii} = dataSetWeights{ii}.';
                end
            end
        end
    end
    
    % Physical Constants --------------------------------------------------
    
    if strcmpi(varargin{i}, 'DiffusionCoefficient')
        D = varargin{i+1};
        if isscalar(D), D = D * ones(1,3); end
        validateattributes(D, {'numeric'}, ...
            {'vector', 'positive', 'finite', 'real', 'numel', 3}, ...
            'fitStaticInterpolatedLandscapes', 'D');
    end
    
    if strcmpi(varargin{i}, 'PointDiffusionCoefficient')
        D0 = varargin{i+1};
        validateattributes(D0, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'}, ...
            'fitStaticInterpolatedLandscapes', 'D0');
    end

    % Manifold/Point Set/Potential Properties -----------------------------

    if strcmpi(varargin{i}, 'PointPotential')
        U0 = varargin{i+1};
        if ~isempty(U0)
            validateattributes(U0, {'numeric'}, {'vector', ...
                'finite', 'real', 'numel', numPoints}, ...
                'fitStaticInterpolatedLandscapes', 'U0');
            if (size(U0,2) ~= 1), U0 = U0.'; end
        end
    end

    if strcmpi(varargin{i}, 'BasePotential')
        UB = varargin{i+1};
        if ~isempty(UB)
            validateattributes(UB, {'numeric'}, {'vector', ...
                'finite', 'real', 'numel', numPoints}, ...
                'fitStaticInterpolatedLandscapes', 'UB');
            if (size(UB,2) ~= 1), UB = UB.'; end
        end
    end

    if strcmpi(varargin{i}, 'Laplacian'), L = varargin{i+1}; end

    if strcmpi(varargin{i}, 'MassMatrix'), M = varargin{i+1}; end

    % 'interpolatePotentialKHarmonic' Options -----------------------------

    if strcmpi(varargin{i}, 'TikhonovRegularization')
        regSigma = varargin{i+1};
        validateattributes(regSigma, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'}, ...
            'fitStaticInterpolatedLandscapes', 'regSigma');
    end

    if strcmpi(varargin{i}, 'RemoveOutliers')
        removeOutliers = varargin{i+1};
        validateattributes(removeOutliers, {'logical'}, {'scalar'}, ...
            'fitStaticInterpolatedLandscapes', 'removeOutliers');
    end

    if strcmpi(varargin{i}, 'OutlierThreshold')
        outlierThreshold = varargin{i+1};
        if ~isempty(outlierThrehsold)
            validateattribtues(outlierThreshold, {'numeric'}, ...
                {'vector', 'numel', 2, 'finite', 'real'}, ...
                'fitStaticInterpolatedLandscapes', 'outlierThreshold');
            assert(outlierThreshold(2) > outlierThreshold(1), ...
                ['Outlier threshold must have a second element ' ...
                'that is greater than its first element']); 
        end
    end

    if strcmpi(varargin{i}, 'OutlierNeighbors')
        outlierNNSize = varargin{i+1};
        validateattributes(outlierNNSize, {'numeric'}, ...
            {'positive', 'integer', 'scalar', 'finite', 'real', ...
            'fitStaticInterpolatedLandscapes', 'outlierNNSize'});
    end

    if strcmpi(varargin{i}, 'NormalizeMassMatrix')
        normalizeMassMatrix = varargin{i+1};
        validateattributes(normalizeMassMatrix, {'logical'}, {'scalar'}, ...
            'fitStaticInterpolatedLandscapes', 'normalizeMassMatrix');
    end

    % 'interpolateValuesAlongPath' Options --------------------------------

    if strcmpi(varargin{i}, 'PathLengths')
        allPathLengths = varargin{i+1};
        validateattributes(allPathLengths, {'cell'}, ...
            {'vector', 'numel', numPaths}, ...
            'fitStaticInterpolatedLandscapes', 'allPathLengths');
    end

    if strcmpi(varargin{i}, 'PathInterpolationMethod')
        pathInterpMethod = lower(varargin{i+1});
        validateattributes(pathInterpMethod, {'char'}, {'vector'}, ...
            'fitStaticInterpolatedLandscapes', 'PathInterpolationMethod');
    end

    if strcmpi(varargin{i}, 'PathCollisionMethod')
        pathCollisionMethod = lower(varargin{i+1});
        validateattributes(pathCollisionMethod, {'char'}, {'vector'}, ...
            'fitStaticInterpolatedLandscapes', 'pathCollisionMethod');
    end

    % 'computeTransitionMatrix' Options -----------------------------------

    if strcmpi(varargin{i}, 'ClipThreshold')
        clipThreshold = varargin{i+1};
        validateattribtues(clipThreshold, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'}, ...
            'fitStaticInterpolatedLandscapes', 'clipThreshold');
    end

    if strcmpi(varargin{i}, 'StrictNormalization')
        strictNormalization = varargin{i+1};
        validateattributes(strictNormalization, {'logical'}, {'scalar'}, ...
            'fitStaticInterpolatedLandscapes', 'strictNormalization');
    end

    if strcmpi(varargin{i}, 'VolumeElementType')
        volumeType = lower(varargin{i+1});
        validateattributes(volumeType, {'char'}, {'vector'}, ...
            'fitStaticInterpolatedLandscapes', 'volumeType');
        assert(ismember(volumeType, allVolumeTypes), ...
            'Invalid volume element type');
    end

    % General Options -----------------------------------------------------

    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'}, ...
            'fitStaticInterpolatedLandscapes', 'useGPU');
    end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'}, ...
            'fitStaticInterpolatedLandscapes', 'verbose');
    end

end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

%--------------------------------------------------------------------------
% OPTIMIZATION INPUT PROCESSING
%--------------------------------------------------------------------------

if ~any(isSaddle), enforceSaddles = false; end
numConstHeights = sum(~isnan(constFixHeights(:)));

if strcmpi(simTimeHandling, 'constant')
    error('Constant time handling is not yet implemented');
    assert(strcmpi(errorType, 'symKLD'), ['Constant simulation time ' ...
        'handling is currently only capable of handling the ' ...
        'symmetric KL divergence error']);
end

% Process the supplied initial guess --------------------------------------
if ~isempty(initGuess)

    assert(numel(initGuess) == (2*numFixPoints+3), ...
        'Initial guess is improperly sized');

    if ~isnan(constScalarMetric(1))
        initGuess(numFixPoints+1) = constScalarMetric(1);
    end
    if ~isnan(constScalarMetric(2))
        initGuess(end-1) = constScalarMetric(2);
    end
    assert((initGuess(numFixPoints+1) > 0) && ...
        (initGuess(end-1) > 0), ['Scalar metrics ' ...
        'must be positive in the initial guess']);

    if any(~isnan(constFixHeights(:,1)))
        initGuess(~isnan(constFixHeights(:,1))) = ...
            constFixHeights(~isnan(constFixHeights(:,1)), 1);
    end
    if any(~isnan(constFixHeights(:,2)))
        initGuess(find(~isnan(constFixHeights(:,2)))+numFixPoints+1) = ...
            constFixHeights(~isnan(constFixHeights(:,2)), 2);
    end

    if ~isnan(constHeightSum(1))
        assert(abs(sum(initGuess(1:numFixPoints)) ...
            - constHeightSum(1)) < 1e-12, ...
            ['User supplied initial guess for landscape 0 does not ' ...
            'adhere to the fixed point height sum constraint']);
    end
    
    if ~isnan(constHeightSum(2))
        assert(abs(sum(initGuess((numFixPoints+1)+(1:numFixPoints))) - ...
            constHeightSum(2)) < 1e-12, ...
            ['User supplied initial guess for landscape 1 does not ' ...
            'adhere to the fixed point height sum constraint']);
    end
    
    if ~isnan(constInterpParam)
        initGuess(end) = constInterpParam;
    end
    assert(isreal(initGuess(end)), ['Initial guess for interpolation ' ...
        'parameter must be real']);
    if enforceUnitInterpParam
        assert( (0 <= initGuess(end)) && (initGuess(end) <= 1), ...
            ['Initial guess for interpolation parameter must lie on ' ...
            'the range [0, 1] if the unit constraint is enforced']);
    end

end

% Process the data points/data times/initial conditions -------------------

numDataPointsPerSet = cell(3, 1);
for i = 1:3, numDataPointsPerSet{i} = zeros(1, numDataSets(i)); end

numTotalDataPoints = zeros(1, 3);
for expID = 1:3
    for i = 1:numDataSets(expID)
        
        validateattributes(dataProb{expID}{i}, {'numeric'}, ...
            {'2d', 'finite', 'real', 'nonnegative', 'nrows', numPoints});
        numDataPointsPerSet{expID}(i) = size(dataProb{expID}{i}, 2);
        
        if isempty(dataTimes{expID}{i})
            
            dataTimes{expID}{i} = 0:(numDataPointsPerSet{expID}(i)-1);
            
        else
            
            validateattributes(dataTimes{expID}{i}, {'numeric'}, ...
                {'vector', 'real', 'nonnegative', 'finite', 'numel', ...
                numDataPointsPerSet{expID}(i)});
            if (size(dataTimes{expID}{i}, 1) ~= 1)
                dataTimes{expID}{i} = dataTimes{expID}{i}.';
            end
            
            % Sort the input data points so that time is monotonically
            % increasing
            [dataTimes{expID}{i}, DI] = sort(dataTimes{expID}{i});
            dataProb{expID}{i} = dataProb{expID}{i}(:, DI);
            
        end
        
        assert(isequal(dataTimes{expID}{i}, unique(dataTimes{expID}{i})), ...
            ['Multiple data points have been supplied with the same ' ...
            'physical time for landscape %d data set %d'], expID, i);
        
        if (numDataPointsPerSet{expID}(i) == 1)
            assert(dataTimes{expID}{i}(1) > 0, ['Only the initial ' ...
                'condition was supplied for landscape %d data set %d'], ...
                expID, i);
        end
        
        if isempty(initConditions{expID}{i})
            
            assert(dataTimes{expID}{i}(1) == 0, ['No initial ' ...
                'condition supplied for landscape %d data set %d'], ...
                expID, i);
            
            initConditions{expID}{i} = dataProb{expID}{i}(:,1);
            
            % Don't count the initial condition in data sets
            % TODO: CHECK IF THIS IS OKAY FOR CONSTANT TIME HANDLING
            if ~strcmpi(simTimeHandling, 'constant')
                dataProb{expID}{i}(:,1) = [];
                dataTimes{expID}{i}(1) = [];
                numDataPointsPerSet{expID}(i) = ...
                    numDataPointsPerSet{expID}(i)-1;
            end
            
        else
            
            validateattributes(initConditions{expID}{i}, {'numeric'}, ...
                {'vector', 'nonnegative', 'finite', 'real'});
            
            if (numel(initConditions{expID}{i}) == 2)
                
                assert(ismember(initConditions{expID}{i}(1), ...
                    (1:numPoints)), ['Invalid index for initial ' ...
                    'condition in landscape %d data set %d'], expID, i);
                assert(initConditions{expID}{i}(2) > 0, ...
                    ['Invalid variance for initial condition in ' ...
                    'landscape %d data set %d'], expID, i);
                
                curIC = X-repmat(X(initConditions{expID}{i}(1), :), [numPoints, 1]);
                curIC = sum(curIC.^2, 2);
                curIC = exp(-curIC ./ (2 * initConditions{expID}{i}(2)^2));
                curIC  = curIC ./ (2 * pi * initConditions{expID}{i}(2)^2)^(dim/2);
                curIC = exp(U0/D0) .* curIC ./ numPoints;
                curIC = curIC ./ sum(curIC(:));
                initConditions{expID}{i} = curIC(:);
                
            elseif (numel(initConditions{expID}{i}) ~= numPoints)
                
                error(['Initial conditions for data set %d ' ...
                    'are improperly sized'], i);
                
            end
            
            if (abs(sum(initConditions{expID}{i})-1) > 1e-12)
                warning(['Initial condition supplied for landscape %d ' ...
                    'data set %d is not properly normalized'], expID, i);
            end
            
            clear curIC
            
        end
        
    end
    
    numTotalDataPoints(expID) = sum(numDataPointsPerSet{expID});
    
end

for expID = 1:3
    if isempty(dataSetWeights{expID})
        
        dataSetWeights{expID} = ones(1, numTotalDataPoints(expID));
        
    else
        
        validateattributes(dataSetWeights{expID}, {'numeric'}, ...
            {'vector', 'finite', 'real', 'positive'}, ...
            'fitStaticInterpolatedLandscapes', 'dataSetWeights');
        
        if numel(dataSetWeights{expID}) == numDataSets(expID)
            dataSetWeights{expID} = repelem(dataSetWeights{expID}, ...
                numDataPointsPerSet(expID));
        end
        
        assert(numel(dataSetWeights{expID}) == numTotalDataPoints(expID), ...
            ['Data set weights are improperly sized for landscape ' ...
            '%d (make sure you are properly handling initial ' ...
            'conditions)'], expID);
        
        if (size(dataSetWeights{expID}, 1) ~= 1)
            dataSetWeights{expID} = dataSetWeights{expID}.';
        end
        
    end
end

% Normalize the data set weights
dataSetWeights = [dataSetWeights{:}];
dataSetWeights = dataSetWeights ./ sum(dataSetWeights);
dataSetWeights = mat2cell(dataSetWeights, 1, numTotalDataPoints).';
assert(isequal(size(dataSetWeights), [3 1]), ...
    'Data set weight normalization failed');
for i = 1:numel(dataSetWeights)
    assert(numel(dataSetWeights{i}) == numTotalDataPoints(i), ...
        'Data set weight normalizatin failed');
end

%--------------------------------------------------------------------------
% MANIFOLD/POINT SET/POTENTIAL OPTION PROCESSING
%--------------------------------------------------------------------------

% Estimate point set pseudo potential from point cloud---------------------
if isempty(U0)
    if verbose, disp('Computing point set potential:'); end
    U0 = -D0 * log(gaussianKDE(X, X, [], sqrt(2 * dt), verbose));
end

if isempty(UB), UB = U0; end

% Compute point cloud volume element --------------------------------------

if verbose
    fprintf('Computing transition matrix volume element... ');
end

if strcmpi(volumeType, 'GraphLaplacian')

    volumeElement = exp(U0 ./ D0);

elseif strcmpi(volumeType, 'LaplaceBeltrami')

    affinityOptions = struct();
    affinityOptions.Sigma = dt;
    affinityOptions.NumNeighbors = numPoints;
    affinityOptions.Verbose = false;
    K = affinityMatrix(X, affinityOptions);

    mapOptions = struct();
    mapOptions.Normalization = 'LaplaceBeltrami';
    mapOptions.NumVectors = 0;
    mapOptions.Verbose = false;
    [~, ~, ~, ~, DAlpha] = diffusionMap(K, mapOptions);
    volumeElement = full(DAlpha);

    clear affinityOptions K mapOptions DAlpha

else

    error('Invalid transition matrix volume element type');

end

if verbose, fprintf('Done\n'); end

% Check Laplacian/mass matrix ---------------------------------------------
if (isempty(L) || isempty(M))

    if verbose, fprintf('Building Laplacian/mass matrix... '); end
    [L, M] = diffusionMapLaplacian(X, dt);
    if verbose, fprintf('Done\n'); end

else

    validateattributes(L, {'numeric'}, {'2d', 'finite', 'real', ...
        'ncols', numPoints, 'nrows', numPoints});
    assert(issymmetric(L), 'Laplacian is not symmetric');

    % Check for positive-definiteness
    [~, isPD] = chol(L); isPD = isPD == 0;
    if ~isPD
        [~, isPD] = chol(-L); isPD = isPD == 0;
        if isPD
            L = -L;
        else
            error('Laplacian is neither positive nor negative definite');
        end
    end

    validateattributes(M, {'numeric'}, {'2d', 'finite', 'real', ...
        'ncols', numPoints, 'nrows', numPoints});
    assert(isdiag(M), 'Mass matrix is not diagonal');
    assert(all(diag(M) > 0), 'Mass matrix contains non-positive masses');

    clear isPD

end

if normalizeMassMatrix, M = M ./ max(abs(diag(M))); end

% Build quadratic coefficients --------------------------------------------
% We explicitly build the biharmonic operator here.
% 'interpolatePotentialKHarmonic' would have more flexibility but is
% generally slower.

Q = L*(M\L);
if ~issymmetric(Q), Q = (Q + Q.') ./ 2; end % Correct for roundoff error
if (regSigma > 0), Q = Q + regSigma .* eye(size(Q)); end
clear L M

% Pre-compute quadratic solver information for constructing the
% interpolated potential
if precomputeQuadProg

    fprintf('Pre-computing quadratic solver info\n')
    [~, tmpKnownIDx] = interpolateValuesAlongPath( ones(numPaths, 2), ...
        allPaths, 'PathLengths', allPathLengths, ...
        'InterpolationMethod', pathInterpMethod, ...
        'CollisionMethod', pathCollisionMethod);
    F = min_quad_with_fixed_precompute(Q, tmpKnownIDx, []);

else

    F = [];

end

%==========================================================================
% BUILD OPTIMIZATION PROBLEM
%==========================================================================
% The saddle inequality constraint, the positive metric constraints, the
% unit interval interpolation parameter constraints, and the total height
% sum constraint are the only non-trivial constraints. If any of these are
% supplied, we formulate the full constrained problem and run a
% corresponding constrained minimization. If none of these constraints
% are enforced, we enforce any remaining constraints by construction and
% run an unconstrained minimization

% Build inequality constraints --------------------------------------------

A = []; b = [];

if enforceSaddles

    if verbose, disp('Adding saddle height inequality constraint'); end

    saddleIDx = find(isSaddle);
    saddleInPath = ismember(fixInPathIDx, saddleIDx);
    assert(all(ismember(sum(saddleInPath, 2), [0 1])), ...
        'Some input paths pair saddles to saddles');

    A = fixInPathIDx(any(saddleInPath, 2), :);
    A(~saddleInPath(:,1), :) = A(~saddleInPath(:,1), [2 1]);

    A = full(sparse( ...
        [repmat((1:size(A,1)).', [1 2]); ...
        repmat((1:size(A,1)).', [1 2])+size(A,1)], ...
        [A; A+(numFixPoints+1)], ...
        repmat([ones(size(A,1), 1), -ones(size(A,1), 1)], [2 1]), ...
        2*size(A,1), 2*numFixPoints+3 ));
    
    b = -1e-12 * ones(size(A,1), 1);

end

% Build equality constraints ----------------------------------------------

Aeq = []; beq = [];

if any(~isnan(constHeightSum))

    if verbose, disp('Adding height sum constraint'); end

    if ~isnan(constHeightSum(1))
        Aeq = [Aeq; ones(1, numFixPoints) zeros(1, numFixPoints+3)];
        beq = [beq; constHeightSum(1)];
    end
    
    if ~isnan(constHeightSum(2))
        Aeq = [Aeq; zeros(1, numFixPoints+1), ones(1, numFixPoints), 0 0];
        beq = [beq; constHeightSum(2)];
    end

end

if (numConstHeights > 0)

    if verbose, disp('Adding fixed height constraint'); end

    if any(~isnan(constFixHeights(:,1)))
        
        Aeq = [Aeq; ...
            full(sparse(1:sum(~isnan(constFixHeights(:,1))), ...
            find(~isnan(constFixHeights(:,1))), ...
            1, sum(~isnan(constFixHeights(:,1))), 2*numFixPoints+3))];
        
        beq = [beq; reshape( ...
            constFixHeights(~isnan(constFixHeights(:,1)), 1), [], 1)];
        
    end
    
    if any(~isnan(constFixHeights(:,2)))
        
        Aeq = [Aeq; ...
            full(sparse(1:sum(~isnan(constFixHeights(:,2))), ...
            find(~isnan(constFixHeights(:,2)))+(numFixPoints+1), ...
            1, sum(~isnan(constFixHeights(:,2))), 2*numFixPoints+3))];
        
        beq = [beq; reshape( ...
            constFixHeights(~isnan(constFixHeights(:,2)), 2), [], 1)];
        
    end

end

if any(~isnan(constScalarMetric))

    if verbose, disp('Adding fixed scalar metric constraint'); end

    % There is no need to enforce a positive metric if it is already fixed
    enforcePositiveMetric = false;

    if ~isnan(constScalarMetric(1))
        Aeq = [Aeq; full(sparse(1, numFixPoints+1, 1, 1, 2*numFixPoints+3))];
        beq = [beq; constScalarMetric(1)];
    end
    
    if ~isnan(constScalarMetric(2))
        Aeq = [Aeq; full(sparse(1, 2*numFixPoints+2, 1, 1, 2*numFixPoints+3))];
        beq = [beq; constScalarMetric(2)];
    end

end

if ~isnan(constInterpParam)
    
    if verbose
        disp('Adding fixed interpolation parameter constraint');
    end
    
    % There is no need to enforce an interpolation parameter on the range
    % [0, 1] if it is already fixed
    enforceUnitInterpParam = false;
    
    Aeq = [Aeq; full(sparse(1, 2*numFixPoints+3, 1, 1, 2*numFixPoints+3))];
    beq = [beq; constInterpParam];
    
end
    
% Set bound constraints ---------------------------------------------------

if isempty(lowerBounds)

    if enforcePositiveMetric
        if verbose, disp('Enforcing positive metric'); end
        lb = [-inf(numFixPoints, 1); 1e-12; -inf(numFixPoints, 1); 1e-12];
    else
        lb = -inf(2*numFixPoints+2, 1);
    end
    
    if enforceUnitInterpParam
        if verbose, disp('Enforcing unit interval interpolation'); end
        lb = [lb; 0];
    else
        lb = [lb; -inf];
    end
    
else

    lb = lowerBounds;
    if enforcePositiveMetric
        assert((lb(numFixPoints+1) > 0) && (lb(end-1) > 0), ...
	    ['Lower bound constraint inconsistent ' ...
            'with positive metric']);
    end
    if enforceUnitInterpParam
        assert(lb(end) >= 0, ...
	    ['Lower bound constraint inconsistent ' ...
	    'with unit interval interpolation parameter constraint']);
    end

end

if isempty(upperBounds)
    
    ub = inf(2*numFixPoints+2, 1);
    if enforceUnitInterpParam
        ub = [ub; 1];
    else
        ub = [ub; inf];
    end

else

    ub = upperBounds;
    if enforceUnitInterpParam
        assert(ub(end) <= 1, ...
	    ['Upper bound constraint inconsistent ' ...
	    'with unit interpolation parameter constraint']);
    end

end

% Determine if unconstrained minimization is feasible ---------------------

constrainedValues = nan(2*numFixPoints+3, 1);
runConstrainedMinimization = true;
if ~enforceSaddles && ~enforcePositiveMetric ...
        && ~enforceUnitInterpParam && isempty(constHeightSum)

    runConstrainedMinimization = false;
    
    constrainedValues(1:numFixPoints) = constFixHeights(:,1);
    constrainedValues((numFixPoints+1)+(1:numFixPoints)) = ...
        constFixHeights(:,2);
    
    constrainedValues(numFixPoints+1) = constScalarMetric(1);
    constrainedValues(2*numFixPoints+2) = constScalarMetric(2);
    
    constrainedValues(end) = constInterpParam;

    initGuess(~isnan(constrainedValues)) = [];

    clear A b Aeq Beq lb ub

end

clear upperBounds lowerBounds

%==========================================================================
% RUN OPTIMIZATION
%==========================================================================
if strcmpi(simTimeHandling, 'constant')
    optFun = @simulateLandscapeDynamicsConstTimeScale;
else
    optFun = @simulateLandscapeDynamics;
end

if runConstrainedMinimization

    if verbose, disp('Running constrained optimization'); end

    options = optimoptions('fmincon', optOptions{:});
    if options.UseParallel
        if (verbose && useGPU)
            disp(['Parallel finite difference computations ' ...
                'override GPU option']);
        end
        useGPU = false;
    end

    [optVals, optErr, ~, optOutput] = ...
        fmincon(optFun, initGuess, A, b, Aeq, beq, lb, ub, [], options);

else

    if verbose, disp('Running unconstrained optimization'); end

    options = optimoptions('fminunc', optOptions{:});
    if options.UseParallel
        if (verbose && useGPU)
            disp(['Parallel finite difference computations ' ...
                'override GPU option']);
        end
        useGPU = false;
    end

    [optVals, optErr, ~, optOutput] = ...
        fminunc(optFun, initGuess, options);

end

optConstrainedValues = constrainedValues;
optConstrainedValues(isnan(constrainedValues)) = optVals;

optFixHeights0 = optConstrainedValues(1:numFixPoints);
optScalarMetric0 = optConstrainedValues(numFixPoints+1);

optFixHeights1 = optConstrainedValues(1+numFixPoints+(1:numFixPoints));
optScalarMetric1 = optConstrainedValues(2*numFixPoints+2);

optInterpParam = optConstrainedValues(2*numFixPoints+3);

optTimes = cell(3, 1);
optTimeScales = [NaN NaN NaN];

if verbose, fprintf('Consolidating output... '); end

%--------------------------------------------------------------------------
% FORMAT OUTPUT FOR LANDSCAPE 0
%--------------------------------------------------------------------------

% Convert fixed point height list into path end point values
optEndPointVals0 = optFixHeights0(fixInPathIDx);
[optKnownU0, optKnownIDx0] = ...
    interpolateValuesAlongPath(optEndPointVals0, ...
    allPaths, 'PathLengths', allPathLengths, ...
    'InterpolationMethod', pathInterpMethod, ...
    'CollisionMethod', pathCollisionMethod);

% Compute interpolated potential (FAST)
optUI0 = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
    optKnownIDx0, optKnownU0, [], [], F);

% Handle interpolated potential outliers
if removeOutliers
    if isempty(outlierThreshold)
        optOutlierThreshold = [min(optKnownU0), max(optKnownU0)] + ...
            1e-14 * [-1 1];
    else
        optOutlierThreshold = outlierThreshold;
    end
    optUI0 = removeScalarOutliersFromPointCloud( ...
        X, optUI0, optOutlierThreshold, outlierNNSize);
end

optUD0 = UB + optUI0; % Combine to compute dynamical potential

% This is a large dense matrix - we don't keep multiple copies
optT = computeTransitionMatrix(X, optUD0, dt, ...
    'PointPotential', U0, 'ScalarMetric', optScalarMetric0, ...
    'DiffusionCoefficient', D(1), 'PointDiffusionCoefficient', D0, ...
    'ClipThreshold', clipThreshold, ...
    'StrictNormalization', strictNormalization, ...
    'VolumeElementType', volumeType, 'VolumeElement', volumeElement);

optSimProb = cell(numDataSets(1), 1);
for i = 1:numDataSets(1)
    optSimProb{i} = evolveProbabilities(initConditions{1}{i}, optT, ...
        (numSimTimes-1), 'NumViewTimes', -1, 'TimeStep', dt, ...
        'StrictNormalization', strictNormalization, ...
        'UseGPU', useGPU);
end

if strcmpi(simTimeHandling, 'constant')

    optTimeScales(1) = fitProbSimTimeScale(dataProb{1}, dataTimes{1}, ...
        optSimProb, dt, 'UseGPU', useGPU);

    optTimes{1} = cellfun(@(x) optTimeScales(1) * x, ...
        dataTimes{1}, 'Uni', false);
    
else

    simTimes = (0:(numSimTimes-1)) * dt;
    optCount = 0;
    optTimes{1} = cell(1, numDataSets(1));
    for i = 1:numDataSets(1)

        optMatchTimes = -ones(numDataPointsPerSet{1}(i), 1);
        for ii = 1:numDataPointsPerSet{1}(i)

            optCount = optCount + 1;
            
            curOptErr = computeSimulationError( ...
                    dataProb{1}{i}(:, ii), optSimProb{i}, errorType, X );

            if strcmpi(simTimeHandling, 'none')

                [~, optMatchTimes(ii)] = min(curOptErr);

            elseif strcmpi(simTimeHandling, 'causal')

                if (ii > 1), curOptErr(1:optMatchTimes(ii-1)) = inf; end
                [~, optMatchTimes(ii)] = min(curOptErr);

            else

                error(['Invalid simulation time handling in ' ...
                    'output consolidation']);

            end

        end

        optTimes{1}{i} = simTimes(optMatchTimes);

    end

end

%--------------------------------------------------------------------------
% FORMAT OUTPUT FOR LANDSCAPE 1
%--------------------------------------------------------------------------

% Convert fixed point height list into path end point values
optEndPointVals1 = optFixHeights1(fixInPathIDx);
[optKnownU1, optKnownIDx1] = ...
    interpolateValuesAlongPath(optEndPointVals1, ...
    allPaths, 'PathLengths', allPathLengths, ...
    'InterpolationMethod', pathInterpMethod, ...
    'CollisionMethod', pathCollisionMethod);

% Compute interpolated potential (FAST)
optUI1 = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
    optKnownIDx1, optKnownU1, [], [], F);

% Handle interpolated potential outliers
if removeOutliers
    if isempty(outlierThreshold)
        optOutlierThreshold = [min(optKnownU1), max(optKnownU1)] + ...
            1e-14 * [-1 1];
    else
        optOutlierThreshold = outlierThreshold;
    end
    optUI1 = removeScalarOutliersFromPointCloud( ...
        X, optUI1, optOutlierThreshold, outlierNNSize);
end

optUD1 = UB + optUI1; % Combine to compute dynamical potential

% This is a large dense matrix - we don't keep multiple copies
optT = computeTransitionMatrix(X, optUD1, dt, ...
    'PointPotential', U0, 'ScalarMetric', optScalarMetric1, ...
    'DiffusionCoefficient', D(3), 'PointDiffusionCoefficient', D0, ...
    'ClipThreshold', clipThreshold, ...
    'StrictNormalization', strictNormalization, ...
    'VolumeElementType', volumeType, 'VolumeElement', volumeElement);

optSimProb = cell(numDataSets(3), 1);
for i = 1:numDataSets(3)
    optSimProb{i} = evolveProbabilities(initConditions{3}{i}, optT, ...
        (numSimTimes-1), 'NumViewTimes', -1, 'TimeStep', dt, ...
        'StrictNormalization', strictNormalization, ...
        'UseGPU', useGPU);
end

if strcmpi(simTimeHandling, 'constant')

    optTimeScales(3) = fitProbSimTimeScale(dataProb{3}, dataTimes{3}, ...
        optSimProb, dt, 'UseGPU', useGPU);

    optTimes{3} = cellfun(@(x) optTimeScales(3) * x, ...
        dataTimes{3}, 'Uni', false);
    
else

    simTimes = (0:(numSimTimes-1)) * dt;
    optCount = 0;
    optTimes{3} = cell(1, numDataSets(3));
    for i = 1:numDataSets(3)

        optMatchTimes = -ones(numDataPointsPerSet{3}(i), 1);
        for ii = 1:numDataPointsPerSet{3}(i)

            optCount = optCount + 1;
            
            curOptErr = computeSimulationError( ...
                    dataProb{3}{i}(:, ii), optSimProb{i}, errorType, X );

            if strcmpi(simTimeHandling, 'none')

                [~, optMatchTimes(ii)] = min(curOptErr);

            elseif strcmpi(simTimeHandling, 'causal')

                if (ii > 1), curOptErr(1:optMatchTimes(ii-1)) = inf; end
                [~, optMatchTimes(ii)] = min(curOptErr);

            else

                error(['Invalid simulation time handling in ' ...
                    'output consolidation']);

            end

        end

        optTimes{3}{i} = simTimes(optMatchTimes);

    end

end

%--------------------------------------------------------------------------
% FORMAT OUTPUT FOR INTERPOLATED LANDSCAPE
%--------------------------------------------------------------------------

optIntUD = (1-optInterpParam) .* (optUD0 ./ optScalarMetric0) + ...
    optInterpParam .* (optUD1 ./ optScalarMetric1);

% This is a large dense matrix - we don't keep multiple copies
optT = computeTransitionMatrix(X, optIntUD, dt, ...
    'PointPotential', U0, 'ScalarMetric', 1, ...
    'DiffusionCoefficient', D(2), 'PointDiffusionCoefficient', D0, ...
    'ClipThreshold', clipThreshold, ...
    'StrictNormalization', strictNormalization, ...
    'VolumeElementType', volumeType, 'VolumeElement', volumeElement);

optSimProb = cell(numDataSets(2), 1);
for i = 1:numDataSets(2)
    optSimProb{i} = evolveProbabilities(initConditions{2}{i}, optT, ...
        (numSimTimes-1), 'NumViewTimes', -1, 'TimeStep', dt, ...
        'StrictNormalization', strictNormalization, ...
        'UseGPU', useGPU);
end

if strcmpi(simTimeHandling, 'constant')

    optTimeScales(2) = fitProbSimTimeScale(dataProb{2}, dataTimes{2}, ...
        optSimProb, dt, 'UseGPU', useGPU);

    optTimes{2} = cellfun(@(x) optTimeScales(2) * x, ...
        dataTimes{2}, 'Uni', false);
    
else

    simTimes = (0:(numSimTimes-1)) * dt;
    optCount = 0;
    optTimes{2} = cell(1, numDataSets(2));
    for i = 1:numDataSets(2)

        optMatchTimes = -ones(numDataPointsPerSet{2}(i), 1);
        for ii = 1:numDataPointsPerSet{2}(i)

            optCount = optCount + 1;
            
            curOptErr = computeSimulationError( ...
                    dataProb{2}{i}(:, ii), optSimProb{i}, errorType, X );

            if strcmpi(simTimeHandling, 'none')

                [~, optMatchTimes(ii)] = min(curOptErr);

            elseif strcmpi(simTimeHandling, 'causal')

                if (ii > 1), curOptErr(1:optMatchTimes(ii-1)) = inf; end
                [~, optMatchTimes(ii)] = min(curOptErr);

            else

                error(['Invalid simulation time handling in ' ...
                    'output consolidation']);

            end

        end

        optTimes{2}{i} = simTimes(optMatchTimes);

    end

end

if verbose, fprintf('Done\n'); end

%**************************************************************************
%**************************************************************************
%                       SIMULATE LANDSCAPE DYNAMICS
%**************************************************************************
%**************************************************************************

    function E = simulateLandscapeDynamics(x)

        locConstrainedValues = constrainedValues;
        locConstrainedValues(isnan(constrainedValues)) = x;

        fixHeights0 = locConstrainedValues(1:numFixPoints);
        scalarMetric0 = locConstrainedValues(numFixPoints+1);
        
        fixHeights1 = ...
            locConstrainedValues((numFixPoints+1)+(1:numFixPoints));
        scalarMetric1 = locConstrainedValues(2*numFixPoints+2);
        
        interpParam = locConstrainedValues(end);
        
        E = 0;
        
        %==================================================================
        % PROCESS LANDSCAPE 0
        %==================================================================

        % Convert fixed point height list into path end point values
        endPointVals0 = fixHeights0(fixInPathIDx);
        [knownU0, knownIDx0] = interpolateValuesAlongPath(endPointVals0, ...
            allPaths, 'PathLengths', allPathLengths, ...
            'InterpolationMethod', pathInterpMethod, ...
            'CollisionMethod', pathCollisionMethod);
        
        % Compute interpolated potential (FAST)
        UI0 = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
            knownIDx0, knownU0, [], [], F);

        % Handle interpolated potential outliers
        if removeOutliers
            if isempty(outlierThreshold)
                curOutlierThreshold = [min(knownU0), max(knownU0)] + ...
                    1e-14 * [-1 1];
            else
                curOutlierThreshold = outlierThreshold;
            end
            UI0 = removeScalarOutliersFromPointCloud( ...
                X, UI0, curOutlierThreshold, outlierNNSize);
        end

        UD0 = UB + UI0; % Combine to compute dynamical potential

        % This is a large dense matrix - we don't keep multiple copies
        T = computeTransitionMatrix(X, UD0, dt, ...
            'PointPotential', U0, 'ScalarMetric', scalarMetric0, ...
            'DiffusionCoefficient', D(1), ...
            'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', clipThreshold, ...
            'StrictNormalization', strictNormalization, ...
            'VolumeElementType', volumeType, ...
            'VolumeElement', volumeElement);

        % NOTE: We perform this operation serially so that gradients can be
        % estimated in parallel, if desired
        count = 0; % An index into the data set weight vector
        for k = 1:numDataSets(1)

            curDataProb = dataProb{1}{k};
            initProb = initConditions{1}{k};

            simProb = evolveProbabilities(initProb, T, (numSimTimes-1), ...
                'NumViewTimes', -1, 'TimeStep', dt, 'UseGPU', useGPU, ...
                'StrictNormalization', strictNormalization);

            matchTimes = -ones(numDataPointsPerSet{1}(k), 1);
            for j = 1:(numDataPointsPerSet{1}(k))

                count = count + 1; % Increment in the index
                
                err = computeSimulationError( ...
                    curDataProb(:,j), simProb, errorType, X );

                if strcmpi(simTimeHandling, 'none')
                    
                    E = E + dataSetWeights{1}(count) .* min(err);

                elseif strcmpi(simTimeHandling, 'causal')

                    if (j > 1), err(1:matchTimes(j-1)) = inf; end
                    [minErr, matchTimes(j)] = min(err);
                    E = E + dataSetWeights{1}(count) .* minErr;

                else

                    error(['Invalid simulation time handling in ' ...
                        'optimization function']);
                    
                end
                
            end
            
        end
        
        %==================================================================
        % PROCESS LANDSCAPE 1
        %==================================================================
        
        % Convert fixed point height list into path end point values
        endPointVals1 = fixHeights1(fixInPathIDx);
        [knownU1, knownIDx1] = interpolateValuesAlongPath(endPointVals1, ...
            allPaths, 'PathLengths', allPathLengths, ...
            'InterpolationMethod', pathInterpMethod, ...
            'CollisionMethod', pathCollisionMethod);
        
        % Compute interpolated potential (FAST)
        UI1 = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
            knownIDx1, knownU1, [], [], F);

        % Handle interpolated potential outliers
        if removeOutliers
            if isempty(outlierThreshold)
                curOutlierThreshold = [min(knownU1), max(knownU1)] + ...
                    1e-14 * [-1 1];
            else
                curOutlierThreshold = outlierThreshold;
            end
            UI1 = removeScalarOutliersFromPointCloud( ...
                X, UI1, curOutlierThreshold, outlierNNSize);
        end

        UD1 = UB + UI1; % Combine to compute dynamical potential

        % This is a large dense matrix - we don't keep multiple copies
        T = computeTransitionMatrix(X, UD1, dt, ...
            'PointPotential', U0, 'ScalarMetric', scalarMetric1, ...
            'DiffusionCoefficient', D(3), ...
            'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', clipThreshold, ...
            'StrictNormalization', strictNormalization, ...
            'VolumeElementType', volumeType, ...
            'VolumeElement', volumeElement);

        % NOTE: We perform this operation serially so that gradients can be
        % estimated in parallel, if desired
        count = 0; % An index into the data set weight vector
        for k = 1:numDataSets(3)

            curDataProb = dataProb{3}{k};
            initProb = initConditions{3}{k};

            simProb = evolveProbabilities(initProb, T, (numSimTimes-1), ...
                'NumViewTimes', -1, 'TimeStep', dt, 'UseGPU', useGPU, ...
                'StrictNormalization', strictNormalization);

            matchTimes = -ones(numDataPointsPerSet{3}(k), 1);
            for j = 1:(numDataPointsPerSet{3}(k))

                count = count + 1; % Increment in the index
                
                err = computeSimulationError( ...
                    curDataProb(:,j), simProb, errorType, X );

                if strcmpi(simTimeHandling, 'none')
                    
                    E = E + dataSetWeights{3}(count) .* min(err);

                elseif strcmpi(simTimeHandling, 'causal')

                    if (j > 1), err(1:matchTimes(j-1)) = inf; end
                    [minErr, matchTimes(j)] = min(err);
                    E = E + dataSetWeights{3}(count) .* minErr;

                else

                    error(['Invalid simulation time handling in ' ...
                        'optimization function']);

                end

            end

        end
        
        %==================================================================
        % PROCESS INTERPOLATED LANDSCAPE
        %==================================================================
        
        intUD = (1-interpParam) .* (UD0 ./ scalarMetric0) + ...
            interpParam .* (UD1 ./ scalarMetric1);
        
        % This is a large dense matrix - we don't keep multiple copies
        T = computeTransitionMatrix(X, intUD, dt, ...
            'PointPotential', U0, 'ScalarMetric', 1, ...
            'DiffusionCoefficient', D(2), ...
            'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', clipThreshold, ...
            'StrictNormalization', strictNormalization, ...
            'VolumeElementType', volumeType, ...
            'VolumeElement', volumeElement);

        % NOTE: We perform this operation serially so that gradients can be
        % estimated in parallel, if desired
        count = 0; % An index into the data set weight vector
        for k = 1:numDataSets(2)

            curDataProb = dataProb{2}{k};
            initProb = initConditions{2}{k};

            simProb = evolveProbabilities(initProb, T, (numSimTimes-1), ...
                'NumViewTimes', -1, 'TimeStep', dt, 'UseGPU', useGPU, ...
                'StrictNormalization', strictNormalization);

            matchTimes = -ones(numDataPointsPerSet{2}(k), 1);
            for j = 1:(numDataPointsPerSet{2}(k))

                count = count + 1; % Increment in the index
                
                err = computeSimulationError( ...
                    curDataProb(:,j), simProb, errorType, X );

                if strcmpi(simTimeHandling, 'none')
                    
                    E = E + dataSetWeights{2}(count) .* min(err);

                elseif strcmpi(simTimeHandling, 'causal')

                    if (j > 1), err(1:matchTimes(j-1)) = inf; end
                    [minErr, matchTimes(j)] = min(err);
                    E = E + dataSetWeights{2}(count) .* minErr;

                else

                    error(['Invalid simulation time handling in ' ...
                        'optimization function']);

                end

            end

        end

    end

%**************************************************************************
%**************************************************************************
%               SIMULATE LANDSCAPE DYNAMICS CONSTANT TIME SCALE
%**************************************************************************
%**************************************************************************

    function E = simulateLandscapeDynamicsConstTimeScale(x)

        locConstrainedValues = constrainedValues;
        locConstrainedValues(isnan(constrainedValues)) = x;

        scalarMetric = locConstrainedValues(end);
        fixHeights = locConstrainedValues(1:numFixPoints);

        % Convert fixed point height list into path end point values
        endPointVals = fixHeights(fixInPathIDx);
        [knownU, knownIDx] = interpolateValuesAlongPath(endPointVals, ...
            allPaths, 'PathLengths', allPathLengths, ...
            'InterpolationMethod', pathInterpMethod, ...
            'CollisionMethod', pathCollisionMethod);

        % Compute interpolated potential (FAST)
        UI = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
            knownIDx, knownU, [], [], F);

        % Handle interpolated potential outliers
        if removeOutliers
            if isempty(outlierThreshold)
                curOutlierThreshold = [min(knownU), max(knownU)] + ...
                    1e-14 * [-1 1];
            else
                curOutlierThreshold = outlierThreshold;
            end
            UI = removeScalarOutliersFromPointCloud( ...
                X, UI, curOutlierThreshold, outlierNNSize);
        end

        U = UB + UI; % Combine to compute dynamical potential

        T = computeTransitionMatrix(X, U, dt, ...
            'PointPotential', U0, 'ScalarMetric', scalarMetric, ...
            'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', clipThreshold, ...
            'StrictNormalization', strictNormalization, ...
            'VolumeElementType', volumeType, ...
            'VolumeElement', volumeElement);

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

function err = computeSimulationError(measProb, simProb, errorType, X)
%COMPUTESIMULATIONERROR A helper function that computes the error between
%the measured probability at a single experimental time point and a set of
%simulated probabilities.
%
%   WARNING: THIS FUNCTION IS NOT INTENDED FOR EXTERNAL USE. NO INTERNAL
%   QUALITY CHECKS ARE PERFORMED
%
%   INPUT PARAMETERS:
%
%       - measProb:     #N x 1 vector of measured probabilities
%
%       - simProb:      #N x #T matrix of simulated probabilities
%
%       - errorType:    Character vector indicating error type.
%
%           - 'symKLD': symmetric K-L divergence D_KL(M||S) + D_KL(S||M)
%
%           - 'simKLD': K-L divergence relative to the simulated
%           probabilities D_KL(S||M). Heuristically, this divergence is
%           mode seeking
%
%           - 'measKLD': K-L divergence relative to measured probabilities
%           D_KL(M||S). Heuristically this divergence is mean seeking
%
%           - 'MSE': Normalized L-2 error || M - S ||^2 ./ || M ||^2
%
%           - 'geoSphere': Note that the element-wise square root of any
%           N-dimensional discrete probability distribution lies on the
%           N-sphere (since sum(sqrt(P).^2) == 1). Given that embedding,
%           this error computes the geodesic distance along the sphere
%           between two probability distributions
%
%           - 'EMD': A "fast" approximation of the earth movers distance,
%           treating (X, P) as signatures (i.e. a set of spatial points
%           with corresponding weights). In some conceptual sense, this is
%           the best error (since it takes spatial distance into account),
%           but in practice it is too slow to use.
%
%       - X:            #N x dim set of input points on which the dynamics
%                       are defined. Only used if the error type is 'EMD'
%
%   OUTPUT PARAMETERS:
%
%       - err:          1 x #T set of error values for each simulated
%                       probability distribution
%
%   by Dillon Cislo 2024/06/25

if strcmpi(errorType, 'symKLD')
    
    err = (measProb - simProb) .* log2(measProb ./ simProb);
    err(isnan(err)) = 0;
    err = sum(err, 1);
    
elseif strcmpi(errorType, 'simKLD')
    
    err = simProb .* log2(simProb ./ measProb);
    err(isnan(err)) = 0;
    err = sum(err, 1);
    
elseif strcmpi(errorType, 'dataKLD')
    
    err = measProb .* log2(measProb ./ simProb);
    err(isnan(err)) = 0;
    err = sum(err, 1);
    
elseif strcmpi(errorType, 'MSE')
    
    err = sum((measProb-simProb).^2, 1) ./ sum(measProb.^2, 1);
    
elseif strcmpi(errorType, 'geoSphere')
    
    err = acos(sum(sqrt(measProb .* simProb), 1));
    
elseif strcmpi(errorType, 'EMD')
    
    err = zeros(1, size(simProb, 2));
    for tt = 1:size(simProb, 2)
        [~, wasserDist, ~] = PPMMOMT(X, X, simProb(:, tt), measProb, 100);
        err(tt) = wasserDist(end);
    end
    
else
    
    error('Invalid error type');
    
end

end