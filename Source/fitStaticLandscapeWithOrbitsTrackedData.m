function [optErr, optFixHeights, optLoopHeights, optLoopSpeeds, ...
    optD, optScalarMetric, optOutput, optRotV] = ...
    fitStaticLandscapeWithOrbitsTrackedData( ...
    X, dataTracks, dataSteps, dt, allPaths, allLoops, varargin)
%FITSTATICLANDSCAPEWITHORBITSTRACKEDDATA Fits a single static dynamical
%landscape to a set of input time-series data by maximizing the probability
%of the tracked data. The degrees of freedom to be fit are:
%
% (1) The heights of potential minima/saddles/points on loops connected to
%     minima or saddles (stored in the same vector)
% (2) The signed speeds around each periodic orbit (stored in
%     the same vector)
% (3) The heights of each periodic orbit (stored in the same vector)
% (4) A scalar (positive) diffusion constant D
% (5) A uniform scalar metric (literally a single scalar constant), and
% 
%Options are included to fix the diffusion constant, scalar metric, (a
%subset of) the minima/saddle heights, (a subset of) the loop heights, or a
%subset of) the loop speeds to user specified values.
%
%   INPUT PARAMETERS:
%
%       - X:            #N x dim set of input points on which the dynamics
%                       are defined
%
%       - dataTracks:   #NT x #DT matrix of indices into X defining a set
%                       of tracked paths through the dynamical manifold.
%                       Each row corresponds to an independent track. A
%                       cell array containing multiple experimental runs of
%                       different lengths can also be supplied.
%
%       - dataSteps:    The number of transition matrix iterations
%                       corresponding to the elapsed time between points in
%                       the tracks. It is assumed that this number is
%                       identical for all tracks and all experiments. This
%                       number is absolutely crucial to the fit and should
%                       be investigated empirically prior to optimization.
%
%       - dt:           The uniform simulation time step. It is assumed
%                       that all simulation runs use the same time step, 
%                       i.e. the times for all simulations are given by
%                       [0 1 2 ... (numSimTimes-1)] * dt
%
%       - allPaths:     #P x 1 cell array. allPaths{i} is an ordered
%                       list of point IDs defining that path. The end
%                       points of each allPaths{i} correspond to the
%                       minima/saddles in the point set. Paths may also
%                       terminate on a loop. The ordering of the values in
%                       the output 'fixHeights' is determined by the
%                       sorted, unique IDs of the path end points
%
%       - allLoops      #L x 1 cell array. allLoops{i} is an ordered list
%                       of point IDs defining each loop and the positive
%                       orientation around that loop. 
%
%   OPTIONAL INPUT ARGUMENTS (Name, Value)-Pairs:
%
%       - ('InitialGuess', initGuess = []); An initial guess for the
%       parameters supplied as a vector of the form:
%
%           [ (heights); (loop heights); (loop speeds); (D); (scalarMetric) ]
%
%       The user is responsible for ensuring the initial guess is feasible
%       given the various imposed constraints. NOTE: the time scale output
%       for 'constant' time handling is always found by a suboptimization
%       problem at each iteration and is not specified by the initial
%       guess.
%
%       - ('IsSaddle', isSaddle = []): A logical vector indicating which
%       points in the fixed point height list are saddles. Ordering is
%       determined from the 'allPaths' variable. This field must be
%       specified in order to enforce the saddles. Note that this includes
%       loop heights as well.
%
%       - ('EnforceSaddles', enforceSaddles = false): Whether or not to
%       enforce the constraint that index-1 saddles have a greater
%       potential height than the corresponding minima (i.e. prevent
%       saddles from becoming minima). Unless you have strong prior
%       knowledge of the system, this should generally be set to false.
%
%       - ('EnforcePositiveDiffusion', enforcePositiveD = true): Whether to
%       set a bound constraint on the diffusion coefficient. Be sure to
%       choose a good initial condition if you turn this off!
%
%       - ('EnforcePositiveMetric', enforcePositiveMetric = true): Whether
%       or not to set a bound constraint on the scalar metric. Be sure to
%       choose a good initial condition if you turn this off!
%
%       - ('ConstHeightSum', constHeightSum = []): A user supplied constant
%       that constrains the total sum of the minima/saddle/loop heights.
%
%       - ('ConstFixedHeights', constFixHeights = []): A set of user
%       supplied values for a subset of the minima/saddle heights, supplied
%       a (numFixPoints) x 1 vector. Non-NaN entries correspond to
%       specified values. If supplied, these fields are held fixed over
%       optimization
%
%       - ('ConstLoopHeights', constLoopHeights = []): A set of user
%       supplied values for a subset of the loop heights, supplied
%       a (numLoops) x 1 vector. Non-NaN entries correspond to
%       specified values. If supplied, these fields are held fixed over
%       optimization
%
%       - ('ConstLoopSpeeds', constLoopSpeeds = []): A set of user
%       supplied values for a subset of the loop speeds, supplied
%       a (numLoops) x 1 vector. Non-NaN entries correspond to
%       specified values. If supplied, these fields are held fixed over
%       optimization
%
%       - ('ConstDiffusionCoefficient', constD = []): The diffusion
%       coefficient for the dynamical drift-diffusion process
%
%       - ('ConstScalarMetric', constScalarMetric = []): The uniform scalar
%       metric, (i.e. the same for all input points) that re-scales the
%       dynamical velocity (i.e. v = -(1/scalarMetric) * \nabla U). If
%       supplied, this property is held fixed over optimization
%
%       - ('PrecomputeQuadProg', precomputeQuadProg = true): Whether or not
%       to precompute the solver information needed to compute the
%       interpolated potential and rotational velocities. This can help to
%       significantly speed up the code, but can also lead to OOM error for
%       large problems run in parallel.
%
%       - ('RotationProblemType', rotMethod = 'bilaplacian-flat'): The
%       structure of the quadratic problem used to compute the rotational
%       velocities. If set to 'bilaplacian', the quadratic problem is set
%       up using a true connection Laplacian. If set to
%       'bilalplacian-flat', the scalar problem is solved using the
%       standard Laplacian separately for each coordinate dimension.
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
%       - ('DataSetWeights', dataSetWeights = []): The relative weight of
%       each transition in each data set to the total error computation.
%       Weights should sum to one. All time points within each data set
%       will be weighted identically. Note that this means longer data sets
%       will be weighted more heavily in summation by default unless
%       weights are picked accordingly.
%
%   Physical Constants ----------------------------------------------------
%   You are allowed to set these for the sake of completeness, but you are
%   strongly advised to just leave them equal to one
%
%       - ('PointDiffusionCoefficient', D0 = 1): The diffusion coefficient
%       for the drift-diffusion process assumed to generate the input point
%       set
%
%   Manifold/Point Set/Potential Properties -------------------------------
%   These fields are computed directly from (X, dt) if they are not
%   supplied
%
%       - ('IntrinsicDimension', intDim = []): The intrinsic dimensionality
%       of the point set manifold. Assumed to be the full dimension of the
%       ambient space if not supplied.
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
%       - ('ConnectionLaplacian', Lconn = []): (#N * intDim) x #N point
%       cloud connection Laplacian
%
%       - ('ConnectionMassMatrix', Mconn = []): (#N * intDim) x #N
%       diagonal mass matrix corresponding to Lconn
%
%       - ('PointCloudTangentBases', allBases = {}): #N x 1 cell array.
%       allBases{i} is a dim x intDim array of orthonormal tangent space
%       basis vectors for the point at X(i).
%
%       - ('ParallelTransportMaps', allPTMaps = {}): #P x #P cell array.
%       allPTMaps{i,j} is the orthogonal transformation that maps vectors
%       in T_{X(j)}M -> T_{X(i)}M. allPTMaps{i,j} is empty if A(i,j) is
%       false. allPTMaps{i,i} is always the identity. note that
%       allPTMaps{i,j} acts on ROW VECTORS from the RIGHT!
%
%   Loop Tangent Vector Options -------------------------------------------
%
%       - ('LoopTangentVectors', allLoopTangentVectors = {}): A #L x 1 cell
%       array containing the unit tangent vectors at each point along each
%       loop.
%
%       - ('LoopSmoothIters', loopSmoothIters = 0): The number of moving
%       average smoothing iterations used to smooth the loop tangent
%       vectors if none are explicitly supplied.
%
%   'interpolatePotentialKHarmonic' Options -------------------------------
%
%       - ('TikhonovRegularization', regSigma = 1e-12): A small positive
%       quantity used to make the quadratic problem positive definite
%
%       - ('RemoveOutliers', removeOutliers = true): Whether or not to
%       remove outliers generated during the interpolation process
%
%       - ('OutlierThreshold', outlierThreshold = []): The threshold
%       above and below which outliers are removed. If empty and
%       removeOutliers == true, this is automatically set from the values
%       of interpVals
%
%       - ('RotationalVelocityOutlierThreshold', rotOutlierThreshold = []):
%       The threshold on velocity norm above and below which outliers are
%       removed. If empty and removeOutliers == true, this is automatically
%       set from the values of interpVals
%
%       - ('OutlierNeighbors', outlierNNSize = 10): The number of neighbors
%       over which to average to in order to find the new value for any
%       removed outliers
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
%       set of positive weights).
%
%       - ('PathInterpolationMethod', pathInterpMethod = 'weighted'):
%       Whether to interpolate using the path length weights or to just
%       interpolate according to the number of points along the path. If no
%       path lengths are supplied, 'weighted' and 'unweighted' are
%       equivalent.
%
%       - ('PathCollisionMethod', pathCollisionMethod = 'mean'): How to
%       handle the case where multiple paths intersect at a subset of the
%       points.
%
%   'computeLogTransitionMatrix' Options ----------------------------------
%
%       - ('ClipThreshold', clipThreshold = 1e-14): Exponential
%       distributions produce insanely small values. Entries |T(i,j)| <
%       this threshold are just set to zero. BE CAREFUL HERE - this has not
%       been thoroughly tested and it may produce reducible transition
%       matrices
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
%       - optErr:        The average K-L divergence error between the
%                           optimized simulated time courses and their
%                           corresponding data sets
%
%       - optFixHeights:    1 x #H list of optimized minima/saddle heights.
%                           Ordering in this list is determined from the
%                           'allPaths' input variable
%
%       - optLoopHeights:   1 x #L list of optimized loop heights.
%
%       - optLoopSpeeds:    1 x #L list of optimized loop speeds.
%
%       - optD:             The optimized dynamical diffusion coefficient
%
%       - optScalarMetric:  The optimized uniform scalar metric
%
%       - optOutput:        A struct containing information about the
%                           optimization process. See 'fmincon' or
%                           'fminunc'
%
%       - optRotV:          #N x dim matrix of optimized rotational
%                           velocities
%
%   by Dillon Cislo 2026/06/17

%==========================================================================
% INPUT PROCESSING
%==========================================================================
validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'}, ...
    'fitStaticLandscapeWithOrbitsTrackedData', 'X');
numPoints = size(X,1); dim = size(X,2);

% This is probably not the most efficient way to format this, but it does
% streamline multiple experiment handling
if ~iscell(dataTracks), dataTracks = {dataTracks}; end

% More explicit data handling is performed after optional input handling
validateattributes(dataTracks, {'cell'}, {'vector'}, ...
    'fitStaticLandscapeWithOrbitsTrackedData', 'dataTracks');
numDataSets = numel(dataTracks);
validateattributes(dataSteps, {'numeric'}, {'scalar', 'integer', ...
    'positive', 'finite', 'real'}, ...
    'fitStaticLandscapeWithOrbitsTrackedData', 'dataSteps');

validateattributes(dt, {'numeric'}, {'scalar', 'positive', ...
    'finite', 'real'}, ...
    'fitStaticLandscapeWithOrbitsTrackedData', 'dt');

validateattributes(allPaths, {'cell'}, {'vector'}, ...
    'fitStaticLandscapeWithOrbitsTrackedData', 'allPaths');
numPaths = numel(allPaths);
assert(numPaths > 0, 'No paths supplied');
cellfun(@(x) validateattributes(x, {'numeric'}, {'vector', 'integer', ...
    'positive', 'finite', 'real', '<=', numPoints}, ...
    'fitStaticLandscapeWithOrbitsTrackedData', 'allPaths'), ...
    allPaths, 'Uni', false);
assert(all(cellfun(@(x) numel(x) > 1, allPaths, 'Uni', true)), ...
    'Paths must have at least two points');
assert(all(cellfun(@(x) isequal(x, unique(x, 'stable')), ...
    allPaths, 'Uni', true)), 'Paths contain duplicate points');
allPaths = cellfun(@(x) x(:), allPaths, 'Uni', false);
allPaths = allPaths(:);

validateattributes(allLoops, {'cell'}, {'vector'}, ...
    'fitStaticLandscapeWithOrbitsTrackedData', 'allLoops');
numLoops = numel(allLoops);
assert(numLoops > 0, ['You should not be using this function without ' ...
    'any periodic orbits. Consider "fitStaticLandscape" instead.']);
cellfun(@(x) validateattributes(x, {'numeric'}, {'vector', 'integer', ...
    'positive', 'finite', 'real', '<=', numPoints}, ...
    'fitStaticLandscapeWithOrbitsTrackedData', 'allLoops'), ...
    allLoops, 'Uni', false);
allLoops = cellfun(@(x) x(1:(end-(x(end) == x(1)))), ...
    allLoops, 'Uni', false);
allLoops = cellfun(@(x) x(:), allLoops, 'Uni', false);
assert(all(cellfun(@(x) numel(x) > 2, allLoops, 'Uni', true)), ...
    'Loops must have at least three points');
assert(all(cellfun(@(x) isequal(x, unique(x, 'stable')), ...
    allLoops, 'Uni', true)), 'Loops contain duplicate points');
allLoops = allLoops(:);

% Clip paths so that they never contain more than one loop point either at
% the end or the beginning
allLoopIDx = vertcat(allLoops{:});
allLoopIDx = allLoopIDx(:);
assert(numel(allLoopIDx) == numel(unique(allLoopIDx)), ...
    'Loops must not share points');
pathEdgeKeepIDx = cell(numPaths, 1);
for i = 1:numPaths
    rmIDx = ismember(allPaths{i}, allLoopIDx);
    hasLoopIntersection = any(rmIDx);
    if ~hasLoopIntersection
        keepIDx = true(size(allPaths{i}));
    elseif find(~rmIDx, 1, 'last' ) < find(rmIDx, 1, 'first')
        % If loop points come at the end of the path
        rmIDx = [false; rmIDx(2:end) & rmIDx(1:(end-1))];
        keepIDx = ~rmIDx;
    elseif find(rmIDx, 1, 'last') < find(~rmIDx, 1, 'first')
        % If loop points come at the start of the path
        rmIDx = [rmIDx(2:end) & rmIDx(1:(end-1)); false];
        keepIDx = ~rmIDx;
    else
        error('Path %d contains invalid loop intersections', i)
    end
    % Edge j survives iff both endpoint vertices survive.
    pathEdgeKeepIDx{i} = keepIDx(1:(end-1)) & keepIDx(2:end);
    allPaths{i} = allPaths{i}(keepIDx);
    if hasLoopIntersection
        assert(sum(ismember(allPaths{i}, allLoopIDx)) == 1, ...
            'Failed to clip loop points from path %d', i);
    end
end

% fixPointIDx: The unique set of indices in 'X' corresponding to
% minima/saddles/loop termini, i.e. numel(fixPointIDx) == (# minima) + (#
% saddles) + (# loops)
% fixInPathIDx: #Px2 array of indices into fixPointIDx
% mapping minima/saddles/loop back into their corresponding paths
fixInPathIDx = cellfun(@(x) [x(1); x(end)], allPaths, 'Uni', false);
fixInPathIDx = cell2mat(fixInPathIDx);
fixPointIDx = unique(reshape(fixInPathIDx.', [], 1));
fixPointIDx(ismember(fixPointIDx, allLoopIDx)) = [];
fixPointIDx = [fixPointIDx; cellfun(@(x) x(1), allLoops, 'Uni', true)];
for i = 1:numLoops
    fixInPathIDx(ismember(fixInPathIDx, allLoops{i})) = allLoops{i}(1);
end
[~, fixInPathIDx] = ismember(fixInPathIDx, fixPointIDx);
fixInPathIDx = reshape(fixInPathIDx, [2 numPaths]).';
numFixPoints = numel(fixPointIDx);
numNonLoopFixPoints = numFixPoints - numLoops;
assert(all(fixInPathIDx(:) > 0) && ...
    all(ismember((1:numNonLoopFixPoints).', fixInPathIDx(:))), ...
    'Failed to assign path endpoints');

%--------------------------------------------------------------------------
% OPTIONAL INPUT PROCESSING
%--------------------------------------------------------------------------

% Optimization options
initGuess = [];
isSaddle = false(1, numFixPoints);
enforceSaddles = false;
enforcePositiveDiffusion = true;
enforcePositiveMetric = true;
constHeightSum = [];
constFixHeights = nan(numNonLoopFixPoints, 1);
constLoopHeights = nan(numLoops, 1);
constLoopSpeeds = nan(numLoops, 1);
constD = [];
constScalarMetric = [];
precomputeQuadProg = true;
rotMethod = 'bilaplacian-flat';
optOptions = {};
upperBounds = [];
lowerBounds = [];
dataSetWeights = [];

allRotMethods = {'bilaplacian', 'bilaplacian-flat'};

% Physical constants
D0 = 1;

% Manifold/point set/potential properties
intDim = [];
U0 = [];
UB = [];
L = [];
M = [];
Lconn = [];
Mconn = [];
allBases = {};
allPTMaps = {};

% Loop tangent vector options
allLoopTangentVectors = {};
loopSmoothIters = 0;

% 'interpolatePotentialKHarmonic' options
regSigma = 1e-12;
removeOutliers = true;
outlierThreshold = [];
rotOutlierThreshold = [];
outlierNNSize = 10;
normalizeMassMatrix = true;

% 'interpolateValuesAlongPath' options
allPathLengths = {};
pathInterpMethod = 'weighted';
pathCollisionMethod = 'mean';

% 'computeLogTransitionMatrix' options
clipThreshold = 1e-14;
volumeType = 'graphlaplacian';
allVolumeTypes = {'graphlaplacian', 'laplacebeltrami'};

% General options
useGPU = true;
verbose = false;

supportedOptions = {'IsSaddle', ...
    'EnforceSaddles', 'ConstHeightSum', ...
    'ConstFixedHeights', 'ConstScalarMetric', 'Verbose', ...
    'OptimizationOptions', 'ConstDiffusionCoefficient', ...
    'PointDiffusionCoefficient', 'PointPotential', 'BasePotential', ...
    'Laplacian', 'MassMatrix', 'TikhonovRegularization', ...
    'RemoveOutliers', 'OutlierThreshold', 'OutlierNeighbors', ...
    'NormalizeMassMatrix', 'PathLengths', 'PathInterpolationMethod', ...
    'PathCollisionMethod', 'ClipThreshold', ...
    'UseGPU', 'InitialGuess', 'EnforcePositiveMetric', ...
    'PrecomputeQuadProg', 'VolumeElementType', 'UpperBounds', ...
    'LowerBounds', 'DataSetWeights', ...
    'EnforcePositiveDiffusion', 'ConstLoopHeights', 'ConstLoopSpeeds', ...
    'IntrinsicDimension', 'ConnectionLaplacian', 'ConnectionMassMatrix', ...
    'RotationProblemType', 'LoopTangentVectors', 'LoopSmoothIters', ...
    'RotationalVelocityOutlierThreshold', 'PointCloudTangentBases', ...
    'ParallelTransportMaps'};
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
                'fitStaticLandscapeWithOrbitsTrackedData', 'initGuess');
            if (size(initGuess, 2) ~= 1), initGuess = initGuess.'; end
        end
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
                '<=', numFixPoints}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'isSaddle');
            isSaddle = ismember((1:numFixPoints).', isSaddle);
        end
    end

    if strcmpi(varargin{i}, 'EnforceSaddles')
        enforceSaddles = varargin{i+1};
        validateattributes(enforceSaddles, {'logical'}, {'scalar'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'enforceSaddles');
    end

    if strcmpi(varargin{i}, 'EnforcePositiveDiffusion')
        enforcePositiveDiffusion = varargin{i+1};
        validateattributes(enforcePositiveDiffusion, {'logical'}, ...
            {'scalar'}, mfilename, 'enforcePositiveDiffusion');
    end

    if strcmpi(varargin{i}, 'EnforcePositiveMetric')
        enforcePositiveMetric = varargin{i+1};
        validateattributes(enforcePositiveMetric, {'logical'}, {'scalar'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'enforcePositiveMetric');
    end

    if strcmpi(varargin{i}, 'ConstHeightSum')
        constHeightSum = varargin{i+1};
        if ~isempty(constHeightSum)
            validateattributes(constHeightSum, {'numeric'}, ...
                {'scalar', 'finite', 'real'}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'constHeightSum');
        end
    end
    
    if strcmpi(varargin{i}, 'ConstFixedHeights')
        constFixHeights = varargin{i+1};
        if ~isempty(constFixHeights)
            assert(isvector(constFixHeights) && ...
                numel(constFixHeights) == numNonLoopFixPoints, ...
                'Constrained heights are improperly sized');
            constFixHeights = constFixHeights(:);
        else
            constFixHeights = nan(numNonLoopFixPoints, 1);
        end
    end

    if strcmpi(varargin{i}, 'ConstLoopHeights')
        constLoopHeights = varargin{i+1};
        if ~isempty(constLoopHeights)
            assert(isvector(constLoopHeights) && ...
                numel(constLoopHeights) == numLoops, ...
                'Constrained loop heights are improperly sized');
            constLoopHeights = constLoopHeights(:);
        else
            constLoopHeights = nan(numLoops, 1);
        end
    end

    if strcmpi(varargin{i}, 'ConstLoopSpeeds')
        constLoopSpeeds = varargin{i+1};
        if ~isempty(constLoopSpeeds)
            assert(isvector(constLoopSpeeds) && ...
                numel(constLoopSpeeds) == numLoops, ...
                'Constrained loop speeds are improperly sized');
            constLoopSpeeds = constLoopSpeeds(:);
        else
            constLoopSpeeds = nan(numLoops, 1);
        end
    end

    if strcmpi(varargin{i}, 'ConstDiffusionCoefficient')
        constD = varargin{i+1};
        if ~isempty(constD)
            validateattributes(constD, {'numeric'}, ...
                {'scalar', 'positive', 'finite', 'real'}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'constD');
        end
    end

    if strcmpi(varargin{i}, 'ConstScalarMetric')
        constScalarMetric = varargin{i+1};
        if ~isempty(constScalarMetric)
            validateattributes(constScalarMetric, {'numeric'}, ...
                {'scalar', 'positive', 'finite', 'real'}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'constScalarMetric');
        end
    end

    if strcmpi(varargin{i}, 'PrecomputeQuadProg')
        precomputeQuadProg = varargin{i+1};
        validateattributes(precomputeQuadProg, {'logical'}, {'scalar'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'precomputeQuadProg');
    end

    if strcmpi(varargin{i}, 'RotationProblemType')
        rotMethod = varargin{i+1};
        validateattributes(rotMethod, {'char'}, {'vector'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'rotMethod');
        assert(ismember(lower(rotMethod), allRotMethods), ...
            'Invalid rotation problem type supplied');
    end

    if strcmpi(varargin{i}, 'OptimizationOptions')
        optOptions = varargin{i+1};
        validateattributes(optOptions, {'cell'}, {'vector'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'optOptions');
    end

    if strcmpi(varargin{i}, 'UpperBounds')
        upperBounds = varargin{i+1};
        if ~isempty(upperBounds)
            validateattributes(upperBounds, {'numeric'}, {'vector', ...
                'numel', numFixPoints+numLoops+2, 'nonnan'}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'upperBounds');
            if (size(upperBounds, 2) ~= 1)
                upperBounds = upperBounds .';
            end
        end
    end

    if strcmpi(varargin{i}, 'LowerBounds')
        lowerBounds = varargin{i+1};
        if ~isempty(lowerBounds)
            validateattributes(lowerBounds, {'numeric'}, {'vector', ...
                'numel', numFixPoints+numLoops+2, 'nonnan'}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'lowerBounds');
            if (size(lowerBounds, 2) ~= 1)
                lowerBounds = lowerBounds .';
            end
        end
    end

    if strcmpi(varargin{i}, 'DataSetWeights')
        dataSetWeights = varargin{i+1};
        if ~isempty(dataSetWeights)
            if iscell(dataSetWeights)
                validateattributes(dataSetWeights, {'cell'}, ...
                    {'vector', 'numel', numDataSets}, ...
                    'fitStaticLandscapeWithOrbitsTrackedData', 'dataSetWeights');
            else
                validateattributes(dataSetWeights, {'numeric'}, ...
                    {'vector', 'finite', 'positive', 'real', 'numel', numDataSets}, ...
                    'fitStaticLandscapeWithOrbitsTrackedData', 'dataSetWeights');
                dataSetWeights = num2cell(dataSetWeights(:));
            end
            if (size(dataSetWeights, 1) ~= 1)
                dataSetWeights = dataSetWeights.';
            end
        end
    end

    % Physical Constants --------------------------------------------------
    
    if strcmpi(varargin{i}, 'PointDiffusionCoefficient')
        D0 = varargin{i+1};
        validateattributes(D0, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'D0');
    end

    % Manifold/Point Set/Potential Properties -----------------------------

    if strcmpi(varargin{i}, 'IntrinsicDimension')
        intDim = varargin{i+1};
        if ~isempty(intDim)
            validateattributes(intDim, {'numeric'}, {'scalar', ...
                'positive', 'integer', 'finite', 'real'}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'intDim');
        end
    end

    if strcmpi(varargin{i}, 'PointPotential')
        U0 = varargin{i+1};
        if ~isempty(U0)
            validateattributes(U0, {'numeric'}, {'vector', ...
                'finite', 'real', 'numel', numPoints}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'U0');
            if (size(U0,2) ~= 1), U0 = U0.'; end
        end
    end

    if strcmpi(varargin{i}, 'BasePotential')
        UB = varargin{i+1};
        if ~isempty(UB)
            validateattributes(UB, {'numeric'}, {'vector', ...
                'finite', 'real', 'numel', numPoints}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'UB');
            if (size(UB,2) ~= 1), UB = UB.'; end
        end
    end

    if strcmpi(varargin{i}, 'Laplacian'), L = varargin{i+1}; end

    if strcmpi(varargin{i}, 'MassMatrix'), M = varargin{i+1}; end

    if strcmpi(varargin{i}, 'ConnectionLaplacian')
        Lconn = varargin{i+1};
    end

    if strcmpi(varargin{i}, 'ConnectionMassMatrix')
        Mconn = varargin{i+1};
    end

    if strcmpi(varargin{i}, 'PointCloudTangentBases')
        allBases = varargin{i+1};
    end

    if strcmpi(varargin{i}, 'ParallelTransportMaps')
        allPTMaps = varargin{i+1};
    end

    % Loop Tangent Vector Options -----------------------------------------

    if strcmpi(varargin{i}, 'LoopTangentVectors')
        allLoopTangentVectors = varargin{i+1};
        if ~isempty(allLoopTangentVectors)
            validateattributes(allLoopTangentVectors, {'cell'}, {'vector', ...
                'numel', numLoops}, 'fitStaticLandscapeWithOrbitsTrackedData', ...
                'allLoopTangentVectors');
            if size(allLoopTangentVectors, 2) ~= 1
                allLoopTangentVectors = allLoopTangentVectors.';
            end
            cellfun(@(x, y) validateattributes(x, {'numeric'}, {'2d', ...
                'finite', 'real', 'ncols', dim, 'nrows', numel(y)}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'allLoopTangentVectors'), ...
                allLoopTangentVectors, allLoops, 'Uni', false);
        end
    end

    if strcmpi(varargin{i}, 'LoopSmoothIters')
        loopSmoothIters = varargin{i+1};
        validateattributes(loopSmoothIters, {'numeric'}, {'scalar', ...
            'integer', 'finite', 'real'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'loopSmoothIters');
    end

    % 'interpolatePotentialKHarmonic' Options -----------------------------

    if strcmpi(varargin{i}, 'TikhonovRegularization')
        regSigma = varargin{i+1};
        validateattributes(regSigma, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'regSigma');
    end

    if strcmpi(varargin{i}, 'RemoveOutliers')
        removeOutliers = varargin{i+1};
        validateattributes(removeOutliers, {'logical'}, {'scalar'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'removeOutliers');
    end

    if strcmpi(varargin{i}, 'OutlierThreshold')
        outlierThreshold = varargin{i+1};
        if ~isempty(outlierThreshold)
            validateattributes(outlierThreshold, {'numeric'}, ...
                {'vector', 'numel', 2, 'finite', 'real'}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'outlierThreshold');
            assert(outlierThreshold(2) > outlierThreshold(1), ...
                ['Outlier threshold must have a second element ' ...
                'that is greater than its first element']); 
        end
    end

    if strcmpi(varargin{i}, 'RotationalVelocityOutlierThreshold')
        rotOutlierThreshold = varargin{i+1};
        if ~isempty(rotOutlierThreshold)
            validateattributes(rotOutlierThreshold, {'numeric'}, ...
                {'vector', 'numel', 2, 'finite', 'real'}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'rotOutlierThreshold');
            assert(rotOutlierThreshold(2) > rotOutlierThreshold(1), ...
                ['Outlier threshold must have a second element ' ...
                'that is greater than its first element']); 
        end
    end

    if strcmpi(varargin{i}, 'OutlierNeighbors')
        outlierNNSize = varargin{i+1};
        validateattributes(outlierNNSize, {'numeric'}, ...
            {'positive', 'integer', 'scalar', 'finite', 'real'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'outlierNNSize');
    end

    if strcmpi(varargin{i}, 'NormalizeMassMatrix')
        normalizeMassMatrix = varargin{i+1};
        validateattributes(normalizeMassMatrix, {'logical'}, {'scalar'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'normalizeMassMatrix');
    end

    % 'interpolateValuesAlongPath' Options --------------------------------

    if strcmpi(varargin{i}, 'PathLengths')
        allPathLengths = varargin{i+1};
        validateattributes(allPathLengths, {'cell'}, ...
            {'vector', 'numel', numPaths}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'allPathLengths');
        allPathLengths = allPathLengths(:);
        for jj = 1:numPaths

            validateattributes(allPathLengths{jj}, {'numeric'}, ...
                {'vector', 'positive', 'finite', 'real', ...
                'numel', numel(pathEdgeKeepIDx{jj})}, ...
                'fitStaticLandscapeWithOrbitsTrackedData', 'allPathLengths');

            allPathLengths{jj} = allPathLengths{jj}(:);
            allPathLengths{jj} = allPathLengths{jj}(pathEdgeKeepIDx{jj});

            assert(numel(allPathLengths{jj}) == numel(allPaths{jj}) - 1, ...
                'Path length clipping failed for path %d', jj);

        end
    end

    if strcmpi(varargin{i}, 'PathInterpolationMethod')
        pathInterpMethod = lower(varargin{i+1});
        validateattributes(pathInterpMethod, {'char'}, {'vector'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'PathInterpolationMethod');
    end

    if strcmpi(varargin{i}, 'PathCollisionMethod')
        pathCollisionMethod = lower(varargin{i+1});
        validateattributes(pathCollisionMethod, {'char'}, {'vector'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'pathCollisionMethod');
    end

    % 'computeTransitionMatrix' Options -----------------------------------

    if strcmpi(varargin{i}, 'ClipThreshold')
        clipThreshold = varargin{i+1};
        validateattributes(clipThreshold, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'clipThreshold');
    end

    if strcmpi(varargin{i}, 'VolumeElementType')
        volumeType = lower(varargin{i+1});
        validateattributes(volumeType, {'char'}, {'vector'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'volumeType');
        assert(ismember(volumeType, allVolumeTypes), ...
            'Invalid volume element type');
    end

    % General Options -----------------------------------------------------

    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'useGPU');
    end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'verbose');
    end

end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

%--------------------------------------------------------------------------
% OPTIMIZATION INPUT PROCESSING
%--------------------------------------------------------------------------

if ~any(isSaddle), enforceSaddles = false; end
constHeightsAndSpeeds = [constFixHeights; constLoopHeights; constLoopSpeeds];
numConstHeightsAndSpeeds = sum(~isnan(constHeightsAndSpeeds));

% Process the supplied initial guess --------------------------------------
if isempty(initGuess)
   
    initGuess = [zeros(numFixPoints+numLoops, 1); 1; 1];

    % Ensure the constant height sum constraint is observed if constrained
    % heights are supplied
    if ~isempty(constHeightSum)

        % Unconstrained values are NaN
        tmpHeightVals = [constFixHeights; constLoopHeights];
        tmpFreeIDx = isnan(tmpHeightVals);
        tmpFixedSum = sum(tmpHeightVals(~tmpFreeIDx));

        assert(any(tmpFreeIDx) || abs(tmpFixedSum - constHeightSum) < 1e-12, ...
            ['Fixed heights are inconsistent with the supplied ' ...
            'ConstHeightSum constraint']);

        tmpHeightVals(tmpFreeIDx) = ...
            (constHeightSum - tmpFixedSum) ./ sum(tmpFreeIDx);

        initGuess(1:numFixPoints) = tmpHeightVals;

        clear tmpHeightVals tmpFreeIDx tmpFixedSum

    end

end

assert(numel(initGuess) == (numFixPoints+numLoops+2), ...
    'Initial guess is improperly sized');

if ~isempty(constD)
    initGuess(end-1) = constD;
end
assert(initGuess(end-1) > 0, ['Diffusion coefficient ' ...
    'must be positive in the initial guess']);

if ~isempty(constScalarMetric)
    initGuess(end) = constScalarMetric;
end
assert(initGuess(end) > 0, ['Scalar metric ' ...
    'must be positive in the initial guess']);

if (numConstHeightsAndSpeeds > 0)
    tmpIDx = find(~isnan(constHeightsAndSpeeds));
    initGuess(tmpIDx) = constHeightsAndSpeeds(tmpIDx);
end

if ~isempty(constHeightSum)
    assert(abs(sum(initGuess(1:numFixPoints))-constHeightSum) < 1e-12, ...
        ['User supplied initial guess does not adhere to the ' ...
        'fixed point/loop height sum constraint']);
end


% Estimate point set pseudo potential from point cloud---------------------
if isempty(U0)
    if verbose, disp('Computing point set potential:'); end
    U0 = -D0 * log(gaussianKDE(X, X, [], sqrt(2 * dt), verbose));
end

if isempty(UB), UB = U0; end

% Process the data points/data times/initial conditions -------------------

if isempty(dataSetWeights)
    % Default is uniform weighting of transitions within data sets
    dataSetWeights = num2cell(ones(numDataSets, 1));
end

for i = 1:numDataSets

    curTracks = dataTracks{i};
    validateattributes(curTracks, {'numeric'}, {'2d', 'finite', ...
        'real', 'integer', 'positive', '<=', numPoints}, ...
        'fitStaticLandscapeWithOrbitsTrackedData', 'dataTracks');
    assert(size(curTracks, 2) > 1, ...
        'Each track must contain at least two points');

    % Tracks get converted from sequences of indices in X to the
    % corresponding sequences of transition probabilities Tij (X(j) ->
    % X(i))
    curTracks = sub2ind([numPoints, numPoints], curTracks(:, 2:end), ...
        curTracks(:, 1:(end-1)));

    % Ensure the data set weights are properly sized
    if isscalar(dataSetWeights{i})
        validateattributes(dataSetWeights{i}, {'numeric'}, {'scalar', ...
            'finite', 'real',  'positive'}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'dataSetWeights');
        dataSetWeights{i} = dataSetWeights{i} * ones(numel(curTracks), 1);
    else
        validateattributes(dataSetWeights{i}, {'numeric'}, {'2d', ...
            'finite', 'real',  'positive', 'nrows', size(curTracks,1), ...
            'ncols', size(curTracks,2)}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'dataSetWeights');
        dataSetWeights{i} = dataSetWeights{i}(:);
    end

    dataTracks{i} = curTracks(:);

end

% For processing we just want simple column vectors of weights and linear
% indices
dataTracks = vertcat(dataTracks{:});
dataSetWeights = vertcat(dataSetWeights{:});
dataSetWeights = dataSetWeights ./ sum(dataSetWeights);

if useGPU
    dataTracks = gpuArray(dataTracks);
    dataSetWeights = gpuArray(dataSetWeights);
end

%--------------------------------------------------------------------------
% MANIFOLD/POINT SET/POTENTIAL OPTION PROCESSING
%--------------------------------------------------------------------------

% Intrinsic dimension is ambient dimension by default
if isempty(intDim), intDim = dim; end

% Compute point cloud volume element --------------------------------------

if verbose
    fprintf('Computing transition matrix volume element... ');
end

if strcmpi(volumeType, 'GraphLaplacian')

    volumeElement = exp(U0 ./ D0) ./ numPoints;

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

% Check connection Laplacian/mass matrix/point cloud tangent space --------

if strcmpi(rotMethod, 'bilaplacian-flat') || (numLoops == 0)

    Lconn = [];
    Mconn = [];
    allBases = {};
    % allPTMaps = {};

elseif strcmpi(rotMethod, 'bilaplacian')
    assert(intDim < dim, ['It is severely wasteful to use a full ' ...
        'connection Laplacian on a flat manifold'])

    if removeOutliers
        warning('Outlier removal is NOT geometry aware for non-flat problems');
    end

    % Validate point cloud tangent space
    if (isempty(allBases) || isempty(allPTMaps))
        assert(isempty(Lconn) && isempty(Mconn), ['If you are supplying' ...
            'a connection Laplacian/mass matrix, you must also supply ' ...
            'the corresponding tangent bases and parallel transport maps']);

        if verbose, fprintf('Building point cloud tangent space... '); end
        [allBases, allPTMaps] = buildPointCloudTangentSpace( ...
            X, intDim, 18, 'knn', false);
        if verbose, fprintf('Done\n'); end

    else

        % It is up to the user to ensure that the basis vectors are
        % orthonormal
        validateattributes(allBases, {'cell'}, ...
            {'vector', 'numel', numPoints}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'allBases');
        cellfun(@(x) validateattributes(x, {'numeric'}, ...
            {'2d', 'finite', 'real', 'nrows', dim, 'ncols', intDim}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'allBases'), allBases, ...
            'Uni', false);

        % It is up to the user to determine if the appropriate entries are
        % proper orthogonal transformations
        validateattributes(allPTMaps, {'cell'}, ...
            {'2d', 'nrows', numPoints, 'ncols', numPoints}, ...
            'fitStaticLandscapeWithOrbitsTrackedData', 'allBases');

    end

    % Validate connection Laplacian and mass matrix
    if (isempty(Lconn) || isempty(Mconn))
        
        if verbose, fprintf('Building connection Laplacian/mass matrix... '); end
        [Lconn, Mconn] = connectionLaplacian(allPTMaps, L, M, true, false);
        if verbose, fprintf('Done\n'); end

    else

        validateattributes(Lconn, {'numeric'}, {'2d', 'finite', 'real', ...
            'ncols', intDim * numPoints, 'nrows', intDim * numPoints});
        assert(issymmetric(Lconn), 'Connection Laplacian is not symmetric');

        % Check for positive-definiteness
        [~, isPD] = chol(Lconn); isPD = isPD == 0;
        if ~isPD
            [~, isPD] = chol(-Lconn); isPD = isPD == 0;
            if isPD
                Lconn = -Lconn;
            else
                error('Connection Laplacian is neither positive nor negative definite');
            end
        end

        validateattributes(Mconn, {'numeric'}, {'2d', 'finite', 'real', ...
            'ncols', intDim * numPoints, 'nrows', intDim * numPoints});
        assert(isdiag(Mconn), 'Connection mass matrix is not diagonal');
        assert(all(diag(Mconn) > 0), 'Connection mass matrix contains non-positive masses');

        clear isPD

    end



else

    error('Invalid rotation problem type supplied');

end

clear allPTMaps

% Compute loop tangent vectors, necessary ---------------------------------
if isempty(allLoopTangentVectors)
    if verbose, fprintf('Building loop tangent vectors... '); end
    allLoopTangentVectors = cellfun(@(x) ...
        computePathTangentVectors(X, x, loopSmoothIters, true), ...
        allLoops, 'Uni', false);
    if verbose, fprintf('Done\n'); end
end

% Build quadratic coefficients --------------------------------------------
% We explicitly build the biharmonic operator here.
% 'interpolatePotentialKHarmonic' would have more flexibility but is
% generally slower.

Q = L*(M\L);
if ~issymmetric(Q), Q = (Q + Q.') ./ 2; end % Correct for roundoff error
if (regSigma > 0), Q = Q + regSigma .* eye(size(Q)); end
clear L M

if (isempty(Lconn) || isempty(Mconn))
    Qconn = [];
else
    Qconn = Lconn * (Mconn \ Lconn);
    if ~issymmetric(Qconn)
        Qconn = (Qconn + Qconn.') ./ 2; % Correct for roundoff error
    end
    if (regSigma > 0)
        Qconn = Qconn + regSigma .* eye(size(Qconn));
    end
end
clear Lconn Mconn

if isempty(Qconn)
    knownLoopIDx = [];
else
    knownLoopIDx = allLoopIDx + numPoints * (0:(intDim-1));
    knownLoopIDx = knownLoopIDx(:);
end

% Pre-compute quadratic solver information for constructing the
% interpolated potential
if precomputeQuadProg

    fprintf('Pre-computing quadratic solver info\n')
    [~, tmpKnownIDx] = interpolateValuesAlongPath(ones(numPaths, 2), ...
        allPaths, 'PathLengths', allPathLengths, ...
        'InterpolationMethod', pathInterpMethod, ...
        'CollisionMethod', pathCollisionMethod);
    tmpKnownIDx = unique([tmpKnownIDx; allLoopIDx], 'stable');
    F = min_quad_with_fixed_precompute(Q, tmpKnownIDx, []);

    if isempty(Qconn)
        Frot = min_quad_with_fixed_precompute(Q, allLoopIDx, []);
    else
        Frot = min_quad_with_fixed_precompute(Qconn, knownLoopIDx, []);
    end

else

    F = []; Frot = [];

end

%==========================================================================
% BUILD OPTIMIZATION PROBLEM
%==========================================================================
% The saddle inequality constraint, the total height sum constraint, and
% the various bound constraints are the only non-trivial constraints. If
% any of these are supplied, we formulate the full constrained problem and
% run a corresponding constrained minimization. If none of these
% constraints are enforced, we enforce any remaining constraints by
% construction and run an unconstraind minimization

% Build inequality constraints --------------------------------------------

A = []; b = [];

if enforceSaddles

    if verbose, disp('Adding saddle height inequality constraint'); end

    saddleIDx = find(isSaddle);
    saddleInPath = ismember(fixInPathIDx, saddleIDx);
    assert(all(ismember(sum(saddleInPath, 2), [0 1])), ...
        'Some input paths pair saddles to saddles');

    A = fixInPathIDx(any(saddleInPath, 2), :);
    A(~saddleInPath(any(saddleInPath, 2), 1), :) = ...
        A(~saddleInPath(any(saddleInPath, 2), 1), [2 1]);

    A = full(sparse( repmat((1:size(A,1)).', [1 2]), A, ...
        [-ones(size(A,1), 1), ones(size(A,1), 1)], ...
        size(A,1), numFixPoints+numLoops+2 ));

    b = -1e-12 * ones(size(A,1), 1);

end

% Build equality constraints ----------------------------------------------

Aeq = []; beq = [];

if ~isempty(constHeightSum)

    if verbose, disp('Adding height sum constraint'); end

    Aeq = [Aeq; ones(1, numFixPoints), zeros(1, numLoops), 0, 0];
    beq = [beq; constHeightSum];

end

if (numConstHeightsAndSpeeds > 0)

    if verbose, disp('Adding fixed height/speed constraints'); end

    Aeq = [Aeq; ...
        full(sparse(1:numConstHeightsAndSpeeds, ...
        find(~isnan(constHeightsAndSpeeds)), ...
        1, numConstHeightsAndSpeeds, numFixPoints+numLoops+2))];
    beq = [beq; reshape(constHeightsAndSpeeds(~isnan(constHeightsAndSpeeds)), [], 1)];

end

if ~isempty(constD)

    if verbose, disp('Adding fixed diffusion coefficient constraint'); end

    % There is no need to enforce a positive diffusion coefficient if it is
    % already fixed
    enforcePositiveDiffusion = false;

    Aeq = [Aeq; full(sparse(1, numFixPoints+numLoops+1, 1, 1, ...
        numFixPoints+numLoops+2))];
    beq = [beq; constD];

end

if ~isempty(constScalarMetric)

    if verbose, disp('Adding fixed scalar metric constraint'); end

    % There is no need to enforce a positive metric if it is already fixed
    enforcePositiveMetric = false;

    Aeq = [Aeq; full(sparse(1, numFixPoints+numLoops+2, 1, 1, ...
        numFixPoints+numLoops+2))];
    beq = [beq; constScalarMetric];

end

% Set bound constraints ---------------------------------------------------

if isempty(lowerBounds)

    lb = -inf(numFixPoints+numLoops+2, 1);

    if enforcePositiveDiffusion
        if verbose, disp('Enforcing positive diffusion'); end
        lb(end-1) = 1e-12;
    end

    if enforcePositiveMetric
        if verbose, disp('Enforcing positive metric'); end
        lb(end) = 1e-12;
    end

else

    lb = lowerBounds;

    if enforcePositiveDiffusion
        assert(lb(end-1) > 0, ['Lower bound constraint inconsistent ' ...
            'with positive diffusion']);
    end

    if enforcePositiveMetric
        assert(lb(end) > 0, ['Lower bound constraint inconsistent ' ...
            'with positive metric']);
    end

end

if isempty(upperBounds)
    ub = inf(numFixPoints+numLoops+2, 1);
else
    ub = upperBounds;
end

% Determine if unconstrained minimization is feasible ---------------------

constrainedValues = nan(numFixPoints+numLoops+2, 1);
runConstrainedMinimization = true;
if ( ~enforceSaddles && ~enforcePositiveDiffusion && ~enforcePositiveMetric ...
        && isempty(constHeightSum) && all(isinf(lb)) && all(isinf(ub)) )

    runConstrainedMinimization = false;
    constrainedValues(1:(numFixPoints+numLoops)) = constHeightsAndSpeeds;

    if ~isempty(constD)
        constrainedValues(end-1) = constD;
    end

    if ~isempty(constScalarMetric)
        constrainedValues(end) = constScalarMetric;
    end

    initGuess(~isnan(constrainedValues)) = [];

    clear A b Aeq Beq lb ub

end

clear upperBounds lowerBounds

%==========================================================================
% RUN OPTIMIZATION
%==========================================================================

optFun = @simulateLandscapeDynamics;

if runConstrainedMinimization

    if verbose, disp('Running constrained optimization'); end

    options = optimoptions('fmincon', optOptions{:});
    if options.UseParallel
        if (verbose && useGPU)
            disp(['Parallel finite difference computations ' ...
                'override GPU option']);
        end
        useGPU = false;
        if isa(dataTracks, 'gpuArray')
            dataTracks = gather(dataTracks);
            dataSetWeights = gather(dataSetWeights);
        end
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
        if isa(dataTracks, 'gpuArray')
            dataTracks = gather(dataTracks);
            dataSetWeights = gather(dataSetWeights);
        end
    end

    [optVals, optErr, ~, optOutput] = ...
        fminunc(optFun, initGuess, options);

end

optConstrainedValues = constrainedValues;
optConstrainedValues(isnan(constrainedValues)) = optVals;

optFixHeights = optConstrainedValues(1:numNonLoopFixPoints);
optLoopHeights = optConstrainedValues(numNonLoopFixPoints + (1:numLoops));
optLoopSpeeds = optConstrainedValues(numNonLoopFixPoints + numLoops + (1:numLoops));
optD = optConstrainedValues(numFixPoints+numLoops+1);
optScalarMetric = optConstrainedValues(numFixPoints+numLoops+2);

%--------------------------------------------------------------------------
% FORMAT OUTPUT
%--------------------------------------------------------------------------
% All we need to do here is compute the optimal rotational velocity. We
% don't need to simulate anything.

if verbose, fprintf('Consolidating output... '); end

% Loop velocities are just speeds times tangent vectors
optLoopVelocities = cellfun(@(x, y)  x .* y, ...
    num2cell(reshape(optLoopSpeeds, [], 1)), allLoopTangentVectors, ...
    'Uni', false);
optLoopVelocities = vertcat(optLoopVelocities{:});

if strcmpi(rotMethod, 'bilaplacian-flat')

    % We solve a separate scalar problem for each ambient dimension
    optRotV = zeros(size(X));
    for dd = 1:dim
        optRotV(:,dd) = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
            allLoopIDx, optLoopVelocities(:,dd), [], [], Frot);
    end

elseif strcmpi(rotMethod, 'bilaplacian')

    % Loop velocities must be projected onto local tangent spaces
    optLoopVelocities = mat2cell(optLoopVelocities, ...
        ones(1, numel(allLoopIDx)), dim);
    optLoopVelocities = cellfun(@(x, y) x * y, optLoopVelocities, ...
        allBases(allLoopIDx), 'Uni', false);
    optLoopVelocities = vertcat(optLoopVelocities{:});

    % We solve a single intrinsic problem
    optRotV = min_quad_with_fixed(Qconn, zeros(intDim * numPoints, 1), ...
        knownLoopIDx, optLoopVelocities(:), [], [], Frot);
    optRotV = reshape(optRotV, numPoints, intDim);

    % We lift both the loop velocities and the full velocites back
    % into the ambient dimensional space
    optLoopVelocities = mat2cell(optLoopVelocities, ...
        ones(1, numel(allLoopIDx)), intDim);
    optLoopVelocities = cellfun(@(x, y) x * y.', optLoopVelocities, ...
        allBases(allLoopIDx), 'Uni', false);
    optLoopVelocities = vertcat(optLoopVelocities{:});

    optRotV = mat2cell(optRotV, ones(1, numPoints), intDim);
    optRotV = cellfun(@(x, y) x * y.', optRotV, allBases, 'Uni', false);
    optRotV = vertcat(optRotV{:});

else

    error('Invalid rotation problem type supplied');

end

% Handle interpolated velocity outliers. NOTE: This is NOT geometry
% aware for non-flat problems
if removeOutliers
    for dd = 1:dim
        if isempty(rotOutlierThreshold)
            optOutlierThreshold = [min(optLoopVelocities(:,dd)), ...
                max(optLoopVelocities(:,dd))] + ...
                1e-14 * [-1 1];
        else
            optOutlierThreshold = rotOutlierThreshold;
        end
        optRotV(:,dd) = removeScalarOutliersFromPointCloud( ...
            X, optRotV(:,dd), optOutlierThreshold, outlierNNSize);
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

        D = locConstrainedValues(end-1);
        scalarMetric = locConstrainedValues(end);
        fixHeights = locConstrainedValues(1:numFixPoints);
        loopHeights = locConstrainedValues(numNonLoopFixPoints + (1:numLoops));
        loopHeights = loopHeights(:);
        loopSpeeds = locConstrainedValues(numFixPoints + (1:numLoops));
        loopSpeeds = loopSpeeds(:);

        % Convert fixed point height list into path end point values
        endPointVals = fixHeights(fixInPathIDx);
        [knownU, knownIDx] = interpolateValuesAlongPath(endPointVals, ...
            allPaths, 'PathLengths', allPathLengths, ...
            'InterpolationMethod', pathInterpMethod, ...
            'CollisionMethod', pathCollisionMethod);
        loopU = cellfun(@(x, y)  x .* ones(numel(y), 1), ...
            num2cell(loopHeights), allLoops, 'Uni', false);
        knownU = [knownU; vertcat(loopU{:})];
        [knownIDx, uniqueIDx, ~] = unique([knownIDx; allLoopIDx], 'stable');
        knownU = knownU(uniqueIDx);

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

        % Loop velocities are just speeds times tangent vectors
        loopVelocities = cellfun(@(x, y)  x .* y, ...
            num2cell(loopSpeeds), allLoopTangentVectors, ...
            'Uni', false);
        loopVelocities = vertcat(loopVelocities{:});

        if strcmpi(rotMethod, 'bilaplacian-flat')

            % We solve a separate scalar problem for each ambient dimension
            rotV = zeros(size(X));
            for d = 1:dim
                rotV(:,d) = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
                    allLoopIDx, loopVelocities(:,d), [], [], Frot);
            end

        elseif strcmpi(rotMethod, 'bilaplacian')

            % Loop velocities must be projected onto local tangent spaces
            loopVelocities = mat2cell(loopVelocities, ...
                ones(1, numel(allLoopIDx)), dim);
            loopVelocities = cellfun(@(x, y) x * y, loopVelocities, ...
                allBases(allLoopIDx), 'Uni', false);
            loopVelocities = vertcat(loopVelocities{:});

            % We solve a single intrinsic problem
            rotV = min_quad_with_fixed(Qconn, zeros(intDim * numPoints, 1), ...
                knownLoopIDx, loopVelocities(:), [], [], Frot);
            rotV = reshape(rotV, numPoints, intDim);

            % We lift both the loop velocities and the full velocites back
            % into the ambient dimensional space
            loopVelocities = mat2cell(loopVelocities, ...
                ones(1, numel(allLoopIDx)), intDim);
            loopVelocities = cellfun(@(x, y) x * y.', loopVelocities, ...
                allBases(allLoopIDx), 'Uni', false);
            loopVelocities = vertcat(loopVelocities{:});

            rotV = mat2cell(rotV, ones(1, numPoints), intDim);
            rotV = cellfun(@(x, y) x * y.', rotV, allBases, 'Uni', false);
            rotV = vertcat(rotV{:});

        else

            error('Invalid rotation problem type supplied');

        end

        % Handle interpolated velocity outliers. NOTE: This is NOT geometry
        % aware for non-flat problems
        if removeOutliers
            for d = 1:dim
                if isempty(rotOutlierThreshold)
                    curOutlierThreshold = [min(loopVelocities(:,d)), ...
                        max(loopVelocities(:,d))] + ...
                        1e-14 * [-1 1];
                else
                    curOutlierThreshold = rotOutlierThreshold;
                end
                rotV(:,d) = removeScalarOutliersFromPointCloud( ...
                    X, rotV(:,d), curOutlierThreshold, outlierNNSize);
            end
        end

        % Compute the single step log transition probabilities. NOTE:
        % Output will stay on GPU if useGPU == true. keepGPU == true is
        % safe even if useGPU == false
        logT = computeLogTransitionMatrix(X, U, dt, ...
            'PointPotential', U0, 'ScalarMetric', scalarMetric, ...
            'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', clipThreshold, ...
            'VolumeElementType', volumeType, ...
            'VolumeElement', volumeElement, ...
            'VectorField', rotV, ...
            'UseGPU', useGPU, 'KeepGPU', true);

        % Perform log-stabilized matrix multiplication to find the
        % multi-step transition probabilities
        if (dataSteps > 1)
            singleStepLogT = logT;
            for k = 2:dataSteps
                oldLogT = logT;
                for j = 1:numPoints
                    logT(:,j) = logsumexp(singleStepLogT + ...
                        oldLogT(:,j).', 2);
                end
            end
        end

        % We are just minimizing the weighted sum of negative log
        % transition probabilities over all tracks
        E = -sum(dataSetWeights .* logT(dataTracks));
        if useGPU, E = gather(E); end

    end

end


% % UNUSED RIGHT NOW: KEEP IN CASE WE WANT TO RELAX LAPLACIAN POSITIVE
% % DEFINITENESS CONSTRAINT LATER
% function [A, signFlip] = orientSymmetricSemidefinite(A, matrixName)
% %ORIENTSYMMETRICSEMIDEFINITE Checks whether a symmetric matrix is positive
% %or negative semi-definite. If negative semi-definite, this function
% %returns a flipped positive semi-definite version
% 
% eigTol = 1e-10 * max(1, norm(A, 'fro'));
% 
% ev = eig(full((A + A.') ./ 2));
% 
% isPSD = all(ev >= -eigTol);
% isNSD = all(ev <= eigTol);
% 
% assert(isPSD || isNSD, ...
%     '%s is neither positive nor negative semidefinite', matrixName);
% 
% signFlip = false;
% if isNSD
%     A = -A;
%     signFlip = true;
% end
% 
% end