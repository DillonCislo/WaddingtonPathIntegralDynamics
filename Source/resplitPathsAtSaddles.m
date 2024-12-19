function [allPaths, fixPointIDx, fixInPathIDx, allPathLengths, allPathWeights] = ...
    resplitPathsAtSaddles(oldPaths, U, isSaddle, varargin)
%RESPLITPATHSATSADDLES This function re-splits a set of paths between
%points in a cloud that run between sinks of a function defined on the
%points so that the new saddles are located at the point along the path
%with the highest value of the function. NOTE: this function only works on
%"index-1 saddles" (saddles that are sandwiched by two sinks)
%
%   INPUT PARAMETERS:
%
%       - oldPaths:         #P x 1 cell array. oldPaths{i} is an ordered
%                           set of point IDs into X defining that path.
%                           Paths may run between two sinks or between a
%                           sink and a saddle, but never between two
%                           saddles. It is expected that each saddle is the
%                           start of exactly one path and the end of
%                           exactly one path.
%
%       - U:                #N x 1 scalar function defined on input points
%
%       - isSaddle:         #FP x 1 logical vector indicating which fixed
%                           points are saddles
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('DistanceMatrix', distMatrix = []): #N x #N pairwise distance
%       matrix, i.e. distMatrix(i,j) is the distance between cell i and
%       cell j. Used to compute path lengths
%
%       - ('PointLocations', X = []): #N x dim list of point cloud
%       coordinates. Used to compute path lengths
%
%       - ('TransitionMatrix', T = []): #N x #N transition matrix. Used to
%       compute path weights
%
%       - ('FixPointIDx', oldFixPointIDx = []): #FP x 1 vector of indices into
%       X indicating the fixed points that define the input paths. If no
%       fixed point list is supplied, it is extracted as the sorted, unique
%       list of path end points.
%
%       - ('FixInPathIDx', oldFixInPathIDx = []): #P x 2 matrix of indices
%       into oldFixPointIDx the define the end points of each path. If no
%       matrix is supplied, it is deduced from oldPaths and oldFixPointIDx
%
%       - ('InvalidSaddleIDx', invalidSaddleIDx = []): A list of points in
%       the cloud that cannot be used as saddles. For example these might
%       include points that are too close to sink. Can be supplied as a
%       list of IDs or as a logical vector with #N elements
%
%
%   OUTPUT PARAMETERS:
%
%       - allPaths:         #Px1 cell array. allPaths{i} is an ordered set
%                           point IDs into X defining the resliced paths
%
%       - fixPointIDx:      #FP x 1 vector of indices into X indicating the
%                           fixed points that define the resliced paths.
%
%       - fixInPathIDx:     #P x 2 matrix of indices fixPointIDx the define
%                           the end points of each resliced path.
%
%       - allPathWeights:   #Px1 cell array. allPathWeights{i} is a vector
%                           of path edge weights (i.e. allPathWeights{i}(j)
%                           is the transition probability from
%                           allPaths{i}(j)->allPaths{i}(j+1))
%
%       - allPathLengths:   #Px1 cell array. allPathLengths{i} is a vector
%                           of physical distances between points in the
%                           corresponding path (i.e. allPathLengths{i}(j)
%                           is the distance between
%                           X(allPaths{i}(j), :)->X(allPaths{i}(j+1)), :))
%
%   by Dillon Cislo 2024/11/18

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------

validateattributes(oldPaths, {'cell'}, {'vector'}, mfilename, 'oldPaths');
% numPaths = numel(oldPaths);
cellfun(@(x) validateattributes(x, {'numeric'}, {'vector', 'integer', ...
    'positive', 'finite', 'real'}, mfilename, 'oldPaths'), ...
    oldPaths, 'Uni', false);
assert(all(cellfun(@(x) numel(x) > 1, oldPaths, 'Uni', true)), ...
    'Paths must have at least two points');
oldPaths = cellfun(@(x) x(:), oldPaths, 'Uni', false);
% numPointsPerPath = cellfun(@numel, oldPaths, 'Uni', true);
maxPathPointID = max(vertcat(oldPaths{:}));

validateattributes(U, {'numeric'}, {'vector', 'finite', 'real'}, ...
    mfilename, 'U');
if (size(U,2) ~= 1), U = U.'; end
numPoints = numel(U);

assert(maxPathPointID <= numPoints, 'Path contains an out-of-bounds ID');

% OPTIONAL INPUT PROCESSING -----------------------------------------------
oldFixPointIDx = [];
oldFixInPathIDx = [];
X = [];
T = [];
distMatrix = [];
invalidSaddleIDx = [];

supportedOptions = {'DistanceMatrix', 'PointLocations', ...
    'TransitionMatrix', 'FixPointIDx', 'FixInPathIDx', ...
    'InvalidSaddleIDx'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'DistanceMatrix')
        distMatrix = varargin{i+1};
        if ~isempty(distMatrix)
            validateattributes(distMatrix, {'numeric'}, ...
                {'2d', 'finite', 'real', 'square', 'nonnegative', ...
                'nrows', numPoints, 'ncols', numPoints}, ...
                mfilename, 'distMatrix');
        end
    end

    if strcmpi(varargin{i}, 'PointLocations')
        X = varargin{i+1};
        if ~isempty(X)
            validateattributes(X, {'numeric'}, {'2d', 'finite', ...
                'real', 'nrows', numPoints}, ...
                mfilename, 'X');
        end
    end

    if strcmpi(varargin{i}, 'TransitionMatrix')
        if ~isempty(T)
            validateattributes(T, {'numeric'}, ...
                {'2d', 'finite', 'real', 'square', 'nonnegative', ...
                'nrows', numPoints, 'ncols', numPoints}, ...
                mfilename, 'T')
            if ( max(abs(sum(T,1) - 1)) > 1e-12 )
                warning('Input transition matrix is NOT properly normalized');
            end
        end
    end

    if strcmpi(varargin{i}, 'FixPointIDx')
        oldFixPointIDx = varargin{i+1};
        if ~isempty(oldFixPointIDx)
            validateattributes(oldFixPointIDx, {'numeric'}, ...
                {'vector', 'positive', 'integer', 'finite', 'real', ...
                '<=', numPoints}, mfilename, 'oldFixPointIDx');
            if (size(oldFixPointIDx, 2) ~= 1)
                oldFixPointIDx = oldFixPointIDx.';
            end
        end
    end

    if strcmpi(varargin{i}, 'FixInPathIDx')
        oldFixInPathIDx = varargin{i+1};
        if ~isempty(oldFixInPathIDx)
            validateattributes(oldFixInPathIDx, {'numeric'}, {'2d', ...
                'positive', 'integer', 'finite', 'real', 'ncols', 2}, ...
                mfilename, 'fixInPathIDx');
        end
    end

    if strcmpi(varargin{i}, 'InvalidSaddleIDx')
        invalidSaddleIDx = varargin{i+1};
        if ~isempty(invalidSaddleIDx)
            if islogical(invalidSaddleIDx)
                validateattributes(invalidSaddleIDx, {'logical'}, ...
                    {'vector', 'numel', numPoints}, ...
                    mfilename, 'invalidSaddleIDx');
                invalidSaddleIDx = reshape(find(invalidSaddleIDx), [], 1);
            else
                validateattributes(invalidSaddleIDx, {'numeric'}, ...
                    {'vector', 'positive', 'finite', 'real', ...
                    'integer', '<=', numPoints}, ...
                    mfilename, 'invalidSaddleIDx');
                invalidSaddleIDx = invalidSaddleIDx(:);
            end
        end
    end

end

% PROCESS FIXED POINTS ----------------------------------------------------
fixPointIDx = oldFixPointIDx;
fixInPathIDx = oldFixInPathIDx;

% Extract fixed point IDs if necessary
if isempty(fixPointIDx)

    fixPointIDx = cellfun(@(x) [x(1); x(end)], oldPaths, 'Uni', false);
    fixPointIDx = cell2mat(fixPointIDx);
    fixPointIDx = unique(fixPointIDx(:));

end

assert(isequal(sort(fixPointIDx), unique(fixPointIDx)), ...
    '''fixPointIDx'' contains duplicate entries');
numFixPoints = numel(fixPointIDx);

% Determine the location of path end points in the fixed point list
if isempty(fixInPathIDx)

    fixInPathIDx = cellfun(@(x) [x(1), x(end)], oldPaths, 'Uni', false);
    fixInPathIDx = vertcat(fixInPathIDx{:});
    [~, fixInPathIDx] = ismember(fixInPathIDx, fixPointIDx);

end

assert(size(fixInPathIDx, 1) == numel(oldPaths), ...
    '''fixInPathIDx'' is improperly sized');
assert(isequal(unique(fixInPathIDx(:)), (1:numFixPoints).'), ...
    ['Some supplied fixed points do not seem to be represented in ' ...
    'the path list']);

% Process the saddle point indicator input
if islogical(isSaddle)
    assert(isvector(isSaddle) && ...
        (numel(isSaddle) == numFixPoints), ['Logical saddle ' ...
        'input is improperly sized']);
    if (size(isSaddle, 2) ~= 1), isSaddle = isSaddle.'; end
else
    validateattributes(isSaddle, {'numeric'}, ...
        {'vector', 'integer', 'positive', 'finite', 'real', ...
        '<=', numFixedPoints}, mfilename, 'isSaddle');
    isSaddle = ismember((1:numFixedPoints).', isSaddle);
end

% ADDITIONAL PATH PROCESSING ----------------------------------------------
% Here we check that (1) each path connects either a sink to a sink or a
% sink to a saddle and (2) that each saddle in the inset and outset of
% exactly one path (i.e. this is an index-1 saddle).

isSaddleEndPoint = isSaddle(fixInPathIDx);
assert(all(sum(isSaddleEndPoint, 2) <= 1), ...
    'All paths must join either a sink to a sink or a sink to a saddle');

startCounts = histcounts(fixInPathIDx(:,1), 0.5:1:(numFixPoints+0.5));
endCounts = histcounts(fixInPathIDx(:,2), 0.5:1:(numFixPoints+0.5));
assert(all(startCounts(isSaddle) == 1) && all(endCounts(isSaddle) == 1), ...
    ['All saddles must be the inset of at exactly one path AND the '...
    'outset of at exactly one path']);

%--------------------------------------------------------------------------
% RE-SPLIT PATHS AT SADDLES
%--------------------------------------------------------------------------
allPaths = oldPaths;

% Construct the primal paths that link sinks to saddles to sinks.
saddleInFixIDx = find(isSaddle);
primalPathIDx = cell(sum(isSaddle), 1);
primalPaths = cell(sum(isSaddle), 1);
for i = 1:sum(isSaddle)

    primalPathIDx{i} = [find(fixInPathIDx(:,2) == saddleInFixIDx(i)), ...
        find(fixInPathIDx(:,1) == saddleInFixIDx(i))];
    assert(numel(primalPathIDx{i}) == 2, ...
        'Invalid primal path construction (multiple saddle end points)');

    primalPaths{i} = vertcat(oldPaths{primalPathIDx{i}(1)}, ...
        oldPaths{primalPathIDx{i}(2)}(2:end));
    assert(isequal(primalPaths{i}, unique(primalPaths{i}, 'stable')), ...
        'Invalid primal path construction (duplicate IDs)');

end

% Add the primal paths that link sinks to sinks directly
isSinkToSinkPath = find(sum(isSaddleEndPoint, 2) == 0);
primalPathIDx = [primalPathIDx; num2cell(isSinkToSinkPath)];
primalPaths = [primalPaths; oldPaths(isSinkToSinkPath)];

% Valid new saddles are (1) not sinks and (2) not shared by multiple primal
% paths
allPathVertexIDx = unique(vertcat(oldPaths{:}));
appearanceCount = cellfun(@(x) ismember(allPathVertexIDx, x), ...
    primalPaths, 'Uni', false);
appearanceCount = sum(horzcat(appearanceCount{:}), 2);
validSaddleIDx = allPathVertexIDx(appearanceCount == 1);
validSaddleIDx = setdiff(validSaddleIDx, fixPointIDx(~isSaddle));
validSaddleIDx = setdiff(validSaddleIDx, invalidSaddleIDx);

% Re-split primal paths that link sinks to saddles to sinks
for i = 1:sum(isSaddle)
    
    % Find the new saddle point
    curPrimalPath = primalPaths{i};
    UPath = U(curPrimalPath);
    UPath(~ismember(curPrimalPath, validSaddleIDx)) = NaN;
    assert(~all(isnan(UPath)), 'No valid saddles in primal path');
    [~, newMaxInPathID] = max(UPath);

    % Update the corresponding fixed point ID
    fixPointIDx(saddleInFixIDx(i)) = curPrimalPath(newMaxInPathID);

    % Update the output path list
    allPaths{primalPathIDx{i}(1)} = curPrimalPath(1:newMaxInPathID);
    allPaths{primalPathIDx{i}(2)} = curPrimalPath(newMaxInPathID:end);

end

%--------------------------------------------------------------------------
% RECOMPUTE PATH PROPERTIES
%--------------------------------------------------------------------------
% This probably isn't the most efficient way to split up these
% computations, but I want there to be transparency in the output if path
% properties are not recomputed

numPaths = numel(allPaths);

allPathLengths = {};
if ((nargout > 3) && ~(isempty(X) && isempty(distMatrix)))

    allPathLengths = cell(numPaths, 1);
    for i = 1:numPaths

        curPath = [allPaths{i}(1:(end-1)), allPaths{i}(2:end)];

        if isempty(distMatrix)
            curPathLengths = X(curPath(:,2), :) - X(curPath(:,1), :);
            curPathLengths = sqrt(sum(curPathLengths.^2, 2));
        else
            dmIDx = sub2ind(size(distMatrix), curPath(:,2), curPath(:,1));
            curPathLengths = distMatrix(dmIDx);
        end

        curPathLengths = [0; cumsum(curPathLengths)];
        allPathLengths{i} = curPathLengths;

    end

end

allPathWeights = {};
if ((nargout > 4) && ~isempty(T))

    allPathWeights = cell(numPaths, 1);
    for i = 1:numPaths

        curPath = [allPaths{i}(1:(end-1)), allPaths{i}(2:end)];
        curPathWeights = nan(size(curPath,1), 1);
        for j = 1:size(curPath,1)
            curPathWeights(j) = T(curPath(j,2), curPath(j,1));
        end
        allPathWeights{i} = curPathWeights;

    end

end

end