function [allPaths, allPathLengths, allPathWeights] = ...
    splitPathsAtIDx(oldPaths, splitIDx, X, T, distMatrix)
%SPLITPATHSATIDX Splits a set of paths (definied by an ordered list of
%point IDs) at a specified index (this is an index into the path NOT the
%point clout). Recomputes properties of the new paths if the necessary
%fields are supplied.
%
%   INPUT PARAMETERS:
%
%       - oldPaths:         #OP x 1 cell array. oldPaths{i} is an ordered
%                           set of point IDs into X defining that path
%
%       - splitIDx:         #OP x 1 cell array. splitIDx{i} is a set of
%                           indices into oldPaths{i} defining where the
%                           splits should take place.
%
%       - X:                #X x dim list of input points
%
%       - T:                #N x #N (right Markov) transition matrix . The
%                           value of T(i,j) is the probability of
%                           transitioning from j->i. The columns of T
%                           should be normalized (sum(T,1) == 1)
%
%       - distMatrix:       #N x #N pairwise distance matrix, i.e.
%                           distMatrix(i,j) is the distance between cell i
%                           and cell j
%
% 	OUTPUT PARAMETERS:
%
%       - allPaths:         #Px1 cell array. allPaths{i} is a vector of
%                           point IDs (beginning with pairIDx(i,1) and
%                           ending with pairIDx(i,2)) denoting the most
%                           probable path between its end points
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
%   by Dillon Cislo 2024/04/03

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 3), X = []; end
if (nargin < 4), T = []; end
if (nargin < 5), distMatrix = []; end

validateattributes(oldPaths, {'cell'}, {'vector'}, ...
    'splitPathsAtIDx', 'oldPaths');
numPaths = numel(oldPaths);
cellfun(@(x) validateattributes(x, {'numeric'}, {'vector', 'integer', ...
    'positive', 'finite', 'real'}, 'splitPathsAtIDx', 'oldPaths'), ...
    oldPaths, 'Uni', false);
assert(all(cellfun(@(x) numel(x) > 1, oldPaths, 'Uni', true)), ...
    'Paths must have at least two points');
oldPaths = cellfun(@(x) x(:), oldPaths, 'Uni', false);
numPointsPerPath = cellfun(@numel, oldPaths, 'Uni', true);
maxPathPointID = max(vertcat(oldPaths{:}));

validateattributes(splitIDx, {'cell'}, {'vector', 'numel', numPaths}, ...
    'splitPathsAtIDx', 'splitIDx');
for i = 1:numPaths
    if ~isempty(splitIDx{i})
        validateattributes(splitIDx{i}, {'numeric'}, ...
            {'vector', 'integer', '>=', 2, 'finite', 'real', '<=', ...
            numPointsPerPath(i)-1}, 'splitPathAtIDx', 'splitIDx');
        splitIDx{i} = splitIDx{i}(:);
    end
end

numPoints = -1;
if ~isempty(X)

    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'}, ...
        'splitPathsAtIDx', 'X');
    numPoints = size(X,1); % dim = size(X,2);
    assert(maxPathPointID <= numPoints, 'Path index out of bounds');

end

if ~isempty(T)

    validateattributes(T, {'numeric'}, ...
        {'2d', 'finite', 'real', 'square', 'nonnegative'}, ...
        'splitPathsAtIDx', 'T')

    if (numPoints > 1)
        assert(size(T,1) == numPoints, ...
            'Transition matrix is improperly sized (numPoints)');
    else
        assert(size(T,1) >= maxPathPointID, ...
            'Transition matrix is improperly sized (maxPathPointID)');
    end

    if ( max(abs(sum(T,1) - 1)) > 1e-12 )
        warning('Input transition matrix is NOT properly normalized');
    end

end

if ~isempty(distMatrix)

    validateattributes(distMatrix, {'numeric'}, ...
        {'2d', 'finite', 'real', 'square', 'nonnegative'}, ...
        'splitPathsAtIDx', 'distMatrix');

    if (numPoints > 1)
        assert(size(distMatrix,1) == numPoints, ...
            'Distance matrix is improperly sized (numPoints)');
    elseif ~isempty(T)
        assert(isequal(size(T), size(distMatrix)), ...
            'Distance matrix is improperly sized (T)');
    else
        assert(size(distMatrix,1) >= maxPathPointID, ...
            'Distance matrix is improperly sized (maxPathPointID)');
    end

end

%--------------------------------------------------------------------------
% SPLIT PATHS
%--------------------------------------------------------------------------

allPaths = oldPaths;
for i = 1:numPaths

    if isempty(splitIDx{i})
        allPaths{i} = allPaths(i);
        continue;
    end

    newPath = ones(size(allPaths{i}));
    newPath(splitIDx{i}) = 2;
    newPath = repelem(allPaths{i}, newPath);

    allPaths{i} = mat2cell(newPath, [splitIDx{i}(1); ...
        diff(splitIDx{i}(1:(end-1)))+1; ...
        numPointsPerPath(i) - splitIDx{i}(end) + 1], 1);

end

allPaths = vertcat(allPaths{:});

%--------------------------------------------------------------------------
% RECOMPUTE PATH PROPERTIES
%--------------------------------------------------------------------------
% This probably isn't the most efficient way to split up these
% computations, but I want there to be transparency in the output if path
% properties are not recomputed

numPaths = numel(allPaths);

allPathLengths = {};
if ((nargout > 1) && ~(isempty(X) && isempty(distMatrix)))

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
if ((nargout > 2) && ~isempty(T))

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