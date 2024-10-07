function [allPaths, allPathWeights, allPathLengths] = ...
    computeMostProbablePaths(X, T, pairIDx, distMatrix)
%COMPUTEMOSTPROBABLEPATHS Computes the most probable paths between pairs of
%points given a (right Markov) transition matrix defined on those points
%
%   INPUT PARAMETERS:
%
%       - X:            #N x dim list of input points
%
%       - T:            #N x #N (right Markov) transition matrix. The value
%                       of T(i,j) is the probability of transitioning from
%                       j->i. The columns of T should be normalized
%                       (sum(T,1) == 1)
%
%       - pairIDx:      #Px2 matrix of start/end point pair IDs
%
%       - distMatrix:   #N x #N pairwise distance matrix, i.e.
%                       distMatrix(i,j) is the distance between cell i and
%                       cell j
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
%   by Dillon Cislo 2024/03/21

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 4), distMatrix = []; end

if ~isempty(X)
    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
    numPoints = size(X,1); % dim = size(X,2);
else
    numPoints = -1;
end

if ~isempty(distMatrix)
    validateattributes(distMatrix, {'numeric'}, {'2d', ...
        'finite', 'real', 'nonnegative', 'square'});
    if (numPoints > 0)
        assert(size(distMatrix,1) == numPoints, ['Distance matrix ' ...
            'is improperly sized']);
    else
        numPoints = size(distMatrix,1);
    end
end

assert(~(isempty(X) && isempty(distMatrix)), ['You must supply ' ...
    'either a complete input point set or a distance matrix']);

validateattributes(T, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', numPoints, 'ncols', numPoints})
if ( max(abs(sum(T,1) - 1)) > 1e-12 )
    warning('Input transition matrix is NOT properly normalized');
end

validateattributes(pairIDx, {'numeric'}, {'2d', 'finite', 'real', ...
    'ncols', 2, 'positive', '<=', numPoints});
numPairs = size(pairIDx, 1);
if (size(unique(pairIDx, 'rows'), 1) ~= numPairs)
    warning('Input pair ID list does not appear to have unique entries');
end

%--------------------------------------------------------------------------
% COMPUTE PROBABLE PATHS
%--------------------------------------------------------------------------

% NOTE: MATLAB's 'digraph(A)' takes an adjacency matrix where A(i,j) is the
% edge weight from node i->j, which is the OPPOSITE of our transition
% matrix convention -- hence the transpose
diffGraph = digraph(-log(T.'));

allPaths = cell(numPairs, 1);
allPathWeights = cell(numPairs, 1);
allPathLengths = cell(numPairs, 1);

for i = 1:numPairs
    
    allPaths{i} = shortestpath(diffGraph, pairIDx(i,1), pairIDx(i,2)).';
    
    curPath = [allPaths{i}(1:(end-1)), allPaths{i}(2:end)];
    if ~isempty(curPath)
        curPathWeights = nan(size(curPath,1), 1);
        for j = 1:size(curPath,1)
            curPathWeights(j) = T(curPath(j,2), curPath(j,1));
        end
        allPathWeights{i} = curPathWeights;
    end

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

