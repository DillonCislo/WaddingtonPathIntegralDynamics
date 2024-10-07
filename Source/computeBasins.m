function [basinProb, basinCounts, basinIDx, incPointIDx] = computeBasins( ...
    X, T, pointProb, basinLocIDx, varargin)
%COMPUTEBASINS Assigns points in an input point cloud to a discrete set of
%"basin" locations based on a user specified method and computes the total
%probability for each basin.
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
%       - pointProb:    #N x 1 vector of probabilities on each point
%
%       - basinLocIDx:  #B x 1 vector of basin location IDs (i.e. a set of
%                       indices into the point cloud 'X')
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('BasinMethod', basinMethod = 'expectedValue'): The method by
%       which each point in the cloud is associated to a basin:
%                   
%           (1) 'distance': points are associated to the closest basin
%           relative to a given distance metric. A pre-computed distance
%           matrix can be supplied. Otherwise, pairwise distances between
%           points and basins are computed using 'pdist2'.
%
%           (2) 'mostProbablePath': computes the most probable path from
%           each point to each basin location using the transition matrix T
%           and then assigns points to basins according to which
%           point-to-basin transition is most probable. This method does
%           not really have a great interpretation from a stochastic
%           dynamics perspective and should probably be avoided.
%
%           (3) 'expectedValue' (*): computes the probability of a random
%           walk (that starts at time s <= t and evolves according to T) of
%           reaching the basin point at time t. This is effectively solving
%           the backwards Kolmogorov equation and is the most principled
%           method for computing actual basins of attraction.
%
%       - ('NumSteps', numSteps = 5): The number of backwards steps used to
%       compute the solution to the backwards Kolmogorov equation if
%       basinMethod == 'expectedValue'. This value should be large enough
%       for expected values of distant points to be non-zero, but small
%       enough to not smear out probabilities.
%
%       - ('DistanceMatrix', distMatrix = []): #N x #N pairwise feature
%       space distance matrix, i.e. distMatrix(i,j) is the distance between
%       cell i and cell j. NOTE: This input is only used if basinMethod
%       == 'distance'! Otherwise the necessary fields are computed from
%       the transition matrix input.
%
%       - ('DistanceType', distType = 'euclidean'): The distance metric
%       used to compute the pairwise feature space distance between points
%       if basinMethod == 'distance'. See 'pdist2' for more details.
%
%       - ('ProbabilityThreshold', probThreshold = 0): The threshold above
%       which points are included in the aggregation of basin probabilities
%       (i.e. low probability points are not counted towards the total)
%
%       - ('MergeBasins', mergeBasins = {}): A cell array where each
%       elements contains a (non-intersecting) list of basins that will me
%       merged in the final output (this is subtly different than just
%       supplying an abbreviated list of input basins!). NOTE: Each element
%       of each basin is an index into 'basinLocIDx'. Each set of basins to
%       be merged will keep the ID of the FIRST basin in the corresponding
%       list. Also, if a given point is excluded from this ANY merged
%       basin, then it will be summarily excluded from ALL associated
%       merged basins
%
%       - ('ExcludeFromBasins', excludeFromBasins = []): #N x #B logical
%       matrix. If excludeFromBasins(i,j) is true, the ith point cannot be
%       assigned to the jth basin.
%
%   OUTPUT PARAMETERS:
%
%       - basinProb:        #MB x 1 vector of total probabilities
%                           associated to each basin
%
%       - basinCounts:      #MB x 1 vector of the number of points
%                           associated to each basin 
%
%       - basinIDx:         #N x 1 vector of basin IDs (NOTE: This is an
%                           index into 'basinLocIDx' (after
%                           mergers/simplifications) NOT an index into the
%                           point cloud 'X')
%
%       - incPointIDx:      #N x 1 logical vector indicating if a
%                           particular point contributed to to its basin's
%                           aggregate probability (i.e. if its probability
%                           was above the threshold)
%
%   by Dillon Cislo 2024/05/20

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(pointProb, {'numeric'}, {'vector', 'finite', ...
    'nonnegative', 'real'}, 'computeBasins', 'pointProb');
if (size(pointProb, 2) ~= 1), pointProb = pointProb.'; end
numPoints = numel(pointProb);
if (abs(sum(pointProb)-1) > 1e-12)
    warning('Input probabilities are NOT properly normalized');
end

if ~isempty(X)
    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real', ...
        'nrows', numPoints}, 'computeBasins', 'X');
    numPoints = size(X,1); % dim = size(X,2);
end

if ~isempty(T)
    validateattributes(T, {'numeric'}, {'2d', 'finite', 'real', ...
        'nrows', numPoints, 'ncols', numPoints}, ...
        'computeBasins', 'T');
    if ( max(abs(sum(T,1) - 1)) > 1e-12 )
        warning('Input transition matrix is NOT properly normalized');
    end
end

validateattributes(basinLocIDx, {'numeric'}, {'vector', 'finite', ...
    'positive', 'real', 'integer', '<=', numPoints}, ...
    'computeBasins', 'basinLocIDx');
if (size(basinLocIDx, 2) ~= 1), basinLocIDx = basinLocIDx.'; end
numBasins = numel(basinLocIDx);

% OPTIONAL INPUT PROCESSING -----------------------------------------------

distMatrix = [];
distType = 'euclidean';
basinMethod = 'expectedvalue';
numSteps = 5;
probThreshold = 0;
mergeBasins = {};
excludeFromBasins = [];

allBasinMethods = {'distance', 'mostprobablepath', 'expectedvalue'};

supportedOptions = {'DistanceMatrix', 'DistanceType', ...
    'BasinMethod', 'NumSteps', 'ProbabilityThreshold', ...
    'MergeBasins', 'ExcludeFromBasins'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'DistanceMatrix')
        distMatrix = varargin{i+1};
        if ~isempty(distMatrix)
            validateattributes(distMatrix, {'numeric'}, {'2d', ...
                'finite', 'real', 'nonnegative', 'square', ...
                'nrows', numPoints}, 'computeBasins', 'distMatrix');
        end
    end

    if strcmpi(varargin{i}, 'DistanceType')
        distType = lower(varargin{i+1});
        validateattributes(distType, {'char'}, {'vector'}, ...
            'computeBasins', 'distType');
    end


    if strcmpi(varargin{i}, 'BasinMethod')
        basinMethod = lower(varargin{i+1});
        validateattributes(basinMethod, {'char'}, {'vector'}, ...
            'computeBasins', 'basinMethod');
        assert(ismember(basinMethod, allBasinMethods), ...
            'Invalid basin method supplied');
    end

    if strcmpi(varargin{i}, 'NumSteps')
        numSteps = varargin{i+1};
        validateattributes(numSteps, {'numeric'}, {'scalar', ...
            'positive', 'finite', 'real', 'integer'}, ...
            'computeBasins', 'numSteps');
    end

    if strcmpi(varargin{i}, 'ProbabilityThreshold')
        probThreshold = varargin{i+1};
        validateattributes(probThreshold, {'numeric'}, {'scalar', ...
            'finite', 'real'}, 'computeBasins', 'probThreshold');
    end

    if strcmpi(varargin{i}, 'MergeBasins')
        mergeBasins = varargin{i+1};
        if (isvector(mergeBasins) && ~iscell(mergeBasins))
            mergeBasins = {mergeBasins};
        end
        assert(iscell(mergeBasins), ['List of basins to be merged ' ...
            'must be supplied either as a single vector of basin IDs ' ...
            'or a cell array of vectors of basin IDs']);
        mergeBasins = mergeBasins(:);
    end

    if strcmpi(varargin{i}, 'ExcludeFromBasins')
        excludeFromBasins = varargin{i+1};
        if ~isempty(excludeFromBasins)
            validateattributes(excludeFromBasins, {'logical'}, ...
                {'2d', 'nrows', numPoints, 'ncols', numBasins}, ...
                'computeBasins', 'excludeFromBasins');
            assert(~any(all(excludeFromBasins, 2)), ...
                'You must be able to assing points to at least one basin');
        end
    end

end

if ~isempty(mergeBasins)

    cellfun(@(x) validateattributes(x, {'numeric'}, {'vector', ...
        'finite', 'real', 'integer', 'positive', '<=' numBasins}), ...
        mergeBasins, 'Uni', false);
    mergeBasins = cellfun(@(x) x(:), mergeBasins, 'Uni', false);

    % Check that merger subsets are non-intersecting
    if (numel(mergeBasins) > 1)

        pairIDx = nchoosek(1:numel(mergeBasins), 2);
        checkIntersections = mergeBasins(pairIDx);
        if (size(checkIntersections, 2) == 1)
            checkIntersections = checkIntersections.';
        end
        checkIntersections = cellfun(@(x,y) numel(intersect(x,y)) == 0, ...
            checkIntersections(:,1), checkIntersections(:,2), ...
            'Uni', true);
        assert(all(checkIntersections), ['Lists of basin IDs to be ' ...
            'merged must be non-intersecting']);

        clear pairIDx checkIntersections

    end

end

%--------------------------------------------------------------------------
% COMPUTE BASINS
%--------------------------------------------------------------------------

incPointIDx = (pointProb > probThreshold);

if strcmpi(basinMethod, 'distance')

    if isempty(distMatrix)

        assert(~isempty(X), ['Please supply either a distance matrix ' ...
            'OR explicit point cloud coordinates']);

        % basinIDx = knnsearch(X(basinLocIDx, :), X);
        distMatrix = pdist2(X, X(basinLocIDx, :), distType);

    else

        distMatrix = distMatrix(:, basinLocIDx);

    end

elseif strcmpi(basinMethod, 'mostProbablePath')

    assert(~isempty(T), 'Please supply a transition matrix');

    % NOTE: MATLAB's 'digraph(A)' takes an adjacency matrix where A(i,j) is
    % the edge weight from node i->j, which is the OPPOSITE of our
    % transition matrix convention
    diffGraph = digraph(-log(T.'));
    distMatrix = distances(diffGraph, basinLocIDx, ...
        'Method', 'positive').';

elseif strcmpi(basinMethod, 'expectedValue')

    assert(~isempty(T), 'Please supply a transition matrix');

    distMatrix = zeros(numPoints, numel(basinLocIDx));
    distMatrix(basinLocIDx + (0:(numel(basinLocIDx)-1)).' .* numPoints) = 1;
    for i = 1:numSteps
        distMatrix = (T.') * distMatrix;
    end

else

    error('Invalid basin method supplied')

end

% Handle basin mergers
for i = 1:numel(mergeBasins)

    mergeIDx = mergeBasins{i};

    if ismember(basinMethod, {'distance', 'mostprobablepath'})
        distMatrix(:, mergeIDx(1)) = min(distMatrix(:, mergeIDx), [], 2);
    elseif strcmpi(basinMethod, 'expectedvalue')
        distMatrix(:, mergeIDx(1)) = max(distMatrix(:, mergeIDx), [], 2);
    else
        error('Invalid basin method supplied');
    end

    distMatrix(:, mergeIDx(2:end)) = NaN;

    if ~isempty(excludeFromBasins)
        excludeFromBasins(:, mergeIDx) = ...
            repmat(any(excludeFromBasins(:, mergeIDx), 2), ...
            [1, numel(mergeIDx)]);
    end

end

rmBasinIDx = any(isnan(distMatrix), 1);
distMatrix(:, rmBasinIDx) = [];
numBasins = size(distMatrix, 2);

if ~isempty(excludeFromBasins)
    excludeFromBasins(:, rmBasinIDx) = [];
    distMatrix(excludeFromBasins) = Inf;
end

if ismember(basinMethod, {'distance', 'mostprobablepath'})
    [~, basinIDx] = min(distMatrix, [], 2);
elseif strcmpi(basinMethod, 'expectedvalue')
    [~, basinIDx] = max(distMatrix, [], 2);
else
    error('Invalid basin method supplied');
end

% Handle basin mergers (OLD METHOD)
% [~, basinIDx] = min(distMatrix, [], 2);
% for i = 1:numel(mergeBasins)
% 
%     mergeIDx = mergeBasins{i};
%     oldBasinIDx = 1:numBasins;
% 
%     newBasinIDx = oldBasinIDx;
%     newBasinIDx(mergeIDx) = mergeIDx(1);
%     [~, newBasinIDx] = ismember(newBasinIDx, ...
%         setdiff(oldBasinIDx, mergeIDx(2:end)));
% 
%     basinIDx = changem(basinIDx, newBasinIDx, oldBasinIDx);
%     mergeBasins = cellfun(@(x) changem(x, newBasinIDx, oldBasinIDx), ...
%         mergeBasins, 'Uni', false);
%     numBasins = numBasins - numel(mergeIDx) + 1;
% 
% end

% Number of points associated to each basin
basinCounts = histcounts(basinIDx(incPointIDx), 0.5:(numBasins+0.5)).';

% Total probability associated to each basin (will only be normalized to 1
% if all points in the cloud are included in the aggregation)
basinProb = accumarray(basinIDx(incPointIDx), pointProb(incPointIDx), ...
    [numBasins 1]);

end

