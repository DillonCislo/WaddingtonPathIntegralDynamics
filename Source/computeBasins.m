function [basinProb, basinCounts, basinIDx, incPointIDx] = computeBasins( ...
    X, T, pointProb, basinLocIDx, varargin)
%COMPUTEBASINS Assigns points in an input point cloud to a discrete set of
%"basin" locations based on a user specified distance measure and computes
%the total probability for each basin.
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
%       - ('DistanceMatrix', distMatrix = []): #N x #N pairwise distance
%       matrix, i.e. distMatrix(i,j) is the distance between cell i and
%       cell j
%
%       - ('DistanceMethod', distMethod = 'euclidean'): The method by which
%       distance from each point in the cloud to a basin location is
%       calculated:
%                   
%           (1) 'euclidean' is just a raw Euclidean distance between points
%           (2) 'probability' computes the most probable path from each
%           point to each basin location using the transition matrix T and
%           then assigns points to basins according to which point-to-basin
%           transition is most probable
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
%       list
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
distMethod = 'euclidean';
probThreshold = 0;
mergeBasins = {};

allDistMethods = {'euclidean', 'probability'};

supportedOptions = {'DistanceMatrix', 'DistanceMethod', ...
    'ProbabilityThreshold', 'MergeBasins'};
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

    if strcmpi(varargin{i}, 'DistanceMethod')
        distMethod = lower(varargin{i+1});
        validateattributes(distMethod, {'char'}, {'vector'}, ...
            'computeBasins', 'distMethod');
        assert(ismember(distMethod, allDistMethods), ...
            'Invalid distance method supplied');
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
        checkIntersections = cellfun(@(x,y) numel(setdiff(x,y)) == 0, ...
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

if strcmpi(distMethod, 'euclidean')

    if isempty(distMatrix)

        assert(~isempty(X), ['Please supply either a distance matrix ' ...
            'OR explicit point cloud coordinates']);
        basinIDx = knnsearch(X(basinLocIDx, :), X);

    else

        basinIDx = distMatrix(:, basinLocIDx);
        [~, basinIDx] = min(basinIDx, [], 2);

    end

elseif strcmpi(distMethod, 'probability')

    assert(~isempty(T), 'Please supply a transition matrix');

    diffGraph = digraph(-log(T));
    transDistMatrix = distances(diffGraph, basinLocIDx, ...
        'Method', 'positive');

    [~, basinIDx] = min(transDistMatrix, [], 1);
    basinIDx = basinIDx.';

else

    error('Invalid distance method supplied')

end

% Handle basin mergers
for i = 1:numel(mergeBasins)

    mergeIDx = mergeBasins{i};
    oldBasinIDx = 1:numBasins;

    newBasinIDx = oldBasinIDx;
    newBasinIDx(mergeIDx) = mergeIDx(1);
    [~, newBasinIDx] = ismember(newBasinIDx, ...
        setdiff(oldBasinIDx, mergeIDx(2:end)));

    basinIDx = changem(basinIDx, newBasinIDx, oldBasinIDx);
    mergeBasins = cellfun(@(x) changem(x, newBasinIDx, oldBasinIDx), ...
        mergeBasins, 'Uni', false);
    numBasins = numBasins - numel(mergeIDx) + 1;

end

% Number of points associated to each basin
basinCounts = histcounts(basinIDx(incPointIDx), 0.5:(numBasins+0.5)).';

% Total probability associated to each basin (will only be normalized to 1
% if all points in the cloud are included in the aggregation)
basinProb = accumarray(basinIDx(incPointIDx), pointProb(incPointIDx), ...
    [numBasins 1]);

end

