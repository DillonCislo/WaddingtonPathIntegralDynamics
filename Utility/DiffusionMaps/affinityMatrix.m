function [K, nnDists, nnIDx, sigmaOut] = affinityMatrix(X, affinityOptions)
%AFFINITYMATRIX Calculates the affinity matrix used to generate a diffusion
%map embedding for high-dimensional feature data
%
%   INPUT PARAMETERS:
%
%       - X:        #N x #D high dimensional point cloud
%
%   OPTIONAL INPUTS (Name, Value)-Pairs:
%
%       - ('DistanceMatrix', distMatrix = []): A #N x #N (or #N x k)
%       precomputed distance matrix for the points in X
%
%       - ('DistanceType', distType = 'euclidean'): The metric used to
%       calculate the raw pair-wise distances between data points. See
%       documentation for 'knnsearch'.
%
%       - ('NumNeighbors', kNN = '10'): The number of neareast neigbors
%       retained in the affinity matrix computation. Higher numbers of
%       neighbors will result in a denser affinity matrix.
%
%       - ('SelfTune', selfTune = 0): The Kth neighbor used to calculate a
%       local scaling measure that sets the width of the affinity matrix
%       kernel. See 'Self-Tuning Spectral Clustering' by Zelnik-Manor and
%       Perona (2004). The authors in that paper use selfTune = 7.
%
%       - ('Sigma', sigma = -1): The bandwidth of the affinity matrix
%       kernel. Any value less than 0 results in an automatic estimate of
%       the bandwidth based on the distribution of nearest neighbor
%       distances. This property is ignored if self-tuning is enabled.
%       NOTE: The Gaussian kernel depends on sigma as
%       ~exp(-vals.^2 / (4*sigma)), which matches the convention in the
%       diffusion map literature, but differs from the usual normal
%       distribution convention.
%
%       - ('Verbose', verbose = false): Report the elapsed time for the
%       pairwise distance calculation
%
%   OUTPUT PARAMETERS:
%
%       - K:            #X x #X symmetric affinity matrix
%
%       - nnDists:      #X x k list of kNN nearest neighbor distances for
%                       each input point
%
%       - nnIDx:        #X x k list of data point IDs corresponding to the
%                       distances in nnDists
%
%       - sigmaOut:     The bandwidth of the affinit matrix kernel. If a
%                       positive, bandwidth was supplied by the user, then
%                       this is equal to affinityOptions.Sigma. Otherwise,
%                       it is equal either to (1) the scalar estimate of
%                       the bandwidth or (2) an #N x 1 vector of the local
%                       bandwidths determined by the self-tuning algorithm,
%                       depeneding on the user supplied options.
%
%   by Dillon Cislo 12/10/2022

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if ~isempty(X)
    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
    numPoints = size(X,1); % dims = size(X,2);
else
    numPoints = -1;
end

if (nargin < 2), affinityOptions = struct(); end
assert(isstruct(affinityOptions), ...
    'Input options must be supplied as a struct');
oldFieldNames = fieldnames(affinityOptions);
newFieldNames = lower(oldFieldNames);
for i = 1:numel(oldFieldNames)
    affinityOptions = renameStructField(affinityOptions, ...
        oldFieldNames{i}, newFieldNames{i});
end

if isfield(affinityOptions, 'distancetype')
    distType = lower(affinityOptions.distancetype);
    validateattributes(distType, {'char'}, {'vector'});
    affinityOptions = rmfield(affinityOptions, 'distancetype');
else
    distType = 'euclidean';
end

if isfield(affinityOptions, 'distancematrix')
    distMatrix = affinityOptions.distancematrix;
    if ~isempty(distMatrix)
        validateattributes(distMatrix, {'numeric'}, {'2d', ...
            'nonnegative', 'finite', 'real', 'square'});
        if (numPoints > 0)
            assert(isequal(size(distMatrix), numPoints * [1 1]), ...
                'Distance matrix is improperly sized');
        else
            numPoints = size(distMatrix,1);
        end
    end
    affinityOptions = rmfield(affinityOptions, 'distancematrix');
else
    distMatrix = [];
end
    
if isfield(affinityOptions, 'numneighbors')
    kNN = affinityOptions.numneighbors;
    validateattributes(kNN, {'numeric'}, ...
        {'scalar', 'integer', 'finite', 'positive', 'real'});
    affinityOptions = rmfield(affinityOptions, 'numneighbors');
else
    kNN = 10;
end
kNN = min(numPoints, kNN);

if isfield(affinityOptions, 'selftune')
    selfTune = affinityOptions.selftune;
    validateattributes(selfTune, {'numeric'}, ...
        {'scalar', 'integer', 'finite', 'nonnegative', 'real'});
    affinityOptions = rmfield(affinityOptions, 'selftune');
    if (selfTune > 0)
        assert(selfTune <= kNN, ['Self-tuning neighbor ID must be ' ...
            'less than or equal to the number of neighbors ' ...
            'retained in the pairwise distance calculation']);
    end
else
    selfTune = 0;
end

if isfield(affinityOptions, 'sigma')
    sigma = affinityOptions.sigma;
    validateattributes(sigma, {'numeric'}, ...
        {'scalar', 'finite', 'real'});
    affinityOptions = rmfield(affinityOptions, 'sigma');
    if (selfTune > 0)
        warning(['User supplied neighborhood size parameter '...
            'being overridden by self-tune option']);
    end
else
    sigma = -1;
end

if isfield(affinityOptions, 'verbose')
    verbose = affinityOptions.verbose;
    validateattributes(verbose, {'logical'}, {'scalar'});
    affinityOptions = rmfield(affinityOptions, 'verbose');
else
    verbose = false;
end

assert(isempty(fieldnames(affinityOptions)), ...
    'Invalid option type supplied');

%--------------------------------------------------------------------------
% GENERATE AFFINITY MATRIX
%--------------------------------------------------------------------------

% Calculate pair-wise distances between points
if verbose, tic; end
if isempty(distMatrix)
    
    assert(~isempty(X), ['Please supply either an input point set ' ...
        'or a pre-computed distance matrix']);
    
    % NOTE: Output is (numPoints) x (kNN)
    [nnIDx, nnDists] = knnsearch(X, X, 'K', kNN, ...
        'SortIndices', true, 'Distance', distType);
    assert(isequal(size(nnDists), [numPoints, kNN]), ...
        'Invalid distance output size');
    
else
    
    nnIDx = zeros(numPoints, kNN);
    nnDists = zeros(numPoints, kNN);
    for i = 1:numPoints
        [nnDists(i,:), nnIDx(i,:)] = mink(distMatrix(i,:), kNN);
    end
    
end
if verbose, toc; end

% Generate the index/value structure for the sparse matrix output
rowIDx = repmat((1:numPoints).', 1, kNN); rowIDx = rowIDx(:);
colIDx = nnIDx(:);
vals = nnDists(:);

% Set local scale for each sample
if selfTune
    
    % The distance of the sth neighbor for each point
    % i.e. sigma_i = d(x_i, x_is)
    sigmaKVec = nnDists(:, selfTune); 
    
    % Local scale for each pair is just product of distances for each point
    % i.e. sigma_i * sigma_j
    autoTuneVals = repmat(sigmaKVec, 1, kNN) .* sigmaKVec(nnIDx);
    autoTuneVals = autoTuneVals(:);
    
    vals = exp(-vals.^2 ./ (autoTuneVals+eps));

    sigmaOut = (autoTuneVals + eps) ./ 4;
    
else
    
    % if (sigma <= 0), sigma = median(vals); end
    if (sigma <= 0), sigma = (mean(nnDists(:,2))+2*std(nnDists(:,2)))/4; end

    % vals = exp(-vals.^2 / (2*sigma.^2));
    vals = exp(-vals.^2 / (4*sigma));

    sigmaOut = sigma;
    
end

% Construct symmetrized affinity matrix
% K = sparse(rowIDx, colIDx, vals, numPoints, numPoints);
% K = (K + K.')/2;
K = sparse( [rowIDx; colIDx], [colIDx; rowIDx], [vals; vals]./2, ...
    numPoints, numPoints );

end

