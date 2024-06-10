function diffCoordsOOS = nystroemOOS(XOOS, X, diffCoords, lambda, ...
    K, affinityOptions, mapOptions)
%NYSTROEMOOS Perform Nystroem out of sample extension to determine the
%diffusion coordinates of a new set of points from the diffusion
%coordinates computed for an initial set of points. At its core, this is
%just linear interpolation of the pre-computed diffusion eigenvectors,
%where the interpolation weights are determined by the affinities between
%the new points and the old points
%
%   INPUT PARAMETERS:
%
%       - XOOS:             #M x dim set of out-of-sample point coordinates
%
%       - X:                #N x dim set of sample point coordinates
%
%       - diffCoords:       #N x numEigs diffusion coordinates for the
%                           in-sample points, where 'numEigs' is the number
%                           if eigenvectors retained during computation.
%                           NOTE: We omit the leading constant eigenvector
%                           in the 'diffusionMap' function
%
%       - lambda:           numEigs x 1 or (numEigs+1) x 1 vector of
%                           diffusioneigenvalues. If numel(lambda) ==
%                           numEigs+1, we automatically remove the leading
%                           constant eigenvalue == 1.
%
%       - K:                #N x #N sparse affinity matrix for the
%                           in-sample points. If K is empty, then it is
%                           recomputed
%
%       - affinityOptions:  A struct containing options relating to
%                           affinity matrix construction (see
%                           'affinityMatrix.m' for more details)
%
%       - mapOptions:       A struct containing options related to
%                           diffusion map construction (see
%                           'diffusionMap.m' for more details)
%
%   OUTPUT PARAMETERS:
%
%       - diffCoordsOOS:    #M x numEigs diffusion coordinates for the
%                           out-of-sample points
%
%   by Dillon Cislo 2024/06/10

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 4), lambda = []; end
if (nargin < 5), K = []; end
if (nargin < 6), affinityOptions = struct(); end
if (nargin < 7), mapOptions = struct(); end

validateattributes(XOOS, {'numeric'}, {'2d', 'finite', 'real'});
numOOSPoints = size(XOOS, 1); dim = size(XOOS, 2);

validateattributes(X, {'numeric'}, {'2d', 'finite', 'real', ...
    'ncols', dim});
numPoints = size(X,1);

validateattributes(diffCoords, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', numPoints});
numEigs = size(diffCoords, 2);

validateattributes(lambda, {'numeric'}, {'vector', 'finite', 'real'});
if any(abs(lambda) > (1+1e-12))
    error(['Supplied eigenvalues fall outside the range [-1, 1]. ' ...
        'Are you sure you computed the eigenvalues of a ' ...
        'Markov matrix?']);
end
if (size(lambda, 2) ~= 1), lambda = lambda.'; end
if (numel(lambda) == numEigs+1)
    assert((abs(1-lambda(1)) < 1e-12) && (lambda(1) > 0), ...
        ['If numel(lambda) == size(diffCoords,2)+1, then the ' ...
        'first eigenvalue must == 1']);
    lambda(1) = [];
end
assert(numel(lambda) == numEigs, ...
    'Invalid diffusion eigenvalues supplied');

if ~isempty(K)

    validateattributes(K, {'numeric'}, {'2d', 'finite', 'real', ...
        'nrows', numPoints, 'ncols', numPoints, 'nonnegative'});
    assert(issymmetric(K), 'In-sample affinity matrix must be symmetric');

end

% Affinity Matrix Options Processing --------------------------------------

assert(isstruct(affinityOptions), ...
    'Input affinity matrix options must be supplied as a struct');
oldFieldNames = fieldnames(affinityOptions);
newFieldNames = lower(oldFieldNames);
for i = 1:numel(oldFieldNames)
    affinityOptions = renameStructField(affinityOptions, ...
        oldFieldNames{i}, newFieldNames{i});
end

if isfield(affinityOptions, 'distancetype')
    distType = lower(affinityOptions.distancetype);
    validateattributes(distType, {'char'}, {'vector'});
else
    distType = 'euclidean';
end

if isfield(affinityOptions, 'selftune')
    selfTune = affinityOptions.selftune;
    validateattributes(selfTune, {'numeric'}, ...
        {'scalar', 'integer', 'finite', 'nonnegative', 'real'});
    if (selfTune > 0)
        assert(selfTune > 1, ['Self-tuning neighbor ID must be > 1, ' ...
            'if that option is selected']);
        assert(selfTune <= numPoints, ['Self-tuning neighbor ID must ' ...
            'be less than or equal to the number of in-sample points']);
    end
else
    selfTune = 0;
end

if isfield(affinityOptions, 'sigma')
    sigma = affinityOptions.sigma;
    validateattributes(sigma, {'numeric'}, ...
        {'scalar', 'finite', 'real'});
    if ((selfTune > 0) && ~isempty(K))
        warning(['User supplied neighborhood size parameter '...
            'being overridden by self-tune option']);
    end
else
    sigma = -1;
end

% Generate an affinity kernel matrix if necessary
if isempty(K)
    [K, ~, ~, sigmaAM] = affinityMatrix(X, affinityOptions);
    if (sigma <= 0) && (selfTune == 0)
       sigma = sigmaAM;
    end
end

% Determine a sigma value if necessary
if (sigma <= 0) && (selfTune == 0)

    [~, nnDists] = knnsearch(X, X, 'K', 2, ...
        'SortIndices', true, 'Distance', distType);
    sigma = (mean(nnDists(:,2))+2*std(nnDists(:,2)))/4;

end

% Diffusion Map Options Processing ----------------------------------------

assert(isstruct(mapOptions), ...
    'Input options must be supplied as a struct');
oldFieldNames = fieldnames(mapOptions);
newFieldNames = lower(oldFieldNames);
for i = 1:numel(oldFieldNames)
    mapOptions = renameStructField(mapOptions, ...
        oldFieldNames{i}, newFieldNames{i});
end

if isfield(mapOptions, 'alpha')
    alpha = mapOptions.alpha;
    validateattributes(alpha, {'numeric'}, ...
        {'scalar', 'finite', 'positive', 'real'});
    alphaSupplied = true;
else
    alpha = 0;
    alphaSupplied = false;
end

if isfield(mapOptions, 'normalization')
    normType = lower(mapOptions.normalization);
    assert(ismember(normType, ...
        {'markov', 'fokkerplanck', 'laplacebeltrami'}));
else
    normType = [];
end

if (~isempty(normType) && alphaSupplied && verbose)
    warning(['Overriding input alpha value with standard ' ...
        'normalization type: ' normType]);
end

switch normType
    case 'markov'
        alpha = 0;
    case 'laplacebeltrami'
        alpha = 1; 
    case 'fokkerplanck'
        alpha = 1/2;
end

% if isfield(mapOptions, 't')
%     t = mapOptions.t;
%     validateattributes(t, {'numeric'}, ...
%         {'scalar', 'finite', 'positive', 'real'});
% else
%     t = 1;
% end

%--------------------------------------------------------------------------
% COMPUTE OUT-OF-SAMPLE EXTENSION
%--------------------------------------------------------------------------

% Compute the squared distances between out-of-sample points and in-sample
% points. Output is #(OOS) x #(IS)
if strcmpi(distType, 'euclidean')
    squaredDistsOOS = pdist2(XOOS, X, 'squaredeuclidean');
else
    squaredDistsOOS = pdist2(XOOS, X, distType);
    squaredDistsOOS = squaredDistsOOS.^2;
end

% Compute the basic extended affinity kernel values
if (selfTune > 0)

    [~, nnDistsIS] = knnsearch(X, X, 'K', selfTune, ...
        'SortIndices', true, 'Distance', distType);
    nnDistsIS = nnDistsIS(:, end); % #(IS) x 1
    [~, nnDistsOOS] = knnsearch(X, XOOS, 'K', selfTune, ...
        'SortIndices', true, 'Distance', distType);
    nnDistsOOS = nnDistsOOS(:, end); % #(OOS) x 1

    autoTuneVals = nnDistsOOS .* nnDistsIS.'; %

    KOOS = exp(-squaredDistsOOS.^2 ./ (autoTuneVals+eps));

else

    assert(sigma > 0, 'Sigma must be positive');
    KOOS = exp(-squaredDistsOOS ./ (4 * sigma));

end

% Compute renormalized extended transition matrix values
DRight = sum(K, 2) + eps;
invDAlphaRight = spdiags( 1./ (DRight.^alpha), 0, size(K,1), size(K,1));
extMB = KOOS * invDAlphaRight;

DLeft = sum(extMB, 2);
invDAlphaLeft = spdiags( 1 ./ DLeft, 0, size(KOOS,1), size(KOOS,1));
extMB = invDAlphaLeft * extMB; % #(OOS) x #(IS)

% Compute the out-of-sample diffusion coordinates
diffCoordsOOS = full(extMB * diffCoords * diag(1./lambda));


end

