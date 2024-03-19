function [moving, wasserDist, allMoving] = ...
    PPMMOMT(source, target, ws, wt, maxIter, wdDelta)
%PPMMOMT Estimate the optimal mass transport map transforming a source set
%of ND data into a target set of ND data using the projection pursuit Monge
%map. An implementation of "Large-scale optimal transport map estimation
%using projection pursuit" by Meng et al. (2019).
%
%   INPUT PARAMETERS:
%
%       - source:   #N x #D matrix. Each row corresponds to a single
%                   D-dimensional observation.
%
%       - target:   #N x #D matrix. Each row corresponds to a single
%                   D-dimensional observation.
%
%       - ws:       #N x 1 vector of source point weights. Weights should
%                   be normalized so that sum(ws) == N
%
%       - wt:       #N x 1 vector of target point weights. Weights should
%                   be normalized so that sum(wt) == N
%
%       - maxIter:  The number of iterations to run the mass transport
%                   process (100)
%
%       - wdDelta:  The transport estimation procedure will end when the
%                   change in the Wasserstein distance between two
%                   iterations is less than this value (1e-6)
%
%   OUTPUT PARAMETERS:
%
%       - moving:       #N x #D matrix. Each row corresponds to the
%                       transformed coordinates of the corresponding 
%                       observation in the source matrix
%
%       - wasserDist:   #T x 1 Wasserstein distance
%
%       - allMoving:    #N x #D x #T array
%
%   by Dillon Cislo 01/27/2023

%--------------------------------------------------------------------------
% Input Processing
%--------------------------------------------------------------------------
if (nargin < 1), error('Please supply source matrix'); end
if (nargin < 2), error('Please supply target matrix'); end

validateattributes(source, {'numeric'}, {'2d', 'finite', 'real'});
validateattributes(target, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(source,1); dim = size(source,2);
assert(isequal(size(source), size(target)), ...
    'Source and target point clouds must be the same size');

if ((nargin < 3) || isempty(ws))
    ws = ones(numPoints, 1);
else
    validateattributes(ws, {'numeric'}, ...
        {'vector', 'finite', 'real', 'nonnegative'});
    if (size(ws,2) ~= 1), ws = ws.'; end
    ws = numPoints .* ws ./ sum(ws); % Normalize source weights
end

if ((nargin < 4) || isempty(wt))
    wt = ones(numPoints, 1);
else
    validateattributes(wt, {'numeric'}, ...
        {'vector', 'finite', 'real', 'nonnegative'});
    if (size(wt,2) ~= 1), wt = wt.'; end
    wt = numPoints .* wt ./ sum(wt); % Normalize target weights
end

if (nargin < 5), maxIter = 100; end
validateattributes(maxIter, {'numeric'}, ...
    {'scalar', 'integer', 'finite', 'positive', 'real'});

if (nargin < 6), wdDelta = 1e-6; end
validateattributes(wdDelta, {'numeric'}, ...
    {'scalar', 'finite', 'nonnegative', 'real'});

%--------------------------------------------------------------------------
% Perform PPMM
%--------------------------------------------------------------------------

moving = source;

if (nargout >= 3)
    allMoving = nan(numPoints, dim, maxIter);
    allMoving(:,:,1) = moving;
else
    allMoving = [];
end

wasserDist = nan(maxIter, 1);
wasserDist(1) = 0;

for iter = 2:maxIter
    
    % Calculate the SAVE direction
    saveDir = SAVEDirection(moving, target, ws, wt);
    
    % Calculate the updated transformed point cloud
    moving = projOMT1D(moving, target, ws, wt, saveDir);
    
    % Calculate the Wasserstein distance
    wasserDist(iter) = sqrt(mean(sum((moving-source).^2, 2)));
    
    if (nargout >= 3), allMoving(:,:,iter) = moving; end
    
    if (abs(wasserDist(iter)-wasserDist(iter-1)) < wdDelta), break; end
    
end

rmIDx = isnan(wasserDist);
wasserDist(rmIDx) = [];
if (nargout >= 3), allMoving(:,:,rmIDx) = []; end

end

function C = fastCov(X, w)
% Fast calculation of the (unbiased) weighted covariance matrix. 
% WARNING: INTENDED FOR INTERNAL USE. NO CHECKS ARE PERFORMED.
%
%   INPUT PARAMETERS:
%
%       - X:    #N x #D data matrix. Each row corresponds to a data point
%       - w:    #N x 1 data point weight vector. Weights should be
%               normalized so that sum(w) == N 
%
%   OUTPUT PARAMETERS:
%
%       - C:    #D x #D sample covariance matrix
%
%   by Dillon Cislo 01/27/2023

% Find the weighted mean of the input data
XW = repmat(w, 1, size(X,2)) .* X;
COM = mean(XW,1);

% Center the data relative to the weighted mean
X0 = X - repmat(COM, size(X,1), 1);

% Calculate the (unbiased) weighted covariance matrix
C = repmat(sqrt(w), 1, size(X,2)) .* X0;
C = (C.' * C) ./ (size(X,1)-1);

end

function saveDir = SAVEDirection(X, Y, wx, wy)
% Select the most "informative" direction describing the diffrence between
% two probability distributions using the sliced average variance estimator
% (SAVE). Roughly speaking, suppose X and Y are sampled from two
% probability distributions pX and pY. The SAVE algorithm will estimate the
% direction that maximizes the discrepancy between the variances of the
% marginal distributions of the data projected along that direction.
% WARNING: INTENDED FOR INTERNAL USE. NO CHECKS ARE PERFORMED.
%
%   INPUT PARAMETERS:
%
%       - X:    #N x #D data matrix. Each row corresponds to a data point
%       - Y:    #N x #D data matrix. Each row corresponds to a data point
%       - wx:   #N x 1 data point weight vector. Weights should be
%               normalized so that sum(wx) == N 
%       - wy:   #N x 1 data point weight vector. Weights should be
%               normalized so that sum(wy) == N 
%
%   OUTPUT PARAMETERS:
%
%       - saveDir:    #D x 1 column vector contaning the estimated SAVE
%                     direction
%
%   by Dillon Cislo 01/27/2023

% Calculate the weighted covariance matrix of the pooled data set
Z = [X; Y]; W = [wx; wy];
C = fastCov(Z, W);

% Find the matrix square root of the inverse of the covariance matrix
sqrtInvC = sqrtm(inv(C));

% Find the weighted mean of the pooled data set
COMZ = mean(repmat(W, 1, size(Z,2)) .* Z, 1);
COMZ = repmat(COMZ, size(X,1), 1); % Re-size for convenience

% Center the deta sets around the weighted mean
X0 = X - COMZ; Y0 = Y - COMZ;

% Find the weighted covariance matrix of the product of each centered data
% set with the square root of the inverse of the pooled covariance matrix
VX = fastCov(X0 * sqrtInvC, wx);
VY = fastCov(Y0 * sqrtInvC, wy);

% Calculate the normalized SAVE direction
SAVEMatrix = ((VX-eye(size(X,2)))^2 + (VY-eye(size(X,2)))^2) ./ 4;
[eigVec, eigVal] = eig(SAVEMatrix);
[~, sortIDx] = sort(diag(eigVal), 'descend');
eigVec = eigVec(:, sortIDx);

saveDir = eigVec(:,1);
saveDir = saveDir ./ sqrt(sum(saveDir.^2));

end

function moving = projOMT1D(source, target, ws, wt, n)
% Perform a projected 1D optimal mass transport between a source set and a
% target set projected along a given direction
% WARNING: INTENDED FOR INTERNAL USE. NO CHECKS ARE PERFORMED.
%
%   INPUT PARAMETERS:
%
%       - source:   #N x #D matrix. Each row corresponds to a single
%                   D-dimensional observation.
%
%       - target:   #N x #D matrix. Each row corresponds to a single
%                   D-dimensional observation.
%
%       - ws:       #N x 1 vector of source point weights. Weights should
%                   be normalized so that sum(ws) == N
%
%       - wt:       #N x 1 vector of target point weights. Weights should
%                   be normalized so that sum(wt) == N
%
%       - n:        #D x 1 column vector denoting the direction along which
%                   the input data will be projected
%
%   OUTPUT PARAMETERS
%
%       - moving:   #N x #D matrix. 
%
%   by Dillon Cislo 01/27/2023

numPoints = size(source, 1);
projSource = source * n;
projTarget = target * n;

% Sort the projected target data/weights
[sortProjTarget, sortIDx] = sort(projTarget);
sortWT = wt(sortIDx); 

% The 'stretched' empirical CDF for the sorted projected target values that
% satisfies 0 <= TCDF <= numPoints.
TCDF = numPoints * (cumsum(sortWT)-sortWT/2) ./ sum(sortWT);

% An interpolant representing the inverse CDF of the target data
invTCDF = griddedInterpolant(TCDF, sortProjTarget, 'linear', 'linear');

% The 'stetched' empirical CDF for the sorted projected source values
[~, sortIDx] = sort(projSource); % Sort the source values
sortWS = ws(sortIDx); % Sort the source weights
SCDF = numPoints * (cumsum(sortWS)-sortWS/2) / sum(sortWS);

% "Unsort" the empirical CDF to match the data point ordering
[~, sortIDx] = sort(sortIDx);
SCDF = SCDF(sortIDx);

% Evaluate the interpoland to find the 1D transformation
projSourceNew = invTCDF(SCDF);

% Update the moving point cloud
moving = source + (projSourceNew - projSource) * n.';

end

