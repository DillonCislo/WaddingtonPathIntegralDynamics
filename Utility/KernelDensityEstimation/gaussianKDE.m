function pointDensity = gaussianKDE(P, Q, covK, bwK, verbose, weights)
%GAUSSIANKDE Simple kernel density estimation for an input set of points
%using a Gaussian kernel with user specified parameters:
%
%	K = \mathcal{N}({\bf 0}, {\bf covK})
%
%Uses a naive O(N^2) exhaustive computation. Nothing fancy. A typical
%usecase may be a simple product kernel
%
%   \rho(Q_i) = sum_{k=1)^{N} exp(-||Q_i-P_k||^2 / (2 * sigma^2)) ./ (N * (2*pi*sigma^2)^(d/2))
%
%which can be accomplished using either gaussianKDE(P, Q, sigma^2, [], [])
%or gaussianKDE(P, Q, [], sigma, []);
%
%   INPUT PARAMETERS:
%
%       - P:    #NP x #D input point cloud defining the kernel density
%               estimator
%
%       - Q:    #NQ x #D query point cloud specifying the locations at
%               which the density is evaluated
%
%       - covK: #D x #D covariance matrix for the Gaussian kernel
%
%       - bwK:  #D x #D bandwidth matrix
%
%       - verbose: Whether or not to report progress
%
%       - weights: 1x#NP set of weights for each input point. Values in
%                  'weights' must sum to one
%
%   OUTPUT PARAMETERS:
%
%       - pointDensity:     #NQ x 1 vector of pointwise density estimates
%                           at the query points
%
%   by Dillon Cislo 02/09/2023

% Input Processing --------------------------------------------------------

if (nargin < 1), error('Please supply point coordinates'); end
if ((nargin < 2) || isempty(Q)), Q = P; end
if ((nargin < 3) || isempty(covK)), covK = 1; end
if ((nargin < 4) || isempty(bwK)), bwK = 1; end
if ((nargin < 5) || isempty(verbose)), verbose = false; end
if ((nargin < 6) || isempty(weights))
    weights = ones(size(P,1), 1) ./ size(P,1);
end

% Process input suppot point cloud
validateattributes(P, {'numeric'}, {'2d', 'finite', 'real'});
numDataPoints = size(P,1); dim = size(P,2);

% Process input suppot point cloud
validateattributes(Q, {'numeric'}, {'2d', 'finite', 'real'});
numQueryPoints = size(Q,1);
assert(size(Q,2) == dim, ['Dimensionality of query points must match ' ...
    'dimensionality of data points']);

% Process input covariance
validateattributes(covK, {'numeric'}, {'finite', 'real'});
if isscalar(covK)
    
    covK =  covK * eye(dim);   
    
elseif isvector(covK)  
    
    assert(numel(covK) == dim, ['Number of elements in the vector ' ...
        'representing isotropic kernel covarience must equal ' ...
        'the dimensionality of the input points']);
    covK = diag(covK);
    
elseif ismatrix(covK)
    
    assert(size(covK) == (dim * [1 1]), ...
        'Covariance matrix must be a #D x #D matrix');
    assert(issymmetric(covK), 'Covariance matrix must be symmetric');
    
else
    
    error('Invalid input covariance format');
    
end
invCov = inv(covK);
detCov = det(covK);


% Process input bandwidth matrix
validateattributes(bwK, {'numeric'}, {'finite', 'real'});
if isscalar(bwK)
    
    bwK = bwK * eye(dim);
    
elseif isvector(bwK)
    
    assert(numel(bwK) == dim, ['Number of elements in the vector ' ...
        'representing isotropic bandwidth must equal ' ...
        'the dimensionality of the input points']);
    bwK = diag(bwK);
    
elseif ismatrix(bwK)

    assert(size(bwK) == (dim * [1 1]), ...
        'Bandwidth matrix must be a #D x #D matrix');
    
else
    
    error('Invalid input bandwidth format');
    
end
invBW = inv(bwK);
detBW = det(bwK);
assert(detBW ~= 0, 'Bandwidth matrix must be nonsingular');

validateattributes(verbose, {'logical'}, {'scalar'});

% Validate input weights
validateattributes( weights, {'numeric'}, {'vector', 'nonnegative', ...
    'finite', 'real', 'numel', numDataPoints} );
if (size(weights, 1) ~= 1), weights = weights.'; end
weights = weights ./ sum(weights); % Normalize just in case

% Estimate Densities at Query Points --------------------------------------

pointDensity = nan(numQueryPoints, 1);

try
    
    % Set up a parallel data queue to handle real-time progres outout
    parDQ = parallel.pool.DataQueue;
    afterEach(parDQ, @updateParallelProgressBar);
    updateParallelProgressBar(1, numQueryPoints);
    
    parfor i = 1:numQueryPoints
        
        % The separation vectors between the current query point and all data
        % points ([dim, numDataPoints] set of column vectors)
        dij = (repmat(Q(i,:), numDataPoints, 1) - P).';
        
        % Multiply the separation vectors by the bandwidth
        % (A new [dim, numDataPoints] set of column vectors)
        dij = invBW * dij;
        
        % Take the dot product of the (bandwidth scaled) separation vectors to
        % find the argument of the exponential in the kernel
        pointDensity(i) = sum(weights .* exp(-dot(dij, invCov * dij, 1)/2));
        
        % if verbose, progressbar(i, numQueryPoints), end
        if verbose, send(parDQ, []); end
        
    end
    
catch
    
    for i = 1:numQueryPoints
        
        progressbar(i, numQueryPoints);
        
        % The separation vectors between the current query point and all data
        % points ([dim, numDataPoints] set of column vectors)
        dij = (repmat(Q(i,:), numDataPoints, 1) - P).';
        
        % Multiply the separation vectors by the bandwidth
        % (A new [dim, numDataPoints] set of column vectors)
        dij = invBW * dij;
        
        % Take the dot product of the (bandwidth scaled) separation vectors to
        % find the argument of the exponential in the kernel
        pointDensity(i) = sum(weights .* exp(-dot(dij, invCov * dij, 1)/2));
        
    end
    
end

% Normalize density estimates
pointDensity = pointDensity ./ (detBW * sqrt(detCov) * sqrt(2 * pi)^dim);

end


function updateParallelProgressBar(curIter, maxIter)
% Helper function to update the progressbar in parallel loops
    
    persistent parCurIter parMaxIter
    
    if (nargin == 2)
        
        parCurIter = curIter;
        parMaxIter = maxIter;
        
    else
        
        progressbar(parCurIter, parMaxIter)
        parCurIter = parCurIter + 1;
        
    end
    
end

