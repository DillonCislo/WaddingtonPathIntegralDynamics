function [U, gradU, hessU, density, gradDensity, hessDensity] = ...
    estimatePotentialAndGrad(Q, P, sigma, varargin)
%ESTIMATEPOTENTIALANDGRAD Estimate a scalar potential and its derivatives
%using kernel density estimation. Given an input set of data points this
%function first computes the density at a set of query points:
%
%   density(Q_i) = sum_{k=1)^{N} w_k exp(-||Q_i-P_k||^2 / (2 * sigma^2))
%
%for a scalar standard deviation sigma assumed to be constant for all data
%points. Note that this is not normalized! The associated scalar potential
%is given by U(Q_i) = -log(density(Q_i)). This simple form enables easy
%computation of analytic gradients and Hessians. See 'gaussianKDE' for a
%more flexible KDE implementation without derivatives.
%
%   INPUT PARAMETERS:
%
%       - Q:        #NQ x dim query point cloud specifying the locations at
%                   which the density is evaluated -OR- a two element
%                   vector with Q(1) = #NQ and Q(2) == dim
%
%       - P:        #NP x dim input point cloud defining the kernel density
%                   estimator -OR- a two-element vector with P(1) == #NP
%                   and P(2) == dim
%
%       - sigma:    The standard deviation in the Gaussian kernel
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('Weights', weights = ones(#NP, 1) / #NP): A #NP x 1 set of
%       weights for each input point. Values in 'weights' should sum to one
%
%       - ('Verbose', verbose = false): Whether to produce verbose output
%
%       - ('ExcludeSelf', excludeSelf = false): Whether to exclude
%       data points that are too close to the query point when computing
%       the density
%
%       - ('SelfTolerance', selfTol = 1e-14): The distance tolerance within
%       which points are determined to be identical
%
%   OUTPUT PARAMETERS:
%
%       - U:            #NQ x 1 scalar potential values
%
%       - gradU:        #NQ x dim scalar potential gradient values
%
%       - hessU:        dim x dim x #NQ scalar potential Hessian values
%
%       - density:      #NQ x 1 (unnormalized) density values
%
%       - gradDensity:  #NQ x dim density gradient values
%
%       - hessDensity:  dim x dim x #NQ density Hessian values
%
%   by Dillon Cislo 2024/04/24

%--------------------------------------------------------------------------
% Input Processing
%--------------------------------------------------------------------------

if (numel(Q) == 2)
    numQueries = Q(1); dim = Q(2);
    validateattributes(numQueries, {'numeric'}, {'integer', ...
        'scalar', 'positive', 'finite', 'real'});
    validateattributes(dim, {'numeric'}, {'integer', ...
        'scalar', 'positive', 'finite', 'real'});
else
    validateattributes(Q, {'numeric'}, {'2d', 'finite', 'real'});
    numQueries = size(Q,1); dim = size(Q,2);
end

if (numel(P) == 2)
    numData = P(1);
    assert(P(2) == dim, ['Dimensionality of query points must match ' ...
        'dimensionality of data points']);
    validateattributes(numData, {'numeric'}, {'integer', ...
        'scalar', 'positive', 'finite', 'real'});
else
    validateattributes(P, {'numeric'}, {'2d', 'finite', 'real'});
    numData = size(P,1);
    assert(size(P,2) == dim, ['Dimensionality of query points must ' ...
        'match dimensionality of data points']);
end

validateattributes(sigma, {'numeric'}, {'scalar', ...
    'finite', 'positive', 'real'});

% Optional Input Processing -----------------------------------------------

weights = ones(numData, 1) ./ numData;
verbose = false;
excludeSelf = false;
selfTol = 1e-14;

for i = 1:length(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end

    if strcmpi(varargin{i}, 'ExcludeSelf')
        excludeSelf = varargin{i+1};
        validateattributes(excludeSelf, {'logical'}, {'scalar'});
    end

    if strcmpi(varargin{i}, 'SelfTolerance')
        selfTol = varargin{i+1};
        validateattributes(selfTol, {'numeric'}, {'scalar', ...
            'nonnegative', 'finite', 'real'});
    end
    
    if strcmpi(varargin{i}, 'Weights')
        weights = varargin{i+1};
        validateattributes(weights, {'numeric'}, {'vector', ...
            'finite', 'real', 'nonnegative', 'numel', numData});
        weights = weights ./ sum(weights);
        if (size(weights, 2) ~= 1), weights = weights.'; end
    end

end

%--------------------------------------------------------------------------
% Compute Scalar Potential and Derivatives
%--------------------------------------------------------------------------

U = nan(numQueries, 1);
density = nan(numQueries, 1);

computeGradients = (nargout > 1);
if computeGradients
    gradU = nan(numQueries, dim);
    gradDensity = nan(numQueries, dim);
else
    gradU = [];
    gradDensity = [];
end

computeHessians = (nargout > 2);
if computeHessians
    hessU = nan(dim, dim, numQueries);
    hessDensity = nan(dim, dim, numQueries);
else
    hessU = [];
    hessDensity = [];
end

try

    % Set up a parallel data queue to handle real-time progres outout
    parDQ = parallel.pool.DataQueue;
    afterEach(parDQ, @updateParallelProgressBar);
    updateParallelProgressBar(1, numQueries);

    parfor i = 1:numQueries

        % The separation vectors between the current query point and
        % all data points ([numData, dim] set of row vectors)
        dij = repmat(Q(i,:), numData, 1) - P;

        if excludeSelf

            squaredDists = dot(dij, dij, 2);
            goodIDx = sqrt(squaredDists) >= selfTol;
            dij = dij(goodIDx, :);
            squaredDists = squaredDists(goodIDx);
            curWeights = weights(goodIDx);
            curWeights = curWeights ./ sum(curWeights);

            expKernel = curWeights .* exp(-squaredDists ./ (2 * sigma^2));
            numCurData = numel(expKernel);

        else

            expKernel = weights .* exp(-dot(dij, dij, 2) ./ (2 * sigma^2));
            numCurData = numData;

        end

        density(i) = sum(expKernel);
        U(i) = -log(density(i));

        if computeGradients
            
            gradDensity(i,:) = sum(-dij .* expKernel ./ sigma^2, 1);
            gradU(i,:) = -gradDensity(i,:) ./ density(i);

        end

        if computeHessians

            curHess = pagemtimes(permute(dij, [2 3 1]), ...
                permute(dij, [3, 2, 1]));

            diagIDx = 1:(dim+1):dim^2;
            diagIDx = repmat(diagIDx, 1, numCurData) + ...
                kron((0:(numCurData-1)) * dim^2, ones(1, dim));
            curHess(diagIDx) = curHess(diagIDx) - sigma^2;

            curHess = curHess .* permute(expKernel / sigma^4, [3, 2, 1]);

            hessDensity(:,:,i) = sum(curHess, 3);
            hessU(:,:,i) = gradU(i,:).' * gradU(i,:) - ...
                hessDensity(:,:,i) ./ density(i);

        end

        if verbose, send(parDQ, []); end

    end

catch

    for i = 1:numQueries

        if verbose, progressbar(i, numQueries); end

        % The separation vectors between the current query point and
        % all data points ([numData, dim] set of row vectors)
        dij = repmat(Q(i,:), numData, 1) - P;

        if excludeSelf

            squaredDists = dot(dij, dij, 2);
            goodIDx = sqrt(squaredDists) >= selfTol;
            dij = dij(goodIDx, :);
            squaredDists = squaredDists(goodIDx);
            curWeights = weights(goodIDx);
            curWeights = curWeights ./ sum(curWeights);

            expKernel = curWeights .* exp(-squaredDists ./ (2 * sigma^2));
            numCurData = numel(expKernel);


        else

            expKernel = weights .* exp(-dot(dij, dij, 2) ./ (2 * sigma^2));
            numCurData = numData;

        end

        density(i) = sum(expKernel);
        U(i) = -log(density(i));

        if computeGradients
            
            gradDensity(i,:) = sum(-dij .* expKernel ./ sigma^2, 1);
            gradU(i,:) = -gradDensity(i,:) ./ density(i);

        end

        if computeHessians

            curHess = pagemtimes(permute(dij, [2 3 1]), ...
                permute(dij, [3, 2, 1]));

            diagIDx = 1:(dim+1):dim^2;
            diagIDx = repmat(diagIDx, 1, numCurData) + ...
                kron((0:(numCurData-1)) * dim^2, ones(1, dim));
            curHess(diagIDx) = curHess(diagIDx) - sigma^2;

            curHess = curHess .* permute(expKernel / sigma^4, [3, 2, 1]);

            hessDensity(:,:,i) = sum(curHess, 3);
            hessU(:,:,i) = gradU(i,:).' * gradU(i,:) - ...
                hessDensity(:,:,i) ./ density(i);

        end

    end

end

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


