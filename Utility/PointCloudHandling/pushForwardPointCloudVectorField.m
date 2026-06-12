function [vPrime, allPushForwards, allUndeformedBases, ...
    allBasisMaps, allDeformedBases] = ...
    pushForwardPointCloudVectorField(x, xPrime, v, dim, k, verbose)
%PUSHFORWARDPOINTCLOUDVECTORFIELD Push forward a vector field defined on an
%N-D point cloud along a map to a deformed M-D point cloud by fitting a
%transformation in a k-nearest-neighborhood.
%
%   INPUT PARAMETERS:
%
%       - x:        #P x domainDim set of undeformed point cloud
%                   coordinates
%
%       - xPrime:  #P x imageDim set of deformed point cloud coordinates
%
%       - v:        #P x domainDim vector field defined on the undeformed
%                   point cloud
%
%       - dim:      The intrinsic dimensionality of the manifold. If this
%                   field is empty or nonpositive it defaults to
%                   min(imageDim, domainDim)
%
%       - k:        The number of nearest-neighbors that are used to define
%                   the affine transformation approximating the local map
%                   x(p)->xPrime(p). Defaults to 15
%
%       - verbose:  Whether to produce verbose progress output. Default is
%                   false
%
%   OUTPUT PARAMETERS:
%
%       - vPrime:               #P x imageDim transformed vector field
%                               defined on the deformed point cloud
%
%       - allPushForwards:      #P x 1 cell array. Each entry contains a
%                               domainDim x imageDim matrix defining the
%                               complete pushward map from x->xPrime
%
%
%       - allUndeformedBases:   #P x 1 cell array. Each entry contains a
%                               domainDim x dim matrix, the columns of
%                               which form an orthonormal basis of the
%                               input manifold at the corresponding point
%
%       - allBasisMaps:         #P x 1 cell array. Each entry contains a
%                               dim x dim matrix defining the linear
%                               transformation from the undeformed tangent
%                               space to the deformed tangent space
%
%       - allDeformedBases:     #P x 1 cell array. Each entry contains an
%                               imageDomain x dim matrix, the columns of
%                               which form an orthonormal basis of the
%                               transformed manifold at the corresponding
%                               point
%
%   by Dillon Cislo 06/03/2024

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 4), dim = []; end
if (nargin < 5), k = 15; end
if (nargin < 6), verbose = false; end

validateattributes(x, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(x, 1); domainDim = size(x, 2);

validateattributes(xPrime, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', numPoints});
imageDim = size(xPrime, 2);

validateattributes(v, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', numPoints, 'ncols', domainDim});

if (isempty(dim) || (dim <= 0))
    dim = min(domainDim, imageDim);
end
validateattributes(dim, {'numeric'}, {'scalar', 'finite', 'real', ...
    'positive', 'integer', '<=', min(domainDim, imageDim)});

validateattributes(k, {'numeric'}, {'scalar', 'finite', 'real', ...
    'positive', 'integer'});

validateattributes(verbose, {'logical'}, {'scalar'});

%--------------------------------------------------------------------------
% PUSH FORWARD VECTOR FIELD
%--------------------------------------------------------------------------

% Find the nearest neighbors of the original points
knnIDx = knnsearch(x, x, 'K', k+1);

vPrime = zeros(numPoints, imageDim);
allPushForwards = cell(numPoints, 1);
allUndeformedBases = cell(numPoints, 1);
allBasisMaps = cell(numPoints, 1);
allDeformedBases = cell(numPoints, 1);

try

    if verbose

        DQ = parallel.pool.DataQueue;
        afterEach(DQ, @parallelProgressBar)
        parallelProgressBar(0, numPoints);

    end

    parfor pID = 1:numPoints

        if verbose, send(DQ, []); end

        % Fit a dim-dimensional hyperplane to each point neighborhood in the
        % UNDEFORMED configuration using local PCA (domainDim x dim)
        x_nn = x(knnIDx(pID, :), :);
        x_nn = x_nn - mean(x_nn, 1);
        [undeformedBasis, undeformedEigs] = eig(x_nn.' * x_nn, 'vector');
        [~, sortIDx] = sort(abs(undeformedEigs), 'descend');
        undeformedBasis = undeformedBasis(:, sortIDx(1:dim));

        % Project the undeformed difference vectors onto the local tangent
        % space
        u_nn = x(knnIDx(pID, :), :) - x(pID, :);
        u_nn = u_nn * undeformedBasis;

        % Fit a dim-dimensional hyperplane to each point neighborhood in the
        % DEFORMED configuration using local PCA (imageDim x dim)
        xPrime_nn = xPrime(knnIDx(pID, :), :);
        xPrime_nn = xPrime_nn - mean(xPrime_nn, 1);
        [deformedBasis, deformedEigs] = eig(xPrime_nn.' * xPrime_nn, 'vector');
        [~, sortIDx] = sort(abs(deformedEigs), 'descend');
        deformedBasis = deformedBasis(:, sortIDx(1:dim));

        % Project the deformed difference vectors onto the local tangent space
        uPrime_nn = xPrime(knnIDx(pID, :), :) - xPrime(pID, :);
        uPrime_nn = uPrime_nn * deformedBasis;

        % Find the linear transformation between the undeformed->deformed
        % tangent spaces (dim x dim)
        basisMap = lsqminnorm(u_nn, uPrime_nn);

        % Compose transformations to generate complete pushforward map
        pushForward = undeformedBasis * basisMap * deformedBasis.';

        vPrime(pID, :) = v(pID, :) * pushForward;
        allPushForwards{pID} = pushForward;
        allUndeformedBases{pID} = undeformedBasis;
        allBasisMaps{pID} = basisMap;
        allDeformedBases{pID} = deformedBasis;

    end

    if verbose
        clear DQ parallelProgressBar % Clears persistent variables
    end

catch

    for pID = 1:numPoints

        if verbose, progressbar(pID, numPoints); end

        % Fit a dim-dimensional hyperplane to each point neighborhood in the
        % UNDEFORMED configuration using local PCA (domainDim x dim)
        x_nn = x(knnIDx(pID, :), :);
        x_nn = x_nn - mean(x_nn, 1);
        [undeformedBasis, undeformedEigs] = eig(x_nn.' * x_nn, 'vector');
        [~, sortIDx] = sort(abs(undeformedEigs), 'descend');
        undeformedBasis = undeformedBasis(:, sortIDx(1:dim));

        % Project the undeformed difference vectors onto the local tangent
        % space
        u_nn = x(knnIDx(pID, :), :) - x(pID, :);
        u_nn = u_nn * undeformedBasis;

        % Fit a dim-dimensional hyperplane to each point neighborhood in the
        % DEFORMED configuration using local PCA (imageDim x dim)
        xPrime_nn = xPrime(knnIDx(pID, :), :);
        xPrime_nn = xPrime_nn - mean(xPrime_nn, 1);
        [deformedBasis, deformedEigs] = eig(xPrime_nn.' * xPrime_nn, 'vector');
        [~, sortIDx] = sort(abs(deformedEigs), 'descend');
        deformedBasis = deformedBasis(:, sortIDx(1:dim));

        % Project the deformed difference vectors onto the local tangent space
        uPrime_nn = xPrime(knnIDx(pID, :), :) - xPrime(pID, :);
        uPrime_nn = uPrime_nn * deformedBasis;

        % Find the linear transformation between the undeformed->deformed
        % tangent spaces (dim x dim)
        basisMap = lsqminnorm(u_nn, uPrime_nn);

        % Compose transformations to generate complete pushforward map
        pushForward = undeformedBasis * basisMap * deformedBasis.';

        vPrime(pID, :) = v(pID, :) * pushForward;
        allPushForwards{pID} = pushForward;
        allUndeformedBases{pID} = undeformedBasis;
        allBasisMaps{pID} = basisMap;
        allDeformedBases{pID} = deformedBasis;

    end

end

end