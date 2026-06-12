function [allBases, allPTMaps] = buildPointCloudTangentSpace( ...
    X, dim, k, A, verbose)
%POINTCLOUDTANGENTSPACE Build a local tangent space for each point in a
%d-dimensional point cloud manifold embedded in D-dimensional
%Euclidean space and build parallel transport maps between them.
%
%This construction follows the one in "Parallel Transport Unfolding" by
%Budninsky et al (2019). Tangent spaces are computed by fitting a dim-plane
%to each points k-nearest neighbors. Note that this method computes
%arbitrarily oriented frames! The parallel transport maps are simply
%orthogonal transformations \in O(dim) that align the frames (one could
%specialize to SO(dim) by systematically orienting all frames along a
%minimum spanning tree, for instance, such that all transformations become
%pure rotations). 
%
%NOTE: the output of allPTMaps{i,j} acts on ROW VECTORS from the RIGHT, not
%on column vectors from the left! See below.
%
%An ambient space vector field V (i.e. an #P x ambiDim matrix) can be
%projected onto the tangent space (i.e. a #P x dim matrix) in the
%following way:
%
%   tanV = mat2cell(V, ones(1, ambiDim), ambiDim);
%   tanV = cellfun(@(x,y) x * y, tanV, allBases, 'Uni', false);
%   tanV = vertcat(tanV{:});
%
%where tanV(i,a) is the component of the vector at point X(i) along the
%local tangent space direction a. The vector at point X(i) can then be
%mapped into the tangent space at point X(j) using:
%
%   tanV(i,:) * allPTMaps{j,i} (Acts on row vectors from the right!)
%
%   INPUT PARAMETERS:
%
%       - X:        #P x ambiDim set of point cloud coordinates.
%
%       - dim:      The intrinsic dimnesionality of the manifold. If this
%                   field is empty or non-positive it defaults to the
%                   embeddeding space dimension ambiDim.
%
%       - k:        The number of nearest neighbors used to fit the local
%                   tangent spaces. Defaults to 15.
%
%       - A:        A #P x #P symmetric logical adjacency matrix, such
%                   that A(i,j) true means that we construct a parallel
%                   transport map from T_{X(j)}M -> T_{X(i)}M. Users can
%                   also supply the strings 'full' (all points connected)
%                   or 'knn' (use symmetrized k-nn graph). Note that
%                   default constructions include the self-edge. Default is
%                   'knn'.
%
%       - verbose:  Whether to produce verbose progress output. Default is
%                   false.
%
%   OUTPUT PARAMETERS:
%
%       - allBases:     #P x 1 cell array. allBases{i} is a ambiDim x dim
%                       array of orthonormal tangent space basis vectors
%                       for the point at X(i).
%
%       - allPTMaps:    #P x #P cell array. allPTMaps{i,j} is the
%                       orthogonal transformation that maps vectors in
%                       T_{X(j)}M -> T_{X(i)}M. allPTMaps{i,j} is empty if
%                       A(i,j) is false. allPTMaps{i,i} is always the
%                       identity. note that allPTMaps{i,j} acts on ROW
%                       VECTORS from the RIGHT!
%
%   by Dillon Cislo 2026/02/04

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 2), dim = []; end
if (nargin < 3), k = 15; end
if (nargin < 4), A = 'knn'; end
if (nargin < 5), verbose = false; end

validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X, 1); ambiDim = size(X, 2);

if (isempty(dim) || (dim <= 0)), dim = ambiDim; end
validateattributes(dim, {'numeric'}, {'scalar', 'finite', 'real', ...
    'positive', 'integer', '<=', ambiDim});

validateattributes(k, {'numeric'}, {'scalar', 'finite', 'real', ...
    'positive', 'integer', '<=', numPoints-1});

% Find the nearest neighbors of the original points
knnIDx = knnsearch(X, X, 'K', k+1);

validateattributes(verbose, {'logical'}, {'scalar'});

if isempty(A), A = 'knn'; end
if ischar(A)
    A = lower(A);
    if strcmpi(A, 'full')
        A = true([numPoints, numPoints]);
    elseif strcmpi(A, 'knn')
        A = sparse(reshape(repmat((1:numPoints).', [1, k+1]), [], 1), ...
            knnIDx(:), 1, numPoints, numPoints);
        A = full((A + A.') > 0);
    else
        error('Invalid adjacency matrix character vector definition');
    end
else
    validateattributes(A, {'numeric'}, {'2d', 'nonnegative', 'finite', ...
        'real', 'nrows', numPoints, 'ncols', numPoints});
    A = full(A > 0);
end
assert(issymmetric(A), 'The supplied adjacency matrix is non-symmetric');

% Handle the trivial case
if (dim == ambiDim)
    allBases = repmat({eye(ambiDim)}, [numPoints, 1]);
    allPTMaps = repmat(allBases, [1, numPoints]);
    allPTMaps(~A) = repmat({[]}, [nnz(~A), 1]);
    return
end

%--------------------------------------------------------------------------
% COMPUTE TANGENT SPACE FRAME FIELD
%--------------------------------------------------------------------------

allBases = cell(numPoints, 1);

if verbose, disp('Building tangent space frame field:'); end

try
    
    error('CPU parallelism is slow here');
    
    if verbose
        DQ = parallel.pool.DataQueue;
        afterEach(DQ, @parallelProgressBar)
        parallelProgressBar(0, numPoints);
    end
    
    parfor pID = 1:numPoints

        if verbose, send(DQ, []); end

        % Fit a dim-dimensional hyperplane to each point neighborhood
        % using local PCA (ambiDim x dim)
        X_nn = X(knnIDx(pID, :), :);
        X_nn = X_nn - mean(X_nn, 1);
        [curBasis, curEigs] = eig(X_nn.' * X_nn, 'vector');
        [~, sortIDx] = sort(abs(curEigs), 'descend');
        curBasis = curBasis(:, sortIDx(1:dim));

        allBases{pID} = curBasis;

    end
    
    if verbose
        clear DQ parallelProgressBar % Clears persistent variables
    end
    
catch
    
    for pID = 1:numPoints

        if verbose, progressbar(pID, numPoints); end

        % Fit a dim-dimensional hyperplane to each point neighborhood
        % using local PCA (ambiDim x dim)
        X_nn = X(knnIDx(pID, :), :);
        X_nn = X_nn - mean(X_nn, 1);
        [curBasis, curEigs] = eig(X_nn.' * X_nn, 'vector');
        [~, sortIDx] = sort(abs(curEigs), 'descend');
        curBasis = curBasis(:, sortIDx(1:dim));

        allBases{pID} = curBasis;

    end
    
end

%--------------------------------------------------------------------------
% COMPUTE PARALLEL TRANSPORT MAPS
%--------------------------------------------------------------------------

allPTMaps = cell(numPoints, numPoints);
if (nargout < 2), return; end


if verbose, disp(' '); disp('Building parallel transport maps:'); end

[rows, cols] = ind2sub(size(A), find(A(:)));
pairIDx = unique(sort([rows, cols], 2), 'rows');
numPairs = size(pairIDx, 1);

try
    
    error('CPU parallelism is slow here');
    
    if verbose
        DQ = parallel.pool.DataQueue;
        afterEach(DQ, @parallelProgressBar)
        parallelProgressBar(0, numPairs);
    end
    
    allPTVec = cell(numPairs, 1);
    parfor pairID = 1:numPairs

        if verbose, send(DQ, []); end

        curPair = pairIDx(pairID, :);
        if ~A(curPair(1), curPair(2)), continue; end
        basisI = allBases{curPair(1)};
        basisJ = allBases{curPair(2)};
        
        % Proof that this construction optimizes ||T_I - T_J R||^2 is given
        % in Budninskiy et al.
        [U, ~, V] = svd(basisI.' * basisJ);
        ptIJ = V * U.';
        
        allPTVec{pairID} = ptIJ;
        
    end
    
    if verbose
        clear DQ parallelProgressBar % Clears persistent variables
    end
    
    for pairID = 1:numPairs
        allPTMaps{pairIDx(pairID,1), pairIDx(pairID,2)} = allPTVec{pairID};
        allPTMaps{pairIDx(pairID,2), pairIDx(pairID,1)} = allPTVec{pairID}.';
    end
    
catch
    
    for pairID = 1:numPairs

        if verbose, progressbar(pairID, numPairs); end

        curPair = pairIDx(pairID, :);
        if ~A(curPair(1), curPair(2)), continue; end
        basisI = allBases{curPair(1)};
        basisJ = allBases{curPair(2)};
        
        % Proof that this construction optimizes ||T_I - T_J R||^2 is given
        % in Budninskiy et al.
        [U, ~, V] = svd(basisI.' * basisJ);
        ptIJ = V * U.';
        
        allPTMaps{curPair(1), curPair(2)} = ptIJ;
        allPTMaps{curPair(2), curPair(1)} = ptIJ.';
        
    end
    
end

% Include the identity self-map
diagIDx = ((1:numPoints)-1) .* (numPoints+1) + 1;
allPTMaps(diagIDx) = repmat({eye(dim)}, [numPoints, 1]);
        
end

