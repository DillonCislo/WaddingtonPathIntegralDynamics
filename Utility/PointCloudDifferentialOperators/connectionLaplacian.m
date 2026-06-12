function [Lconn, Mconn] = connectionLaplacian( ...
    allPTMaps, L, M, strictProps, verbose)
%CONNECTIONLAPLACIAN Builds a vector connection Laplacian on a
%simplicial complex representing a dim-dimensional Riemannian manifold
%embedded in ambiDim-dimensional Euclidean space. In addition to the
%discrete manifold the ingredients to build connection Laplacian include
%(1) A representation of a scalar Laplacian (e.g the Laplace-Beltrami
%operator) and (2) a set of parallel transport maps between the tangent
%spaces of adjacent points. Generically, a connection Laplacian can be
%defined as the Hessian of the (discrete) vector Dirichlet energy:
%
%   E_D = 1/2 \sum_{ij \in E} w_{ij} ||V_i - R_{ij} V_j||^2 
%       = -1/2 V.' * Lconn V
%
%where V_i is a dim-dimensional tangent vector defined in the tangent space
%of X_i (NOT the ambient space!) and in the second line equality V has been
%flattened into an (#P * dim) x 1 column vector. Here, w_{ij} are the
%(symmetric) off-diagonal elements of the weak Laplacian L and we have
%adopted the convention that L and Lconn are negative semi-definite
%operators. R_{ij} is the transformation from the tangent space at X_j to
%the tangent space at X_j.
%
%   INPUT PARAMETERS:
%
%       - allPTMaps:    #P x #P cell array. allPTMaps{i,j} is the dim x dim
%                       orthogonal transformation that maps vectors in
%                       T_{X(j)}M -> T_{X(i)}M. allPTMaps{i,j} is empty if
%                       A(i,j) is false.
%
%       - L:            #P x #P weak scalar Laplace operator. Ideally, L is
%                       a symmetric, negative semi-definite matrix with no
%                       negative off-diagonal entries.
%
%       - M:            #P x #P diagonal mass matrix. An empty matrix can
%                       be supplied if no output is necessary.
%
%       - strictProps:  Whether to strictly enforce the properties of the
%                       scalar Laplace operator and mass matrix. Defaults
%                       to true.
%
%       - verbose:      Whether to produce verbose progress output.
%                       Defaults to false.
%
%   OUTPUT PARAMETERS:
%
%       - Lconn:        (#P * dim) x (#P * dim) weak connection Laplacian
%                       operator.
%
%       - Mconn:        (#P * dim) x (#P * dim) tiled mass matrix for use
%                       with Lconn.
%
%   by Dillon Cislo  2026/02/05

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 3) || (nargout < 2), M = []; end
if (nargin < 4), strictProps = true; end
if (nargin < 5), verbose = false; end

numPoints = size(allPTMaps, 1);
validateattributes(allPTMaps, {'cell'}, ...
    {'2d', 'nrows', numPoints, 'ncols', numPoints});

validateattributes(L, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', numPoints, 'ncols', numPoints});

% Extract adjacency matrix from weak Laplacian
% A = abs(L) == 0;
A = abs(L) > eps; % This cutoff may have unintended effects...
diagIDx = ((1:numPoints)-1) .* (numPoints+1) + 1;
assert(all(A(diagIDx)), 'Weak Laplacian has zero diagonal element');
assert(issymmetric(A), 'Adjacency matrix defined by L is not symmetric');

% Check parallel transport maps for consistency. We do NOT check that
% allPTMaps{i,j} == inv(allPTMaps{j,i})
allPTMaps(~A(:)) = cell(sum(~A(:)), 1); % Remove unnecessary maps
assert(isequal(~cellfun('isempty', allPTMaps), A), ...
    'Missing parallel transport map for simplex edges');
firstID = find(A(:), 1, 'first');
assert(ismatrix(allPTMaps{firstID}), 'Non-matrix entry in allPTMaps');
dim = size(allPTMaps{firstID}, 1); % Intrinsic manifold dimension
cellfun(@(x) validateattributes(x, {'numeric'}, ...
    {'2d', 'finite', 'real', 'ncols', dim, 'nrows', dim}), ...
    allPTMaps(A(:)), 'Uni', false);
allPTMaps(diagIDx) = repmat({eye(dim)}, [numPoints, 1]); % Self-map is identity

validateattributes(strictProps, {'logical'}, {'scalar'});
validateattributes(verbose, {'logical'}, {'scalar'});

if ~isempty(M)
    validateattributes(M, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', numPoints, 'ncols', numPoints, 'nonnegative'});
end

if strictProps
    
    assert(issymmetric(L), 'Weak Laplacian is not symmetric');
    [~, isPD] = chol(L); isPD = isPD == 0; % L is positive definite
    if ~isPD
        [~, isPD] = chol(-L); isPD = isPD == 0; % L is negative definite
        if ~isPD % Check if we are close to being negative definite
            if (mean(diag(L)) > 0), L = -L; end
            [~, isPD] = chol(-L + 1e-8 * ones(numPoints)); isPD = isPD == 0;
            assert(isPD, ['Weak Laplacian cannot be made close ' ...
                'to negative definite']);
        end
    else
        L = -L; % Make L negative-definite
    end
    assert(~any(L(setdiff(1:numel(L(:)), diagIDx)) < 0), ...
        'Weak Laplacian contains negative off-diagonal elements');
    
    if ~isempty(M)
        assert(isdiag(M), 'Mass matrix is not diagonal');
    end
    
end

%--------------------------------------------------------------------------
% BUILD CONNECTION LAPLACIAN
%--------------------------------------------------------------------------

% Explicit construction (slow and heavy, but retained for clarity) --------
% I = []; J = []; V = [];
% for i = 1:numPoints
%     if verbose, progressbar(i, numPoints); end
%     
%     for j = 1:numPoints
%         
%         if ~A(i,j), continue; end % L(i,j) == 0
%         R = allPTMaps{i,j};
%         
%         for a = 1:dim
%             for b = 1:dim
%                 
%                 I = [I; i + (a-1) * numPoints];
%                 J = [J; j + (b-1) * numPoints];
%                 V = [V; L(i,j) .* R(b,a)]; % NOTICE R(b,a) ORDER HERE!
%                 
%             end
%         end
%     end
% end
% 
% Lconn0 = sparse(I, J, V, numPoints * dim, numPoints * dim);

% Chunked, (partially) vectorized construction ----------------------------
[Ai, Aj] = find(A);
Lij = L(A(:));
nnzL = numel(Lij);

% Pre-compute the per-block row/col offsets for each edge
aOff = (0:(dim-1)).' .* numPoints;
one_dim = ones(dim, 1);

% Choose chunk size so that chunkSize * dim^2 is manageable in RAM
chunkSize = max(1, floor(5e6 / dim^2));

Lconn = spalloc(numPoints * dim, numPoints * dim, nnzL * dim.^2);
for chunkMin = 1:chunkSize:nnzL
    if verbose, progressbar(chunkMin, nnzL); end
    chunkMax = min(nnzL, chunkMin+chunkSize-1);
    chunkIDx = chunkMin:chunkMax;
    % curChunkSize = numel(chunkIDx);
    
    % Fetch all R blocks for this chunk
    Rchunk = allPTMaps(sub2ind([numPoints, numPoints], ...
        Ai(chunkIDx), Aj(chunkIDx)));
    
    % Stack R(:) -> (dim^2) x curChunkSize
    Rchunk = cellfun(@(R) R(:), Rchunk, 'UniformOutput', false);
    Rchunk = [Rchunk{:}];
    
    % Build row/column indices for this chunk
    rows_a = Ai(chunkIDx).'+ aOff;	% dim x curChunkSize
    cols_b = Aj(chunkIDx).'+ aOff;	% dim x curChunkSize
    I = kron(rows_a, one_dim);  % dim^2 x curChunkSize
    J = kron(one_dim, cols_b);	% dim^2 x curChunkSize
    
    % Scale each column by Lij to build values
    V = Rchunk .* (Lij(chunkIDx).'); % dim^2 x curChunkSize
    
    Lconn = Lconn + sparse(I, J, V, numPoints * dim, numPoints * dim);
    
end

if (nnz(Lconn) > 0.2 * numel(Lconn)), Lconn = full(Lconn); end

if ~isempty(M)
    
    Mconn = repmat({M}, [dim, 1]);
    Mconn = blkdiag(Mconn{:});
    if ~issparse(Lconn), Mconn = full(Mconn); end
    
else
    
    Mconn = [];

end

end

