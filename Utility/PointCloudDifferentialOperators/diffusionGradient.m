function gradS = diffusionGradient(X, S, varargin)
%DIFFUSIONGRADIENT Compute the gradient of a scalar function defined on a
%dim-D point cloud embedded in R^N (d<=N) using the method of 
% "Approximating Gradients for Meshes and Point Clouds via Diffusion
% Metric" by Luo, Safa, and Wang (2009)
%
%   INPUT PARAMETERS:
%
%       - X:    #N x ambiDim set of point cloud coordinates
%
%       - S:    #N x 1 scalar function defined on the input point set
%   
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('IntrinsicDimension', dim = ambiDim): The intrinsic dimension of
%       the manifold
%
%       - ('NumEigenvectors', numEigs = 500): The number of Laplacian
%       eigenvectors to retain in the gradient computation (fewer
%       eigenvectors means greater smoothing)
%
%       - ('EigMethod', eigMethod = 'complete'): Whether to perform a
%       complete eigendecomposition of the Laplace-Beltrami operator (i.e.
%       using 'eig') or an incomplete one (i.e. using 'eigs')
%
%       - ('KNN', knn = 15): The number of nearest neighbors that are used
%       to define the local tangent space at each point pulling back the
%       vector field from diffusion space to ambient space
%
%       - ('TimeStep', timeStep = -1): The diffusion time used in the
%       computation to determine the magnitude of the desired gradient
%       field. Luo et al. claim that this doesn't really matter so long as
%       timeStep >> d^2 where d = the average distance of any point to its
%       k-nearest neighbors, but this claim is NOT true, even for very well
%       behaved low dimensional point clouds. Take the time to explore this
%       parameter
%
%       - ('Verbose', verbose = false): Whether to produce verbose progress
%       output
%
%   'diffusionMapLaplacian' Parameters ------------------------------------
%
%       - ('Laplacian', L = []): #N x #N point cloud Laplace-Beltrami
%       operator
%
%       - ('MassMatrix', M = []): #N x #N diagonal mass matrix
%       corresponding to L
%
%       - ('LaplacianSigma', lapSigma = -1): Bandwidth of the affinity
%       matrix kernel used to construct the Laplace-Beltrami operator
%
%       - ('LaplacianKNN', lapKNN = -1): The number of nearest neighbors
%       retained in the affinity matrix computation needed to construct the
%       Laplace-Beltrami operator. Full connectivity is recommended.
%
%   OUTPUT PARAMETERS:
%
%       - gradS:    #N x ambiDim gradient vector defined on input points
%
%   by Dillon Cislo 2024/06/03

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------

validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1); ambiDim = size(X,2);

validateattributes(S, {'numeric'}, {'vector', 'finite', 'real', ...
    'numel', numPoints});
if (size(S,2) ~= 1), S = S.'; end

% OPTIONAL INPUT PROCESSING -----------------------------------------------

dim = ambiDim;
numEigs = 500;
eigMethod = 'complete';
knn = 15;
timeStep = -1;
verbose = false;

L = [];
M = [];
lapSigma = -1;
lapKNN = -1;

supportedOptions = {'IntrinsicDimension', 'NumEigenvectors', ...
    'Eigmethod', 'KNN', 'TimeStep', 'Verbose', 'Laplacian', ...
    'MassMatrix', 'LaplacianSigma', 'LaplacianKNN'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'IntrinsicDimension')
        dim = varargin{i+1};
        validateattributes(dim, {'numeric'}, {'scalar', 'integer', ...
            'positive', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'NumEigenvectors')
        numEigs = varargin{i+1};
        validateattributes(numEigs, {'numeric'}, {'scalar', 'integer', ...
            'positive', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'EigMethod')
        eigMethod = lower(varargin{i+1});
        validateattributes(eigMethod, {'char'}, {'vector'});
        assert(ismember(eigMethod, {'complete', 'incomplete'}), ...
            'Invalid eigensystem computation method supplied');
    end

    if strcmpi(varargin{i}, 'KNN')
        knn = varargin{i+1};
        validateattributes(knn, {'numeric'}, {'scalar', 'integer', ...
            'positive', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'TimeStep')
        timeStep = varargin{i+1};
        validateattributes(timeStep, {'numeric'}, {'scalar', ...
            'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'Laplacian'), L = varargin{i+1}; end

    if strcmpi(varargin{i}, 'MassMatrix'), M = varargin{i+1}; end

    if strcmpi(varargin{i}, 'LaplacianSigma')
        lapSigma = varargin{i+1};
    end

    if strcmpi(varargin{i}, 'LaplacianKNN')
        lapKNN = varargin{i+1};
    end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end

end

assert(dim <= numEigs, ['You must retain more Laplacian eigenvectors ' ...
    'than the intrinsic dimension of the manifold']);

% Compute time step if necessary
if (timeStep <= 0)
    [~, knnDists] = knnsearch(X, X, 'K', 2);
    timeStep = 30 * mean(knnDists(:,end));
end

% Check Laplacian/mass matrix ---------------------------------------------
% NOTE: Luo et al. use the POSITIVE-DEFINITE Laplacian

if (isempty(L) || isempty(M))

    if verbose, fprintf('Building Laplacian/mass matrix... '); end
    [L, M] = diffusionMapLaplacian(X, lapSigma, lapKNN);
    L = -L;
    if verbose, fprintf('Done\n'); end

else

    validateattributes(L, {'numeric'}, {'2d', 'finite', 'real', ...
        'ncols', numPoints, 'nrows', numPoints});
    assert(issymmetric(L), 'Laplacian is not symmetric');

    % Check for positive-definiteness
    [~, isPD] = chol(L); isPD = isPD == 0;
    if ~isPD
        [~, isPD] = chol(-L); isPD = isPD == 0;
        if isPD
            L = -L;
        else
            error('Laplacian is neither positive nor negative definite');
        end
    end

    validateattributes(M, {'numeric'}, {'2d', 'finite', 'real', ...
        'ncols', numPoints, 'nrows', numPoints});
    assert(isdiag(M), 'Mass matrix is not diagonal');
    assert(all(diag(M) > 0), 'Mass matrix contains non-positive masses');

    clear isPD

end

% Compute Laplacian eigensystem -------------------------------------------
% NOTE: Since L is symmetric PD and M is diagonal PD the matrix product
% M^{-1} L should have real eigenvalues/eigenfunctions. Numerical round-off
% can mess this up though.

if verbose, fprintf('Computing Laplacian eigensystem... '); end

if strcmpi(eigMethod, 'complete')

    % Sparse matrices not supported for generalized eigenvalue problem
    [lapVecs, lapEigs] = eig(full(L), full(M));
    lapEigs = diag(lapEigs);
    [~, sortIDx] = sort(abs(lapEigs), 'ascend');
    lapEigs = lapEigs(sortIDx(1:numEigs));
    lapVecs = lapVecs(:, sortIDx(1:numEigs));

elseif strcmpi(eigMethod, 'incomplete')

    [lapVecs, lapEigs] = eigs(L, M, numEigs, 'smallestabs');
    lapEigs = diag(lapEigs);

else

    error('Invalid eigensystem solution method supplied');

end

lapEigs = real(lapEigs);
lapVecs = real(lapVecs);
lapEigs(1) = round(lapEigs(1), 12);
assert(~any(lapEigs < 0), ['\nLaplacian eigensystem computation ' ...
    'failed. Negative eigenvalues detected']);

if verbose, fprintf('Done\n'); end

%--------------------------------------------------------------------------
% COMPUTE DIFFUSION GRADIENTS
%--------------------------------------------------------------------------

% Project the input function onto the basis of Lapacian eigenvectors
alphaS = full(lapVecs.' * M * S);

% Construct the diffusion gradient vector (see Luo et al. for more details)
diffGrad = exp(lapEigs .* timeStep ./ 2) .* alphaS; % numEigs x 1
diffGrad = repmat(diffGrad.', numPoints, 1); % #N x numEigs

% Compute the diffusion embedding of the input manifold. Note the sign in
% the exponent
diffCoords = ...
    repmat(exp(-lapEigs .* timeStep ./ 2).', numPoints, 1) .* lapVecs;

% Pull the vector field back into the input space (technically we are
% pushing it forward. Not sure if this is always kosher. Diffusion
% embedding is guaranteed to at least always be a homeomorphism in the
% continuous case).
if verbose, disp('Pulling diffusion gradients back to input space'); end
[pbProjDiffGrad, ~, allUndeformedBases, ~, ~] = ...
    pushForwardPointCloudVectorField(diffCoords, X, diffGrad, ...
    dim, knn, verbose);

% Find the magnitudes of the projected diffusion gradients onto each local
% tangent space
projDiffGrad = cellfun(@(x, y, z) sum(((x * y) .* z), 2).', ...
    mat2cell(diffGrad, ones(1, numPoints), numEigs), ...
    allUndeformedBases, allUndeformedBases, 'Uni', false);
projDiffGrad = cell2mat(projDiffGrad);
normProjDiffGrad = sqrt(sum(projDiffGrad.^2, 2));

% Compute the numerical gradients
gradS = normProjDiffGrad .* normalizerow(pbProjDiffGrad) ./ ...
    sqrt(2 * timeStep * (4 * pi * timeStep).^(dim/2));

end

