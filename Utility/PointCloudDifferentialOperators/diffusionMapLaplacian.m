function [L, M, sigma] = diffusionMapLaplacian(X, sigma, kNN, normalizeDensity)
%DIFFUSIONMAPLAPLACIAN Construct a point cloud Laplace-Beltrami operator
%based on a diffusion map algorithm. For any manifold with a boundary, the
%Laplace-Beltrami operator needs to be interpreted as acting with Neumann
%boundary conditions. See "Geometric diffusions as a tool for harmonic
%analysis and structure definition of data: Diffusion Maps" by Coifman,
%Lafon, et al. (2005) for more details.  NOTE: We allow the user to set the
%number of nearest neighbors retained in the affinity matrix computation,
%but suggest using a fully connected graph to avoid subtle issues in
%asymptotic convergence.
%
%   INPUT PARAMETERS:
%
%       - X:                    #N x dim set of input points
%
%       - sigma:                Bandwidth of the affinity matrix kernel
%
%       - kNN:                  The number of nearest neighbors retained in
%                               the affinity matrix computation
%
%       - normalizeDensity:     Whether to normalize density in the
%                               diffusion map computation (true)
%
%   OUTPUT PARAMETERS:
%
%       - L:        #N x #N Laplace-Beltrami operator. L is a symmetric,
%                   negative semi-definite matrix with no negative
%                   off-diagonal entries.
%
%       - M:        #N x #N diagonal mass matrix
%
%       - sigma:    Returns the computed value of sigma. If a specific
%                   value is supplied, then this is identical to the input. 
%
%   by Dillon Cislo 2024/03/21

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
% NOTE: More input checks are performed in the diffusion map functions
if (nargin < 4), normalizeDensity = true; end

validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1); % dims = size(X,2);

if (nargin < 2), sigma = -1; end
validateattributes(sigma, {'numeric'}, {'scalar', 'finite', 'real'});

if (nargin < 3), kNN = numPoints; end
validateattributes(kNN, {'numeric'}, ...
    {'scalar', 'integer', 'finite', 'real'});
if (kNN <= 0), kNN = numPoints; end

%--------------------------------------------------------------------------
% COMPUTE LAPLACIAN
%--------------------------------------------------------------------------

% Compute affinity matrix
affinityOptions = struct();
affinityOptions.Sigma = sigma;
affinityOptions.NumNeighbors = kNN;
affinityOptions.Verbose = false;

[K, ~, ~, sigma] = affinityMatrix(X, affinityOptions);

% Compute transition probability matrix corresponding to the
% Laplace-Beltrami operator on input point set
mapOptions = struct();
mapOptions.Normalization = 'LaplaceBeltrami';
mapOptions.NumVectors = 0;
mapOptions.Verbose = false;
mapOptions.normalizeDensity = normalizeDensity;

[~, ~, ~, KAlpha, DAlpha] = diffusionMap(K, mapOptions);

% Generate the mass matrix
M = spdiags(DAlpha, 0, numPoints, numPoints);

% Convert diffusion map output into Laplace-Beltrami operator
L = KAlpha - M;

% Split the normalization by the time step across the mass matrix and the
% Laplace-Beltrami operator
L = L ./ sqrt(sigma);
M = M .* sqrt(sigma);

% Use dense matrices for fully connected operators
if (kNN == numPoints)
    L = full(L);
    M = full(M);
end

end

