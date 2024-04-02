function U = interpolatePotentialKHarmonic(X, interpVals, interpIDx, k, varargin)
%INTERPOLATEPOTENTIAL Interpolate a smooth potential on a set of points by
%minimizing a k-harmonic energy.
%
%   INPUT PARAMETERS:
%
%       - X:            #N x dim set of input points
%
%       - interpVals:   #IV x 1 vector of known values to interpolate.
%                       These essentially become Dirichlet boundary
%                       conditions during minimization
%
%       - interpIDx:    #IV x 1 vector of point IDs corresponding to the
%                       values in 'interpVals'
%
%       - k:            The power of the Laplacian (k = 1 is harmonic, 
%                       k = 2 is biharmonic, etc). Defaults to biharmonic.
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('TikhonovRegularization', regSigma = 1e-12): A small positive
%       quantity used to make the quadratic problem positive definite.
%
%       - ('RemoveOutliers', removeOutliers = true): Whether or not to
%       remove outliers generated during the interpolation process.
%
%       - ('OutlierThreshold', outlierThreshold = []): The threshold
%       above and below which outliers are removed. If empty and
%       removeOutliers == true, this is automatically set from the values
%       of interpVals
%
%       - ('OutlierNeighbors', outlierNNSize = 10): The number of neighbors
%       over which to average to in order to find the new value for any
%       removed outliers.
%
%       - ('Laplacian', L = []): #N x #N point cloud Laplace-Beltrami
%       operator
%
%       - ('MassMatrix', M = []): #N x #N diagonal mass matrix
%       corresponding to L
%
%       - ('NormalizeMassMatrix', normalizeMassMatrix = true): Whether or
%       not to normalize the mass matrix by its largest value (see
%       'kharmonic.m')
%
%       - ('TimeStep', dt = -1): The time step used to construct the
%       Laplace-Beltrami operator and mass matrix on the point cloud if the
%       operators are not supplied by the user
%
%       - ('Verbose', verbose = false): Whether or not to report verbose
%       progress output
%
%   OUTPUT PARAMETERS:
%
%       - U:            #N x 1 interpolated potential
%
%   by Dillon Cislo 2024/03/27

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1); % dim = size(X,2);

validateattributes(interpVals, {'numeric'}, {'vector', 'finite', 'real'});
assert(numel(interpVals) < numPoints, ['Number of interpolation ' ...
    'values exceeds the number of points']);
if (size(interpVals, 2) ~= 1), interpVals = interpVals.'; end

validateattributes(interpIDx, {'numeric'}, {'vector', 'integer', ...
    'positive', 'finite', 'real', '<=', numPoints});
if (size(interpIDx, 2) ~= 1), interpIDx = interpIDx.'; end
assert(isequal(size(interpIDx), size(interpVals)), ...
    'Interpolation input is improperly sized');
assert(isequal(sort(interpIDx), unique(interpIDx)), ...
    'Interpolation indices contain duplicate values');

% Handle the trivial case of uniform values
if (numel(unique(interpVals)) == 1)
    U = unique(interpVals) .* ones(numPoints, 1);
    return;
end

if (nargin < 4), k = 2; end
if (isempty(k) || (k <= 0)), k = 2; end
validateattributes(k, {'numeric'}, {'scalar', 'positive', ...
    'integer', 'finite', 'real'});

% OPTIONAL INPUT PROCESSING -----------------------------------------------

regSigma = 1e-12;
removeOutliers = true;
outlierThreshold = [];
outlierNNSize = 10;
L = [];
M = [];
normalizeMassMatrix = true;
dt = -1;
verbose = false;

for i = 1:numel(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'TikhonovRegularization')
        regSigma = varargin{i+1};
        validateattributes(regSigma, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'RemoveOutliers')
        removeOutliers = varargin{i+1};
        validateattributes(removeOutliers, {'logical'}, {'scalar'});
    end

    if strcmpi(varargin{i}, 'OutlierThreshold')
        outlierThreshold = varargin{i+1};
        if ~isempty(outlierThreshold)
            validateattribtues(outlierThreshold, {'numeric'}, ...
                {'vector', 'numel', 2, 'finite', 'real'});
            assert(outlierThreshold(2) > outlierThreshold(1), ...
                ['Outlier threshold must have a second element ' ...
                'that is greater than its first element']); 
        end
    end

    if strcmpi(varargin{i}, 'OutlierNeighbors')
        outlierNNSize = varargin{i+1};
        validateattributes(outlierNNSize, {'numeric'}, ...
            {'positive', 'integer', 'scalar', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'Laplacian'), L = varargin{i+1}; end

    if strcmpi(varargin{i}, 'MassMatrix'), M = varargin{i+1}; end

    if strcmpi(varargin{i}, 'NormalizeMassMatrix')
        normalizeMassMatrix = varargin{i+1};
        validateattributes(normalizeMassMatrix, {'logical'}, {'scalar'});
    end

    if strcmpi(varargin{i}, 'TimeStep')
        dt = varargin{i+1};
        validateattributes(dt, {'numeric'}, {'scalar', 'finite', 'real'});
        if (dt <= 0), dt = -1; end
    end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end

end

if isempty(outlierThreshold)
    outlierThreshold = [min(interpVals), max(interpVals)] + 1e-14 * [-1 1];
end

if (isempty(L) || isempty(M))

    if verbose, fprintf('Building Laplacian/mass matrix... '); end
    [L, M] = diffusionMapLaplacian(X, dt);
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

end

if normalizeMassMatrix, M = M ./ max(abs(diag(M))); end

%--------------------------------------------------------------------------
% INTERPOLATE POTENTIAL
%--------------------------------------------------------------------------

if verbose, fprintf('Computing interpolated potential... '); end

% Build the k-laplacian, quadratic coefficients. NOTE: We have already
% checked to ensure that L is positive definite
switch k

    case 1
        Q = L;

    case 2
        Q = L*(M\L);

    case 3
        Q = L*((M\L)*(M\L));
        Q = 0.5*(Q+Q');

    otherwise
        Q = L;
        for ii = 2:k
            Q = Q*(M\L);
        end
end

if (regSigma > 0), Q = Q + regSigma .* eye(size(Q)); end

U = min_quad_with_fixed(Q, zeros(numPoints, 1), interpIDx, interpVals);

if verbose, fprintf('Done\n'); end

%--------------------------------------------------------------------------
% REMOVING OUTLIERS
%--------------------------------------------------------------------------

if removeOutliers

    if verbose
        fprintf('Handling interpolated potential outliers... ');
    end

    rmIDx = find( (U < outlierThreshold(1)) | ...
        (outlierThreshold(2) < U) );

    [nnIDx, nnDists] = knnsearch(X, X, 'K', numPoints);
    nnIDx(:, 1) = []; nnDists(:, 1) = [];
    assert(all(nnDists(:) > 0), 'Input point set contains duplicates');

    for i = 1:numel(rmIDx)

        curID = rmIDx(i); % The current outlier

        % Determine the averaging neighborhood
        curNNIDx = ~ismember(nnIDx(curID, :), rmIDx);
        assert(sum(curNNIDx) >= outlierNNSize, ...
            'Too few neighbors for outlier removal');
        curNNIDx = find(curNNIDx, outlierNNSize, 'first');
        curNNDists = nnDists(curID, curNNIDx);
        curNNIDx = nnIDx(curID, curNNIDx);

        % Averaging weights are proportional to inverse distance to the
        % outlier
        rmWeights = (1./curNNDists) ./ sum(1./curNNDists);

        U(curID) = sum(rmWeights .* U(curNNIDx) .');

    end

    if verbose, fprintf('Done\n'); end

end

end

