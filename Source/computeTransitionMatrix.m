function T = computeTransitionMatrix(X, U, dt, varargin)
%COMPUTETRANSITIONMATRIX Compute the transition matrix that discretizes the
%short time Fokker-Planck drift-diffusion dynamics on a set of points
%subject to a user defined potential. Explicitly, T(i,j) defines the
%probability (NOT probability density) of transitioning from X(j) -> X(i).
%It is assumed that the points themselves are sampled from a
%drift-diffusion process (NOT necessarily the one defined by U!) defined by
%a potential U0. NOTE: You are allowed to set the various diffusion
%coefficients for the sake of generalizability, but we STRONGLY recommend
%you just leave them equal to one.
%
%   INPUT PARAMETERS:
%
%       - X:    #N x dim set of input points
%
%       - U:    #N x 1 scalar potential defined on the input points that
%               defines the drift-diffusion dynamics
%
%       - dt:   The short time step over which the transition matrix
%               approximates the drift-diffusion dynamics
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('PointPotential', U0 = U): The auxilliary potential defining the
%       equilibrium distribution from which the point set is assumed to be
%       sampled. By default, we assume this is equal to the dynamical
%       potential
%
%       - ('ScalarMetric', scalarMetric = []): The conformal factor of a
%       scalar metric, defined on each input point, that re-scales the
%       dynamical velocity (i.e. v = -(1/scalarMetric) * \nabla U)
%
%       - ('DiffusionCoefficient', D = 1): The diffusion coefficient for
%       the dynamical drift-diffusion process
%
%       - ('PointDiffusionCoefficient', D0 = 1): The diffusion coefficient
%       for the drift-diffusion process assumed to generate the input point
%       set
%
%       - ('ClipThreshold', clipThreshold = 0): Exponential
%       distributions produce insanely small values. Entries |T(i,j)| <
%       this threshold are just set to zero. BE CAREFUL HERE - I HAVE NOT
%       TESTED THIS THOROUGHLY YET
%
%       - ('StrictNormalization', strictNormalization = true): Whether or
%       not to distribute round-off error in the column-wise normalization.
%
%   OUTPUT PARAMETERS:
%
%       - T:    #N x #N (left Markov) transition matrix
%
%   by Dillon Cislo 2024/03/21

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1); % dim = size(X,2);

validateattributes(U, {'numeric'}, {'vector', 'finite', 'real', ...
    'numel', numPoints});
if (size(U,2) ~= 1), U = U.'; end

validateattributes(dt, {'numeric'}, ...
    {'scalar', 'positive', 'finite', 'real'});

% OPTIONAL INPUT PROCESSING -----------------------------------------------

U0 = U;
scalarMetric = [];
D = 1;
D0 = 1;
clipThreshold = 0;
strictNormalization = true;

supportedOptions = {'PointPotential', 'ScalarMetric', ...
    'DiffusionCoefficient', 'PointDiffusionCoefficient', ...
    'ClipThreshold', 'StrictNormalization'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'PointPotential')
        U0 = varargin{i+1};
        validateattributes(U0, {'numeric'}, {'vector', ...
            'finite', 'real', 'numel', numPoints});
        if (size(U0,2) ~= 1), U0 = U0.'; end
    end
    
    if strcmpi(varargin{i}, 'ScalarMetric')
        scalarMetric = varargin{i+1};
    end
    
    if strcmpi(varargin{i}, 'DiffusionCoefficient')
        D = varargin{i+1};
        validateattributes(D, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'});
    end
    
    if strcmpi(varargin{i}, 'PointDiffusionCoefficient')
        D0 = varargin{i+1};
        validateattributes(D0, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'ClipThreshold')
        clipThreshold = varargin{i+1};
        validateattributes(clipThreshold, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'StrictNormalization')
        strictNormalization = varargin{i+1};
        validateattributes(strictNormalization, {'logical'}, {'scalar'});
    end
    
end

if ~isempty(scalarMetric)

    % If the metric is constant over all space it is faster to simply
    % re-scale U than to perform the full metric calculation
    if isscalar(scalarMetric)

        validateattributes(scalarMetric, {'numeric'}, {'scalar', ...
            'finite', 'positive', 'real'});

        U = U ./ scalarMetric;
        scalarMetric = [];

    else

        validateattributes(scalarMetric, {'numeric'}, {'vector', ...
            'finite', 'positive', 'real', 'numel', numPoints});
        if (size(scalarMetric,2) ~= 1)
            scalarMetric = scalarMetric.';
        end

    end

end

%--------------------------------------------------------------------------
% COMPUTE TRANSITION MATRIX
%--------------------------------------------------------------------------

% Fast computation of squared Euclidean distances
T = X * X.';
T = -(diag(T) + diag(T).' - 2 .* T) ./ (4 * D * dt);

if isempty(scalarMetric)
    
    U = U ./ (2 * D);
    T = T + (U.' - U);
    
else
    
    U = U ./ D;
    T = T + ((U.' - U) ./ (scalarMetric.' + scalarMetric));
    
end

T = exp(T);
T = exp(U0 ./ D0) .* T;

% Normalize transition matrix to be a right Markov matrix
nanIDx = isnan(T(:));
if any(nanIDx)
    warning('\nTransition matrix contains NaN prior to normalization');
    T(nanIDx) = 0;
end

infIDx = isinf(T(:));
if any(infIDx)
    warning('\nTransition matrix contains Inf prior to normalization');
    T(infIDx) = 0;
end

if (clipThreshold > 0), T(T(:) < clipThreshold) = 0; end

normT = sum(T,1);
assert(~any(normT == 0), 'Column-wise normalization constant equals 0');
T = T ./ normT;

if strictNormalization
    normT = sum(T,1);
    for i = 1:size(T, 2)
        if (normT(i) > 1)
            [~, maxID] = max(T(:, i));
            T(maxID, i) = T(maxID, i) + (1 - normT(i));
        end
    end
end

assert(~any(T(:) < 0), 'Negative transition probabilities on output');
    
end
