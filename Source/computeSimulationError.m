function err = computeSimulationError(measProb, simProb, errorType, X)
%COMPUTESIMULATIONERROR A helper function that computes the error between
%the measured probability at a single experimental time point and a set of
%simulated probabilities.
%
%   INPUT PARAMETERS:
%
%       - measProb:     #N x 1 vector of measured probabilities
%
%       - simProb:      #N x #T matrix of simulated probabilities
%
%       - errorType:    Character vector indicating error type.
%
%           - 'symKLD': symmetric K-L divergence D_KL(M||S) + D_KL(S||M)
%
%           - 'simKLD': K-L divergence relative to the simulated
%           probabilities D_KL(S||M). Heuristically, this divergence is
%           mode seeking
%
%           - 'measKLD': K-L divergence relative to measured probabilities
%           D_KL(M||S). Heuristically this divergence is mean seeking
%
%           - 'MSE': Normalized L-2 error || M - S ||^2 ./ || M ||^2
%
%           - 'geoSphere': Note that the element-wise square root of any
%           N-dimensional discrete probability distribution lies on the
%           N-sphere (since sum(sqrt(P).^2) == 1). Given that embedding,
%           this error computes the geodesic distance along the sphere
%           between two probability distributions
%
%           - 'EMD': A "fast" approximation of the earth movers distance,
%           treating (X, P) as signatures (i.e. a set of spatial points
%           with corresponding weights). In some conceptual sense, this is
%           the best error (since it takes spatial distance into account),
%           but in practice it is too slow to use.
%
%       - X:            #N x dim set of input points on which the dynamics
%                       are defined. Only used if the error type is 'EMD'
%
%   OUTPUT PARAMETERS:
%
%       - err:          1 x #T set of error values for each simulated
%                       probability distribution
%
%   by Dillon Cislo 2024/06/25

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(measProb, {'numeric'}, {'vector', 'nonnegative', ...
    'finite', 'real'}, mfilename, 'measProb');
if (size(measProb, 2) ~= 1), measProb = measProb.'; end
if (abs(sum(measProb)-1) > 1e-14)
    warning('Measured probability input does not appear to be normalized');
end

validateattributes(simProb, {'numeric'}, {'2d', 'nonnegative', ...
    'finite', 'real', 'nrows', numel(measProb)}, mfilename, 'simProb');
if any(abs(sum(simProb,1)-1) > 1e-14)
    warning('Simulated probability input does not appear to be normalized');
end

validateattributes(errorType, {'char'}, {'vector'}, mfilename, 'errorType');

if (nargin < 4), X = []; end
if ~isempty(X)
    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real', ...
        'nrows', numel(measProb)}, mfilename, 'X')
end

%--------------------------------------------------------------------------
% COMPUTE ERROR
%--------------------------------------------------------------------------


if strcmpi(errorType, 'symKLD')
    
    err = (measProb - simProb) .* log(measProb ./ simProb);
    err(isnan(err)) = 0;
    err = sum(err, 1);
    
elseif strcmpi(errorType, 'simKLD')
    
    err = simProb .* log(simProb ./ measProb);
    err(isnan(err)) = 0;
    err = sum(err, 1);
    
elseif strcmpi(errorType, 'dataKLD')
    
    err = measProb .* log(measProb ./ simProb);
    err(isnan(err)) = 0;
    err = sum(err, 1);
    
elseif strcmpi(errorType, 'MSE')
    
    err = sum((measProb-simProb).^2, 1) ./ sum(measProb.^2, 1);
    
elseif strcmpi(errorType, 'geoSphere')
    
    err = acos(sum(sqrt(measProb .* simProb), 1));
    
elseif strcmpi(errorType, 'EMD')

    assert(~isempty(X), ['Please supply point locations to ' ...
        'compute the earth mover''s distance']);
    
    err = zeros(1, size(simProb, 2));
    for tt = 1:size(simProb, 2)
        [~, wasserDist, ~] = PPMMOMT(X, X, simProb(:, tt), measProb, 100);
        err(tt) = wasserDist(end);
    end
    
else
    
    error('Invalid error type');
    
end

end
