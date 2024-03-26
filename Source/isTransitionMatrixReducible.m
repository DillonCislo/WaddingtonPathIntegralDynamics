function [isReducible, isAperiodic] = ...
    isTransitionMatrixReducible(T, zeroTol)
%ISTRANSITIONMATRIXREDUCIBLE Check if a transition probability matrix is
%reducible. A transition matrix is irreducible if any state in the system
%can be reached from any other state in a finite number of steps. By the
%Perron-Frobenius theorem an irreducible transition matrix has a unique
%stationary limiting distribution. Furthermore if T is aperiodic, then this
%limiting distribution can be reached from any given initial condition
%
%   INPUT PARAMETERS:
%
%       - T:        #N x #N (left Markov) transition probability matrix
%
%       - zeroTol:  Tolerance below which elements of T are set zero prior
%                   to the irreducibility computation.
%                   
%   OUTPUT PARAMETERS:
%
%       - isRedicible:  Logical indicating if T is reducible
%
%       - isAperiodic:  Logical indicating if T is aperiodic
%
%   by Dillon Cislo 2024/03/26

% Input Processing --------------------------------------------------------
validateattributes(T, {'numeric'}, ...
    {'2d', 'nonnegative', 'finite', 'real', 'square', '<=', 1});
numStates = size(T,1);

if (nargin < 2), zeroTol = numStates * eps(1); end
validateattributes(zeroTol, {'numeric'}, ...
    {'scalar', 'nonnegative', 'finite', 'real', '<', 1});

T(T(:) < zeroTol) = 0;

% Check Reducibility ------------------------------------------------------
% A transition matrix is reducible if the corresponding digraph has
% multiple strongly connected components

G = digraph(T);
SCC = conncomp(G, 'OutputForm', 'cell', 'Type', 'strong');
isReducible = numel(SCC) > 1;

% Check Periodicity -------------------------------------------------------
isAperiodic = NaN;

if (nargout > 1)

    % Check if each strongly connected component has at least one
    % self-loop. This is a sufficient, but not a necessary condition
    selfTransitionProb = diag(T);
    hasSelfLoop = cellfun(@(x) any(selfTransitionProb(x) > 0), ...
        SCC, 'UniformOutput', 'true');
    
    if all(hasSelfLoop)

        isAperiodic = true;

    else

        warning(['Periodicity test was inconclusive. Consider a more ' ...
            'rigorous test if necessary']);

    end

end

end

