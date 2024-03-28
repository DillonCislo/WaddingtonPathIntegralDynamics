function [allProb, viewTimes] = evolveProbabilities(initProb, T, ...
    numTimeSteps, numViewTimes, dt, strictNormalization, useGPU)
%EVOLVEPROBABILITIES Time evolve an initial probability distribution using
%the (left Markov) transition probability operator
%
%   INPUT PARAMETERS:
%
%       - initProb:             #N x 1 initial probability distribution
%
%       - T:                    #N x #N transition probability matrix
%
%       - numTimeSteps:         The number of time steps over which to
%                               evolve the initial probability
%
%       - numViewTimes:         The number of intermediate probability
%                               distributions to record in the output
%
%       - dt:                   The time step accrued for each application
%                               of the transition probability matrix. 
%
%       - strictNormalization:  Whether or not to re-normalize the
%                               probability at every step to avoid problems
%                               with roundoff error
%
%       - useGPU:               Whether or not to use the GPU to accelerate
%                               time evolution
%
%   OUTPUT PARAMETERS:
%
%       - allProb:      #N x numViewTimes matrix of time-evolved
%                       probability distributions
%
%       - viewTimes     1 x numViewTimes vector of times corresponding to
%                       the entries in 'allProb'
%
%   by Dillon Cislo 2024/03/27

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(initProb, {'numeric'}, {'vector', 'nonnegative', 'finite', ...
    'real' , '<=', 1});
normP0 = sum(initProb);
if (abs(1-normP0) > 2*eps(1))
    warning(['Input probability distribution was not properly ' ...
        'normalized. Renormalizing... ']);
    initProb = initProb ./ normP0;
end
if (size(initProb, 2) ~= 1), initProb = initProb.'; end
numStates = numel(initProb);

validateattributes(T, {'numeric'}, {'2d', 'finite', 'real', ...
    'nonnegative', 'nrows', numStates, 'ncols', numStates});
normT = sum(T,1);
if any(abs(1-normT) > 2*eps(1))
    warning(['Input transition matrix was not properly ' ...
        'normalized. Renormalizing... ']);
    T = T ./ normT;
end

validateattributes(numTimeSteps, {'numeric'}, {'scalar', 'integer', ...
    'positive', 'finite', 'real'});

if ((nargin < 4) || isempty(numViewTimes))
    numViewTimes = min(numTimeSteps+1, 500);
end
validateattributes(numViewTimes, {'numeric'}, {'scalar', 'integer', ...
    'positive', 'finite', 'real', '<=', numTimeSteps+1});

if ((nargin < 5) || isempty(dt)), dt = 1; end
validateattributes(dt, {'numeric'}, {'scalar', 'positive', ...
    'finite', 'real'});

if ((nargin < 6) || isempty(strictNormalization))
    strictNormalization = true;
end
validateattributes(strictNormalization, {'logical'}, {'scalar'});

if ((nargin < 7) || isempty(useGPU)), useGPU = true; end
validateattributes(useGPU, {'logical'}, {'scalar'});
if useGPU, try gpuDevice; catch, useGPU = false; end; end


%--------------------------------------------------------------------------
% TIME EVOLVE PROBABILITIES
%--------------------------------------------------------------------------

viewTIDx = round(linspace(1, numTimeSteps+1, numViewTimes));
assert(isequal(viewTIDx, unique(viewTIDx)), ...
    'View time assignment failed');

curProb = initProb;
viewTimes = zeros(1, numViewTimes);
allProb = zeros(numStates, numViewTimes);
if (numViewTimes > 1), allProb(:,1) = curProb; end

if useGPU
    T = gpuArray(T);
    curProb = gpuArray(curProb);
    allProb = gpuArray(allProb);
end

for i = 1:numTimeSteps

    curProb = T * curProb;
    if strictNormalization, curProb = curProb ./ sum(curProb); end

    [isViewTime, viewID] = ismember((i+1), viewTIDx);
    if isViewTime
        viewTimes(viewID) = i * dt;
        allProb(:, viewID) = curProb;
    end

end

allProb = gather(allProb);

end