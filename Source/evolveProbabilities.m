function [allProb, viewTimes] = ...
    evolveProbabilities(initProb, T, TSchedule, varargin)
%EVOLVEPROBABILITIES Time evolve an initial probability distribution using
%the (left Markov) transition probability operator
%
%   INPUT PARAMETERS:
%
%       - initProb:             #N x 1 initial probability distribution
%
%       - T:                    #N x #N transition probability matrix -OR-
%                               #NT x 1 cell array of transition
%                               probability matrices
%
%       - TSchedule:            The number of time steps over which to
%                               evolve the initial probability -OR-
%                               #NT x 1 array of indices into 'T'
%                               indicating which transition matrix to use
%                               at each time step        
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('NumViewTimes', numViewTimes = -1): The number of intermediate
%       probability distributions to record in the output. Non-positive
%       values report all time steps.
%
%       - ('TimeStep', dt = 1): The time step accrued for each application
%       of the transition probability matrix.
%
%       - ('LogTransitionMatrix', logT = []): #N x #N log transition
%       matrix. Used for stable computation of graph edge weights with
%       small temperatures/scalar metrics. This overrides the 'T' input if
%       supplied
%
%       - ('StrictNormalization', strictNormalization = true):  Whether or
%       not to re-normalize the probability at every step to avoid problems
%       with roundoff error.
%
%       - ('UseGPU', useGPU = true): Whether or not to use the GPU to
%       accelerate time evolution
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
validateattributes(initProb, {'numeric'}, {'vector', 'nonnegative', ...
    'finite', 'real' , '<=', 1}, ...
    'evolveProbabilities', 'initProb');
if (abs(1-sum(initProb)) > 1e-14)
    warning(['Input probability distribution is not properly ' ...
        'normalized.']);
end
if (size(initProb, 2) ~= 1), initProb = initProb.'; end
numStates = numel(initProb);

% OPTIONAL INPUT PROCESSING -----------------------------------------------

numViewTimes = -1;
dt = 1;
strictNormalization = true;
useGPU = true;
logT = {};

supportedOptions = {'NumViewTimes', 'TimeStep', ...
    'StrictNormalization', 'UseGPU', 'LogTransitionMatrix'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'NumViewTimes')
        numViewTimes = varargin{i+1};
        validateattributes(numViewTimes, {'numeric'}, {'scalar', ...
            'integer', 'finite', 'real'}, ...
            'evolveProbabilities', 'numViewTimes');
    end

    if strcmpi(varargin{i}, 'TimeStep')
        dt = varargin{i+1};
        validateattributes(dt, {'numeric'}, {'scalar', 'positive', ...
            'finite', 'real'}, ...
            'evolveProbabilities', 'dt');
    end

    if strcmpi(varargin{i}, 'StrictNormalization')
        strictNormalization = varargin{i+1};
        validateattributes(strictNormalization, {'logical'}, {'scalar'}, ...
            'evolveProbabilities', 'strictNormalization');
    end

    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'}, ...
            'evolveProbabilities', 'useGPU');
    end

    if strcmpi(varargin{i}, 'LogTransitionMatrix')
        % Validation checks are performed in the next section
        logT = varargin{i+1};
    end

end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

% PROCESS TRANSITION MATRICES ---------------------------------------------

assert(~(isempty(T) && isempty(logT)), ['You must supply either a ' ...
        'transition matrix OR a log transition matrix']);

% Supplying a log-transition matrix overrides the non-stablized computation
if ~isempty(logT)

    if (ismatrix(logT) && ~iscell(logT)), logT = {logT}; end
    
    assert(iscell(logT), ['Log transition matrix must be supplied as a ' ...
        'single matrix or as a cell array of matrices']);

    cellfun(@(x) validateattributes(x, {'numeric'}, {'2d', 'finite', ...
        'real', 'nrows', numStates, 'ncols', numStates}, ...
        'evolveProbabilities', 'logT'), ...
        logT, 'Uni', false);

    numTransMatrices = numel(logT);

else

    if (ismatrix(T) && ~iscell(T)), T = {T}; end

    assert(iscell(T), ['Transition matrix must be supplied as a single ' ...
        'matrix or as a cell array of matrices']);

    cellfun(@(x) validateattributes(x, {'numeric'}, {'2d', 'finite', ...
        'real', 'nonnegative', 'nrows', numStates, 'ncols', numStates}, ...
        'evolveProbabilities', 'T'), ...
        T, 'Uni', false);

    % We only check for normalization on the non-stabilized branch
    checkNormT = cellfun(@(x) max(abs(1-sum(x,1))), T, 'Uni', true);
    if (any(checkNormT > 1e-14))
        warning(['The following transition matrices are not ' ...
            'properly normalized: ' repmat('%d ', [1 sum(checkNormT)])], ...
            find(checkNormT));
    end

    numTransMatrices = numel(T);

end

if isscalar(TSchedule)
    
    assert(numTransMatrices == 1, ['Mutliple transition matrices ' ...
        'supplied with only a scalar schedule']);
    numTimeSteps = TSchedule;
    validateattributes(numTimeSteps, {'numeric'}, {'scalar', ...
        'integer', 'positive', 'finite', 'real'}, ...
        'evolveProbabilities', 'TSchedule');
    TSchedule = ones(numTimeSteps, 1);

else

    validateattributes(TSchedule, {'numeric'}, {'vector', 'positive', ...
        'finite', 'real', 'integer', '<=', numTransMatrices}, ...
        'evolveProbabilities', 'TSchedule');
    if (size(TSchedule, 2) ~= 1), TSchedule = TSchedule.'; end
    numTimeSteps = numel(TSchedule);

end

assert(numViewTimes <= (numTimeSteps+1), 'Too many view times requested');
if (numViewTimes <= 0), numViewTimes = numTimeSteps+1; end

%--------------------------------------------------------------------------
% TIME EVOLVE PROBABILITIES
%--------------------------------------------------------------------------

viewTIDx = round(linspace(1, numTimeSteps+1, numViewTimes));
assert(isequal(viewTIDx, unique(viewTIDx)), 'View time assignment failed');

curProb = initProb;
viewTimes = zeros(1, numViewTimes);
allProb = zeros(numStates, numViewTimes);
if (numViewTimes > 1), allProb(:,1) = curProb; end

if useGPU

    if ~isempty(logT)
        for i = 1:numTransMatrices, logT{i} = gpuArray(logT{i}); end
    else
        for i = 1:numTransMatrices, T{i} = gpuArray(T{i}); end
    end

    curProb = gpuArray(curProb);
    allProb = gpuArray(allProb);

end

if ~isempty(logT)

    % Log-stabilized computation ------------------------------------------
    curProb = log(curProb);
    for i = 1:numTimeSteps

        curProb = logsumexp(logT{TSchedule(i)} + curProb.', 2);

        [isViewTime, viewID] = ismember((i+1), viewTIDx);
        if isViewTime
            viewTimes(viewID) = i * dt;
            allProb(:, viewID) = exp(curProb);
            if strictNormalization
                allProb(:, viewID) = ...
                    allProb(:, viewID) ./ sum(allProb(:, viewID));
            end
        end

    end

else

    % Unstabilized standard computation -----------------------------------
    for i = 1:numTimeSteps

        curProb = T{TSchedule(i)} * curProb;
        if strictNormalization, curProb = curProb ./ sum(curProb); end

        [isViewTime, viewID] = ismember((i+1), viewTIDx);
        if isViewTime
            viewTimes(viewID) = i * dt;
            allProb(:, viewID) = curProb;
        end

    end

end

allProb = gather(allProb);

end