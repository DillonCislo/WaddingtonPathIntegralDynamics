function optParam = fitSimTimeScale(dataProb, dataTimes, ...
    simProb, simTimes, timeFunc, numOptParam, varargin)
%FITSIMTIMESCALE Fits the parameters of a function that matches a simulated
%probability time series to a measured one by minimizing an average error
%over each data point and its matched simulation time. 
%
%   INPUT PARAMETERS:
%
%       - dataProb:     #NS x #DT matrix of measured probability vectors.
%                       Each column in the matrix corresponds to the
%                       probability of being in one of the possible states
%                       at a particular time. A cell array containing
%                       multiple experimental runs can also be supplied.
%                       Each experiment can have a different number of data
%                       points.
%
%       - dataTimes:    1 x #DT vector of physical times at which the
%                       data in 'dataProb' were measured. If multiple data
%                       sets are supplied, the size of the cell array must
%                       match the size of 'dataProb'
%
%       - simProb:      #NS x #ST matrix of simulated probability vectors.
%                       Each column in the matrix corresponds to the
%                       probability of being in one of the possible states
%                       at a particular time. A cell array containing
%                       different simulations corresponding to each of the
%                       multiple experimental runs can also be supplied.
%                       All simulations are assumed to have the same number
%                       of time points.
%
%       - simTimes:     1 x #ST vector of simulation times corresponding to
%                       each column in 'simProb'. If multiple simulated
%                       data sets are supplied, the size of the cell array
%                       must match the size of 'simProb'. For example, if
%                       all simulation runs use the same constant time
%                       steps then the times would be given by:
%                       [0 1 2 ... (numSimTimes-1)] * dt
%
%       - timeFunc:     A handle to a parameterized function that maps
%                       simulated time to physical times. It is assumed
%                       that this function is differentiable and
%                       invertible, although these assumptions can probably
%                       relaxed. The first argument of the function should
%                       be simulation time, the second argument should be a
%                       vector of optimizable parameters. Subsequent input
%                       arguments are not used.
%
%       - numOptParam:  The number of optimizable parameters that determine
%                       the behavior of 'timeFunc'
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('ErrorType', errorType = 'symKLD'): Character vector indicating
%       error type. See 'computeSimulationError' for more details.
%
%       - ('WeightType', weightType = 'softmax'): Handling for weighting in
%       contributions to total loss function. Handling options are:
%           
%           - 'max': Only the simulated time point whose corresponding time
%           is closest to the data time is counted towards the loss
%           function. Discontinuous, but doesn't require any tuning of a
%           temperature hyperparameter
%
%           - 'softmax': All simulated time points are included in the
%           error function, weighted according to a softmax function
%
%       - ('InverseTemperature', invT = 10): Inverse temperature used in
%       the softmax function for weight handling
%
%       - ('PointLocations', X = []): #NS x dim set of input points on
%       which the dynamics are defined. Only used for some error types
%
%       - ('OptimizationOptions', optOptions = {}): A cell array containing
%       options that can be supplied to a MATLAB 'optimoptions' object to
%       define solver behavior
%
%       - ('OptimizationFields', optFields = struct()): A struct containing
%       a subset the following common optimization arguments: A, b, Aeq,
%       beq, nonlcon, lb, ub, x0. WARNING: No checks are performed
%
%   OUTPUT PARAMETERS:
%
%       - optParam:     #NP x 1 vector of optimized parameters for the time
%                       function
%
%   by Dillon Cislo 2025/01/19

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------               

% This is probably not the most efficient way to format this, but it does
% streamline multiple experiment handling
if ~iscell(dataProb), dataProb = {dataProb}; end
if ~iscell(dataTimes), dataTimes = {dataTimes}; end
if ~iscell(simProb), simProb = {simProb}; end
if ~iscell(simTimes), simTimes = {simTimes}; end

validateattributes(dataProb, {'cell'}, {'vector'});
numDataSets = numel(dataProb);
validateattributes(dataTimes, {'cell'}, {'vector', 'numel', numDataSets});
validateattributes(simProb, {'cell'}, {'vector', 'numel', numDataSets});
validateattributes(simTimes, {'cell'}, {'vector', 'numel', numDataSets});

numStates = size(dataProb{1}, 1);
numSimTimesPerSet = zeros(1, numDataSets);
numDataPointsPerSet = zeros(1, numDataSets);
for i = 1:numDataSets

    validateattributes(dataProb{i}, {'numeric'}, {'2d', 'finite', ...
        'real', 'nonnegative', 'nrows', numStates});
    numDataPointsPerSet(i) = size(dataProb{i}, 2);
    
    if isempty(dataTimes{i})

        dataTimes{i} = 0:(numDataPointsPerSet(i)-1);

    else

        validateattributes(dataTimes{i}, {'numeric'}, {'vector', 'real', ...
            'nonnegative', 'finite', 'numel', numDataPointsPerSet(i)});
        if (size(dataTimes{i}, 1) ~= 1), dataTimes{i} = dataTimes{i}.'; end

        % Sort the input data points so that time is monotonically
        % increasing
        [dataTimes{i}, DI] = sort(dataTimes{i});
        dataProb{i} = dataProb{i}(:, DI);

    end

    assert(isequal(dataTimes{i}, unique(dataTimes{i})), ...
        ['Multiple data points have been supplied with the same ' ...
        'physical time for data set %d'], i);

    if (numDataPointsPerSet(i) == 1)
        assert(dataTimes{i}(1) > 0, ['Only the initial condition ' ...
            'was supplied for data set %d'], i);
    end

    validateattributes(simProb{i}, {'numeric'}, {'2d', 'finite', ...
        'real', 'nonnegative', 'nrows', numStates});
    numSimTimesPerSet(i) = size(simProb{i}, 2);

    if isempty(simTimes{i})

        simTimes{i} = 0:(numSimTimesPerSet(i)-1);

    else

        validateattributes(simTimes{i}, {'numeric'}, {'vector', 'real', ...
            'nonnegative', 'finite', 'numel', numSimTimesPerSet(i)});
        if (size(simTimes{i}, 1) ~= 1), simTimes{i} = simTimes{i}.'; end

        % Sort the input simulated data points so that time is
        % monotonically increasing
        [simTimes{i}, DI] = sort(simTimes{i});
        simProb{i} = simProb{i}(:, DI);

    end

    % This can probably be relaxed
    if (dataTimes{i}(1) == 0)
        assert(isequal(simProb{i}(:,1), dataProb{i}(:,1)), ...
            ['Measured and simulated initial conditions do not match ' ...
            'for data set %d'], i);
    end

end

assert(strcmpi(class(timeFunc), 'function_handle'), ...
    '''timeFunc'' must be supplied as a function handle');
assert(nargin(timeFunc) >= 2, ...
    '''timeFunc'' must take at least two input arguments');

validateattributes(numOptParam, {'numeric'}, {'scalar', 'positive', ...
    'finite', 'real', 'integer'});

% OPTIONAL INPUT PROCESSING -----------------------------------------------

errorType = 'symKLD';
weightType = 'softmax';
invT = 10;
X = [];
optOptions = {};
optFields = struct();

allWeightTypes= {'softmax', 'max'};

supportedOptions = {'ErrorType', 'WeightType', 'InverseTemperature', ...
    'PointLocations', 'OptimizationOptions', ...
    'OptimizationFields'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'ErrorType')
        errorType = varargin{i+1};
        validateattributes(errorType, {'char'}, {'vector'});
    end

    if strcmpi(varargin{i}, 'WeightType')
        weightType = lower(varargin{i+1});
        validateattributes(weightType, {'char'}, {'vector'});
        assert(ismember(weightType, allWeightTypes), ...
            'Invalid weight type supplied');
    end

    if strcmpi(varargin{i}, 'InverseTemperature')
        invT = varargin{i+1};
        validateattributes(invT, {'numeric'}, {'scalar', 'positive', ...
            'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'PointLocations')
        X = varargin{i+1};
        validateattributes(X, {'numeric'}, {'2d', 'finite', 'real', ...
            'nrows', numStates});
    end

    if strcmpi(varargin{i}, 'OptimizationOptions')
        optOptions = varargin{i+1};
        validateattributes(optOptions, {'cell'}, {'vector'});
    end

    if strcmpi(varargin{i}, 'OptimizationFields')
        optFields = varargin{i+1};
        assert(isstruct(optFields), ...
            'Optimization fields must be supplied as a struct');
    end

end

%--------------------------------------------------------------------------
% FIT TIME SCALE
%--------------------------------------------------------------------------

% Compute Errors ----------------------------------------------------------

allErrors = cell(1, numDataSets);
for i = 1:numDataSets

    % Each entry should be #DT x #ST
    allErrors{i} = nan(numDataPointsPerSet(i), numSimTimesPerSet(i));
    for j = 1:numDataPointsPerSet(i)
        allErrors{i}(j,:) = computeSimulationError(dataProb{i}(:,j), ...
            simProb{i}, errorType, X).';
    end

    assert(~any(isnan(allErrors{i}), 'all'), 'Error computation failed');

end

% Set Up/Run Optimization Problem -----------------------------------------

% Set the loss function
if strcmpi(weightType, 'max')
    fun = @hardMaxLoss;
elseif strcmpi(weightType, 'softmax')
    fun = @softMaxLoss;
else
    error('Invalid weight type supplied');
end

% Inequality constraints
if isfield(optFields, 'A'), A = optFields.A; else, A = []; end
if isfield(optFields, 'b'), b = optFields.b; else, b = []; end

% Equality constraints
if isfield(optFields, 'Aeq'), Aeq = optFields.Aeq; else, Aeq = []; end
if isfield(optFields, 'beq'), beq = optFields.beq; else, beq = []; end

% Upper and lower bounds
if isfield(optFields, 'lb'), lb = optFields.lb; else, lb = []; end
if isfield(optFields, 'ub'), ub = optFields.ub; else, ub = []; end

% Nonlinear constraints
if isfield(optFields, 'nonlcon')
    nonlcon = optFields.nonlcon;
else
    nonlcon = [];
end

% Initial condition
if isfield(optFields, 'x0')
    x0 = optFields.x0;
else
    x0 = ones(numOptParam, 1);
end

% Create optimization options struct
options = optimoptions('fmincon', optOptions{:});

optParam = fmincon( fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options );

%--------------------------------------------------------------------------
% LOSS FUNCTIONS
%--------------------------------------------------------------------------

    function E = hardMaxLoss(x)

        E = 0;
        for ii = 1:numDataSets

            % Evaluate time function given current parameter set
            evalSimTimes = timeFunc(simTimes{ii}, x);

            for jj = 1:numDataPointsPerSet(ii)

                % Determine which simulation steps most closely match the
                % data times
                [~, minID] = min(abs(dataTimes{ii}(jj)-evalSimTimes));

                E = E + allErrors{ii}(jj, minID);

            end

        end


    end

    function E = softMaxLoss(x)

        E = 0;
        for ii = 1:numDataSets

            % Evaluate time function given current parameter set
            evalSimTimes = timeFunc(simTimes{ii}, x);

            % Compute the weights (#DT x #ST) -- sum(weights, 2) == 1
            weights = (dataTimes{ii}.' - evalSimTimes).^2;
            
            weights = exp(-invT .* weights);
            weights = weights ./ sum(weights, 2);

            % weights = -invT .* weights - max(-invT .* weights);
            % weights = exp(weights - logsumexp(weights, 2));
            % % weights = weights ./ sum(weights, 2);

            E = E + sum(weights .* allErrors{ii}, 'all');

        end

    end

end
