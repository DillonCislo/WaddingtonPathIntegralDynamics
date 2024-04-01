function [timeScale, KLDErr] = ...
    fitProbSimTimeScale(dataProb, dataTimes, simProb, dt, varargin)
%FITPROBSIMTIMESCALE Determines a time scale that best matches a simulated
%probability time series to a measured one by minimizing an average
%(symmetric) K-L divergence over each data point and its matched simulation
%point. It is assumed that (simulation time) = (timeScale) * (physical
%time). WARNING: This function is just a brute-force exhaustive search over
%all feasible time scales, so it may not scale well for very large
%problems.
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
%       - dt:           The uniform simulation time step. It is assumed
%                       that all simulation runs use the same time step, 
%                       i.e. the times for all simulations are given by
%                       [0 1 2 ... (numSimTimes-1)] * dt
%
%   OPTIONAL INPUT PARAMETERS:
%
%       - ('UseGPU', useGPU = true): Whether or not to use the GPU to
%       accelerate time evolution
%
%   OUTPUT PARAMETERS:
%
%       - timeScale:    The scalar time scale that best matches the data to
%                       the simulated results
%
%       - KLDErr:       The average (symmetric) K-L divergence error
%                       associated to the optimal time scale
%
%   by Dillon Cislo 2024/03/29

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------

% This is probably not the most efficient way to format this, but it does
% streamline multiple experiment handling
if ~iscell(dataProb), dataProb = {dataProb}; end
if ~iscell(dataTimes), dataTimes = {dataTimes}; end
if ~iscell(simProb), simProb = {simProb}; end

validateattributes(dataProb, {'cell'}, {'vector'});
numDataSets = numel(dataProb);
validateattributes(dataTimes, {'cell'}, {'vector', 'numel', numDataSets});
validateattributes(simProb, {'cell'}, {'vector', 'numel', numDataSets});

numStates = size(dataProb{1}, 1);
numSimTimes = size(simProb{1}, 2);
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
        'real', 'nonnegative', 'nrows', numStates, 'ncols', numSimTimes});
    
    if (dataTimes{i}(1) == 0)
        assert(isequal(simProb{i}(:,1), dataProb{i}(:,1)), ...
            ['Measured and simulated initial conditions do not match ' ...
            'for data set %d'], i);
    end

end

validateattributes(dt, {'numeric'}, {'scalar', 'positive', ...
    'finite', 'real'});
simTimes = ((0:(numSimTimes-1)) * dt).';

% OPTIONAL INPUT PROCESSING -----------------------------------------------

useGPU = true;

supportedOptions = {'useGPU'};
checkSupportedOptions(supportedOptions, varargin);

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'useGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'});
    end

end

if useGPU, try gpuDevice; catch, useGPU = false; end; end

%--------------------------------------------------------------------------
% FIT TIME SCALE
%--------------------------------------------------------------------------

maxDataTime = max([dataTimes{:}]);
allTimeScales = simTimes(2:end) ./ maxDataTime;
numTimeScales = numel(allTimeScales);

% For each time scale, determine the index of the simulation that that
% matches the re-scaled data time. dataIDxInSim{i} is a
% (numTimeScales) X (numDataPointsPerSet(i)) matrix.
dataIDxInSim = cellfun(@(x) reshape(knnsearch(simTimes, ...
    reshape(allTimeScales .* x, [], 1)), [numTimeScales, numel(x)]).', ...
    dataTimes, 'UniformOutput', false);

% We now assemble all of the data vectors into a single 3D array. Each
% page corresponds to a different time scale to test
dataProb = repmat([dataProb{:}], [1, 1, numTimeScales]);

% We now assemble all of the simulation vectors tagged in 'dataIDxInSim'
% into a single 3D array. This array will have the same size as the
% augmented 3D array 'dataProb'
simProb = cellfun(@(x, y) reshape(x(:, y), ...
    [numStates, size(y,1), size(y,2)]), simProb, dataIDxInSim, ...
    'UniformOutput', false);
simProb = [simProb{:}];
assert(isequal(size(simProb), size(dataProb)), ...
    'Expansion of data into 3D array failed');

if useGPU
    dataProb = gpuArray(dataProb);
    simProb = gpuArray(simProb);
end

KLDErr = dataProb .* log(dataProb ./ simProb) + ...
    simProb .* log(simProb ./ dataProb);
KLDErr(isnan(KLDErr)) = 0;
KLDErr = sum(sum(KLDErr, 1), 2);

KLDErr = gather(KLDErr);
KLDErr = permute(KLDErr, [3 1 2]);

% KLDErr = inf(numTimeScales, 1);
% for i = 1:numTimeScales
% 
%     curData = [dataProb{:}];
% 
%     curSim = cell(numDataSets, 1);
%     for j = 1:numDataSets
% 
%         curDataIDx = knnsearch(simTimes, allTimeScales(i) * dataTimes{j}.');
%         assert(isequal(curDataIDx, dataIDxInSim{j}(i, :).'), ...
%             'Data ID mismatch');
%         curSim{j} = simProb{j}(:, curDataIDx);
% 
%     end
%     curSim = [curSim{:}];
% 
%     assert(isequal(size(curData), size(curSim)), 'Bad size');
% 
%     KLDErr(i) = sum(sum(curData .* log(curData ./ curSim) + ...
%         curSim .* log(curSim ./ curData), 1), 2);
% 
% end

[KLDErr, minID] = min(KLDErr);
KLDErr = KLDErr ./ sum(numDataPointsPerSet);
timeScale = allTimeScales(minID);

end