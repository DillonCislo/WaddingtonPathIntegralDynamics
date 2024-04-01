% TEST TIME SCALE FIT =====================================================
% This script tests the functionality of the 'fitProbSimTimeScale' function
clear; close all; clc;

numSimTimes = 500;
dt = 1e-3;
numDataSets = 5;
maxNumDataPointsPerSet = 10; % <-- This is typically ~O(1)
trueTimeScale = 10 * rand(1) + 1;
useGPU = true;

%--------------------------------------------------------------------------
% GENERATE SIMULATED DATA
%--------------------------------------------------------------------------
% We just simulate some simple scalar diffusion on the sphere. It doesn't
% really matter how we get this data - we just need some process with a
% reasonable and tunable time scale

fprintf('Generating simulated data... ');

[V, F] = subdivided_sphere(4);
numStates = size(V,1);

[L, iF, il] = intrinsic_delaunay_cotmatrix(V,F);
M = massmatrix_intrinsic(il, iF, numStates, 'voronoi');
clear iF iL

simProb = cell(numDataSets, 1);
parfor i = 1:numDataSets

    curSimProb = zeros(numStates, numSimTimes);
    idx = randi(10, 1);
    idx = randsample(1:numStates, idx);
    curSimProb(idx, 1) = 1 / numel(idx);

    for j = 2:numSimTimes
        curSimProb(:, j) = (M - dt * L) \ (M * curSimProb(:, j-1));
    end

    curSimProb = curSimProb ./ sum(curSimProb, 1);
    simProb{i} = curSimProb;

end
simTimes = (0:(numSimTimes-1)) * dt;

clear idx curSimProb i j

fprintf('Done\n');

%--------------------------------------------------------------------------
% FIT TIME SCALE
%--------------------------------------------------------------------------

fprintf('Fitting time scale: ');

% Extract the 'data' probability vectors - we just sample these directly
% from the simulated probability vectors
dataProb = cell(numDataSets, 1);
dataTimes = cell(numDataSets, 1);
trueDataIDx = cell(numDataSets, 1);
for i = 1:numDataSets

    idx = randi(maxNumDataPointsPerSet, 1);
    idx = sort(randsample(1:numSimTimes, idx));

    trueDataIDx{i} = idx;
    dataProb{i} = simProb{i}(:, idx);
    dataTimes{i} = simTimes(idx) ./ trueTimeScale;

    clear idx

end

tic;
[timeScale, KLDErr] = fitProbSimTimeScale(dataProb, dataTimes, ...
    simProb, dt, 'UseGPU', true);
toc;

% The fit time scale may have a quantitative mismatch with the true time
% scale, but it should still return the exact time point IDs
dataIDx = cellfun(@(x) timeScale .* x, dataTimes, 'Uni', false);
dataIDx = cellfun(@(x) knnsearch(simTimes(:), x(:)).', dataIDx, 'Uni', false);

if isequal(dataIDx, trueDataIDx)
    disp('Method successfully fit time scale!');
else
    error('Method FAILED to fit time scale!');
end

%% View Simulated Data ====================================================
close all; clc;

viewID = 1;
viewProb = simProb{viewID};

fig = figure('Color', 'w', 'Units', 'normalized', ...
    'OuterPosition', [0.5 0 0.5 1]);

for tidx = 1:numSimTimes

    subplot(1, 1, 1, 'replace')

    trisurf(triangulation(F, V), 'FaceVertexCData', viewProb(:, tidx), ...
        'FaceColor', 'interp', 'EdgeColor', 'none');

    axis equal
    colorbar
    set(gca, 'Clim', [0 max(viewProb(:, tidx))]);

    title(sprintf('T = %0.3f/%0.3f', ...
        simTimes(tidx), simTimes(numSimTimes)));

    drawnow

    pause(0.05);


end

clear viewID viewProb fig tidx tIDx



