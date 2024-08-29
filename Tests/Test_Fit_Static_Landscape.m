%% TEST FIT STATIC LANDSCAPE ==============================================
%
%   This is a canned example to test the functionality of the
%   'fitStaticLandscape' function with various differenct constraints
%
%   by Dillon Cislo 2024/04/01
%==========================================================================

%% Generate Analytic Results for Specified Potential ======================
clear; close all; clc;

%--------------------------------------------------------------------------
% Generate Analytic Results For Base Potential
%--------------------------------------------------------------------------
fprintf('Performing symbolic analysis of base potential... ');

syms x y
assume(x, 'real');
assume(y, 'real');

% Generate 2D potential ---------------------------------------------------

p3 = -0.75; p4 = 0.4; % Deep wells 
% p3 = -0.5; p4 = 0.1; % Shallow wells

% 3-well heteroclinic flip potential
U1 = x.^4 + y.^4 + x.^3 - 2.*x.*y.^2 - x.^2 + p3*x + p4*y;

% Perform symbolic analysis -----------------------------------------------
gradU1 = simplify(gradient(U1, [x y]));
% delU1 = simplify(laplacian(U1, [x y]));

% Generate anonymous functions from analytic statements
symU1 = U1;
U1Func = matlabFunction(U1, 'Vars', {x, y});
gradU1Func = matlabFunction(gradU1.', 'Vars', {x, y});
% delU1Func = matlabFunction(delU1, 'Vars', {x, y});

% Find fixed points of the deterministic gradient flow numerically
fixedPts1 = vpasolve([gradU1(1) == 0; gradU1(2) == 0], [x y]);
fixedPts1 = double([fixedPts1.x(:), fixedPts1(:).y]);

% Assess stability of fixed points
J = simplify([gradient(-gradU1(1), [x, y]), gradient(-gradU1(2), [x y])].');
% fpEigV = zeros(size(fixedPts1,1), 
fpLambda1 = zeros(size(fixedPts1,1), 1);
for i = 1:numel(fpLambda1)
    numJ = double(vpa(subs(J, {x, y}, {fixedPts1(i,1), fixedPts1(i,2)})));
    fpLambda1(i) = max(real(eig(numJ)));
end

% Re-order fixed points by largest (real) eigenvalue
[fpLambda1, sortOrder] = sort(fpLambda1, 'ascend');
fixedPts1 = fixedPts1(sortOrder, :);

fprintf('Done\n');

%--------------------------------------------------------------------------
% Generate Analytic Results For Modified Potential
%--------------------------------------------------------------------------
fprintf('Performing symbolic analysis of modified potential... ');

if ~exist('x', 'var'), syms x; assume(x, 'real'); end
if ~exist('y', 'var'), syms y; assume(y, 'real'); end

% Generate 2D potential ---------------------------------------------------

% p3 = -0.5; p4 = 0.1; % Shallow wells
p3 = -0.75; p4 = -0.4; % Deep wells flip Y
% p3 = 0.75; p4 = 0.4; % Deep wells flip X
% p3 = 0.1; p4 = -0.4; % Deep wells flip X and Y

% 3-well heteroclinic flip potential
U2 = x.^4 + y.^4 + x.^3 - 2.*x.*y.^2 - x.^2 + p3*x + p4*y;

% Perform symbolic analysis -----------------------------------------------
gradU2 = simplify(gradient(U2, [x y]));
% delU2 = simplify(laplacian(U2, [x y]));

% Generate anonymous functions from analytic statements
symU2 = U2;
U2Func = matlabFunction(U2, 'Vars', {x, y});
gradU2Func = matlabFunction(gradU2.', 'Vars', {x, y});
% delU2Func = matlabFunction(delU2, 'Vars', {x, y});

% Find fixed points of the deterministic gradient flow numerically
fixedPts2 = vpasolve([gradU2(1) == 0; gradU2(2) == 0], [x y]);
fixedPts2 = double([fixedPts2.x(:), fixedPts2(:).y]);

% Assess stability of fixed points
J = simplify([gradient(-gradU2(1), [x, y]), gradient(-gradU2(2), [x y])].');
% fpEigV = zeros(size(fixedPts22,1), 
fpLambda2 = zeros(size(fixedPts2,1), 1);
for i = 1:numel(fpLambda2)
    numJ = double(vpa(subs(J, {x, y}, {fixedPts2(i,1), fixedPts2(i,2)})));
    fpLambda2(i) = max(real(eig(numJ)));
end

% Re-order fixed points by largest (real) eigenvalue
[fpLambda2, sortOrder] = sort(fpLambda2, 'ascend');
fixedPts2 = fixedPts2(sortOrder, :);

fprintf('Done\n');

%--------------------------------------------------------------------------
% Convert to Numerical Results
%--------------------------------------------------------------------------
fprintf('Converting to numerical results... ');

xLim = [-2 2];
yLim = [-2 2];
numPts = 100;
[X, Y] = meshgrid(linspace(xLim(1), xLim(2), numPts), ...
    linspace(yLim(1), yLim(2), numPts));

% A set of node IDs for each grid point
gridIDx = reshape((1:numel(X)).', size(X));

% A list of boundary node IDs
bdyIDx = false(size(X));
bdyIDx(1,:) = true; bdyIDx(end,:) = true;
bdyIDx(:,1) = true; bdyIDx(:,end) = true;
bdyIDx = find(bdyIDx(:));

gridU1 = U1Func(X(:), Y(:));
% gridGradU1 = gradU1Func(X(:), Y(:));
% gridDelU1 = delU1Func(X(:), Y(:));

fpU2 = U2Func(fixedPts2(:,1), fixedPts2(:,2));
gridU2 = U2Func(X(:), Y(:));
% gridGradU2 = gradU2Func(X(:), Y(:));
% gridDelU2 = delU2Func(X(:), Y(:));

fprintf('Done\n');

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

crange = [min(min(gridU1(:)), min(gridU2(:))), 2];

U1Colors = vals2colormap(gridU1(:), 'parula', crange); 
U1Colors = cat( 3, reshape(U1Colors(:,1), size(X)), ...
    reshape(U1Colors(:,2), size(X)), reshape(U1Colors(:,3), size(X)) );

U2Colors = vals2colormap(gridU2(:), 'parula', crange);
U2Colors = cat( 3, reshape(U2Colors(:,1), size(X)), ...
    reshape(U2Colors(:,2), size(X)), reshape(U2Colors(:,3), size(X)) );

sinkIDx1 = fpLambda1 < 0;
saddleIDx1 = fpLambda1 >= 0;

sinkIDx2 = fpLambda2 < 0;
saddleIDx2 = fpLambda2 >= 0;

numContours = linspace(min(min(gridU1(:)), min(gridU2(:))), 2, 15);

fig = figure;
clear axArray

axArray(1) = subplot(2,2,1);
surf(X, Y, reshape(gridU1, size(X)), U1Colors);
hold on
scatter3(fixedPts1(sinkIDx1, 1), fixedPts1(sinkIDx1, 2), ...
    U1Func(fixedPts1(sinkIDx1, 1), fixedPts1(sinkIDx1, 2)), ...
    'filled', 'r');
scatter3(fixedPts1(saddleIDx1, 1), fixedPts1(saddleIDx1, 2), ...
    U1Func(fixedPts1(saddleIDx1, 1), fixedPts1(saddleIDx1, 2)), ...
    'filled', 'g');
hold off

xlabel('x'); ylabel('y');
axis equal
zlim([1.05 * min(min(gridU1(:)), min(gridU2(:))), 2]);
% colorbar
set(gca, 'Clim', crange);
title('Initial Potential U_1');

subplot(2,2,2)
contourf(X, Y, reshape(gridU1, size(X)), numContours);
hold on
scatter(fixedPts1(sinkIDx1, 1), fixedPts1(sinkIDx1, 2), 'filled', 'r');
scatter(fixedPts1(saddleIDx1, 1), fixedPts1(saddleIDx1, 2), 'filled', 'g');
hold off
colorbar
% set(gca, 'Clim', [min(min(gridU1(:)), min(gridU2(:))), max(gridU1)]);
set(gca, 'Clim', crange);
set(gca, 'YDir', 'normal');
axis equal tight
title('Initial Potential U_1');

axArray(2) = subplot(2,2,3);
surf(X, Y, reshape(gridU2, size(X)), U2Colors);
hold on
scatter3(fixedPts2(sinkIDx2, 1), fixedPts2(sinkIDx2, 2), ...
    U2Func(fixedPts2(sinkIDx2, 1), fixedPts2(sinkIDx2, 2)), ...
    'filled', 'r');
scatter3(fixedPts2(saddleIDx2, 1), fixedPts2(saddleIDx2, 2), ...
    U2Func(fixedPts2(saddleIDx2, 1), fixedPts2(saddleIDx2, 2)), ...
    'filled', 'g');
hold off

xlabel('x'); ylabel('y');
axis equal
zlim([1.05 * min(min(gridU1(:)), min(gridU2(:))), 2]);
% colorbar
set(gca, 'Clim', crange);
title('Modified Potential U_2');

subplot(2,2,4)
contourf(X, Y, reshape(gridU2, size(X)), numContours);
hold on
scatter(fixedPts2(sinkIDx2, 1), fixedPts2(sinkIDx2, 2), 'filled', 'r');
scatter(fixedPts2(saddleIDx2, 1), fixedPts2(saddleIDx2, 2), 'filled', 'g');
hold off
colorbar
% set(gca, 'Clim', [min(min(gridU1(:)), min(gridU2(:))), max(gridU2)]);
set(gca, 'Clim', crange);
set(gca, 'YDir', 'normal');
axis equal tight
title('Modified Potential U_2');

Link = linkprop(axArray, {'CameraUpVector', 'CameraPosition', ...
    'CameraTarget', 'XLim', 'YLim', 'ZLim'});
setappdata(fig, 'StoreTheLink', Link);

clear i J numJ numPts saddleIDx1 sinkIDx1 sortOrder x y
clear UImR UImG UImB UIm imref
clear fpU1 fpU2 saddleIDx2 sinkIDx2
clear D p3 p4 crange
clear gridU1 gridGradU1 gridDelU1 gridU2 gridGradU2 gridDelU2
clear U1 U2 U1Colors U2Colors axArray Link fig numContours

%% Generate Points by Simulating Stochastic ODE ===========================
% Points are generated by simulating drift-diffusion dynamics using the
% BASE potential
close all; clc;

% rng(88, 'twister');
rng(25, 'twister'); % For reproducible random numbers

D0 = 1; % Diffusion coefficient
numPts = 5000; % Total number of points
numTimePts = 50000; % Number of time steps for each simulation
dtSODE = 0.01; % Simulation time step

% Initialize simulation points
simX = repmat(fixedPts1(2,:), numPts, 1) + (0.3 * rand(numPts,2)-0.15);

% The time at which each simulation will be terminated
stopTimes = randsample(numTimePts, numPts, true);

% List of simulations that are still active
activeIDx = stopTimes > 1;
stopTimes(~activeIDx) = [];
activeIDx = find(activeIDx);

t = 2;

fprintf('Generating points via stochastic simulation:\n');
% profile on
while (~isempty(activeIDx) && (t <= numTimePts))
    
    numActive = numel(activeIDx);
    progressbar(numPts-numActive, numPts)
    
    % Update active points
    gradUXY = gradU1Func(simX(activeIDx, 1), simX(activeIDx, 2));
    simX(activeIDx, :) = simX(activeIDx, :) - dtSODE * gradUXY + ...
        sqrt(2 * dtSODE * D0) * normrnd(0, 1, [numActive 2]);
    
    % Determine which points are still active
    curStop = stopTimes == t;
    activeIDx(curStop) = [];
    stopTimes(curStop) = [];
    t = t+1;
    
end
% profile viewer

clear activeIDx stopTimes numActive t gradUXY curStop

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

simColors = U1Func(simX(:,1), simX(:,2));
crange = [min(simColors), 2];

sinkIDx = fpLambda1 < 0;
saddleIDx = fpLambda1 >= 0;

subplot(1,3,1)
scatter(simX(:,1), simX(:,2), [], simColors, 'x');
hold on
scatter(fixedPts1(sinkIDx, 1), fixedPts1(sinkIDx, 2), ...
    'filled', 'r');
scatter(fixedPts1(saddleIDx, 1), fixedPts1(saddleIDx, 2), ...
    'filled', 'g');
hold off
axis equal tight
xlim([-2 2]);
ylim([-2 2]);
xlabel('x', 'Color', 'w'); ylabel('y', 'Color', 'w')
set(gca, 'Color', 'k');
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
set(gca, 'Clim', crange);
title('2D Simulated Point Locations', 'Color', 'w');

subplot(1,3,2)
pcshow([simX(:,1), simX(:,2), U1Func(simX(:,1), simX(:,2))], simColors);
hold on
scatter3(fixedPts1(sinkIDx, 1), fixedPts1(sinkIDx, 2), ...
    U1Func(fixedPts1(sinkIDx, 1), fixedPts1(sinkIDx, 2)), ...
    'filled', 'r');
scatter3(fixedPts1(saddleIDx, 1), fixedPts1(saddleIDx, 2), ...
    U1Func(fixedPts1(saddleIDx, 1), fixedPts1(saddleIDx, 2)), ...
    'filled', 'g');
hold off
xlim(xLim);
ylim(yLim);
xlabel('x'); ylabel('y'), zlabel('U(x,y)');
set(gca, 'Color', 'k');
set(gca, 'Clim', crange);
title('3D Simulated Point Locations');
axis equal tight
camproj('orthographic');

subplot(1,3,3)
nBins = 20;
hist3([simX(:,1), simX(:,2)], 'Edges', {linspace(xLim(1), xLim(2), nBins), ...
    linspace(yLim(1), yLim(2), nBins)});
title('Simulated Point Density Distribution', 'Color', 'w');
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
set(gca, 'ZColor', 'w');
set(gca, 'Color', 'k');
set(gca, 'GridColor', 'w');
grid on

clear timeColors nBins sinkIDx saddleIDx
clear numInitPts numTimePts simColors t crange
clear dtSODE

%% Generate Simulated Probability Density Data ============================
% This data is simulated on the manifold extracted from the stochastic ODE
% in the previous section defined using the BASE potential, but in a
% dynamical potential field extracted from the MODIFIED potential
close all; clc;

%--------------------------------------------------------------------------
% Compute Unstable Manifolds From Base Potential
%--------------------------------------------------------------------------

fprintf('Computing unstable manifolds from base potential... ');

D = 1;
dt = 5e-3;
U0 = U1Func(simX(:,1), simX(:,2));
TUM = computeTransitionMatrix(simX, U0, dt, ...
    'VolumeElementType', 'LaplaceBeltrami');
isIrreducible = ~isTransitionMatrixReducible(TUM);
assert(isIrreducible, ['Point set potential transition matrix is ' ...
    'reducible. Choose a larger time step']);

fixSaddles = true;
if fixSaddles

    pairIDx = [2 5; 5 1; 1 4; 4 3];

else

    pairIDx = [2 1; 1 3];

end

pairIDx = reshape(knnsearch(simX, fixedPts1(pairIDx(:), :)), size(pairIDx));
[allPaths, allPathLengths] = computeMostProbablePaths(simX, TUM, pairIDx);

fixInPathIDx = cellfun(@(x) [x(1); x(end)], allPaths, 'Uni', false);
fixInPathIDx = cell2mat(fixInPathIDx);
[fixPointIDx, ~, fixInPathIDx] = unique(reshape(fixInPathIDx.', [], 1));
fixInPathIDx = reshape(fixInPathIDx, [2 numel(allPaths)]).';

isSaddle = ismember(fixPointIDx, knnsearch(simX, fixedPts1(fpLambda1 > 0, :)));

fprintf('Done\n');

clear isIrreducible TUM

%--------------------------------------------------------------------------
% Compute Interpolated Potential
%--------------------------------------------------------------------------

trueScalarMetric = 1;
trueFixHeights = trueScalarMetric .* U2Func(simX(fixPointIDx, 1), ...
    simX(fixPointIDx, 2)) - U0(fixPointIDx);

fprintf('Interpolating values along path... ')
[knownU, knownIDx] = interpolateValuesAlongPath( ...
    trueFixHeights(fixInPathIDx), allPaths, 'PathLengths', allPathLengths);
fprintf('Done\n');

trueUI = interpolatePotentialKHarmonic(simX, knownU, knownIDx, ...
            2, 'TimeStep', dt, 'Verbose', true);

trueU = (U0 + trueUI);

%--------------------------------------------------------------------------
% Generate Simulated Data
%--------------------------------------------------------------------------

probSigma = 0.05;
numSimTimes = 500;
numDataSets = 3;
assert(numDataSets <= 5, 'Please choose a number of data sets <= 5');

fprintf('Generating simulated data set... ')

trueT = computeTransitionMatrix(simX, trueU, dt, ...
    'PointPotential', U0, 'ScalarMetric', trueScalarMetric, ...
    'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0, ...
    'ClipThreshold', 1e-14, 'StrictNormalization', true, ...
    'VolumeElementType', 'LaplaceBeltrami');

dataProb = cell(numDataSets, 1);
dataTimes = cell(numDataSets, 1);
if (numDataSets == 1)
    initIDx = 5;
else
    initIDx = randsample((1:size(fixedPts1, 1)), numDataSets);
end
initIDx = fixPointIDx(initIDx);

for i = 1:numDataSets

    initProb = simX - repmat(simX(initIDx(i), :), [size(simX,1), 1]);
    initProb = sum(initProb.^2, 2);
    initProb = exp(-initProb ./ (2 * probSigma.^2));
    initProb = initProb ./ (2 * pi * probSigma).^(size(simX,2)/2);
    initProb = exp(U0 ./ D0) .* initProb ./ size(simX, 1);
    initProb = initProb ./ sum(initProb);

    [allProbabilities, viewTimes] = evolveProbabilities( ...
        initProb, trueT, numSimTimes-1, 'TimeStep', dt, 'UseGPU', true, ...
        'StrictNormalization', true);

    % Choose the number of data sets to retain (not including the initial
    % condition)
    numTimePoints = 5; % randi([3 5]);
    keepIDx = knnsearch(viewTimes.', ...
        linspace(0, max(viewTimes), numTimePoints).');
    dataProb{i} = allProbabilities(:, keepIDx);
    dataTimes{i} = viewTimes(keepIDx);

end

clear initIDx probSigma initProb allProbabilities viewTimes
clear numTimePoints keepIDx i

fprintf('Done\n')

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------
close all; clc;

baseCRange = [min(U0), 3];
interpCRange = [min(trueUI), max(trueUI)];
dynCRange = [min(trueU), 3];

subplot(2,2,1)
pcshow([simX, U0], U0, 'MarkerSize', 25);
hold on
scatter3(simX(knownIDx, 1), simX(knownIDx, 2), U0(knownIDx), ...
    80, vals2colormap(U0(knownIDx), 'parula', baseCRange), ...
    'filled', 'MarkerEdgeColor', 'c');
hold off
view([0 90]);
colorbar('Color', 'w');
set(gca, 'Clim', baseCRange);
axis square vis3d
camproj('orthographic')
title('Base Potential U_0', 'Color', 'w')
xlabel('x'); ylabel('y');

subplot(2,2,2)
pcshow([simX, trueUI], trueUI, 'MarkerSize', 25);
hold on
scatter3(simX(knownIDx, 1), simX(knownIDx, 2), trueUI(knownIDx), ...
    80, vals2colormap(trueUI(knownIDx), 'parula', interpCRange), ...
    'filled', 'MarkerEdgeColor', 'c');
hold off
view([0 90]);
colorbar('Color', 'w');
set(gca, 'Clim', interpCRange);
axis square vis3d
camproj('orthographic')
title('Interpolated Potential U_I', 'Color', 'w')
xlabel('x'); ylabel('y');

subplot(2,2,3)
pcshow([simX, trueU], trueU, 'MarkerSize', 25);
hold on
scatter3(simX(knownIDx, 1), simX(knownIDx, 2), trueU(knownIDx), ...
    80, vals2colormap(trueU(knownIDx), 'parula', dynCRange), ...
    'filled', 'MarkerEdgeColor', 'c');
hold off
view([0 90]);
colorbar('Color', 'w');
set(gca, 'Clim', dynCRange);
axis square vis3d
camproj('orthographic')
title('Dynamical Potential U = (U_0 + U_I)/g', 'Color', 'w')
xlabel('x'); ylabel('y');

clear baseCRange interpCRange dynCRange

%% View Simulated Probability Time Course =================================
close all; clc;

viewID = 3;
viewProb = dataProb{viewID};
viewDensity = exp(-U0 ./ D0) .* viewProb;
viewTimes = dataTimes{viewID};

figure('Color', 'k', 'Units', 'normalized', 'OuterPosition', ...
    [0 0 1 1]);

for tidx = 1:size(viewProb, 2)

    subplot(1, 2, 1, 'replace');

    probCRange = prctile(viewProb(:, tidx), [0 98]);

    pcshow([simX, zeros(size(simX,1), 1)], viewProb(:, tidx));
    view([0 90]);
    colorbar('Color', 'w');
    set(gca, 'Clim', probCRange);
    axis square vis3d
    camproj('orthographic')
    title(sprintf(' Probability T = %0.5f', viewTimes(tidx)), ...
        'Color', 'w');
    xlabel('x'); ylabel('y');

    subplot(1, 2, 2, 'replace');

    densCRange = prctile(viewDensity(:, tidx), [0 98]);

    pcshow([simX, zeros(size(simX,1), 1)], viewDensity(:, tidx));
    view([0 90]);
    colorbar('Color', 'w');
    set(gca, 'Clim', densCRange);
    axis square vis3d
    camproj('orthographic')
    title(sprintf('Density T = %0.5f', viewTimes(tidx)), ...
        'Color', 'w');
    xlabel('x'); ylabel('y');

    drawnow

    pause(1);

end


clear viewID viewProb viewTimes probCRange viewDensity densCRange

%% ************************************************************************
% *************************************************************************
%                       FIT STATIC LANDSCAPE
% *************************************************************************
% *************************************************************************
close all; clc;

% Set optimization options
optOptions = {'Display', 'iter', 'FiniteDifferenceType', 'forward', ...
    'UseParallel', false, 'PlotFcn', {'optimplotx', 'optimplotfval'}};

% Generate an initial guess that satisfies the constraints (just a
% potential height sum constraint here)
% initGuess = 0.25 * rand(numel(isSaddle), 1) + 0.5;
% initGuess = sum(trueFixHeights) * (initGuess ./ sum(initGuess));
% initGuess = [initGuess; 1];

% Generic initial guess (no constraints)
initGuess = [zeros(numel(trueFixHeights), 1); 1];
% constFixHeights = nan(size(trueFixHeights));
% constFixHeights(end) = 0; 
constFixHeights = [];

[optErr, fixHeights, scalarMetric, timeScale, fitTimes, optOutput] = ...
    fitStaticLandscape( simX, dataProb, dataTimes, dt, allPaths, ...
    'InitialGuess', initGuess, 'InitialConditions', {}, ...
    'NumSimTimes', numSimTimes, 'IsSaddle', isSaddle, ...
    'EnforceSaddles', false, 'ConstHeightSum', 0, ...
    'SimTimeHandling', 'none', 'OptimizationOptions', optOptions, ...
    'ConstFixedHeights', constFixHeights, 'ConstScalarMetric', [], ...
    'EnforcePositiveMetric', true, ...
    'PointPotential', U0, 'BasePotential', U0, ...
    'Laplacian', [], 'MassMatrix', [], 'PathLengths', allPathLengths, ...
    'VolumeElementType', 'LaplaceBeltrami', ...
    'UseGPU', true, 'Verbose', true, 'ErrorType', 'symKLD' );

