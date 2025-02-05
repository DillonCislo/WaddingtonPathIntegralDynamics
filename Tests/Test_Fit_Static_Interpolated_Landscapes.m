%% TEST FIT STATIC INTERPOLATED LANDSCAPES ================================
%
%   This is a canned example to test the functionality of the
%   'fitStaticInterpolatedLandscapes' function with various different
%   constraints
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
UB = x.^4 + y.^4 + x.^3 - 2.*x.*y.^2 - x.^2 + p3*x + p4*y;

% Perform symbolic analysis -----------------------------------------------
gradUB = simplify(gradient(UB, [x y]));
% delUB = simplify(laplacian(UB, [x y]));

% Generate anonymous functions from analytic statements
symUB = UB;
UBFunc = matlabFunction(UB, 'Vars', {x, y});
gradUBFunc = matlabFunction(gradUB.', 'Vars', {x, y});
% delUBFunc = matlabFunction(delUB, 'Vars', {x, y});

% Find fixed points of the deterministic gradient flow numerically
fixedPtsB = vpasolve([gradUB(1) == 0; gradUB(2) == 0], [x y]);
fixedPtsB = double([fixedPtsB.x(:), fixedPtsB(:).y]);

% Assess stability of fixed points
J = simplify([gradient(-gradUB(1), [x, y]), gradient(-gradUB(2), [x y])].');
% fpEigV = zeros(size(fixedPtsB,1), 
fpLambdaB = zeros(size(fixedPtsB,1), 1);
for i = 1:numel(fpLambdaB)
    numJ = double(vpa(subs(J, {x, y}, {fixedPtsB(i,1), fixedPtsB(i,2)})));
    fpLambdaB(i) = max(real(eig(numJ)));
end

% Re-order fixed points by largest (real) eigenvalue
[fpLambdaB, sortOrder] = sort(fpLambdaB, 'ascend');
fixedPtsB = fixedPtsB(sortOrder, :);

fprintf('Done\n');

%--------------------------------------------------------------------------
% Generate Analytic Results For Modified Potential 0
%--------------------------------------------------------------------------
fprintf('Performing symbolic analysis of modified potential 0... ');

if ~exist('x', 'var'), syms x; assume(x, 'real'); end
if ~exist('y', 'var'), syms y; assume(y, 'real'); end

% Generate 2D potential ---------------------------------------------------

% p3 = -0.5; p4 = 0.1; % Shallow wells
p3 = -0.75; p4 = -0.4; % Deep wells flip Y
% p3 = 0.75; p4 = 0.4; % Deep wells flip X
% p3 = 0.1; p4 = -0.4; % Deep wells flip X and Y

% 3-well heteroclinic flip potential
U0 = x.^4 + y.^4 + x.^3 - 2.*x.*y.^2 - x.^2 + p3*x + p4*y;

% Perform symbolic analysis -----------------------------------------------
gradU0 = simplify(gradient(U0, [x y]));
% delU0 = simplify(laplacian(U0, [x y]));

% Generate anonymous functions from analytic statements
symU0 = U0;
U0Func = matlabFunction(U0, 'Vars', {x, y});
gradU0Func = matlabFunction(gradU0.', 'Vars', {x, y});
% delU0Func = matlabFunction(delU0, 'Vars', {x, y});

% Find fixed points of the deterministic gradient flow numerically
fixedPts0 = vpasolve([gradU0(1) == 0; gradU0(2) == 0], [x y]);
fixedPts0 = double([fixedPts0.x(:), fixedPts0(:).y]);

% Assess stability of fixed points
J = simplify([gradient(-gradU0(1), [x, y]), gradient(-gradU0(2), [x y])].');
% fpEigV = zeros(size(fixedPts0,1), 
fpLambda0 = zeros(size(fixedPts0,1), 1);
for i = 1:numel(fpLambda0)
    numJ = double(vpa(subs(J, {x, y}, {fixedPts0(i,1), fixedPts0(i,2)})));
    fpLambda0(i) = max(real(eig(numJ)));
end

% Re-order fixed points by largest (real) eigenvalue
[fpLambda0, sortOrder] = sort(fpLambda0, 'ascend');
fixedPts0 = fixedPts0(sortOrder, :);

fprintf('Done\n');

%--------------------------------------------------------------------------
% Generate Analytic Results For Modified Potential 1
%--------------------------------------------------------------------------
fprintf('Performing symbolic analysis of modified potential 1... ');

if ~exist('x', 'var'), syms x; assume(x, 'real'); end
if ~exist('y', 'var'), syms y; assume(y, 'real'); end

% Generate 2D potential ---------------------------------------------------

% p3 = -0.5; p4 = 0.1; % Shallow wells
% p3 = -0.75; p4 = -0.4; % Deep wells flip Y
% p3 = 0.75; p4 = 0.4; % Deep wells flip X
p3 = 0.1; p4 = -0.4; % Deep wells flip X and Y

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

gridUB = UBFunc(X(:), Y(:));
% gridGradUB = gradUBFunc(X(:), Y(:));
% gridDelUB = delUBFunc(X(:), Y(:));

fpU0 = U0Func(fixedPts0(:,1), fixedPts0(:,2));
gridU0 = U0Func(X(:), Y(:));
% gridGradU0 = gradU0Func(X(:), Y(:));
% gridDelU0 = delU0Func(X(:), Y(:));

fpU1 = U1Func(fixedPts1(:,1), fixedPts1(:,2));
gridU1 = U1Func(X(:), Y(:));
% gridGradU1 = gradU1Func(X(:), Y(:));
% gridDelU1 = delU1Func(X(:), Y(:));

fprintf('Done\n');

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

crange = [min([gridUB(:); gridU0(:); gridU1(:)]), 2];

UBColors = vals2colormap(gridUB(:), 'parula', crange); 
UBColors = cat( 3, reshape(UBColors(:,1), size(X)), ...
    reshape(UBColors(:,2), size(X)), reshape(UBColors(:,3), size(X)) );

U0Colors = vals2colormap(gridU0(:), 'parula', crange);
U0Colors = cat( 3, reshape(U0Colors(:,1), size(X)), ...
    reshape(U0Colors(:,2), size(X)), reshape(U0Colors(:,3), size(X)) );

U1Colors = vals2colormap(gridU1(:), 'parula', crange);
U1Colors = cat( 3, reshape(U1Colors(:,1), size(X)), ...
    reshape(U1Colors(:,2), size(X)), reshape(U1Colors(:,3), size(X)) );

sinkIDxB = fpLambdaB < 0;
saddleIDxB = fpLambdaB >= 0;

sinkIDx0 = fpLambda0 < 0;
saddleIDx0 = fpLambda0 >= 0;

sinkIDx1 = fpLambda1 < 0;
saddleIDx1 = fpLambda1 >= 0;

numContours = linspace(min([gridUB(:); gridU0(:); gridU1(:)]), 2, 15);

fig = figure;
clear axArray

axArray(1) = subplot(2,3,1);
surf(X, Y, reshape(gridUB, size(X)), UBColors);
hold on
scatter3(fixedPtsB(sinkIDxB, 1), fixedPtsB(sinkIDxB, 2), ...
    UBFunc(fixedPtsB(sinkIDxB, 1), fixedPtsB(sinkIDxB, 2)), ...
    'filled', 'r');
scatter3(fixedPtsB(saddleIDxB, 1), fixedPtsB(saddleIDxB, 2), ...
    UBFunc(fixedPtsB(saddleIDxB, 1), fixedPtsB(saddleIDxB, 2)), ...
    'filled', 'g');
hold off

xlabel('x'); ylabel('y');
axis equal
zlim([1.05 * min([gridUB(:); gridU0(:); gridU1(:)]), 2]);
% colorbar
set(gca, 'Clim', crange);
title('Base Potential U_B');

subplot(2,3,4)
contourf(X, Y, reshape(gridUB, size(X)), numContours);
hold on
scatter(fixedPtsB(sinkIDxB, 1), fixedPtsB(sinkIDxB, 2), 'filled', 'r');
scatter(fixedPtsB(saddleIDxB, 1), fixedPtsB(saddleIDxB, 2), 'filled', 'g');
hold off
colorbar
% set(gca, 'Clim', [min(min(gridUB(:)), min(gridUB(:))), max(gridUB)]);
set(gca, 'Clim', crange);
set(gca, 'YDir', 'normal');
axis equal tight
title('Base Potential U_B');

axArray(2) = subplot(2,3,2);
surf(X, Y, reshape(gridU0, size(X)), U0Colors);
hold on
scatter3(fixedPts0(sinkIDx0, 1), fixedPts0(sinkIDx0, 2), ...
    U0Func(fixedPts0(sinkIDx0, 1), fixedPts0(sinkIDx0, 2)), ...
    'filled', 'r');
scatter3(fixedPts0(saddleIDx0, 1), fixedPts0(saddleIDx0, 2), ...
    U0Func(fixedPts0(saddleIDx0, 1), fixedPts0(saddleIDx0, 2)), ...
    'filled', 'g');
hold off

xlabel('x'); ylabel('y');
axis equal
zlim([1.05 * min([gridUB(:); gridU0(:); gridU1(:)]), 2]);
% colorbar
set(gca, 'Clim', crange);
title('Modified Potential U_0');

subplot(2,3,5)
contourf(X, Y, reshape(gridU0, size(X)), numContours);
hold on
scatter(fixedPts0(sinkIDx0, 1), fixedPts0(sinkIDx0, 2), 'filled', 'r');
scatter(fixedPts0(saddleIDx0, 1), fixedPts0(saddleIDx0, 2), 'filled', 'g');
hold off
colorbar
% set(gca, 'Clim', [min(min(gridU0(:)), min(gridU0(:))), max(gridU0)]);
set(gca, 'Clim', crange);
set(gca, 'YDir', 'normal');
axis equal tight
title('Modified Potential U_0');

axArray(3) = subplot(2,3,3);
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
zlim([1.05 * min([gridUB(:); gridU0(:); gridU1(:)]), 2]);
% colorbar
set(gca, 'Clim', crange);
title('Modified Potential U_1');

subplot(2,3,6)
contourf(X, Y, reshape(gridU1, size(X)), numContours);
hold on
scatter(fixedPts1(sinkIDx1, 1), fixedPts1(sinkIDx1, 2), 'filled', 'r');
scatter(fixedPts1(saddleIDx1, 1), fixedPts0(saddleIDx1, 2), 'filled', 'g');
hold off
colorbar
% set(gca, 'Clim', [min(min(gridU0(:)), min(gridU0(:))), max(gridU0)]);
set(gca, 'Clim', crange);
set(gca, 'YDir', 'normal');
axis equal tight
title('Modified Potential U_1');

Link = linkprop(axArray, {'CameraUpVector', 'CameraPosition', ...
    'CameraTarget', 'XLim', 'YLim', 'ZLim'});
setappdata(fig, 'StoreTheLink', Link);

clear i J numJ numPts saddleIDxB sinkIDxB sortOrder x y
clear UImR UImG UImB UIm imref
clear fpU1 fpU0 saddleIDx0 sinkIDx0 saddleIDx1 sinkIDx1
clear D p3 p4 crange gridUB gridGradUB gridDelUB
clear gridU1 gridGradU1 gridDelU1 gridU0 gridGradU2 gridDelU2
clear UB U0 UBColors U0Colors axArray Link fig numContours

%% Generate Points by Simulating Stochastic ODE ===========================
% Points are generated by simulating drift-diffusion dynamics using the
% BASE potential
close all; clc;

% rng(88, 'twister');
rng(25, 'twister'); % For reproducible random numbers

psD0 = 1; % Diffusion coefficient
numPts = 5000; % Total number of points
numTimePts = 50000; % Number of time steps for each simulation
dtSODE = 0.01; % Simulation time step

% Initialize simulation points
simX = repmat(fixedPtsB(2,:), numPts, 1) + (0.3 * rand(numPts,2)-0.15);

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
    gradUXY = gradUBFunc(simX(activeIDx, 1), simX(activeIDx, 2));
    simX(activeIDx, :) = simX(activeIDx, :) - dtSODE * gradUXY + ...
        sqrt(2 * dtSODE * psD0) * normrnd(0, 1, [numActive 2]);
    
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

simColors = UBFunc(simX(:,1), simX(:,2));
crange = [min(simColors), 2];

sinkIDx = fpLambdaB < 0;
saddleIDx = fpLambdaB >= 0;

subplot(1,3,1)
scatter(simX(:,1), simX(:,2), [], simColors, 'x');
hold on
scatter(fixedPtsB(sinkIDx, 1), fixedPtsB(sinkIDx, 2), ...
    'filled', 'r');
scatter(fixedPtsB(saddleIDx, 1), fixedPtsB(saddleIDx, 2), ...
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
pcshow([simX(:,1), simX(:,2), UBFunc(simX(:,1), simX(:,2))], simColors);
hold on
scatter3(fixedPtsB(sinkIDx, 1), fixedPtsB(sinkIDx, 2), ...
    UBFunc(fixedPtsB(sinkIDx, 1), fixedPtsB(sinkIDx, 2)), ...
    'filled', 'r');
scatter3(fixedPtsB(saddleIDx, 1), fixedPtsB(saddleIDx, 2), ...
    UBFunc(fixedPtsB(saddleIDx, 1), fixedPtsB(saddleIDx, 2)), ...
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

%% Generate Interpolated Potential Landscape ==============================
% The final landscape is produced by linearly interpolating between the
% "boundary" landscapes. Explicitly we have:
%
%   (U_b + U_int) = (1-c) * (U_b + U_i0) / g0 + c * (U_b + U_i1) / g1
%
% for some parameter 0 <= c <= 1. The inclusion of an explicit base
% potential in the dynamical potential for the interpolated dynamics is
% made for compatibility with underlying functions. Note that we absorb the
% scalar metric for the interpolated dynamics into the dynamical potential.
close all; clc;

trueD0 = 1.25;
trueD1 = 0.9;
trueScalarMetric0 = 0.8;
trueScalarMetric1 = 1.2;
trueInterpParam = 0.25;

% % UIntFunc = @(x,y) ...
% %     (1-trueInterpParam) * (UBFunc(x,y) + U0Func(x,y)) ./ trueScalarMetric0 + ...
% %     trueInterpParam * (UBFunc(x,y) + U1Func(x,y)) ./ trueScalarMetric1;
% % 
% % gradUIntFunc = @(x,y) ...
% %     (1-trueInterpParam) * (gradUBFunc(x,y) + gradU0Func(x,y)) ./ trueScalarMetric0 + ...
% %     trueInterpParam * (gradUBFunc(x,y) + gradU1Func(x,y)) ./ trueScalarMetric1;
% 
% UIntFunc = @(x,y) ...
%     (1-trueInterpParam) * U0Func(x,y) ./ trueScalarMetric0 + ...
%     trueInterpParam * U1Func(x,y) ./ trueScalarMetric1;
% 
% gradUIntFunc = @(x,y) ...
%     (1-trueInterpParam) * gradU0Func(x,y) ./ trueScalarMetric0 + ...
%     trueInterpParam * gradU1Func(x,y) ./ trueScalarMetric1;
% 
% %--------------------------------------------------------------------------
% % View Results
% %--------------------------------------------------------------------------
% 
% gridUInt = UIntFunc(X(:), Y(:));
% crange = [min(gridUInt(:)), 2];
% 
% UIntColors = vals2colormap(gridUInt(:), 'parula', crange); 
% UIntColors = cat( 3, reshape(UIntColors(:,1), size(X)), ...
%     reshape(UIntColors(:,2), size(X)), reshape(UIntColors(:,3), size(X)) );
% 
% numContours = linspace(min(gridUInt(:)), 2, 15);
% 
% figure;
% 
% subplot(1,2,1)
% surf(X, Y, reshape(gridUInt, size(X)), UIntColors);
% xlabel('x'); ylabel('y');
% axis equal
% zlim([1.05 * min(gridUInt(:)), 2]);
% % colorbar
% set(gca, 'Clim', crange);
% title('Interpolated Potential U_{int}');
% 
% subplot(1,2,2)
% contourf(X, Y, reshape(gridUInt, size(X)), numContours);
% colorbar
% set(gca, 'Clim', crange);
% set(gca, 'YDir', 'normal');
% axis equal tight
% title('Interpolated Potential U_{int}');
% 
% clear gridUInt UIntColors fig numContours crange

%% Generate Simulated Probability Density Data ============================
% This data is simulated on the manifold extracted from the stochastic ODE
% in the previous section defined using the BASE potential, but in a
% dynamical potential field extracted from the MODIFIED/INTERPOLATED
% potentials
close all; clc;

rng(88, 'twister'); % For reproducible random numbers

%--------------------------------------------------------------------------
% Compute Unstable Manifolds From Base Potential
%--------------------------------------------------------------------------

fprintf('Computing unstable manifolds from base potential... ');

dt = 5e-3;
UB = UBFunc(simX(:,1), simX(:,2));
TUM = computeTransitionMatrix(simX, UB, dt, ...
    'VolumeElementType', 'GraphLaplacian');
isIrreducible = ~isTransitionMatrixReducible(TUM);
assert(isIrreducible, ['Point set potential transition matrix is ' ...
    'reducible. Choose a larger time step']);

fixSaddles = true;
if fixSaddles
    pairIDx = [2 5; 5 1; 1 4; 4 3];
else
    pairIDx = [2 1; 1 3];
end

pairIDx = reshape(knnsearch(simX, fixedPtsB(pairIDx(:), :)), size(pairIDx));
[allPaths, allPathLengths] = computeMostProbablePaths(simX, TUM, pairIDx);

fixInPathIDx = cellfun(@(x) [x(1); x(end)], allPaths, 'Uni', false);
fixInPathIDx = cell2mat(fixInPathIDx);
[fixPointIDx, ~, fixInPathIDx] = unique(reshape(fixInPathIDx.', [], 1));
fixInPathIDx = reshape(fixInPathIDx, [2 numel(allPaths)]).';

isSaddle = ismember(fixPointIDx, knnsearch(simX, fixedPtsB(fpLambdaB > 0, :)));

fprintf('Done\n');

clear isIrreducible TUM pairIDx

%--------------------------------------------------------------------------
% Compute Interpolated Potentials
%--------------------------------------------------------------------------

fprintf('Generating dynamical potentials for landscapes 0 and 1... ')

trueFixHeights0 = trueScalarMetric0 .* U0Func(simX(fixPointIDx, 1), ...
    simX(fixPointIDx, 2)) - UB(fixPointIDx);

trueFixHeights1 = trueScalarMetric1 .* U1Func(simX(fixPointIDx, 1), ...
    simX(fixPointIDx, 2)) - UB(fixPointIDx);

[knownU0, knownIDx0] = interpolateValuesAlongPath( ...
    trueFixHeights0(fixInPathIDx), allPaths, 'PathLengths', allPathLengths);

[knownU1, knownIDx1] = interpolateValuesAlongPath( ...
    trueFixHeights1(fixInPathIDx), allPaths, 'PathLengths', allPathLengths);

trueUI0 = interpolatePotentialKHarmonic(simX, knownU0, knownIDx0, ...
            2, 'TimeStep', dt, 'Verbose', true);

trueUI1 = interpolatePotentialKHarmonic(simX, knownU1, knownIDx1, ...
            2, 'TimeStep', dt, 'Verbose', true);

trueUD0 = (UB + trueUI0);
trueUD1 = (UB + trueUI1);

trueUDInt = (1-trueInterpParam) * trueUD0 ./ trueScalarMetric0 + ...
    trueInterpParam * trueUD1 ./ trueScalarMetric1;

trueDInt = (1-trueInterpParam) * trueD0 + trueInterpParam * trueD1;

fprintf('Done\n');

%--------------------------------------------------------------------------
% Generate Simulated Data
%--------------------------------------------------------------------------

probSigma = 0.05;
numSimTimes = 250;
% numDataSets = [1 1 1];
numDataSets = [2 3 4];
assert(all(numDataSets <= 5), ...
    'Please choose a number of data sets <= 5 for each landscape');

dataProb = cell(3,1);
dataTimes = cell(3,1);
initIDx = cell(3,1);
for expID = 1:3
    
    dataProb{expID} = cell(numDataSets(expID), 1);
    dataTimes{expID} = cell(numDataSets(expID), 1);

    if (numDataSets(expID) == 1)
        if (expID == 1)
            initIDx{expID} = 5;
        elseif (expID == 2)
            initIDx{expID} = 4;
        elseif (expID == 3)
            initIDx{expID} = 4;
        else
            error('Invalid experiment number');
        end
    else
        initIDx{expID} = ...
            randsample((1:size(fixedPtsB, 1)), numDataSets(expID));
    end

    initIDx{expID} = fixPointIDx(initIDx{expID});

end

clear expID

% Generate Simulated Data for Landscape 0 ---------------------------------

fprintf('Generating simulated data set for landscape 0... ')

[trueT, volumeElement] = computeTransitionMatrix(simX, trueUD0, dt, ...
    'PointPotential', UB, 'ScalarMetric', trueScalarMetric0, ...
    'DiffusionCoefficient', trueD0, 'PointDiffusionCoefficient', psD0, ...
    'ClipThreshold', 1e-14, 'StrictNormalization', true, ...
    'VolumeElementType', 'GraphLaplacian');

for i = 1:numDataSets(1)

    initProb = simX - repmat(simX(initIDx{1}(i), :), [size(simX,1), 1]);
    initProb = sum(initProb.^2, 2);
    initProb = exp(-initProb ./ (2 * probSigma.^2));
    initProb = initProb ./ (2 * pi * probSigma).^(size(simX,2)/2);
    initProb = volumeElement .* initProb ./ size(simX, 1);
    initProb = initProb ./ sum(initProb);

    [allProbabilities, viewTimes] = evolveProbabilities( ...
        initProb, trueT, numSimTimes-1, 'TimeStep', dt, 'UseGPU', true, ...
        'StrictNormalization', true);

    % Choose the number of data sets to retain (not including the initial
    % condition)
    numTimePoints = randi([3 5]);
    keepIDx = knnsearch(viewTimes.', ...
        linspace(0, max(viewTimes), numTimePoints).');
    dataProb{1}{i} = allProbabilities(:, keepIDx);
    dataTimes{1}{i} = viewTimes(keepIDx);

end

fprintf('Done\n')

clear trueT i initProb allProbabilities viewTimes numTimePoints keepIDx

% Generate Simulated Data for Landscape 1 ---------------------------------

fprintf('Generating simulated data set for landscape 1... ')

[trueT, ~] = computeTransitionMatrix(simX, trueUD1, dt, ...
    'PointPotential', UB, 'ScalarMetric', trueScalarMetric1, ...
    'DiffusionCoefficient', trueD1, 'PointDiffusionCoefficient', psD0, ...
    'ClipThreshold', 1e-14, 'StrictNormalization', true, ...
    'VolumeElementType', 'GraphLaplacian', ...
    'VolumeElement', volumeElement);

for i = 1:numDataSets(3)

    initProb = simX - repmat(simX(initIDx{3}(i), :), [size(simX,1), 1]);
    initProb = sum(initProb.^2, 2);
    initProb = exp(-initProb ./ (2 * probSigma.^2));
    initProb = initProb ./ (2 * pi * probSigma).^(size(simX,2)/2);
    initProb = volumeElement .* initProb ./ size(simX, 1);
    initProb = initProb ./ sum(initProb);

    [allProbabilities, viewTimes] = evolveProbabilities( ...
        initProb, trueT, numSimTimes-1, 'TimeStep', dt, 'UseGPU', true, ...
        'StrictNormalization', true);

    % Choose the number of data sets to retain (not including the initial
    % condition)
    numTimePoints = randi([3 5]);
    keepIDx = knnsearch(viewTimes.', ...
        linspace(0, max(viewTimes), numTimePoints).');
    dataProb{3}{i} = allProbabilities(:, keepIDx);
    dataTimes{3}{i} = viewTimes(keepIDx);

end

fprintf('Done\n')

clear trueT i initProb allProbabilities viewTimes numTimePoints keepIDx

% Generate Simulated Data for Interpolated Landscape ----------------------

fprintf('Generating simulated data set for interpolated landscape ... ')

[trueT, ~] = computeTransitionMatrix(simX, trueUDInt, dt, ...
    'PointPotential', UB, 'ScalarMetric', 1, ...
    'DiffusionCoefficient', trueDInt, 'PointDiffusionCoefficient', psD0, ...
    'ClipThreshold', 1e-14, 'StrictNormalization', true, ...
    'VolumeElementType', 'GraphLaplacian', ...
    'VolumeElement', volumeElement);

for i = 1:numDataSets(2)

    initProb = simX - repmat(simX(initIDx{2}(i), :), [size(simX,1), 1]);
    initProb = sum(initProb.^2, 2);
    initProb = exp(-initProb ./ (2 * probSigma.^2));
    initProb = initProb ./ (2 * pi * probSigma).^(size(simX,2)/2);
    initProb = volumeElement .* initProb ./ size(simX, 1);
    initProb = initProb ./ sum(initProb);

    [allProbabilities, viewTimes] = evolveProbabilities( ...
        initProb, trueT, numSimTimes-1, 'TimeStep', dt, 'UseGPU', true, ...
        'StrictNormalization', true);

    % Choose the number of data sets to retain (not including the initial
    % condition)
    numTimePoints = randi([3 5]);
    keepIDx = knnsearch(viewTimes.', ...
        linspace(0, max(viewTimes), numTimePoints).');
    dataProb{2}{i} = allProbabilities(:, keepIDx);
    dataTimes{2}{i} = viewTimes(keepIDx);

end

fprintf('Done\n')

clear trueT i initProb allProbabilities viewTimes numTimePoints keepIDx
clear initIDx probSigma

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------
close all; clc;

U0CRange = [min(trueUD0), min(max(trueUD0), 3)];
U1CRange = [min(trueUD1), min(max(trueUD1), 3)];
UIntCRange = [min(trueUDInt), min(max(trueUDInt), 3)];

subplot(2,2,1)
pcshow([simX, trueUD0], trueUD0, 'MarkerSize', 25);
hold on
scatter3(simX(knownIDx0, 1), simX(knownIDx0, 2), trueUD0(knownIDx0), ...
    80, vals2colormap(trueUD0(knownIDx0), 'parula', U0CRange), ...
    'filled', 'MarkerEdgeColor', 'c');
hold off
view([0 90]);
colorbar('Color', 'w');
set(gca, 'Clim', U0CRange);
axis square vis3d
camproj('orthographic')
title('Dynamical Potential U_0', 'Color', 'w')
xlabel('x'); ylabel('y');

subplot(2,2,2)
pcshow([simX, trueUD1], trueUD1, 'MarkerSize', 25);
hold on
scatter3(simX(knownIDx1, 1), simX(knownIDx1, 2), trueUD1(knownIDx1), ...
    80, vals2colormap(trueUD1(knownIDx1), 'parula', U1CRange), ...
    'filled', 'MarkerEdgeColor', 'c');
hold off
view([0 90]);
colorbar('Color', 'w');
set(gca, 'Clim', U1CRange);
axis square vis3d
camproj('orthographic')
title('Dynamical Potential U_1', 'Color', 'w')
xlabel('x'); ylabel('y');

subplot(2,2,3)
pcshow([simX, trueUDInt], trueUDInt, 'MarkerSize', 25);
view([0 90]);
colorbar('Color', 'w');
set(gca, 'Clim', UIntCRange);
axis square vis3d
camproj('orthographic')
title('Dynamical Potential U_{int}', 'Color', 'w')
xlabel('x'); ylabel('y');

clear U0CRange U1CRange UIntCRange
clear knownU0 knownIDx0 knownU1 knownIDx1

%% View Simulated Probability Time Course =================================
close all; clc;

viewExpID = 2;
viewDataSetID = 1;
viewProb = dataProb{viewExpID}{viewDataSetID};
viewDensity = viewProb ./ volumeElement;
viewTimes = dataTimes{viewExpID}{viewDataSetID};

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

clear viewDataSetID viewProb viewTimes probCRange
clear viewExpID viewDensity densCRange

%% ************************************************************************
% *************************************************************************
%                       FIT STATIC LANDSCAPE
% *************************************************************************
% *************************************************************************
close all; clc;

% Set optimization options
optOptions = {'Display', 'iter', 'UseParallel', false, ...
    'FiniteDifferenceType', 'forward', ...
    'PlotFcn', {'optimplotx', 'optimplotfval'}, ...
    'MaxIterations', 1000, 'MaxFunctionEvaluations', 15000};

% Generate an initial guess that satisfies the constraints (just a
% potential height sum constraint here)
% initGuess = 5 * rand(numel(isSaddle), 2) - 2.5;
% initGuess(:,1) = initGuess(:,1)-sum(initGuess(:,1))/numel(initGuess(:,1));
% initGuess(:,2) = initGuess(:,2)-sum(initGuess(:,2))/numel(initGuess(:,2));
% initGuess = [initGuess(:,1); 1; 1; initGuess(:,2); 1; 1; 0.5];
% constFixHeights = [];

% Generic initial guess (no constraints)
initGuess = [zeros(numel(trueFixHeights1), 1); 1; 1; ...
    zeros(numel(trueFixHeights1), 1); 1; 1; 0.5];
% constFixHeights = nan(size(trueFixHeights));
% constFixHeights(end) = 0; 
constFixHeights = [];

[optErr, fixHeights0, D0, scalarMetric0, fixHeights1, D1, ...
    scalarMetric1, interpParam, timeScales, optTimes, optOutput] = ...
    fitStaticInterpolatedLandscapes( simX, dataProb, dataTimes, dt, allPaths, ...
    'InitialGuess', initGuess, 'InitialConditions', {}, ...
    'NumSimTimes', numSimTimes, 'IsSaddle', isSaddle, ...
    'EnforceSaddles', false, 'ConstHeightSum', [0 0], ...
    'SimTimeHandling', 'none', 'OptimizationOptions', optOptions, ...
    'ConstFixedHeights', constFixHeights, 'ConstScalarMetric', [], ...
    'ConstDiffusionCoefficient', [], 'PointDiffusionCoefficient', psD0, ...
    'EnforcePositiveMetric', true, 'EnforceUnitInterpParam', true, ...
    'EnforcePositiveDiffusion', true, ...
    'PointPotential', UB, 'BasePotential', UB, ...
    'Laplacian', [], 'MassMatrix', [], 'PathLengths', allPathLengths, ...
    'VolumeElementType', 'GraphLaplacian', ...
    'UpperBounds', [inf(numel(isSaddle), 1); 2; 2; inf(numel(isSaddle), 1); 2; 2; 1], ...
    'UseGPU', true, 'Verbose', true, 'ErrorType', 'symKLD' );

