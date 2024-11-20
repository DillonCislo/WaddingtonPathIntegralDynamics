%% Test Finite Difference Solution of Fokker-Planck Equation ==============
%
%   by Dillon Cislo 2023/01/05
%
%==========================================================================

[scriptDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
addpath(genpath(fullfile(scriptDir, '..', '..', '..')));
clear scriptDir

%% Generate Analytic Results for Specified Potential ======================
clear; close all; clc;

% Generate Analytic Results -----------------------------------------------
fprintf('Performing symbolic analysis of potential... ');

syms x y
assume(x, 'real');
assume(y, 'real');

% Generate 2D potential
p3 = -0.75; p4 = 0.4; D = 0.5;
U = x.^4 + y.^4 + x.^3 - 2.*x.*y.^2 - x.^2 + p3*x + p4*y;
% U = x.^4 + y.^4 + x.^3 - x.^2 + 25* y.^2 + p3*x;
% U = x.^4/4-25*x.^3/12+9*x.^2/2+25*y.^2/2;
% U = x.^4/4-x.^2 * y / 2 - y/6;
gradU = simplify(gradient(U, [x y]));
delU = simplify(laplacian(U, [x y]));

% Generate anonymous functions from analytic statements
symU = U;
UFunc = matlabFunction(U, 'Vars', {x, y});
gradUFunc = matlabFunction(gradU.', 'Vars', {x, y});
delUFunc = matlabFunction(delU, 'Vars', {x, y});

% Find fixed points of the deterministic gradient flow numerically
fixedPts = vpasolve([gradU(1) == 0; gradU(2) == 0], [x y]);
fixedPts = double([fixedPts.x(:), fixedPts(:).y]);

% Assess stability of fixed points
J = simplify([gradient(-gradU(1), [x, y]), gradient(-gradU(2), [x y])].');
fpLambda = zeros(size(fixedPts,1), 1);
for i = 1:numel(fpLambda)
    numJ = double(vpa(subs(J, {x, y}, {fixedPts(i,1), fixedPts(i,2)})));
    fpLambda(i) = max(real(eig(numJ)));
end

% Re-order fixed points by largest (real) eigenvalue
[fpLambda, sortOrder] = sort(fpLambda, 'ascend');
fixedPts = fixedPts(sortOrder, :);

fprintf('Done\n');

% Convert to Numerical Results --------------------------------------------
fprintf('Converting to numerical results... ');

xLim = [-2 2];
yLim = [-2 2];
numPts = 100;
[X, Y] = meshgrid(linspace(xLim(1), xLim(2), numPts), ...
    linspace(yLim(1), yLim(2), numPts));
dx = X(1,2) - X(1,1);
dy = Y(2,1) - Y(1,1);

% A set of node IDs for each grid point
gridIDx = reshape((1:numel(X)).', size(X));

% A list of boundary node IDs
bdyIDx = false(size(X));
bdyIDx(1,:) = true; bdyIDx(end,:) = true;
bdyIDx(:,1) = true; bdyIDx(:,end) = true;
bdyIDx = find(bdyIDx(:));

fpU = double(vpa(subs(U, {x, y}, {fixedPts(:,1), fixedPts(:,2)})));
U = UFunc(X(:), Y(:));
gradU = gradUFunc(X(:), Y(:));
delU = delUFunc(X(:), Y(:));

fprintf('Done\n');

% View Results ------------------------------------------------------------
sinkIDx = fpLambda < 0;
saddleIDx = fpLambda >= 0;

surf(X, Y, reshape(U, size(X)), reshape(min(U, 2), size(X)));
hold on
scatter3(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), fpU(sinkIDx), ...
    'filled', 'r');
scatter3(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), fpU(saddleIDx), ...
    'filled', 'g');
hold off

xlabel('x'); ylabel('y');

axis equal
% xlim([-1.3 1.3]);
% ylim([-1.3 1.3]);
zlim([min(U(:))-1e-2, 2]);

clear i J numJ numPts saddleIDx sinkIDx sortOrder x y
% clear D p3 p4

%% Solve Fokker-Planck Equation (Finite Differences) ======================
close all; clc;

% Initial density
% P0 = reshape(exp(-UFunc(X(:), Y(:))./D), size(X));
% P0 = (X-fixedPts(2,1)).^2  + (Y-fixedPts(2,2)).^2 < (0.1.^2);
% P0 = (X-0).^2 + (Y-(-1.5)).^2 < (0.1.^2);
% P0 = P0 ./ sum(P0(:));
P0 = zeros(numel(X), 1);
P0(knnsearch([X(:), Y(:)], fixedPts(2,:))) = 1;
P0 = reshape(P0, size(X));

dirBC = [bdyIDx, zeros(size(bdyIDx))]; % Dirichlet boundary conditions
timeSpan = [0 5]; % Time span for integration
FPType = 'Forward';

[FPSol, FPO] = solveFokkerPlanck2DFD(X, Y, P0, 'U', reshape(U, size(X)), ...
    'DirichletBoundaryConditions', [], 'TimeSpan', timeSpan, ...
    'FokkerPlanckType', FPType, 'IntegratorType', 'stiff', ...
    'D', D);

%% ************************************************************************
% *************************************************************************
%      SOLVE FOKKER-PLANCK EQUATION ON FINITE DIFFERENCE GRID
% *************************************************************************
% *************************************************************************

%% Solve Fokker-Planck Equation (Path Integral) ===========================
close all; clc;

simU = U;
simX = [X(:), Y(:)];
D0 = 1;
numPoints = numel(simU);

dt = 1e-2;
scalarMetric = 0.025; % 1;
% simU = UFunc(simX(:,1), simX(:,2));
% volumeType = 'LaplaceBeltrami';
volumeType = 'GraphLaplacian';
if strcmpi(volumeType, 'LaplaceBeltrami')
    simU0 = simU;
elseif strcmpi(volumeType, 'GraphLaplacian')
    simU0 = -log(gaussianKDE(simX, simX, [], sqrt(2 * dt), false, ...
        [], [], true));
else
    error('Invalid volume element type');
end

T = computeTransitionMatrix(simX, simU, dt, ...
    'PointDiffusionCoefficient', D0, 'DiffusionCoefficient', 0.5, ...D, ...
    'ClipThreshold', 1e-12, 'StrictNormalization', true, 'useGPU', true, ...
    'PointPotential', simU0, 'VolumeElementType', volumeType, ...
    'ScalarMetric', scalarMetric);

[allProb, viewTimes] = evolveProbabilities(P0(:), T, 500, ...
    'TimeStep', dt, 'NumViewTimes', 50);

% For deep wells
pathPairIDx = [knnsearch(simX, fixedPts([2, 1], :)), ...
    knnsearch(simX, fixedPts([1, 3], :))];

% For shallow wells
% pathPairIDx = [knnsearch(simX, fixedPts([1, 2], :)), ...
%     knnsearch(simX, fixedPts([2, 3], :))];

[allPaths, allPathLengths, allPathWeights] = computeMostProbablePaths( ...
    simX, T, pathPairIDx, []);

basinLocIDx = knnsearch(simX, fixedPts(fpLambda < 0, :));
[basinProb, basinCounts, basinIDx] = computeBasins(simX, T, ...
    ones(numPoints, 1) ./ numPoints, basinLocIDx, ...
    'DistanceMethod', 'Probability');

for i = 1:numel(basinLocIDx)
    fprintf('Total Probability for (%0.2f, %0.2f) = %0.4f\n', ...
        simX(basinLocIDx(i), 1), simX(basinLocIDx(i), 2), basinProb(i));
end

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

sinkIDx = fpLambda < 0;
saddleIDx = fpLambda >= 0;

% View Potential and Unstable Manifolds -----------------------------------

crange = [min(simU), 2];
pointColors = vals2colormap(simU, 'parula', crange);

figure('Color', 'k');

subplot(1,2,1)
scatter(simX(:,1), simX(:,2), [], pointColors, 'x');

hold on

for i = 1:numel(allPaths)
    line(simX(allPaths{i}, 1), simX(allPaths{i}, 2), ...
        'Color', 'm', 'LineWidth', 2, 'Marker', 'o', ...
        'MarkerFaceColor', 'c', 'MarkerEdgeColor', 'm');
end

scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
    'filled', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
    'filled', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 2);

hold off

axis equal
xlim([-2 2]);
ylim([-2 2]);
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
set(gca, 'ZColor', 'w');
set(gca, 'Color', 'k');
set(gca, 'GridColor', 'w');
grid on
box on
cb = colorbar('Color', 'w');
cb.Label.String = ('U');
set(gca, 'Clim', crange)
title(sprintf(['Volume Element: ' volumeType '\n' ...
    'D_0 = %0.2f, D = %0.2f, dt = %0.4f, g = %0.4f'], ...
    D0, D, dt, scalarMetric), 'Color', 'w')

% View Basins of Attraction -----------------------------------------------

basinColors = distinguishable_colors(numel(basinLocIDx), ...
    [0 0 0; 1 1 1; 0 1 0; 1 0 1; 0 1 1]);

allBasinColors = basinColors(basinIDx, :);

subplot(1,2,2);
scatter(simX(:,1), simX(:,2), [], allBasinColors, 'x');

hold on

for i = 1:numel(allPaths)
    line(simX(allPaths{i}, 1), simX(allPaths{i}, 2), ...
        'Color', 'm', 'LineWidth', 2, 'Marker', 'o', ...
        'MarkerFaceColor', 'c', 'MarkerEdgeColor', 'm');
end

scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
    [], basinColors, 'filled', 'MarkerEdgeColor', 'k', ...
    'LineWidth', 2);
scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
    'filled', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 2);

hold off

axis equal
xlim([-2 2]);
ylim([-2 2]);
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
set(gca, 'ZColor', 'w');
set(gca, 'Color', 'k');
set(gca, 'GridColor', 'w');
grid on
box on
set(gca, 'Clim', [1 numel(basinLocIDx)]);
set(gca, 'Colormap', basinColors);
cb2 = colorbar('Ticks', 1:numel(basinLocIDx), 'Color', 'w');
cb2.Label.String = 'Basin #';
title('Basins of Attraction', 'Color', 'w');


clear crange pointColors sinkIDx saddleIDx cb
clear basinColors allBasinColors cb2

%% View Results -----------------------------------------------------------
close all; clc;

% numTimePoints = 50;
% timePoints = linspace(timeSpan(1), timeSpan(2), numTimePoints);
% % timePoints = linspace(timeSpan(1), 2, numTimePoints);

numTimePoints = numel(viewTimes);
timePoints = viewTimes;

imref = imref2d(size(X), [min(X(:)), max(X(:))], [min(Y(:)), max(Y(:))]);

totalDensityFD = zeros(numTimePoints, 1);
totalDensityPI = zeros(numTimePoints, 1);
fig = figure('Color', 'k', 'units', 'normalized', 'outerposition', [0 0 1 1]);

if any(strcmpi(FPType, {'Forward', 'Backward'}))
    equiDensity = reshape(exp(-U./D), size(X));
    equiDensity = equiDensity ./ sum(equiDensity(:));
else
    equiDensity = reshape(exp(-3 * U ./ (2 * D)), size(X));
end

sinkIDx = fpLambda < 0;
saddleIDx = fpLambda >= 0;

for tidx = 1:numTimePoints
    
    t = timePoints(tidx);
    curDensityFD = reshape(deval(FPSol, t), size(X));
    curDensityPI = reshape(allProb(:, tidx), size(X));
    
    totalDensityFD(tidx) = sum(curDensityFD(:));
    totalDensityPI(tidx) = sum(curDensityPI(:));
    
    densityFDRange = [min(curDensityFD(:)), max(curDensityFD(:))];
    densityImFD = vals2colormap(curDensityFD(:), 'parula', densityFDRange);
    densityImFD = cat(3, reshape(densityImFD(:,1), size(X)), ...
        reshape(densityImFD(:,2), size(X)), reshape(densityImFD(:,3), size(X)));

    densityPIRange = [min(curDensityPI(:)), max(curDensityPI(:))];
    densityImPI = vals2colormap(curDensityPI(:), 'parula', densityPIRange);
    densityImPI = cat(3, reshape(densityImPI(:,1), size(X)), ...
        reshape(densityImPI(:,2), size(X)), reshape(densityImPI(:,3), size(X)));
    
    errMin = 5e-4;
    % densityErr = abs(curDensityFD-equiDensity);
    densityErr = abs(curDensityFD - curDensityPI) ./ abs(max(curDensityFD, errMin));
    % errorRange = [0, max(densityErr(:))];
    errorRange = [0 0.1];
    if ~(errorRange(2) > errorRange(1)), errorRange = [0 1]; end
    errorIm = vals2colormap(densityErr(:), 'parula', errorRange);
    errorIm = cat(3, reshape(errorIm(:,1), size(X)), ...
        reshape(errorIm(:,2), size(X)), reshape(errorIm(:,3), size(X)));
    
    subplot(1,3,1)
    imshow(densityImFD, imref);
    hold on
    scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
        'filled', 'r');
    scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
        'filled', 'g');
    hold off
    
    if (tidx == 1), set(fig, 'outerposition', [0 0 1 1]); end
    
    titleString = sprintf('Finite Difference\nT = %0.3f, Total Density = %0.3f\n', ...
        t, totalDensityFD(tidx));
    title(titleString, 'Color', 'w');
    xlabel('x'); ylabel('y');
    
    set(gca, 'YDir', 'normal');
    set(gca, 'XColor', 'w');
    set(gca, 'YColor', 'w');
    set(gca, 'Clim', densityFDRange);
    axis equal tight
    
    colorbar('Color', 'w')

    subplot(1,3,2)
    imshow(densityImPI, imref);
    hold on
    scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
        'filled', 'r');
    scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
        'filled', 'g');
    hold off
    
    if (tidx == 1), set(fig, 'outerposition', [0 0 1 1]); end
    
    titleString = sprintf('Path Integral\nT = %0.3f, Total Density = %0.3f\n', ...
        t, totalDensityPI(tidx));
    title(titleString, 'Color', 'w');
    xlabel('x'); ylabel('y');
    
    set(gca, 'YDir', 'normal');
    set(gca, 'XColor', 'w');
    set(gca, 'YColor', 'w');
    set(gca, 'Clim', densityPIRange);
    axis equal tight
    
    colorbar('Color', 'w')
    
    subplot(1,3,3)
    imshow(errorIm, imref);
    hold on
    scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
        'filled', 'r');
    scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
        'filled', 'g');
    hold off
    
    if (tidx == 1), set(fig, 'outerposition', [0 0 1 1]); end
    
    titleString = sprintf(['|p_{FD}-p_{PI}| / max(p_{FD}, %0.0e)\n'...
        'T = %0.3f, Max Error = %0.3e\n'], ...
        errMin, t, max(densityErr(:)));
    title(titleString, 'Color', 'w');
    xlabel('x'); ylabel('y');
    
    set(gca, 'YDir', 'normal');
    set(gca, 'XColor', 'w');
    set(gca, 'YColor', 'w');
    set(gca, 'Clim', errorRange);
    axis equal tight
    
    colorbar('Color', 'w')
    
    drawnow
    pause(0.5);
    
end

%% ************************************************************************
% *************************************************************************
%      SOLVE FOKKER-PLANCK EQUATION ON RANDOM POINT SET
% *************************************************************************
% *************************************************************************

%% Generate Random Point Set ==============================================
close all; clc;

D0 = 1;
numPoints = numel(X);
pointSetType = 'potential';

% For reproducible random numbers
rng(25, 'twister');

fprintf('Generating random point set... ');

if strcmpi(pointSetType, 'uniform')

    simX = [diff(xLim) * rand(numPoints, 1) + xLim(1), ...
        diff(yLim) * rand(numPoints, 1) + yLim(1)];

elseif strcmpi(pointSetType, 'quasirandom')

    designSet = sobolset(2, 'Skip', 0, 'Leap', 0);
    designSet = scramble(designSet, 'MatousekAffineOwen');
    simX = net(designSet, numPoints);
    simX = [diff(xLim) .* simX(:,1) + xLim(1), ...
        diff(yLim) .* simX(:,2) + yLim(1)];

    clear designSet 

elseif strcmpi(pointSetType, 'potential')

    constVal = 1.1 .* max(exp(-UFunc(fixedPts(:,1), ...
        fixedPts(:,2)) ./ D0));
    simX = nan(numPoints, 2);

    count = 1;
    while count < (size(simX,1)+1)

        curX = [diff(xLim) * rand(1) + xLim(1), ...
        diff(yLim) * rand(1) + yLim(1)];
        curProb = exp(-UFunc(curX(1), curX(2)) ./ D0);

        if rand(1) < (curProb / constVal)
            simX(count, :) = curX;
            count = count+1;
        end

    end

    clear constVal count curX curProb

else

    error('Invalid point set type');

end

if exist('P0', 'var')

    initXY = P0 > 0;
    initXY = [X(initXY), Y(initXY)];
    if (size(initXY) == 1)
        simX(knnsearch(simX, initXY), :) = initXY;
    end

    clear initXY

end
        

fprintf('Done\n')

%--------------------------------------------------------------------------
% Compute Natural Neighbor Coordinates For Grid Points Relative
% to Random Points
%--------------------------------------------------------------------------

fprintf('Computing natural neighbor coordinates... ');

% Constrouct boundary polygons (squares) for each of the simulation grid
% points
polyX = repmat(X(1) + dx .* (0:(size(X,2))) - dx/2, [size(X,1)+1, 1]);
polyY = repmat(Y(1) + dy .* (0:(size(Y,1))).' - dy/2, [1, size(Y,2)+1]);

polyIDx = reshape((1:numel(polyX)).', size(polyX));
polyIDx = cat(3, polyIDx(1:size(X,1), 1:size(X,2)), ...
    polyIDx(1:size(X,1), 2:(size(X,2)+1)), ...
    polyIDx(2:(size(X,1)+1), 2:(size(X,2)+1)), ...
    polyIDx(2:(size(X,1)+1), 1:size(X,2)));
polyIDx = reshape(polyIDx(:), [numel(X), 4]).';

polyX = polyX([polyIDx; polyIDx(1,:)]);
polyY = polyY([polyIDx; polyIDx(1,:)]);

% Generate a clipped Voronoi diagram of the random points
DT = delaunayTriangulation(simX);
% bdyPoly = mean([xLim; yLim], 2) + ...
%     1.1 * diff([xLim; yLim], 1, 2) .* [-1 1; -1 1] ./ 2;
bdyPoly = [min(polyX(:)), max(polyX(:)); min(polyY(:)), max(polyY(:))];
bdyPoly = [bdyPoly(1,1) bdyPoly(2,1); bdyPoly(1,2) bdyPoly(2,1); ...
    bdyPoly(1,2) bdyPoly(2,2); bdyPoly(1,1) bdyPoly(2,2)];
[v, c] = clippedVoronoiDiagram(DT, bdyPoly);
cv = cellfun(@(x) v([x, x(1)], :), c, 'Uni', false);
careas = cellfun(@(x) area(polyshape(x(:,1), x(:,2))), cv, 'Uni', true);

gridNNI = cell(numPoints, 1);
gridNNJ = cell(numPoints, 1);
gridNNV = cell(numPoints, 1);
parfor i = 1:numPoints
    
    int_areas = polygon_intersection_area(cv{i}, polyX, polyY );

    J = find(int_areas > 0);
    I = repmat(i, [numel(J), 1]);
    V = int_areas(J);

    gridNNI{i} = I;
    gridNNJ{i} = J;
    gridNNV{i} = V;

end

gridNNCoords = sparse(cell2mat(gridNNI), cell2mat(gridNNJ), ...
    cell2mat(gridNNV), numPoints, numel(X));
assert( (max(abs(1-full(sum(gridNNCoords, 1)) ./ (dx * dy))) < 1e-5) && ...
    (max(abs(1-full(sum(gridNNCoords, 2)) ./ careas)) < 1e-5), ...
    'Natural neighbor coordinates do not appear to be normalized');

% gridNNCoords = gridNNCoords ./ sum(gridNNCoords, 1);
gridNNCoords = gridNNCoords ./ sum(gridNNCoords, 2);

fprintf('Done\n');

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

% Reconstruct grid points for visualization purposes
% polyX = repmat(X(1) + dx .* (0:(size(X,2))) - dx/2, [size(X,1)+1, 1]);
% polyY = repmat(Y(1) + dy .* (0:(size(Y,1))).' - dy/2, [1, size(Y,2)+1]);

f = max(cellfun(@numel, c, 'Uni', true));
f = cell2mat(cellfun(@(x) [x, nan(1, f-numel(x)+1)], c, 'Uni', false));

simColors = UFunc(simX(:,1), simX(:,2));
crange = [min(simColors), 2];

% simColors = full(gridNNCoords * P0(:));
% crange = [min(simColors), max(simColors)];

sinkIDx = fpLambda < 0;
saddleIDx = fpLambda >= 0;

figure('Color', 'k')

subplot(1,2,1)
hold on
patch('Faces', f, 'Vertices', v, 'EdgeColor', 'k', ...
    'FaceVertexCData', simColors, ...
    'FaceColor', 'flat', 'LineWidth', 1);
% scatter(simX(:,1), simX(:,2), 'filled', 'c');
% patch('Faces', polyIDx.', 'Vertices', [polyX(:), polyY(:)], ...
%     'EdgeColor', 'c', 'FaceColor', 'none');
scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
    'filled', 'r');
scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
    'filled', 'g');
hold off
axis equal tight
xlim(xLim);
ylim(yLim);
xlabel('x', 'Color', 'w'); ylabel('y', 'Color', 'w')
set(gca, 'Color', 'k');
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
set(gca, 'Clim', crange);
title('2D Random Point Locations', 'Color', 'w');
colorbar('Color', 'w');

subplot(1,2,2)
nBins = 20;
hist3([simX(:,1), simX(:,2)], 'Edges', {linspace(xLim(1), xLim(2), nBins), ...
    linspace(yLim(1), yLim(2), nBins)});
title('Random Point Density Distribution', 'Color', 'w');
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
set(gca, 'ZColor', 'w');
set(gca, 'Color', 'k');
set(gca, 'GridColor', 'w');
grid on
xlim(xLim);
ylim(yLim);
axis square

clear simColors crange sinkIDx saddleIDx nbins
clear polyX polyY polyIDx DT bdyPoly cv gridNNI gridNNJ gridNNV
clear I J V int_areas f pointSetType

%% Solve Fokker-Planck Equation (Path Integral) ===========================
close all; clc;

dt = 1e-2;
scalarMetric = 1;
simU = UFunc(simX(:,1), simX(:,2));
% volumeType = 'LaplaceBeltrami';
volumeType = 'GraphLaplacian';
if strcmpi(volumeType, 'LaplaceBeltrami')
    simU0 = simU;
elseif strcmpi(volumeType, 'GraphLaplacian')
    simU0 = -log(gaussianKDE(simX, simX, [], sqrt(2 * dt), false, ...
        [], [], true));
else
    error('Invalid volume element type');
end

[T, volumeElement] = computeTransitionMatrix(simX, simU, dt, ...
    'PointDiffusionCoefficient', D0, 'DiffusionCoefficient', D, ...
    'ClipThreshold', 1e-12, 'StrictNormalization', true, 'useGPU', true, ...
    'PointPotential', simU0, 'VolumeElementType', volumeType, ...
    'ScalarMetric', scalarMetric);

% simP0 = full(gridNNCoords * P0(:));
% simP0 = simP0 ./ sum(simP0);
simP0 = zeros(size(simX,1), 1);
simP0(knnsearch(simX, fixedPts(2,:))) = 1;
[allProb, viewTimes] = evolveProbabilities( ...
    simP0, T, 500, 'TimeStep', dt, 'NumViewTimes', 50);

% For deep wells
pathPairIDx = [knnsearch(simX, fixedPts([2, 1], :)), ...
    knnsearch(simX, fixedPts([1, 3], :))];

% For shallow wells
% pathPairIDx = [knnsearch(simX, fixedPts([1, 2], :)), ...
%     knnsearch(simX, fixedPts([2, 3], :))];

[allPaths, allPathLengths, allPathWeights] = computeMostProbablePaths( ...
    simX, T, pathPairIDx, []);

basinLocIDx = knnsearch(simX, fixedPts(fpLambda < 0, :));
[basinProb, basinCounts, basinIDx] = computeBasins(simX, T, ...
    ones(numPoints, 1) ./ numPoints, basinLocIDx, ...
    'BasinMethod', 'ExpectedValue');

for i = 1:numel(basinLocIDx)
    fprintf('Total Probability for (%0.2f, %0.2f) = %0.4f\n', ...
        simX(basinLocIDx(i), 1), simX(basinLocIDx(i), 2), basinProb(i));
end

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

sinkIDx = fpLambda < 0;
saddleIDx = fpLambda >= 0;

% View Potential and Unstable Manifolds -----------------------------------

crange = [min(simU), 2];
pointColors = vals2colormap(simU, 'parula', crange);

figure('Color', 'k');

subplot(1,2,1)
scatter(simX(:,1), simX(:,2), [], pointColors, 'x');

hold on

for i = 1:numel(allPaths)
    line(simX(allPaths{i}, 1), simX(allPaths{i}, 2), ...
        'Color', 'm', 'LineWidth', 2, 'Marker', 'o', ...
        'MarkerFaceColor', 'c', 'MarkerEdgeColor', 'm');
end

scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
    'filled', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
    'filled', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 2);

hold off

axis equal
xlim([-2 2]);
ylim([-2 2]);
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
set(gca, 'ZColor', 'w');
set(gca, 'Color', 'k');
set(gca, 'GridColor', 'w');
grid on
box on
cb = colorbar('Color', 'w');
cb.Label.String = ('U');
set(gca, 'Clim', crange)
title(sprintf(['Volume Element: ' volumeType '\n' ...
    'D_0 = %0.2f, D = %0.2f, dt = %0.4f, g = %0.4f'], ...
    D0, D, dt, scalarMetric), 'Color', 'w')

% View Basins of Attraction -----------------------------------------------

basinColors = distinguishable_colors(numel(basinLocIDx), ...
    [0 0 0; 1 1 1; 0 1 0; 1 0 1; 0 1 1]);

allBasinColors = basinColors(basinIDx, :);

subplot(1,2,2);
scatter(simX(:,1), simX(:,2), [], allBasinColors, 'x');

hold on

for i = 1:numel(allPaths)
    line(simX(allPaths{i}, 1), simX(allPaths{i}, 2), ...
        'Color', 'm', 'LineWidth', 2, 'Marker', 'o', ...
        'MarkerFaceColor', 'c', 'MarkerEdgeColor', 'm');
end

scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
    [], basinColors, 'filled', 'MarkerEdgeColor', 'k', ...
    'LineWidth', 2);
scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
    'filled', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 2);

hold off

axis equal
xlim([-2 2]);
ylim([-2 2]);
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
set(gca, 'ZColor', 'w');
set(gca, 'Color', 'k');
set(gca, 'GridColor', 'w');
grid on
box on
set(gca, 'Clim', [1 numel(basinLocIDx)]);
set(gca, 'Colormap', basinColors);
cb2 = colorbar('Ticks', 1:numel(basinLocIDx), 'Color', 'w');
cb2.Label.String = 'Basin #';
title('Basins of Attraction', 'Color', 'w');


clear crange pointColors sinkIDx saddleIDx cb
clear basinColors allBasinColors cb2

%% View Results -----------------------------------------------------------
close all; clc;

% numTimePoints = 50;
% timePoints = linspace(timeSpan(1), timeSpan(2), numTimePoints);
% % timePoints = linspace(timeSpan(1), 2, numTimePoints);

numTimePoints = numel(viewTimes);
timePoints = viewTimes;

f = max(cellfun(@numel, c, 'Uni', true));
f = cell2mat(cellfun(@(x) [x, nan(1, f-numel(x)+1)], c, 'Uni', false));

totalDensityFD = zeros(numTimePoints, 1);
totalDensityPI = zeros(numTimePoints, 1);
fig = figure('Color', 'k', 'units', 'normalized', 'outerposition', [0 0 1 1]);

sinkIDx = fpLambda < 0;
saddleIDx = fpLambda >= 0;

for tidx = 1:numTimePoints
    
    t = timePoints(tidx);
    curDensityPI = allProb(:, tidx) ./ volumeElement;
    curDensityFD = full(gridNNCoords * deval(FPSol, t) ./ (dx * dy)); % .* volumeElement;
    curDensityFD = curDensityFD ./ sum(curDensityFD) .* sum(curDensityPI);
    
    
    totalDensityFD(tidx) = sum(curDensityFD(:));
    totalDensityPI(tidx) = sum(curDensityPI(:));
    
    densityFDRange = [min(curDensityFD(:)), max(curDensityFD(:))];
    densityFDColors = vals2colormap(curDensityFD(:), 'parula', densityFDRange);

    densityPIRange = [min(curDensityPI(:)), max(curDensityPI(:))];
    densityPIColors = vals2colormap(curDensityPI(:), 'parula', densityPIRange);
    
    errMin = 5e-4;
    % densityErr = abs(curDensityFD-equiDensity);
    densityErr = abs(curDensityFD - curDensityPI) ./ abs(max(curDensityFD, errMin));
    % errorRange = [0, max(densityErr(:))];
    errorRange = [0 0.1];
    if ~(errorRange(2) > errorRange(1)), errorRange = [0 1]; end
    errorColors = vals2colormap(densityErr(:), 'parula', errorRange);

    subplot(1,3,1, 'replace')
    patch('Faces', f, 'Vertices', v, 'EdgeColor', 'none', ...
        'FaceVertexCData', densityFDColors , 'FaceColor', 'flat', ...
        'LineWidth', 0.5);
    hold on
    % scatter(X(:), Y(:), [], ...
    %     vals2colormap(deval(FPSol, t), 'parula', densityFDRange), ...
    %     'filled', 'MarkerEdgeColor', 'k');
    scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
        'filled', 'r');
    scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
        'filled', 'g');
    hold off
    
    if (tidx == 1), set(fig, 'outerposition', [0 0 1 1]); end
    
    titleString = sprintf('Finite Difference\nT = %0.3f, Total Density = %0.3f\n', ...
        t, totalDensityFD(tidx));
    title(titleString, 'Color', 'w');
    xlabel('x'); ylabel('y');
    
    set(gca, 'YDir', 'normal');
    set(gca, 'XColor', 'w');
    set(gca, 'YColor', 'w');
    set(gca, 'Clim', densityFDRange);
    axis equal tight
    xlim(xLim);
    ylim(yLim);
    
    colorbar('Color', 'w')

    subplot(1,3,2, 'replace')
    patch('Faces', f, 'Vertices', v, 'EdgeColor', 'none', ...
        'FaceVertexCData', densityPIColors , 'FaceColor', 'flat', ...
        'LineWidth', 0.5);
    hold on
    scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
        'filled', 'r');
    scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
        'filled', 'g');
    hold off
    
    if (tidx == 1), set(fig, 'outerposition', [0 0 1 1]); end
    
    titleString = sprintf('Path Integral\nT = %0.3f, Total Density = %0.3f\n', ...
        t, totalDensityPI(tidx));
    title(titleString, 'Color', 'w');
    xlabel('x'); ylabel('y');
    
    set(gca, 'YDir', 'normal');
    set(gca, 'XColor', 'w');
    set(gca, 'YColor', 'w');
    set(gca, 'Clim', densityPIRange);
    axis equal tight
    xlim(xLim);
    ylim(yLim);
    
    colorbar('Color', 'w')
    
    subplot(1,3,3,'replace')
    patch('Faces', f, 'Vertices', v, 'EdgeColor', 'none', ...
        'FaceVertexCData', errorColors , 'FaceColor', 'flat', ...
        'LineWidth', 0.5);
    hold on
    scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
        'filled', 'r');
    scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
        'filled', 'g');
    hold off
    
    if (tidx == 1), set(fig, 'outerposition', [0 0 1 1]); end
    
    titleString = sprintf(['|p_{FD}-p_{PI}| / max(p_{FD}, %0.0e)\n'...
        'T = %0.3f, Max Error = %0.3e\n'], ...
        errMin, t, max(densityErr(:)));
    title(titleString, 'Color', 'w');
    xlabel('x'); ylabel('y');
    
    set(gca, 'YDir', 'normal');
    set(gca, 'XColor', 'w');
    set(gca, 'YColor', 'w');
    set(gca, 'Clim', errorRange);
    axis equal tight
    xlim(xLim);
    ylim(yLim);
    
    colorbar('Color', 'w')
    
    drawnow
    % pause(0.5);
    
end
