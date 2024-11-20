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

%% Solve Fokker-Planck Equation ===========================================
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

%% View Results -----------------------------------------------------------
close all; clc;

numTimePoints = 50;
% timePoints = linspace(timeSpan(1), timeSpan(2), numTimePoints);
timePoints = linspace(timeSpan(1), 2, numTimePoints);

imref = imref2d(size(X), [min(X(:)), max(X(:))], [min(Y(:)), max(Y(:))]);

totalDensity = zeros(numTimePoints, 1);
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
    curDensity = reshape(deval(FPSol, t), size(X));
    
    totalDensity(tidx) = sum(curDensity(:));
    
    densityRange = [min(curDensity(:)), max(curDensity(:))];
    densityIm = vals2colormap(curDensity(:), 'parula', densityRange);
    densityIm = cat(3, reshape(densityIm(:,1), size(X)), ...
        reshape(densityIm(:,2), size(X)), reshape(densityIm(:,3), size(X)));
    
    densityErr = abs(curDensity-equiDensity);
    errorRange = [0, max(densityErr(:))];
    errorIm = vals2colormap(densityErr(:), 'parula', errorRange);
    errorIm = cat(3, reshape(errorIm(:,1), size(X)), ...
        reshape(errorIm(:,2), size(X)), reshape(errorIm(:,3), size(X)));
    
    subplot(1,2,1)
    imshow(densityIm, imref);
    hold on
    scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
        'filled', 'r');
    scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
        'filled', 'g');
    hold off
    
    if (tidx == 1), set(fig, 'outerposition', [0 0 1 1]); end
    
    titleString = sprintf('T = %0.3f, Total Density = %0.3f\n', ...
        t, totalDensity(tidx));
    title(titleString, 'Color', 'w');
    xlabel('x'); ylabel('y');
    
    set(gca, 'YDir', 'normal');
    set(gca, 'XColor', 'w');
    set(gca, 'YColor', 'w');
    set(gca, 'Clim', densityRange);
    axis equal tight
    
    colorbar('Color', 'w')
    
    subplot(1,2,2)
    imshow(errorIm, imref);
    hold on
    scatter(fixedPts(sinkIDx, 1), fixedPts(sinkIDx, 2), ...
        'filled', 'r');
    scatter(fixedPts(saddleIDx, 1), fixedPts(saddleIDx, 2), ...
        'filled', 'g');
    hold off
    
    if (tidx == 1), set(fig, 'outerposition', [0 0 1 1]); end
    
    titleString = sprintf('T = %0.3f, Max Error = %0.3e\n', ...
        t, max(densityErr(:)));
    title(titleString, 'Color', 'w');
    xlabel('x'); ylabel('y');
    
    set(gca, 'YDir', 'normal');
    set(gca, 'XColor', 'w');
    set(gca, 'YColor', 'w');
    set(gca, 'Clim', errorRange);
    axis equal tight
    
    colorbar('Color', 'w')
    
    drawnow
    
end